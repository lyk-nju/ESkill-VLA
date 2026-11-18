"""Measure online augmentation coverage by task through the actual GroundingDataset."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_data_loader, create_dataset
from .grounding_review import record_run


FLAGS = ('target_swapped', 'copy_attempted', 'copy_applied', 'target_moved', 'augmentation_invalidated')
LIBERO_SUITES = ('libero_object', 'libero_goal', 'libero_spatial', 'libero_10')
EPOCH_FIELDS = (
    'frames', 'text_eligible_frames', 'target_swapped', 'valid_swapped_current',
    'copy_attempted', 'copy_applied', 'copy_count', 'moved_one', 'moved_two', 'moved_three',
    'copy_priority_target_eligible', 'copy_priority_target_selected', 'copy_priority_target_moved',
    'priority_move_kept_after_text', 'target_moved', 'mask_erase_frames', 'target_moved_by_mask_erase',
    'copy_only', 'text_only', 'text_and_copy',
)
_epoch_dataset = None


def _count_episode(job):
    suite, task, episode_id, keys = job
    counts = Counter(dict.fromkeys(EPOCH_FIELDS, 0))
    reasons = Counter()
    for key in keys:
        metadata = _epoch_dataset[key]['metadata']
        info = metadata['augmentation']
        swapped, copied = info['target_swapped'], info['copy_applied']
        moved = info['copy_count']
        counts.update(frames=1, text_eligible_frames=int(info['swap_candidate_count'] > 0),
                      target_swapped=int(swapped), valid_swapped_current=int(swapped and info['current_valid']),
                      copy_attempted=int(info['copy_attempted']), copy_applied=int(copied), copy_count=moved,
                      moved_one=int(moved == 1), moved_two=int(moved == 2), moved_three=int(moved == 3),
                      copy_priority_target_eligible=int(info['copy_priority_target_eligible']),
                      copy_priority_target_selected=int(info['copy_priority_target_selected']),
                      copy_priority_target_moved=int(info['copy_priority_target_moved']),
                      priority_move_kept_after_text=int(info['copy_priority_target_moved'] and not swapped),
                      target_moved=int(info['target_moved']),
                      mask_erase_frames=int(bool(info['mask_erase_targets'])),
                      target_moved_by_mask_erase=int(metadata['entity_id'] in info['mask_erase_targets']),
                      copy_only=int(copied and not swapped), text_only=int(swapped and not copied),
                      text_and_copy=int(swapped and copied))
        reasons[info['copy_reason']] += 1
    return dict(suite=suite, task=task, episode_id=episode_id, **counts), reasons


def _aggregate_epoch(rows, **labels):
    counts = {key: sum(row[key] for row in rows) for key in EPOCH_FIELDS}
    return dict(**labels, episodes=len(rows), **counts,
                text_swap_rate=counts['target_swapped'] / counts['frames'],
                copy_attempt_rate=counts['copy_attempted'] / counts['frames'],
                copy_rate=counts['copy_applied'] / counts['frames'],
                copy_success_given_attempt=(counts['copy_applied'] / counts['copy_attempted']
                                            if counts['copy_attempted'] else None))


def audit_epoch(config, out, *, tasks_per_suite, workers=1):
    """Replay every train frame of selected tasks with the actual one-rank epoch draw keys."""
    if config.platform != 'libero' or config.training_stage != 'grounding':
        raise ValueError('Epoch coverage requires a LIBERO Grounding recipe')
    if config.frame_stride != 1 or config.skill_sampling:
        raise ValueError('Epoch coverage requires every frame and a sampler without replacement')
    out.mkdir(parents=True, exist_ok=False)
    record_run(config, out, 'LIBERO per-suite coverage on every selected training frame')
    config.save(out / 'config.json')
    started = time.monotonic()
    loader = create_data_loader(config, 'train')
    dataset = loader.dataset
    labels, tasks = {}, defaultdict(set)
    for ep in range(len(dataset.paths)):
        meta = dataset.reader(ep).meta
        episode_id = meta['episode_id']
        suite = episode_id.split('_no_noops_', 1)[0]
        if suite not in LIBERO_SUITES:
            continue
        task = meta['instructions'][0]
        labels[ep] = (suite, task, episode_id)
        tasks[suite].add(task)
    rng = np.random.default_rng(config.seed)
    selected = {suite: sorted(rng.choice(sorted(tasks[suite]), tasks_per_suite, replace=False).tolist())
                for suite in LIBERO_SUITES}
    episodes = {ep for ep, (suite, task, _) in labels.items() if task in selected[suite]}
    keys = defaultdict(list)
    for batch in loader.batch_sampler:
        for key in batch:
            ep, _ = dataset.samples[key[0]]
            if ep in episodes:
                keys[ep].append(key)
    # Read by trajectory for I/O locality, retaining the original draw key of each frame.
    jobs = [(*labels[ep], sorted(keys[ep])) for ep in sorted(episodes)]
    frames = sum(len(job[3]) for job in jobs)
    if frames != sum(ep in episodes for ep, _ in dataset.samples):
        raise RuntimeError('Epoch sampler did not visit all selected frames exactly once')
    print(json.dumps(dict(episodes=len(jobs), frames=frames, selected=selected), ensure_ascii=False), flush=True)
    rows, reasons = [], defaultdict(Counter)

    def collect(results):
        for row, episode_reasons in results:
            rows.append(row)
            reasons[row['suite']].update(episode_reasons)
            if len(rows) % 100 == 0:
                print(f'{len(rows)} / {len(jobs)} episodes', flush=True)

    global _epoch_dataset
    _epoch_dataset = dataset
    try:
        if workers == 1:
            collect(map(_count_episode, jobs))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('fork')) as pool:
                collect(pool.map(_count_episode, jobs))
    finally:
        _epoch_dataset = None
    by_suite = [_aggregate_epoch([row for row in rows if row['suite'] == suite], suite=suite)
                for suite in LIBERO_SUITES]
    by_task = [_aggregate_epoch([row for row in rows if (row['suite'], row['task']) == (suite, task)],
                                suite=suite, task=task)
               for suite in LIBERO_SUITES for task in selected[suite]]
    total = _aggregate_epoch(rows)
    result = dict(subset='train', seed=config.seed, epoch=0, world_size=1, frame_stride=config.frame_stride,
                  workers=workers, tasks_per_suite=tasks_per_suite, selected=selected, total=total,
                  copy_reasons=dict(sum(reasons.values(), Counter())), copy_reasons_by_suite=reasons,
                  by_suite=by_suite, by_task=by_task, seconds=time.monotonic() - started)
    (out / 'coverage.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    for name, data in (('by_suite', by_suite), ('by_task', by_task), ('by_episode', rows)):
        with (out / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    print(json.dumps(total), flush=True)
    return result


def audit(config, out, *, samples_per_task=200, subset='train'):
    out.mkdir(parents=True, exist_ok=False)
    record_run(config, out, 'per-task Dataset augmentation coverage; no model or VAE')
    config.save(out / 'config.json')
    started = time.monotonic()
    dataset = create_dataset(config, subset)
    groups, episode_tasks = defaultdict(list), {}
    for ep in range(len(dataset.paths)):
        episode_tasks[ep] = dataset.reader(ep).meta['instructions'][0]
    for index, (ep, _) in enumerate(dataset.samples):
        groups[episode_tasks[ep]].append(index)
    rng = np.random.default_rng(config.seed)
    reports, totals = [], Counter()
    with (out / 'samples.jsonl').open('w') as log:
        for task, indices in sorted(groups.items()):
            selected = sorted(rng.choice(indices, min(samples_per_task, len(indices)), replace=False).tolist())
            counts, reasons = Counter(), Counter()
            episodes = set()
            for draw, index in enumerate(selected):
                row = dataset[(index, 0, draw)]['metadata']
                info = row['augmentation']
                episodes.add(row['episode_id'])
                counts['samples'] += 1
                for key in FLAGS:
                    counts[key] += int(info[key])
                for key in ('copy_budget', 'copy_count', 'copy_priority_target_eligible',
                            'copy_priority_target_selected', 'copy_priority_target_moved'):
                    if key in info:
                        counts[key] += int(info[key])
                for key in ('locatable_object_count', 'swap_candidate_count', 'copy_candidate_count',
                            'transfer_resolved', 'copy_task_protected', 'missing_interaction_lock_count'):
                    if key in info:
                        counts[key] += int(bool(info[key]))
                reasons[info['copy_reason']] += 1
                log.write(json.dumps(dict(episode_id=row['episode_id'], frame=row['frame'],
                    source_task=task, task=row['task'], entity_id=row['entity_id'], augmentation=info)) + '\n')
            rates = {key + '_rate': counts[key] / counts['samples'] for key in FLAGS}
            report = dict(task=task, dataset_frames=len(indices), sampled_episodes=len(episodes),
                          **counts, **rates, copy_success_given_attempt=(
                              counts['copy_applied'] / counts['copy_attempted'] if counts['copy_attempted'] else None),
                          copy_reasons=dict(reasons))
            reports.append(report)
            totals.update(counts)
            print(json.dumps({key: report[key] for key in ('task', 'samples', 'target_swapped', 'copy_applied', 'target_moved')}), flush=True)
    total_frames = sum(len(values) for values in groups.values())
    summary = dict(counts=dict(totals), sampled_rates={key: totals[key] / totals['samples'] for key in FLAGS},
                   frame_weighted_estimated_rates={key: sum(r[key + '_rate'] * r['dataset_frames']
                       for r in reports) / total_frames for key in FLAGS})
    result = dict(scope='stratified sample, uniform frames without replacement within each task; Dataset phase only, '
                       'before batch geometry/color; no VAE/model; not an exhaustive frame count',
                  subset=subset, augment_enabled=dataset.augment, seed=config.seed, samples_per_task=samples_per_task,
                  tasks=len(groups), dataset_frames=total_frames, summary=summary, by_task=reports,
                  seconds=time.monotonic() - started)
    (out / 'coverage.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    fields = ['task', 'dataset_frames', 'samples', 'sampled_episodes',
              'target_swapped', 'copy_attempted', 'copy_applied', 'target_moved',
              'copy_budget', 'copy_count', 'copy_priority_target_eligible', 'copy_priority_target_selected', 'copy_priority_target_moved',
              'target_swapped_rate', 'copy_applied_rate', 'target_moved_rate', 'copy_success_given_attempt']
    with (out / 'by_task.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(reports)
    print(json.dumps(dict(tasks=len(groups), **summary, seconds=result['seconds'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--samples-per-task', type=int, default=200)
    sampling.add_argument('--tasks-per-suite', type=int, help='Audit every train frame of this many tasks per suite')
    parser.add_argument('--workers', type=int, default=1, help='CPU workers for the per-suite audit')
    parser.add_argument('--subset', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    if args.samples_per_task < 1:
        parser.error('--samples-per-task must be positive')
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = TrainConfig.load(args.config)
    if args.tasks_per_suite is not None:
        if args.subset != 'train' or not 1 <= args.tasks_per_suite <= 10:
            parser.error('--tasks-per-suite requires the train split and a count from 1 to 10')
        audit_epoch(config, args.output, tasks_per_suite=args.tasks_per_suite, workers=args.workers)
    else:
        audit(config, args.output, samples_per_task=args.samples_per_task, subset=args.subset)


if __name__ == '__main__':
    main()
