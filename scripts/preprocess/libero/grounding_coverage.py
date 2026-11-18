"""Measure online augmentation coverage by task through the actual GroundingDataset."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import time

import numpy as np

from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_dataset
from .grounding_review import record_run


FLAGS = ('target_swapped', 'copy_attempted', 'copy_applied', 'target_moved', 'augmentation_invalidated')


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
                for key in ('copy_budget', 'copy_count', 'copy_task_source_eligible',
                            'copy_task_source_selected', 'copied_task_source'):
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
              'copy_budget', 'copy_count', 'copy_task_source_eligible', 'copy_task_source_selected', 'copied_task_source',
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
    parser.add_argument('--samples-per-task', type=int, default=200)
    parser.add_argument('--subset', choices=('train', 'val'), default='train')
    args = parser.parse_args()
    if args.samples_per_task < 1:
        parser.error('--samples-per-task must be positive')
    audit(TrainConfig.load(args.config), args.output, samples_per_task=args.samples_per_task, subset=args.subset)


if __name__ == '__main__':
    main()
