"""Exercise saved grounding Episodes through DataLoader and optional tiny-model backward."""
import argparse
from collections import Counter
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from openpi.models.model import to_device
from openpi.augmentation import augment_training_batch
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.grounding_dataset import collate_grounding
from openpi.training.config import TrainConfig, WandbConfig, create_model
from openpi.training.data_loader import create_dataset, create_data_loader, ResumableBatchSampler
from .grounding_review import record_run


def overview(index, out):
    """One saved trajectory per task, showing each Event midpoint for two seconds."""
    import av
    from PIL import Image
    from .interaction_review import SceneRenderer, WIDTH, HEIGHT
    from openpi.policies.eskill_dataset import EpisodeReader
    selected = {}
    for row in json.loads(index.read_text())['episodes']:
        path = index.parent/row['path']
        reader = EpisodeReader(path)
        selected.setdefault(reader.meta['interaction_annotation']['bddl'], path)
    previews = out/'previews'
    previews.mkdir()
    records = []
    with av.open(str(out/'all_tasks.mp4'), 'w') as container:
        stream = container.add_stream('libx264', rate=10)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
        stream.options = {'crf':'20','preset':'fast','threads':'2'}
        for number, (task, path) in enumerate(selected.items(), 1):
            renderer = SceneRenderer(path, number)
            for event in renderer.meta['events']:
                frame = (event['start_frame']+event['stop_frame']-1)//2
                image = renderer.frame(frame)
                name = f'{number:02d}_frame_{frame:04d}.jpg'
                Image.fromarray(image).save(previews/name, quality=93)
                records.append(dict(task=task,episode_id=renderer.meta['episode_id'],frame=frame,
                                    target=event['interaction_target']['entity_id'],preview=name))
                for _ in range(20):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                        container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return dict(tasks=len(selected), observations=records, seconds=len(records)*2,
                scope='saved labels at Event midpoints; manual review pending')


def text_coverage(dataset):
    """Audit genuine Event queries on every frame without decoding RGB or applying augmentation."""
    counts, invalid = Counter(), []
    for episode in range(len(dataset.paths)):
        reader = dataset.reader(episode)
        meta, entities = reader.meta, reader.meta['scene_entities']
        sequences = {k: e['regions']['image'] for k,e in entities.items()
                     if e['kind'] in ('object','fixture','part','region') and 'image' in e['regions']}
        boxes = {k:reader.array(s['boxes_xyxy']) for k,s in sequences.items()}
        valid = {k:reader.array(s['boxes_valid']) for k,s in sequences.items()}
        events = reader.array(meta['event_index']['single'])
        for frame in range(meta['frame_count']):
            current = {k:b[frame] for k,b in boxes.items() if valid[k][frame]}
            event = meta['events'][events[frame]] if events[frame] >= 0 else None
            target = event['interaction_target']['entity_id'] if event else None
            reason = ('no_task' if not meta['instructions'][0].strip() else
                      'valid' if target in current else 'no_event' if event is None else 'missing_box')
            counts[reason] += 1
            if reason != 'valid':
                invalid.append(dict(episode_id=meta['episode_id'], frame=frame, target=target, reason=reason))
    return dict(counts=dict(counts), invalid=invalid,
                scope='full task and bound Event target on all source frames; no random target replacement')


def training_loader_check(config, workers):
    """Use the real loader with a bounded epoch so prefetched workers finish normally."""
    loader = create_data_loader(replace(config, num_workers=workers, device='cpu'), 'train')
    loader.batch_sampler.sampler = list(islice(loader.batch_sampler.sampler, 64*config.batch_size))
    iterator = iter(loader)
    selected_rows, durations = [], []
    for _ in range(len(loader)):
        start = time.monotonic()
        batch = next(iterator)
        durations.append(time.monotonic()-start)
        assert torch.isfinite(batch['inputs']['images']).all()
        assert torch.isfinite(batch['targets']['role_images']).all()
        selected_rows.extend(dict(episode_id=r['episode_id'], frame=r['frame']) for r in batch['metadata'])
    assert next(iterator, None) is None
    warmup = min(8, len(durations)//2)
    report = dict(workers=workers, batches=len(durations), samples=len(selected_rows),
        episodes=len({r['episode_id'] for r in selected_rows}), observations=selected_rows,
        batch_seconds=durations, warmup_batches=warmup,
        measured_samples_per_second=(len(selected_rows)-warmup*config.batch_size)/sum(durations[warmup:]),
        timing_scope='CPU input only, excludes dataset creation and warmup batches; shared host')
    del iterator, loader
    return report


def check(config, out, workers, smoke_device=None):
    out.mkdir(parents=True, exist_ok=False)
    record_run(config, out, 'full grounding input check; optional tiny backward; no formal training')
    config.save(out/'config.json')
    torch.set_num_threads(2)
    reports, identities, datasets, skill_samples = {}, {}, {}, {}
    for split in ('train', 'val'):
        dataset = datasets[split] = create_dataset(config, split)
        identities[split] = {dataset.reader(ep).meta['episode_id'] for ep in range(len(dataset.paths))}
        wanted, midpoints = {}, {}
        for ep in range(len(dataset.paths)):
            meta = dataset.reader(ep).meta
            wanted[ep] = {0, meta['frame_count']-1}
            for event in meta['events']:
                mid = (event['start_frame']+event['stop_frame']-1)//2
                wanted[ep].update((event['start_frame'], mid, event['stop_frame']-1))
                midpoints[ep, mid] = event['skill']
        indices = [i for i, pair in enumerate(dataset.samples) if pair[1] in wanted[pair[0]]]
        if split == 'train':
            for i in indices:
                skill = midpoints.get(dataset.samples[i])
                if skill:
                    skill_samples.setdefault(skill, i)
        sampler = ResumableBatchSampler(indices, config.batch_size, drop_last=False)
        sampler.sample_keys = split == 'train'
        first_keys = next(iter(sampler))
        reference = collate_grounding([dataset[key] for key in first_keys])
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_grounding,
                            num_workers=workers, multiprocessing_context='spawn' if workers else None)
        totals, missing, seen = Counter(), [], set()
        begun = time.monotonic()
        with (out/f'{split}_samples.jsonl').open('w') as log:
            for number, batch in enumerate(loader):
                inputs, targets = batch['inputs'], batch['targets']
                if number == 0:
                    for section in ('inputs', 'targets'):
                        for name, expected in reference[section].items():
                            if isinstance(expected, dict):
                                for key, value in expected.items():
                                    torch.testing.assert_close(batch[section][name][key], value, rtol=0, atol=0)
                            else:
                                torch.testing.assert_close(batch[section][name], expected, rtol=0, atol=0)
                    assert batch['metadata'] == reference['metadata']
                for name in ('images',):
                    assert torch.isfinite(inputs[name]).all()
                assert torch.isfinite(targets['role_images']).all() and torch.isfinite(targets['role_weights']).all()
                assert not (targets['role_valid'] & ~inputs['arm_presence'][:, :, None]).any()
                assert inputs['images'].shape[1:] == (1, 3, config.model.image_size, config.model.image_size)
                assert targets['role_images'].shape[1:] == (2, 2, 3, config.model.canvas_size, config.model.canvas_size)
                totals['samples'] += len(batch['metadata'])
                totals['current_valid'] += int(targets['role_valid'][:, 0, 0].sum())
                totals['subject_valid'] += int(targets['role_valid'][:, 0, 1].sum())
                totals['task_input_valid'] += int(inputs['token_mask'].any(1).sum())
                for j, row in enumerate(batch['metadata']):
                    info = row['augmentation']
                    if info['target_swapped']:
                        assert info['target_visible_fraction'] >= config.grounding_augmentation.target_min_visible_fraction
                    if split == 'val':
                        assert not info['target_swapped'] and not info['copy_attempted'] and info['patch_fraction'] == 0
                    for key in ('copy_attempted', 'copy_applied', 'target_swapped', 'augmentation_invalidated'):
                        totals[key] += int(info[key])
                    seen.add(row['episode_id'])
                    log.write(json.dumps(row, ensure_ascii=False)+'\n')
                    if not targets['role_valid'][j, 0].all() or not inputs['token_mask'][j].any():
                        missing.append(dict(episode_id=row['episode_id'], frame=row['frame'], entity_id=row['entity_id'],
                            current_valid=bool(targets['role_valid'][j, 0, 0]),
                            subject_valid=bool(targets['role_valid'][j, 0, 1]),
                            task_input_valid=bool(inputs['token_mask'][j].any())))
        assert seen == identities[split] and totals['samples'] == len(indices)
        reports[split] = dict(episodes=len(seen), dataset_frames=len(dataset), sampled=dict(totals),
            workers=workers, batch_size=config.batch_size, seconds=time.monotonic()-begun,
            multiworker_first_batch_exact=True, invalid_labels=missing)
        print(json.dumps(dict(split=split, **{k:v for k,v in reports[split].items() if k!='invalid_labels'})), flush=True)
    assert not identities['train'] & identities['val']
    report = dict(scope='all episodes, Event starts/midpoints/ends; input correctness, not policy performance',
                  splits_disjoint=True, splits=reports, smoke=None)
    (out/'input_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    coverage = {split:text_coverage(dataset) for split,dataset in datasets.items()}
    (out/'source_label_coverage.json').write_text(json.dumps(coverage, ensure_ascii=False, indent=2)+'\n')
    report['source_text_coverage'] = {split:value['counts'] for split,value in coverage.items()}
    report['training_loader'] = training_loader_check(config, workers)
    (out/'input_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    if smoke_device:
        torch.manual_seed(config.seed)
        tiny = replace(config, model=ESkillConfig.tiny(visual_primitive_version='v3', grounding_subject=True,
                       freeze_vlm=True, image_size=224, patch_size=14, canvas_size=256, role_queries=16),
                       smoke=True, pretrained_path=None, device=smoke_device, batch_size=2, num_workers=0,
                       precision='float32', backbone_dtype='float32', wandb_info=WandbConfig(mode='disabled'))
        tiny.save(out/'tiny_model_config.json')
        model = create_model(tiny).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=tiny.learning_rate,
                                     betas=tiny.adam_betas, weight_decay=tiny.weight_decay)
        chosen = list(skill_samples.items())
        steps = []
        for step in range(8):
            selected = [chosen[(step*2+j) % len(chosen)] for j in range(2)]
            batch = to_device(collate_grounding([datasets['train'][i] for _,i in selected]), smoke_device)
            batch = augment_training_batch(batch, tiny)
            optimizer.zero_grad(set_to_none=True)
            losses = model(batch['inputs'], batch['targets'])
            loss = sum(tiny.loss_weights[k]*v for k,v in losses.items())
            assert torch.isfinite(loss)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), tiny.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            steps.append(dict(step=step+1, source_skills=[k for k,_ in selected], loss=float(loss.detach()),
                              grad_norm=float(grad), losses={k:float(v.detach()) for k,v in losses.items()}))
        report['smoke'] = dict(steps=steps, device=smoke_device, initialization='random tiny backbone; frozen real FLUX VAE',
                              wandb_mode='disabled', full_pi05_training=False)
        (out/'input_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(report['smoke']), flush=True)
    report['overview'] = overview(Path(config.data_index), out)
    (out/'input_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--smoke-device')
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    if args.index:
        config = replace(config, data_index=str(args.index))
    check(config, args.output, args.workers, args.smoke_device)


if __name__ == '__main__':
    main()
