"""Save full LIBERO camera calibration and 8x4 Motion labels without copying RGB.

Each source model and every joint pose is checked. One three-frame raster check
per BDDL scene is shared only after checking identical compiled camera matrices.
"""
import argparse
from collections import Counter
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
import time

import numpy as np

from openpi.policies.eskill_dataset import ESkillDataset
from scripts.preprocess.artifacts import digest
from scripts.preprocess.export import write_index
from .motion_2d import CHECK, OFFSETS, build_episode, open_scene, read, save

DATA = Path('/data1/yuankai/robot-research/data/atomicdata/libero40')


def verify(index, records):
    """Read the exported calibration through the training Dataset at every frame."""
    dataset = ESkillDataset(index, action_horizon=50, skill_names=[], image_size=224)
    frames = 0
    maximum = 0.
    fields = dict(trace='future_model_xy', trace_valid='future_valid',
                  future_observed='future_observed', future_padded='future_padded',
                  future_frame_indices='future_frame_indices',
                  future_source_frame_indices='future_source_frame_indices')
    by_id = {r['episode_id']: r for r in records}
    for i, meta in enumerate(dataset.metadata):
        reader = dataset.reader(i)
        record = by_id[meta['episode_id']]
        if digest(record['source_episode']) != record['source_episode_sha256']:
            raise ValueError('Original episode changed during export')
        path = Path(record['training_episode']).parent / record['arrays']
        if digest(path) != record['arrays_sha256']:
            raise ValueError('Exported arrays changed')
        with np.load(path) as archive:
            saved = {k: archive[v] for k, v in fields.items()}
        for frame in range(meta['frame_count']):
            target = dataset._motion(reader, 'single', frame)
            if target is None:
                raise ValueError(f'Missing Motion target: {reader.path}, {frame}')
            for field in fields:
                np.testing.assert_array_equal(target[field], saved[field][frame])
            maximum = max(maximum, float(np.abs(target['trace']-saved['trace'][frame]).max()))
            frames += 1
        if (i+1) % 100 == 0:
            print(f'verified {i+1}/{len(records)} episodes, {frames} frames', flush=True)
    return dict(episodes=len(records), frames=frames, dataset_target_max_difference=maximum,
                source_metadata_unchanged=True, saved_arrays_hashes_match=True,
                all_frame_targets_and_masks_equal=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=DATA/'index.json')
    parser.add_argument('--output', type=Path, default=DATA/'motion_2d')
    parser.add_argument('--report-dir', type=Path, default=CHECK/'calibration_full_v1')
    parser.add_argument('--libero-root', type=Path,
                        default=Path('/data1/yuankai/robot-research/workspace/third_party/libero-native'))
    parser.add_argument('--gpu-device', type=int, default=1)
    parser.add_argument('--annotation-ids', nargs='+', type=int, help='Optional small batch for checking the exporter')
    parser.add_argument('--resume', action='store_true', help='Reuse complete episodes after checking their source hashes')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_path = CHECK/'all_rules_v1/sources.json'
    samples = read(source_path)['samples']
    if args.annotation_ids is not None:
        samples = [s for s in samples if s['annotation_id'] in args.annotation_ids]
        if len(samples) != len(set(args.annotation_ids)):
            raise ValueError('Requested annotation IDs are not available')
    source_index = read(args.index)
    paths = {e['episode_id']: args.index.parent/e['path'] for e in source_index['episodes']}
    manifest = dict(source_index=str(args.index), source_index_sha256=digest(args.index),
                    source_manifest=str(source_path), source_manifest_sha256=digest(source_path),
                    annotation_ids=[s['annotation_id'] for s in samples], offsets=list(OFFSETS),
                    model_size=224, tail_policy='hold_last', camera='image',
                    packages={p: version(p) for p in ('mujoco', 'robosuite', 'numpy')},
                    scripts={str(p): digest(p) for p in (Path(__file__), Path(__file__).with_name('motion_2d.py'),
                        CHECK.parents[4]/'src/openpi/transforms.py',
                        CHECK.parents[4]/'src/openpi/policies/eskill_dataset.py')})
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError('Resume requires unchanged sources and preprocessing code')
    save(manifest_path, manifest)
    records = []
    scenes = sorted(samples, key=lambda s: (s['bddl'], s['annotation_id']))
    for _, group in groupby(scenes, key=lambda s: s['bddl']):
        env, reference = None, None
        try:
            for sample in group:
                path = args.output/'episodes'/f"{sample['annotation_id']:04d}"/'motion.json'
                if args.resume and path.exists():
                    record = read(path)
                    if (record['source_episode_sha256'] != digest(paths[sample['key']])
                            or record['arrays_sha256'] != digest(path.parent/record['arrays'])
                            or record['bddl_sha256'] != digest(sample['bddl'])
                            or record['audit_sha256'] != digest(record['audit'])):
                        raise ValueError(f'Saved episode no longer matches: {path}')
                else:
                    if env is None:
                        env = open_scene(sample, args.libero_root, args.gpu_device)
                    path = build_episode(sample, paths[sample['key']], args.output, OFFSETS, 224,
                                         args.libero_root, args.gpu_device, native_env=env, raster_reference=reference)
                    record = read(path)
                if reference is None:
                    reference = path
                records.append(record)
                progress = dict(completed=len(records), total=len(samples), frames=sum(r['frames'] for r in records),
                                last_annotation_id=sample['annotation_id'], elapsed_s=round(time.monotonic()-started, 1))
                save(args.report_dir/'progress.json', progress)
                if len(records) % 25 == 0 or len(records) == len(samples):
                    print(progress, flush=True)
        finally:
            if env is not None:
                env.close()
    records.sort(key=lambda r: r['annotation_id'])
    index = args.output/'index.json'
    write_index(index, [r['training_episode'] for r in records])
    verification = verify(index, records)
    save(args.report_dir/'verification.json', verification)
    coverage_fields = ['valid_by_offset', 'observed_valid_by_offset', 'padded_by_offset', 'invalid_by_offset']
    summary = dict(episodes=len(records), frames=sum(r['frames'] for r in records),
        tasks=len({r['bddl'] for r in records}), suites=dict(Counter(r['suite'] for r in records)),
        training_index=str(index.resolve()), training_index_sha256=digest(index),
        offsets=list(OFFSETS), tail_policy='hold_last', model_size=224,
        coverage={k: np.sum([r['coverage'][k] for r in records], axis=0).tolist() for k in coverage_fields},
        eef_valid=sum(r['coverage']['eef_valid'] for r in records),
        eef_fk_max_m=max(r['checks']['eef_fk_max_m'] for r in records),
        eef_fk_projection_max_px=max(r['checks']['eef_fk_projection_max_px'] for r in records),
        camera_matrix_max_change=max(r['checks']['static_camera_matrix_max_change'] for r in records),
        raster_checked_episodes=sum(bool(r['checks']['marker_checks']) for r in records),
        raster_checked_frames=sum(len(r['checks']['marker_checks']) for r in records),
        marker_max_error_px=max(c['error_px'] for r in records for c in r['checks']['marker_checks']),
        existing_warning_episodes=sum(bool(r['existing_warnings']) for r in records),
        verification=verification, elapsed_s=round(time.monotonic()-started, 1), manual_review='pending')
    save(args.output/'report.json', summary)
    save(args.report_dir/'report.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
