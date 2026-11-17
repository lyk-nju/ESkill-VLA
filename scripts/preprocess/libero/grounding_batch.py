"""Complete native scene labels for LIBERO40, with one isolated renderer per task."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.artifacts import digest
from scripts.preprocess.export import write_index
from .atomic_vla.scene_regions import extract_scene, prepare_online_episode


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def localize_arrays(path, source):
    """Reuse immutable original files with hard links; operational refs become local."""
    meta = json.loads(path.read_text())
    mapping = {}
    for name in ('arrays.npz', 'images.npz'):
        original = (source.parent/name).resolve()
        local = path.parent/name
        if not local.exists():
            os.link(original, local)
        assert os.path.samefile(original, local)
        mapping[original] = name
    def visit(value):
        if isinstance(value, dict):
            if value.get('uri'):
                resolved = (path.parent/value['uri']).resolve()
                if resolved in mapping:
                    value['uri'] = mapping[resolved]
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(meta)
    meta['interaction_annotation']['train_interface'] = 'grounding stage: local scene and visible regions'
    write_json(path, meta)


def check_episode(path, expected, pilot_path):
    reader = EpisodeReader(path)
    meta = reader.meta
    if meta['interaction_annotation']['geometry_errors']:
        raise ValueError(f"Unresolved geometry: {meta['interaction_annotation']['geometry_errors']}")
    pilot = EpisodeReader(pilot_path) if pilot_path.exists() else None
    regions = visible = comparisons = pilot_comparisons = 0
    counts = {}
    for key, entity in meta['scene_entities'].items():
        if entity['kind'] == 'background':
            continue
        seq = entity['regions']['image']
        masks = reader.array(seq['masks'])
        boxes = reader.array(seq['boxes_xyxy'])
        valid = masks.any((1, 2))
        ys, xs = masks.any(2), masks.any(1)
        calculated = np.stack((xs.argmax(1), ys.argmax(1), masks.shape[2]-xs[:, ::-1].argmax(1),
                               masks.shape[1]-ys[:, ::-1].argmax(1)), axis=1).astype(np.float32)
        calculated[~valid] = np.nan
        assert np.array_equal(boxes, calculated, equal_nan=True), (meta['episode_id'], key, 'mask/bbox')
        assert np.array_equal(reader.array(seq['boxes_valid']), valid)
        regions += len(boxes)
        counts[key] = int(valid.sum())
        for frame, box in expected.get(key, []):
            assert np.array_equal(boxes[frame], box), (meta['episode_id'], key, frame, 'sparse reference')
            comparisons += 1
        if pilot and key in pilot.meta['scene_entities']:
            previous = pilot.meta['scene_entities'][key]['regions'].get('image')
            if previous:
                assert np.array_equal(boxes, pilot.array(previous['boxes_xyxy']), equal_nan=True)
                assert np.array_equal(masks, pilot.array(previous['masks']))
                pilot_comparisons += len(boxes)
        if 'visible_masks' in seq:
            seen = reader.array(seq['visible_masks'])
            fraction = seen.sum((1, 2))/np.maximum(masks.sum((1, 2)), 1)
            assert np.isfinite(fraction).all()
            assert np.allclose(fraction, reader.array(entity['visible_fraction']))
            visible += len(seen)
    target_valid = sum(int(reader.array(meta['scene_entities'][e['interaction_target']['entity_id']]
                      ['regions']['image']['boxes_valid'])[e['start_frame']:e['stop_frame']].sum())
                      for e in meta['events'])
    grip = meta['effector_regions']['single']['image']
    source = Path(meta['interaction_annotation']['source_episode'])
    assert digest(source) == meta['interaction_annotation']['source_episode_sha256']
    return dict(episode_id=meta['episode_id'], frames=meta['frame_count'], events=len(meta['events']),
                region_frames=regions, visible_region_frames=visible, entity_valid_frames=counts,
                current_bbox_valid_frames=target_valid, gripper_bbox_valid_frames=int(reader.array(grip['boxes_valid']).sum()),
                sparse_bbox_comparisons=comparisons, pilot_dense_comparisons=pilot_comparisons,
                max_rgb_mae=max(r['rgb_mae'] for r in meta['online_grounding']['visible_alignment']),
                source_unchanged=True, operational_files=['arrays.npz','images.npz','scene_regions.npz','online_grounding.npz'])


def task(config, number, gpu, limit=None):
    sys.path.insert(0, config['libero_root'])
    from libero.libero.envs import OffScreenRenderEnv
    manifest = json.loads(Path(config['inventory']).read_text())
    grouped = defaultdict(list)
    for row in manifest['episodes']:
        grouped[row['bddl']].append(row)
    bddl, rows = list(grouped.items())[number-1]
    if limit:
        rows = rows[:limit]
    expected = defaultdict(lambda: defaultdict(list))
    reference = Path(config['sparse_reference'])/(Path(bddl).stem+'.csv')
    with reference.open() as handle:
        for r in csv.DictReader(handle):
            expected[r['episode_id']][r['instance']].append((int(r['frame']), [float(r[k]) for k in ('x0','y0','x1','y1')]))
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_names=['agentview'], camera_heights=256,
                            camera_widths=256, render_gpu_device_id=gpu, hard_reset=False)
    try:
        env.seed(0)
        env.reset()
        for row in rows:
            start = time.monotonic()
            destination = Path(config['output'])/'episodes'/row['episode_id']
            path, check = destination/'episode.json', destination/'grounding_check.json'
            if check.exists():
                print(f'SKIP {row["episode_id"]}', flush=True)
                continue
            if not path.exists():
                extract_scene(row, manifest['scenes'][bddl], destination, gpu=gpu, include_robot=True, environment=env)
            if 'online_grounding' not in json.loads(path.read_text()):
                prepare_online_episode(path, destination, gpu=gpu, environment=env)
            localize_arrays(path, Path(row['source_episode']))
            pilot = Path(config['pilot'])/'episodes'/row['episode_id']/'episode.json'
            result = check_episode(path, expected[row['episode_id']], pilot)
            result['seconds'] = time.monotonic()-start
            write_json(check, result)
            print(json.dumps(dict(task=number, **{k:result[k] for k in ('episode_id','frames','seconds','max_rgb_mae')})), flush=True)
    finally:
        env.close()


def build(config_path, config):
    root = Path(config['output'])
    (root/'logs').mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(config['inventory']).read_text())
    tasks = list(dict.fromkeys(r['bddl'] for r in manifest['episodes']))
    record = dict(started=datetime.datetime.now().astimezone().isoformat(), pid=os.getpid(), command=sys.argv,
        python=sys.executable, config=config, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        git_status=subprocess.check_output(['git','status','--short'],text=True),
        source_sha256={str(p):digest(p) for p in [Path(__file__),Path(__file__).parent/'atomic_vla/scene_regions.py']},
        inventory_sha256=digest(config['inventory']), model=None, training=False, seed=0,
        gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv'],text=True))
    write_json(root/'run.json', record)
    write_json(root/'config.json', config)
    pending = queue.Queue()
    for i in range(1, len(tasks)+1):
        pending.put(i)
    def worker(gpu):
        while True:
            try:
                number = pending.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable,'-u','-m','scripts.preprocess.libero.grounding_batch','--config',str(config_path),
                   '--task',str(number),'--gpu',str(gpu)]
            with (root/'logs'/f'task_{number:03d}.log').open('a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(f'FINISHED task {number}/{len(tasks)} GPU {gpu}', flush=True)
    with ThreadPoolExecutor(max_workers=len(config['gpus'])) as executor:
        futures = [executor.submit(worker, gpu) for gpu in config['gpus']]
        for future in futures:
            future.result()
    checks, paths = [], []
    for row in manifest['episodes']:
        path = root/'episodes'/row['episode_id']/'episode.json'
        checks.append(json.loads((path.parent/'grounding_check.json').read_text()))
        paths.append(path)
    write_index(root/'index.json', paths)
    summary = dict(episodes=len(paths),tasks=len(tasks),
        **{key:sum(r[key] for r in checks) for key in ('frames','events','region_frames','visible_region_frames',
           'current_bbox_valid_frames','gripper_bbox_valid_frames','sparse_bbox_comparisons','pilot_dense_comparisons')},
        max_rgb_mae=max(r['max_rgb_mae'] for r in checks), source_unchanged=all(r['source_unchanged'] for r in checks),
        finished=datetime.datetime.now().astimezone().isoformat())
    write_json(root/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--task', type=int)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, help='Small renderer check within one task')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.task:
        task(config, args.task, args.gpu, args.limit)
    else:
        build(args.config, config)


if __name__ == '__main__':
    main()
