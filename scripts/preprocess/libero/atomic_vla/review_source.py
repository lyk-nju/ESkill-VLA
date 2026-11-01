"""Prepare a fixed 100-episode cohort and match bounded scene-state excerpts."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import re
import time

import numpy as np

from .source import source_data, rgb

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"
OUT = ROOT / 'check/test/review100_v1'
RAW = Path('/data1/yuankai/robot-research/data/external/atomic_vla/raw/6ce6aaaaabdb')
CACHE = RAW.parents[1] / 'review100_v1'
MIRROR = 'clip-rt/modified_libero_hdf5'
MIRROR_REV = '6a6659f8ac7d580fd594173a0e3abf880c843130'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def download(item, revision):
    import requests
    path = RAW / item['rfilename']
    if path.exists() and sha(path) == item['lfs']['sha256']:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name+'.download')
    url = f"https://huggingface.co/datasets/openvla/modified_libero_rlds/resolve/{revision}/{item['rfilename']}?download=true"
    with requests.get(url, stream=True, timeout=(30, 60)) as response:
        response.raise_for_status()
        with partial.open('wb') as stream:
            for block in response.iter_content(1024**2):
                stream.write(block)
    if sha(partial) != item['lfs']['sha256']:
        raise ValueError(f'Source download hash mismatch: {path}')
    partial.replace(path)
    print(json.dumps({'downloaded': item['rfilename'], 'bytes': path.stat().st_size}), flush=True)


def prepare():
    from scripts.preprocess.libero.check_raw import atomic_positions, arrays, metadata, normalized, SUITES
    from scripts.preprocess.libero.rlds import records
    if (OUT / 'manifest.json').exists():
        print('Fixed manifest already exists; retaining it', flush=True)
        return
    repository = json.loads((RAW / 'repository.json').read_text())
    original = json.loads((ROOT / 'check/raw/manifest.json').read_text())
    existing = {f['path'] for f in original['files']}
    rng = random.Random(42)
    chosen_files = []
    for suite in SUITES:
        items = sorted((x for x in repository['siblings'] if x['rfilename'].startswith(suite+'/') and '.tfrecord-' in x['rfilename']), key=lambda x:x['rfilename'])
        chosen_files += [x for x in items if x['rfilename'] in existing]
        chosen_files += rng.sample([x for x in items if x['rfilename'] not in existing], 2)
    additional = sum(x['size'] for x in chosen_files if not (RAW / x['rfilename']).exists())
    if additional > 2*1024**3:
        raise RuntimeError('Selected source shards exceed the declared download budget')
    save(OUT / 'source_shards.json', {'revision': repository['sha'], 'seed': 42, 'files': chosen_files})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda item: download(item, repository['sha']), chosen_files))
    positions = atomic_positions(RAW)
    annotations = json.loads((RAW / 'libero_lerobot.json').read_text())
    population, hashes = [], []
    for item in sorted(chosen_files, key=lambda x:x['rfilename']):
        relative = item['rfilename']
        digest = sha(RAW / relative)
        if digest != item['lfs']['sha256']:
            raise ValueError('Source shard changed')
        hashes.append({'path': relative, 'sha256': digest, 'bytes': item['size']})
        suite = relative.split('/')[0]
        shard = int(re.search(r'tfrecord-(\d+)-', relative)[1])
        for row, offset, data in records(RAW / relative):
            state, action, joints = arrays(data)
            task = data['steps/language_instruction'][0].decode()
            number = positions[suite, shard, row]
            label = annotations.get(str(number))
            linked = label is not None and {normalized(s['action_name']) for s in label['segments']} == {normalized(task)} and label['all_frames'] == len(state)
            fingerprint = hashlib.sha256(state.tobytes()+action.tobytes())
            for value in data['steps/observation/image']:
                fingerprint.update(value)
            population.append({'key': f'{suite}_s{shard:03d}_r{row:03d}', 'suite': suite,
                'shard': relative, 'record_index': row, 'byte_offset': offset, 'frames': len(state),
                'task': task, 'metadata': metadata(data), 'annotation_candidate_id': number,
                'annotation': label if linked else None, 'annotation_link': 'task_and_exact_length_match' if linked else 'unlinked',
                'main_state_action_sha256': fingerprint.hexdigest(), 'actions_sha256': hashlib.sha256(action.tobytes()).hexdigest()})
        print(json.dumps({'scanned_shard': relative, 'candidates': len(population)}), flush=True)
    # Include the previous ten unchanged, then balance each suite by task exposure.
    by_key = {s['key']: s for s in population}
    selected = [by_key[s['key']] for s in original['samples']]
    used = {s['main_state_action_sha256'] for s in selected}
    for suite in SUITES:
        task_counts = Counter(s['task'] for s in selected if s['suite'] == suite)
        while sum(s['suite'] == suite for s in selected) < 25:
            pool = [s for s in population if s['suite'] == suite and s['main_state_action_sha256'] not in used]
            if not pool:
                raise ValueError(f'Not enough independent candidates for {suite}')
            tasks = sorted({s['task'] for s in pool})
            minimum = min(task_counts[t] for t in tasks)
            task = rng.choice([t for t in tasks if task_counts[t] == minimum])
            sample = rng.choice([s for s in pool if s['task'] == task])
            selected.append(sample); used.add(sample['main_state_action_sha256']); task_counts[task] += 1
    if len(selected) != 100 or len(used) != 100:
        raise ValueError('Cohort must contain 100 distinct main RGB/state/action trajectories')
    for number, sample in enumerate(selected, 1):
        sample['number'] = number
        sample['source_cache'] = str(CACHE / 'source' / (sample['key']+'.npz'))
    selected_by_shard = defaultdict(dict)
    for sample in selected:
        selected_by_shard[sample['shard']][sample['record_index']] = sample
    for relative, wanted in selected_by_shard.items():
        for row, _, data in records(RAW / relative):
            if row not in wanted:
                continue
            sample = wanted[row]
            state, actions, joints = arrays(data)
            images = data['steps/observation/image']
            offsets = np.r_[0, np.cumsum([len(v) for v in images])].astype(np.int64)
            path = Path(sample['source_cache']); path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, state=state, actions=actions, joints=joints,
                image_bytes=np.frombuffer(b''.join(images), np.uint8), image_offsets=offsets)
            sample['source_cache_sha256'] = sha(path)
    save(OUT / 'candidate_index.json', {'candidates': population})
    code = [Path(__file__).with_name('bind_instances.py'), Path(__file__).with_name('binding_rules.json'), Path(__file__).parents[1]/'render.py']
    save(OUT / 'manifest.json', {'source': 'atomic-vla', 'repository': repository['id'], 'revision': repository['sha'],
        'raw_root': str(RAW), 'cache_root': str(CACHE), 'seed': 42, 'candidate_records': len(population),
        'candidate_tasks': len({s['task'] for s in population}), 'selected_tasks': len({s['task'] for s in selected}),
        'sampling': 'Previous 10 first; 25 per suite; task-balanced random selection from 3 shards per suite; no binding-success selection',
        'files': hashes, 'samples': selected, 'source_annotation_sha256': sha(RAW/'libero_lerobot.json'),
        'frozen_code': {str(p): sha(p) for p in code}, 'prepare_script_sha256': sha(__file__),
        'playback_fps': 10, 'source_fps': None, 'manual_review': 'pending'})
    print(json.dumps({'selected': len(selected), 'frames': sum(s['frames'] for s in selected),
                      'tasks': len({s['task'] for s in selected}), 'unlinked': sum(s['annotation'] is None for s in selected)}), flush=True)


def match_task(job):
    import h5py
    from .native_object import RangeFile
    relative, item, samples = job
    report_path = OUT / 'scene_matches' / (Path(relative).stem+'.json')
    if report_path.exists():
        return json.loads(report_path.read_text())
    started = time.monotonic()
    url = f'https://huggingface.co/datasets/{MIRROR}/resolve/{MIRROR_REV}/{relative}'
    remote = RangeFile(url, item['size'], budget=128*1024**2)
    source = {s['key']: source_data(s) for s in samples}
    matches, rows, demo_fingerprints = defaultdict(list), [], []
    with h5py.File(remote, 'r') as h:
        for name, group in h['data'].items():
            actions = group['actions'][()].astype(np.float32)
            digest = hashlib.sha256(actions.tobytes()).hexdigest()
            demo_fingerprints.append({'demo': name, 'frames': len(actions), 'actions_sha256': digest})
            for sample in samples:
                if len(actions) == sample['frames'] and np.array_equal(actions, source[sample['key']]['actions']):
                    matches[sample['key']].append(name)
        for sample in samples:
            data = source[sample['key']]
            row = {'number': sample['number'], 'key': sample['key'], 'matches': matches[sample['key']],
                   'status': 'unmatched', 'manual_review': 'pending'}
            if len(matches[sample['key']]) == 1:
                name = matches[sample['key']][0]; group = h['data'][name]
                states = group['states'][()]
                mirror_joints = group['obs/joint_states'][()].astype(np.float32)
                mirror_state = np.concatenate([group['obs/ee_states'][()], group['obs/gripper_states'][()]], axis=1).astype(np.float32)
                frames = sorted({0, sample['frames']//2, sample['frames']-1})
                comparisons = []
                mirror_rgb = []
                for frame in frames:
                    raw = group['obs/agentview_rgb'][frame]
                    mirror_rgb.append(raw)
                    comparisons.append({'frame': frame, 'rgb_mae_after_180_rotation': float(np.abs(raw[::-1,::-1].astype(float)-rgb(data, frame)).mean())})
                path = CACHE / 'scenes' / (sample['key']+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, states=states, mirror_joints=mirror_joints, mirror_state=mirror_state,
                                    mirror_rgb=np.stack(mirror_rgb), frames=np.asarray(frames))
                row.update(status='actions_exact_unique_match', demo=name, excerpt_path=str(path), excerpt_sha256=sha(path),
                    max_joint_difference=float(np.abs(mirror_joints-data['joints']).max()),
                    max_eef_position_difference_m=float(np.linalg.norm(mirror_state[:,:3]-data['state'][:,:3],axis=1).max()),
                    rgb_comparisons=comparisons, states_shape=list(states.shape))
            rows.append(row)
    remote.session.close()
    result = {'mirror': MIRROR, 'revision': MIRROR_REV, 'remote_file': relative, 'remote_size': item['size'],
        'remote_lfs_sha256': item['lfs']['sha256'], 'full_remote_hash_verified': False,
        'range_bytes': remote.transferred, 'range_requests': remote.requests, 'scanned': demo_fingerprints,
        'samples': rows, 'seconds': round(time.monotonic()-started,3), 'manual_review': 'pending'}
    save(report_path, result)
    return result


def scenes():
    manifest = json.loads((OUT/'manifest.json').read_text())
    repository = json.loads((CACHE/'mirror_repository.json').read_text())
    files = {x['rfilename']: x for x in repository['siblings'] if x['rfilename'].endswith('.hdf5')}
    groups = defaultdict(list)
    for sample in manifest['samples']:
        relative = sample['suite'] + '/' + Path(sample['metadata']['file_path']).name
        groups[relative].append(sample)
    jobs = [(relative, files[relative], samples) for relative, samples in sorted(groups.items())]
    reports = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(match_task, job): job[0] for job in jobs}
        for future in as_completed(futures):
            result = future.result(); reports.append(result)
            print(json.dumps({'scene_task': result['remote_file'], 'samples': len(result['samples']),
                'matched': sum(s['status']=='actions_exact_unique_match' for s in result['samples']),
                'range_mib': round(result['range_bytes']/1024**2,2), 'tasks_complete': len(reports), 'tasks_total': len(jobs)}), flush=True)
    total = sum(r['range_bytes'] for r in reports)
    if total > 6*1024**3:
        raise RuntimeError('Scene excerpt download budget exceeded')
    save(OUT/'scene_summary.json', {'tasks': len(reports), 'range_bytes': total,
        'samples': sorted([s for r in reports for s in r['samples']], key=lambda s:s['number']),
        'manifest_sha256': sha(OUT/'manifest.json'), 'manual_review': 'pending'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','scenes'])
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else scenes()


if __name__ == '__main__':
    main()
