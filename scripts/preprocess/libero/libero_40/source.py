"""Official LIBERO HDF5 inventory and exact action-to-state correspondence."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
from pathlib import Path
import random
import time

import numpy as np

# Reuse I/O helpers; all remote HDF5 URLs below point to the official source.
atomic = importlib.import_module('scripts.preprocess.libero.atomic_vla.review_source')
RangeFile = importlib.import_module('scripts.preprocess.libero.atomic_vla.native_object').RangeFile
sha, save = atomic.sha, atomic.save
from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/libero-40"
OUT = ROOT/'check'
REPOSITORY = 'yifengzhu-hf/LIBERO-datasets'
CACHE = Path('/data1/yuankai/robot-research/data/external/libero_40')
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
PER_FILE_BUDGET = 144*1024**2


def retained_indices(actions):
    """OpenVLA no-op rule; a stationary gripper change remains an action."""
    kept = []
    for i, action in enumerate(actions):
        stationary = np.linalg.norm(action[:-1]) < 1e-4
        unchanged = not kept or action[-1] == actions[kept[-1], -1]
        if not (stationary and unchanged):
            kept.append(i)
    return np.asarray(kept, dtype=np.int64)


def match_indices(raw, target):
    if raw.shape == target.shape and np.array_equal(raw, target):
        return np.arange(len(raw), dtype=np.int64), 'raw_exact'
    indices = retained_indices(raw)
    if raw[indices].shape == target.shape and np.array_equal(raw[indices], target):
        return indices, 'noop_filtered_exact'
    return None, None


def relative_file(sample):
    return sample['suite'].removesuffix('_no_noops')+'/'+Path(sample['metadata']['file_path']).name


def prepare():
    repository = json.loads((OUT/'official_repository.json').read_text())
    if repository['id'] != REPOSITORY:
        raise ValueError('Inventory is not the official LIBERO repository')
    base = atomic.OUT/'manifest.json'
    samples = json.loads(base.read_text())['samples']
    files = [f for f in repository['siblings'] if f['rfilename'].split('/')[0] in SUITES
             and f['rfilename'].endswith('.hdf5')]
    if Counter(f['rfilename'].split('/')[0] for f in files) != Counter({s:10 for s in SUITES}):
        raise ValueError('Expected ten task files in each of the four suites')
    if {relative_file(s) for s in samples} != {f['rfilename'] for f in files}:
        raise ValueError('The fixed cohort does not cover the same official 40 task files')
    selected = [s for s in samples if s['number'] in {11,44,67}]
    used = {relative_file(s) for s in selected}
    rng = random.Random(42)
    for suite, count in zip(SUITES, (3,2,2,3)):
        pool = [s for s in samples if s['suite'].removesuffix('_no_noops') == suite and s['number'] != 15]
        rng.shuffle(pool)
        for sample in pool:
            if sum(s['suite'].removesuffix('_no_noops') == suite for s in selected) == count:
                break
            if relative_file(sample) not in used:
                selected.append(sample); used.add(relative_file(sample))
    manifest = {'repository':REPOSITORY, 'revision':repository['sha'],
                'official_download_script':'https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/utils/download_utils.py',
                'source_manifest':str(base), 'source_manifest_sha256':sha(base),
                'files':files, 'samples':samples,
                'review_samples':sorted(s['number'] for s in selected),
                'review_selection':'011/044/067 + seed42, 3 spatial / 2 object / 2 goal / 3 libero10; distinct tasks; 015 excluded from manual focus',
                'cache':str(CACHE/repository['sha'][:12]), 'manual_review':'pending'}
    save(OUT/'manifest.json',manifest)
    print(json.dumps({'revision':manifest['revision'],'tasks':len(files),'samples':len(samples),
                      'review_samples':manifest['review_samples'],'full_file_bytes':sum(f['size'] for f in files)}),flush=True)


def manifest():
    value = json.loads((OUT/'manifest.json').read_text())
    if sha(value['source_manifest']) != value['source_manifest_sha256']:
        raise ValueError('The fixed source cohort changed')
    return value


def fetch_task(job):
    import h5py
    item, samples, revision, cache, manifest_hash = job
    relative = item['rfilename']
    destination = OUT/'matches'/relative.replace('/','__').replace('.hdf5','.json')
    if destination.exists():
        previous = json.loads(destination.read_text())
        if previous['revision'] != revision or previous['source_manifest_sha256'] != manifest_hash:
            raise ValueError('Existing excerpt belongs to a different source snapshot')
        return previous
    started = time.monotonic()
    sources = {s['key']:atomic.source_data(s) for s in samples}
    remote = RangeFile(f'https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{relative}',
                       item['size'], budget=PER_FILE_BUDGET, block_size=1024**2)
    matches, scanned, records = defaultdict(list), [], []
    try:
        with h5py.File(remote, 'r') as h:
            for name, group in h['data'].items():
                actions = group['actions'][()].astype(np.float32)
                scanned.append({'demo':name,'frames':len(actions), 'retained_frames':len(retained_indices(actions))})
                for sample in samples:
                    indices, method = match_indices(actions, sources[sample['key']]['actions'])
                    if indices is not None:
                        matches[sample['key']].append((name, indices, method))
            for sample in samples:
                candidates = matches[sample['key']]
                row = {'number':sample['number'],'key':sample['key'],'frames':sample['frames'],
                       'repository':REPOSITORY,'revision':revision,'remote_file':relative,
                       'matches':[c[0] for c in candidates],
                       'status':'unmatched' if not candidates else 'ambiguous_actions',
                       'alignment_status':'not_checked','manual_review':'pending'}
                if len(candidates) == 1:
                    name, indices, method = candidates[0]
                    group = h['data'][name]
                    data = sources[sample['key']]
                    states = group['states'][indices]
                    joints = group['obs/joint_states'][indices].astype(np.float32)
                    state = np.concatenate([group['obs/ee_states'][indices], group['obs/gripper_states'][indices]],axis=1).astype(np.float32)
                    frames = np.asarray(sorted({0,sample['frames']//2,sample['frames']-1}))
                    images = group['obs/agentview_rgb'][indices[frames]]
                    path = Path(cache)/'scenes'/(sample['key']+'.npz')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path, states=states, official_joints=joints, official_state=state,
                                        official_rgb=images, frames=frames, source_to_official_indices=indices)
                    model_xml = group.attrs.get('model_file')
                    model_info = None
                    if model_xml is not None:
                        model_xml = model_xml.decode() if isinstance(model_xml,bytes) else str(model_xml)
                        xml_path = path.with_suffix('.model.txt'); xml_path.write_text(model_xml)
                        model_info = {'path':str(xml_path),'sha256':sha(xml_path),'contains_xml':model_xml.lstrip().startswith('<')}
                    row.update(status='actions_exact_unique_match', demo=name, match_method=method,
                               excerpt_path=str(path), excerpt_sha256=sha(path), states_shape=list(states.shape),
                               official_observation_keys=list(group['obs'].keys()),
                               raw_frames=int(group['actions'].shape[0]),
                               original_image_shape=list(group['obs/agentview_rgb'].shape[1:]),
                               model_file=model_info,
                               max_observed_joint_difference_rad=float(np.abs(joints-data['joints']).max()),
                               max_observed_eef_difference_m=float(np.linalg.norm(state[:,:3]-data['state'][:,:3],axis=1).max()))
                records.append(row)
    finally:
        remote.session.close()
    result = {'repository':REPOSITORY,'revision':revision,'remote_file':relative,
              'remote_bytes':item['size'],'remote_lfs_sha256':item['lfs']['sha256'],
              'full_remote_hash_verified':False,'range_bytes':remote.transferred,'range_requests':remote.requests,
              'range_block_bytes':remote.block_size,
              'scanned':scanned,'samples':records,'source_manifest_sha256':manifest_hash,
              'script_sha256':sha(__file__),'seconds':round(time.monotonic()-started,3),'manual_review':'pending'}
    save(destination,result)
    return result


def fetch(workers, sample_number=None):
    data = manifest()
    groups = defaultdict(list)
    for sample in data['samples']:
        groups[relative_file(sample)].append(sample)
    selected_files = {relative_file(s) for s in data['samples'] if sample_number is None or s['number']==sample_number}
    jobs = [(f,groups[f['rfilename']],data['revision'],data['cache'],data['source_manifest_sha256'])
            for f in data['files'] if f['rfilename'] in selected_files]
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_task,job):job[0]['rfilename'] for job in jobs}
        for done, future in enumerate(as_completed(futures),1):
            name = futures[future]
            try:
                result = future.result()
                print(json.dumps({'done':done,'tasks':len(jobs),'file':name,
                                  'matched':sum(s['status']=='actions_exact_unique_match' for s in result['samples']),
                                  'samples':len(result['samples']),'range_mib':round(result['range_bytes']/1024**2,2),
                                  'seconds':result['seconds']}),flush=True)
            except Exception as error:
                errors.append({'file':name,'error':repr(error)})
                print(json.dumps(errors[-1]),flush=True)
    records = [json.loads(p.read_text()) for p in sorted((OUT/'matches').glob('*.json'))]
    transferred = sum(r['range_bytes'] for r in records)
    if transferred > 6*1024**3:
        raise RuntimeError('Declared transfer budget exceeded')
    summary = {'repository':REPOSITORY,'revision':data['revision'],'tasks':len(records),
               'range_bytes':transferred,'errors':errors,
               'samples':sorted([s for r in records for s in r['samples']],key=lambda s:s['number']),
               'manifest_sha256':sha(OUT/'manifest.json'),'manual_review':'pending'}
    save(OUT/'scene_summary.json',summary)
    if errors:
        raise RuntimeError(f'{len(errors)} task reads failed; completed task reports are retained for retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fetch'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    prepare() if args.stage=='prepare' else fetch(args.workers,args.sample)


if __name__=='__main__':
    main()
