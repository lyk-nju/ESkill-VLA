"""Inspect and render the three public LIBERO releases; never run annotation models.

Run from ESkill-VLA with the existing Pi_05 Python environment:
python -m scripts.preprocess.libero.check_raw prepare|render|verify --source lilo-vla|atomic-vla|cyclevla
"""
import argparse
from collections import Counter
import csv
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import random
import re

import av
import numpy as np
from PIL import Image

from .rlds import records, load_records
from scripts.preprocess.libero.render import Renderer, render_video, WIDTH, HEIGHT
from scripts.preprocess.libero.view import from_rlds

from scripts.preprocess import DATA_ROOT

BASE = DATA_ROOT / "libero"
EXTERNAL = Path('/data1/yuankai/robot-research/data/external')
RELEASES = {
    'lilo-vla': ('lilo_vla', '7d7816a00c98'),
    'atomic-vla': ('atomic_vla', '6ce6aaaaabdb'),
    'cyclevla': ('cyclevla', '88c812cd6a2e'),
}
SUITES = ['libero_10_no_noops', 'libero_goal_no_noops',
          'libero_object_no_noops', 'libero_spatial_no_noops']
FPS = 10


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def text_value(value):
    return value.decode() if isinstance(value, bytes) else value


def metadata(data):
    return {key.split('/', 1)[1]: text_value(value[0]) for key, value in data.items()
            if key.startswith('episode_metadata/')}


def arrays(data):
    n = len(data['steps/observation/image'])
    return (np.asarray(data['steps/observation/state'], np.float32).reshape(n, 8),
            np.asarray(data['steps/action'], np.float32).reshape(n, 7),
            np.asarray(data['steps/observation/joint_state'], np.float32).reshape(n, 7))


def tfds_positions(lengths):
    """Default deterministic TFDS interleave: 16 active files, blocks of 16 records.

    This supplies candidate integer IDs, not proof of an author's historical conversion.
    Every used ID must additionally match the published task text and frame count.
    """
    pending = iter(range(len(lengths)))
    slots = [[i, 0] for i in range(min(16, len(lengths)))]
    for _ in slots:
        next(pending)
    output, number = {}, 0
    while any(slot is not None for slot in slots):
        for index, slot in enumerate(slots):
            if slot is None:
                continue
            shard, position = slot
            for row in range(position, min(position + 16, lengths[shard])):
                output[shard, row] = number
                number += 1
            position += 16
            if position >= lengths[shard]:
                following = next(pending, None)
                slots[index] = [following, 0] if following is not None else None
            else:
                slots[index] = [shard, position]
    return output


def atomic_positions(root):
    offset, result = 0, {}
    for suite in SUITES:
        info = json.loads((root / suite / '1.0.0/dataset_info.json').read_text())
        lengths = list(map(int, info['splits'][0]['shardLengths']))
        for (shard, row), number in tfds_positions(lengths).items():
            result[suite, shard, row] = number + offset
        offset += sum(lengths)
    return result


def normalized(task):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', task.lower().replace('_', ' ')).split())


def prepare(source):
    folder, revision = RELEASES[source]
    root, out = EXTERNAL / folder / 'raw' / revision, BASE / source / 'check/raw'
    if (out / 'manifest.json').exists():
        raise FileExistsError('Keep the existing fixed sample manifest')
    repository = json.loads((root / 'repository.json').read_text())
    files = json.loads((root / 'candidate_shards.json').read_text())['files']
    annotations = json.loads((root / 'libero_lerobot.json').read_text()) if source == 'atomic-vla' else {}
    positions = atomic_positions(root) if annotations else {}
    candidates, schemas, hashes = [], {}, []
    for item in sorted(files, key=lambda entry: entry['rfilename']):
        relative = item['rfilename']
        digest = sha(root / relative)
        if digest != item['lfs']['sha256']:
            raise ValueError('Official SHA256 mismatch: ' + relative)
        hashes.append({'path': relative, 'bytes': item['size'], 'sha256': digest})
        shard = int(re.search(r'tfrecord-(\d+)-', relative)[1])
        count = 0
        for row, offset, data in records(root / relative):
            count += 1
            state, action, joints = arrays(data)
            n, meta = len(state), metadata(data)
            assert all(np.isfinite(a).all() for a in (state, action, joints))
            assert len(data['steps/observation/wrist_image']) == n
            assert len(data['steps/language_instruction']) == n
            fingerprint = hashlib.sha256(state.tobytes() + action.tobytes())
            for image in data['steps/observation/image']:
                fingerprint.update(image)
            task = text_value(data['steps/language_instruction'][0])
            candidate = {'key': f's{shard:03d}_r{row:03d}', 'shard': relative, 'record_index': row,
                         'byte_offset': offset, 'frames': n, 'task': task, 'metadata': meta,
                         'main_state_action_sha256': fingerprint.hexdigest(),
                         'source_first_flags': sum(data['steps/is_first']),
                         'source_last_flags': sum(data['steps/is_last'])}
            if source == 'atomic-vla':
                suite = relative.split('/')[0]
                identity = positions[suite, shard, row]
                label = annotations.get(str(identity))
                texts = {normalized(s['action_name']) for s in label['segments']} if label else set()
                task_matches = texts == {normalized(task)}
                frame_matches = label is not None and label['all_frames'] in (n, n-1)
                candidate.update(key=f'{suite}_s{shard:03d}_r{row:03d}',
                                 annotation_candidate_id=identity,
                                 annotation_task_matches=task_matches, annotation_frames_match=frame_matches,
                                 annotation_link='candidate_task_and_length_match' if task_matches and frame_matches else 'unlinked',
                                 annotation=label if task_matches and frame_matches else None)
                candidate['independent_id'] = candidate['key']
            elif source == 'cyclevla':
                candidate['independent_id'] = re.sub(r'_subtask_\d+$', '', meta['file_path'])
            else:
                candidate['independent_id'] = fingerprint.hexdigest()
            candidates.append(candidate)
            schemas[relative] = {key: len(values) for key, values in data.items()}
        info_path = root / (str(Path(relative).parent / 'dataset_info.json'))
        info = json.loads(info_path.read_text())
        assert count == int(info['splits'][0]['shardLengths'][shard])
    # Deduplicate main-view/state/action variants; no quality-driven replacement.
    unique = {}
    for candidate in candidates:
        unique.setdefault(candidate['independent_id'], candidate)
    population = list(unique.values())
    tasks = sorted({c['task'] for c in population})
    rng = random.Random(42)
    chosen_tasks = rng.sample(tasks, min(10, len(tasks)))
    selected, used = [], set()
    for task in chosen_tasks:
        pool = [c for c in population if c['task'] == task and c['independent_id'] not in used]
        if pool:
            candidate = rng.choice(pool)
            selected.append(candidate)
            used.add(candidate['independent_id'])
    if len(selected) < 10:
        pool = [c for c in population if c['independent_id'] not in used]
        selected.extend(rng.sample(pool, 10-len(selected)))
    assert len({c['independent_id'] for c in selected}) == 10
    for number, sample in enumerate(selected, 1):
        sample['number'] = number
        sample['video'] = f"video/{number:02d}_{sample['key']}_raw.mp4"
    save(out / 'candidate_index.json', {'candidates': candidates, 'schemas': schemas,
                                      'records': len(candidates), 'independent_units': len(population)})
    save(out / 'manifest.json', {'source': source, 'repository': repository['id'],
                                'revision': repository['sha'], 'raw_root': str(root),
                                'requirements_sha256': sha(BASE.parents[1] / 'docs/data.md'),
                                'seed': 42, 'candidate_records': len(candidates),
                                'candidate_independent_units': len(population), 'candidate_tasks': len(tasks),
                                'sampling': 'Random shards; distinct task texts then one record per task; unique parent for CycleVLA and main/state/action dedup for LiLo',
                                'files': hashes, 'samples': selected, 'manual_review': 'pending',
                                'render': {'width': WIDTH, 'height': HEIGHT, 'fps': FPS,
                                           'fps_meaning': 'display only; no source timestamps',
                                           'camera': 'main', 'keep_published_duplicates': True},
                                'annotation_sha256': sha(root / 'libero_lerobot.json') if annotations else None})
    print(json.dumps({'source': source, 'candidates': len(candidates), 'unique': len(population),
                      'tasks': len(tasks), 'selected': len(selected), 'frames': sum(s['frames'] for s in selected),
                      'atomic_linked': sum(c.get('annotation_link', '') != 'unlinked' for c in candidates) if annotations else None}))


def rgb(encoded):
    with Image.open(BytesIO(encoded)) as image:
        return image.convert('RGB').copy()


def draw_frame(source, sample, data, frame):
    """Compatibility wrapper; all source-specific mapping is outside the renderer."""
    return Renderer(from_rlds(source, sample, data)).frame(frame)


def render(source):
    out = BASE / source / 'check/raw'
    manifest = json.loads((out / 'manifest.json').read_text())
    if (manifest['render']['width'], manifest['render']['height']) != (WIDTH, HEIGHT):
        raise ValueError('Historical layout manifest: use scripts.preprocess.libero.render_review for new videos')
    root = Path(manifest['raw_root'])
    for sample, data in load_records(root, manifest['samples']):
        target = out / sample['video']
        render_video(from_rlds(source, sample, data), target)
        print(json.dumps({'source': source, 'sample': sample['number'], 'frames': sample['frames'], 'video': str(target)}), flush=True)


def verify(source):
    out = BASE / source / 'check/raw'
    manifest = json.loads((out / 'manifest.json').read_text())
    root = Path(manifest['raw_root'])
    results = []
    for sample, data in load_records(root, manifest['samples']):
        sizes, source_frames = Counter(), 0
        for camera in ['image', 'wrist_image']:
            for encoded in data['steps/observation/' + camera]:
                image = rgb(encoded)
                sizes[str(image.size)] += 1
                source_frames += 1
        state, action, joints = arrays(data)
        assert all(np.isfinite(a).all() for a in (state, action, joints))
        target = out / sample['video']
        with av.open(str(target)) as container:
            stream = container.streams.video[0]
            decoded = 0
            for frame in container.decode(video=0):
                assert frame.width == manifest['render']['width'] and frame.height == manifest['render']['height']
                assert abs(float(frame.pts * frame.time_base) - decoded/FPS) < 1e-6
                decoded += 1
            assert decoded == sample['frames'] and float(stream.average_rate) == FPS
        repeated = sum(data['steps/observation/image'][i] == data['steps/observation/image'][i-1] and
                       np.array_equal(state[i], state[i-1]) and np.array_equal(action[i], action[i-1])
                       for i in range(1, len(state)))
        progress = data['steps/is_terminal']
        annotation = sample.get('annotation')
        invalid = ([s for s in annotation['segments'] if not 0 <= s['start_frame'] <= s['end_frame'] < len(state)]
                   if annotation else [])
        gaps = (sum(a['end_frame']+1 != b['start_frame'] for a, b in zip(annotation['segments'], annotation['segments'][1:]))
                if annotation else None)
        results.append({'number': sample['number'], 'key': sample['key'], 'frames': decoded,
                        'video': sample['video'], 'video_sha256': sha(target), 'bytes': target.stat().st_size,
                        'source_rgb_decoded': source_frames, 'source_sizes': dict(sizes),
                        'finite_signals': True, 'consecutive_repeated_main_state_action': repeated,
                        'first_flags': sum(data['steps/is_first']), 'last_flags': sum(data['steps/is_last']),
                        'is_terminal_range': [min(progress), max(progress)],
                        'invalid_annotation_intervals': invalid, 'annotation_adjacency_issues': gaps,
                        'max_state_xyz_step': float(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1).max()) if len(state)>1 else 0.0,
                        'manual_review': 'pending'})
    assert len(results) == 10
    save(out / 'verification.json', {'source': source, 'crc_checked': True, 'source_sha256_checked': True,
                                     'frames': sum(r['frames'] for r in results), 'samples': results,
                                     'code_sha256': {name: sha(Path(__file__).parent/name) for name in ['check_raw.py', 'rlds.py']},
                                     'manual_review': 'pending'})
    with (out / 'manual_review.csv').open('x', newline='') as stream:
        fields = ['number', 'key', 'video', 'image', 'time_alignment', 'events', 'regions', 'motion', 'decision', 'notes']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**{k: row[k] for k in ['number', 'key', 'video']}, 'decision': 'pending'})
    items = ''.join(f"<h2>{s['number']:02d}: {html.escape(s['task'])}</h2><video controls preload='none' width='960' src='{s['video']}'></video>" for s in manifest['samples'])
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Raw check</title><h1>'+source+' — manual review pending</h1>'+items)
    print(json.dumps({'source': source, 'videos': len(results), 'frames': sum(r['frames'] for r in results),
                      'bytes': sum(r['bytes'] for r in results), 'status': 'technical_checks_passed_manual_pending'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'render', 'verify'])
    parser.add_argument('--source', choices=RELEASES, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'render': render, 'verify': verify}[args.stage](args.source)


if __name__ == '__main__':
    main()
