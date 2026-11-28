"""Replay matched LIBERO states once and save observation-clock intervals for text relabeling."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from importlib.util import find_spec
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import xml.etree.ElementTree as ET

import numpy as np

from scripts.preprocess.artifacts import digest


def ranges(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2).tolist()


def state_windows(pick, place, valid, command, touch, source_contact, both_pads, destination_contact):
    frames = np.arange(len(valid))
    picking = (frames >= pick['start_frame']) & (frames < pick['stop_frame'])
    placing = (frames >= place['start_frame']) & (frames < place['stop_frame'])
    hits = np.flatnonzero(picking & source_contact)
    first_close = np.flatnonzero(picking & ~(command < 0))
    pre = np.zeros_like(valid, dtype=bool)
    if len(hits):
        stop = min(hits[0] - 2, first_close[0] if len(first_close) else pick['stop_frame'])
        pre = picking & valid & (frames < stop) & ~touch & (command < 0)
    held = both_pads & (command > 0) & valid
    carrying = placing & held & np.r_[False, held[:-1]]
    end = np.flatnonzero(placing & (destination_contact | ~(command > 0)))
    if len(end):
        carrying[end[0]:] = False
    return ranges(pre), ranges(carrying)


def contacts(model, states, entities, source, destination):
    import mujoco

    names = [model.geom(i).name for i in range(model.ngeom)]
    owners = {i: key for i, name in enumerate(names) for key, e in entities.items()
              if e['kind'] in ('object', 'fixture') and name.startswith(key + '_')}
    if not {source, destination} <= set(owners.values()):
        raise ValueError(f'Missing contact geometry for {source} or {destination}')
    gripper = {i for i, name in enumerate(names) if name.startswith('gripper0_') and 'visual' not in name}
    pads = {model.geom(f'gripper0_finger{i}_pad_collision').id for i in (1, 2)}
    data = mujoco.MjData(model)
    flags = np.zeros((4, len(states)), bool)
    at_destination = {k: np.zeros(len(states), bool) for k, e in entities.items()
                      if e['kind'] == 'object' and k not in (source, destination)}
    for row, state in enumerate(states):
        data.time = state[0]
        data.qpos[:] = state[1:1 + model.nq]
        data.qvel[:] = state[1 + model.nq:]
        mujoco.mj_forward(model, data)
        touching_pads = set()
        foreign = False
        payload_contact = False
        for c in data.contact:
            if c.dist > 0:
                continue
            i, j = c.geom1, c.geom2
            a, b = owners.get(i), owners.get(j)
            if a == destination or b == destination:
                other = b if a == destination else a
                if other == source:
                    flags[3, row] = True
                if other in at_destination:
                    at_destination[other][row] = True
            if ((a == source and b != source and j not in gripper)
                    or (b == source and a != source and i not in gripper)):
                payload_contact = True  # Still supported, or already touching a potential new goal.
            if j in gripper:
                i, b = j, a
            if i in gripper and b is not None:
                flags[0, row] = True
                if b == source:
                    flags[1, row] = True
                    touching_pads.add(i)
                else:
                    foreign = True
                    if b == destination:
                        flags[3, row] = True
        flags[2, row] = pads <= touching_pads and not foreign and not payload_contact
    return flags, at_destination


def load_model(file, assets):
    import mujoco

    path = Path(file['path'])
    if digest(path) != file['sha256']:
        raise ValueError(f'XML checksum mismatch: {path}')
    tree = ET.parse(path)
    robot_assets = Path(find_spec('robosuite').origin).parent / 'models/assets'
    for node in tree.findall('asset/*'):
        name = node.get('file')
        if name:
            root = robot_assets if 'robosuite/models/assets/' in name else assets
            node.set('file', str((root / name.split('/assets/')[-1]).resolve(strict=True)))
    return mujoco.MjModel.from_xml_string(ET.tostring(tree.getroot(), encoding='unicode'))


def replay(paths, *, assets):
    model = None
    rows = {}
    for path in paths:
        m = json.loads(path.read_text())
        match = m['interaction_annotation']['scene_match']
        if model is None:
            model = load_model(match['model_file'], assets)
        excerpt = Path(match['excerpt_path'])
        if digest(excerpt) != match['excerpt_sha256']:
            raise ValueError(f'State checksum mismatch: {excerpt}')
        with np.load(excerpt) as archive:
            states = archive['states']
        n = m['frame_count']
        if states.shape != (n, 1 + model.nq + model.nv):
            raise ValueError(f'{path}: state shape disagrees with the scene')
        pick, place = m['events']
        source, dest = (e['interaction_target']['entity_id'] for e in (pick, place))
        actions, camera = m['actions'], m['cameras']['image']
        def array(ref):
            with np.load(path.parent / ref['uri']) as archive:
                return archive[ref['key']]
        frames = array(camera['frame_indices'])
        if not np.array_equal(frames, np.arange(n)):
            raise ValueError(f'{path}: expected dense native camera frames')
        channel = actions['gripper_convention']['channels']['single']
        values = np.sign(array(actions['values'])[:, channel]) * actions['gripper_convention']['close_sign']
        pairs = array(actions['conditioning_image_indices']['image'])
        present = array(actions['valid'])[:, channel]
        command = np.full(n, np.nan)
        for row in np.unique(pairs[(pairs >= 0) & (pairs < n)]):
            use = pairs == row
            v = values[use]
            if present[use].all() and np.all(v == v[0]):
                command[row] = v[0]
        flags, at_destination = contacts(model, states, m['scene_entities'], source, dest)
        pre, held = state_windows(pick, place, array(camera['valid']), command, *flags)
        pre_frames = np.zeros(n, bool)
        for start, stop in pre:
            pre_frames[start:stop] = True
        blocked = {key: ranges(mask & pre_frames) for key, mask in at_destination.items() if (mask & pre_frames).any()}
        rows[m['episode_id']] = dict(source=source, destination=dest, task=m['instructions'][0],
                                     pre_grasp=pre, holding=held, sources_touching_destination=blocked)
    return rows


def prepare(index, output, assets, workers):
    if output.resolve() == index.resolve():
        raise ValueError('Output must be a separate file from the dataset index')
    groups = defaultdict(list)
    episodes = {}
    for row in json.loads(index.read_text())['episodes']:
        path = index.parent / row['path']
        m = json.loads(path.read_text())
        eid = row['episode_id']
        episodes[eid] = None  # Explicitly covered, with no eligible windows.
        events = m['events']
        if not eid.startswith(('libero_object_', 'libero_goal_')) or [e['skill'] for e in events] != ['Pick', 'Place']:
            continue
        if not all(e['arm'] == 'single' and e['boundary_valid'] and e['skill_valid']
                   and e['interaction_target']['status'] == 'resolved' for e in events):
            continue
        source, dest = (e['interaction_target']['entity_id'] for e in events)
        match = m['interaction_annotation']['scene_match']
        if (events[0]['stop_frame'] > events[1]['start_frame'] or source == dest
                or any(e['source_annotation']['instance'] != source for e in events)
                or m['scene_entities'][dest]['kind'] not in ('object', 'fixture')
                or match.get('status') != 'actions_exact_unique_match'):
            continue
        groups[match['model_file']['sha256']].append(path)
    job = partial(replay, assets=assets)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(job, groups.values()):
            episodes.update(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile('w', dir=output.parent, delete=False) as stream:
        json.dump(dict(version=2, index_sha256=digest(index), episodes=episodes), stream, separators=(',', ':'))
        stream.write('\n')
    Path(stream.name).replace(output)
    print(f'{sum(v is not None for v in episodes.values())} replayed / {len(episodes)} indexed episodes: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--libero-assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    prepare(args.index, args.output, args.libero_assets, args.workers)
