"""Replay matched MuJoCo states and record robosuite's grasp predicate for each Pick/Place object."""
import argparse
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader, planning_event
from scripts.preprocess.artifacts import digest, write_json


def local_model_xml(xml, libero_assets, robosuite_root):
    """Keep the demonstration's XML and resolve the original machine's asset paths."""
    root = ElementTree.fromstring(xml)
    for node in root.findall('./asset/*[@file]'):
        path = Path(node.get('file'))
        if path.is_file():
            continue
        parts = path.parts
        if 'robosuite' in parts:
            i = max(i for i, part in enumerate(parts) if part == 'robosuite')
            local = Path(robosuite_root).joinpath(*parts[i + 1:])
        elif 'assets' in parts:
            i = max(i for i, part in enumerate(parts) if part == 'assets')
            local = Path(libero_assets).joinpath(*parts[i + 1:])
        else:
            raise FileNotFoundError(path)
        if not local.is_file():
            raise FileNotFoundError(local)
        node.set('file', str(local.resolve()))
    return ElementTree.tostring(root, encoding='unicode')


def grasp_frames(reader, env, states):
    meta = reader.meta
    if meta['arm_ids'] != ['single'] or len(env.robots) != 1:
        raise ValueError('Grasp replay requires one LIBERO arm')
    if states.shape != (meta['frame_count'], len(env.sim.get_state().flatten())):
        raise ValueError('Matched states disagree with the Episode timeline or the saved scene')
    frames, objects = {}, {}
    for t, state in enumerate(states):
        event = planning_event(reader, t)
        if not event or event['skill'] not in ('Pick', 'Place') or not event.get('skill_valid', True):
            continue
        binding = event.get('source_annotation', {})
        if binding.get('status') not in ('rule_bound', 'manual_bound') or binding.get('scope', {}).get('part') != 'whole':
            continue
        name = binding['instance']  # Place interaction_target is the receiver, not the transported object.
        obj = env.env.objects_dict[name]
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        frames[str(t)] = bool(env.env._check_grasp(env.robots[0].gripper, obj))
        objects[event['event_id']] = name
    return frames, objects


def replay(episode, *, model_xml=None, bddl=None, excerpt=None):
    import robosuite
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    reader = EpisodeReader(episode)
    annotation = reader.meta['interaction_annotation']
    match = annotation['scene_match']
    if match['status'] != 'actions_exact_unique_match' or match['key'] != reader.meta['episode_id']:
        raise ValueError('Grasp replay needs the unique action-matched source demonstration')
    xml = match['model_file']
    model_xml = Path(model_xml or xml['path'])
    bddl = Path(bddl or annotation['bddl'])
    excerpt = Path(excerpt or match['excerpt_path'])
    if (not xml.get('contains_xml') or digest(model_xml) != xml['sha256']
            or digest(bddl) != annotation['bddl_sha256'] or digest(excerpt) != match['excerpt_sha256']):
        raise ValueError('The saved XML, BDDL or state excerpt does not match the Episode provenance')
    with np.load(excerpt) as archive:
        states = archive['states']
        official = archive['source_to_official_indices']
    if (official.shape != (len(states),) or not np.issubdtype(official.dtype, np.integer)
            or (official < 0).any() or (np.diff(official.astype(np.int64)) <= 0).any()):
        raise ValueError('Expected a monotonic Episode-frame to HDF5 row mapping')
    relocated = local_model_xml(model_xml.read_text(), get_libero_path('assets'), Path(robosuite.__file__).parent)
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, hard_reset=False)
    try:
        env.reset_from_xml_string(relocated)
        frames, objects = grasp_frames(reader, env, states)
    finally:
        env.close()
    if not frames:
        raise ValueError('Episode contains no uniquely bound and valid whole-object Pick/Place frames')
    return dict(episode_id=reader.meta['episode_id'], source='robosuite._check_grasp on frame-matched MuJoCo states',
                provenance=dict(hdf5=match['remote_file'], revision=match['revision'], demo=match['demo'],
                                xml_sha256=xml['sha256'], excerpt_sha256=match['excerpt_sha256'],
                                bddl_sha256=annotation['bddl_sha256']),
                source_to_hdf5_row=official.tolist(), manipulated_object_by_event=objects, frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model-xml', type=Path, help='Local path to the same source XML recorded in the Episode')
    parser.add_argument('--bddl', type=Path, help='Local path to the same task BDDL')
    parser.add_argument('--excerpt', type=Path, help='Local path to the matched state excerpt NPZ')
    args = parser.parse_args()
    write_json(args.output, replay(args.episode, model_xml=args.model_xml, bddl=args.bddl, excerpt=args.excerpt))


if __name__ == '__main__':
    main()
