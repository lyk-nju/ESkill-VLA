"""Find the start of gripper contact for LIBERO Pick events in matched native states."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts.preprocess.artifacts import digest


CONTACT_MARGIN = 2
STOP_FIELD = 'copy_precontact_stop_frame'


def annotate_episode(meta, annotations):
    for event in meta['events']:
        if event['skill'] != 'Pick':
            continue
        event.pop(STOP_FIELD, None)
        row = annotations.get(event['event_id'])
        if row is not None and row[STOP_FIELD] > event['start_frame']:
            event[STOP_FIELD] = row[STOP_FIELD]


def pick_windows(env, path, meta, picks):
    sim = env.sim
    match = meta['interaction_annotation']['scene_match']
    excerpt = Path(match['excerpt_path'])
    if digest(excerpt) != match['excerpt_sha256']:
        raise ValueError(f'{path}: source state digest disagrees')
    with np.load(excerpt) as archive:
        states = archive['states']
    if states.shape != (meta['frame_count'], len(sim.get_state().flatten())):
        raise ValueError(f'{path}: source state dimensions disagree')

    objects = env.env.objects_dict
    gripper = {sim.model.geom_name2id(name) for name in env.robots[0].gripper.contact_geoms}
    windows = {}
    for event, target in picks:
        target_geoms = {sim.model.geom_name2id(name) for name in objects[target].contact_geoms}
        start, stop = event['start_frame'], event['stop_frame']
        first = None
        for frame in range(max(0, start - CONTACT_MARGIN), stop):
            sim.set_state_from_flattened(states[frame])
            sim.forward()
            pairs = ((contact.geom1, contact.geom2) for contact in sim.data.contact[:sim.data.ncon])
            if any((a in gripper and b in target_geoms) or (b in gripper and a in target_geoms)
                   for a, b in pairs):
                first = frame
                break
        # Without observed contact there is no confirmed end of the approach.
        windows[event['event_id']] = dict(target=target, first_contact_frame=first,
                                         **{STOP_FIELD: start if first is None else max(start, first - CONTACT_MARGIN)})
    return windows


def prepare(index, *, episode_ids=None):
    from libero.libero.envs.env_wrapper import ControlEnv

    grouped = defaultdict(list)
    reports = {}
    for entry in json.loads(index.read_text())['episodes']:
        episode = entry['episode_id']
        if episode_ids is not None and episode not in episode_ids:
            continue
        path = index.parent / entry['path']
        meta = json.loads(path.read_text())
        picks = [(e, e.get('interaction_target', {}).get('entity_id')) for e in meta['events']
                 if e['skill'] == 'Pick' and e['boundary_valid']]
        picks = [(e, target) for e, target in picks if meta['scene_entities'].get(target, {}).get('kind') == 'object']
        annotation = meta.get('interaction_annotation', {})
        match = annotation.get('scene_match', {})
        if not picks or match.get('status') != 'actions_exact_unique_match':
            reports[episode] = dict(path=entry['path'], status='no_matched_pick', events={})
            continue
        grouped[annotation['bddl']].append((entry, path, meta, picks))

    for bddl, episodes in sorted(grouped.items()):
        env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False, has_offscreen_renderer=False, hard_reset=False)
        try:
            env.seed(42)
            env.reset()
            for entry, path, meta, picks in episodes:
                result = pick_windows(env, path, meta, picks)
                reports[entry['episode_id']] = dict(path=entry['path'], status='checked', events=result)
        finally:
            env.close()
        print(f'{Path(bddl).stem}: {len(episodes)} episodes', flush=True)
    return dict(contact_margin_frames=CONTACT_MARGIN, episodes=reports)


def write_annotations(index, report):
    for row in report['episodes'].values():
        if row['status'] != 'checked':
            continue
        path = index.parent / row['path']
        meta = json.loads(path.read_text())
        annotate_episode(meta, row['events'])
        temporary = path.with_suffix('.json.precontact.tmp')
        temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Write the offline contact report here')
    parser.add_argument('--episode-csv', type=Path, help='Optional CSV with an episode_id column')
    parser.add_argument('--write', action='store_true', help='Also save the approach windows in Episode metadata')
    args = parser.parse_args()
    episode_ids = None
    if args.episode_csv:
        with args.episode_csv.open() as stream:
            episode_ids = {row['episode_id'] for row in csv.DictReader(stream)}
    report = prepare(args.index, episode_ids=episode_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    if args.write:
        write_annotations(args.index, report)


if __name__ == '__main__':
    main()
