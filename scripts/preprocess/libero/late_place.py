"""Export held-out, frame-matched late Place observations with verified grasps for Current evaluation."""
import argparse
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from openpi.policies.eskill_dataset import EpisodeReader, planning_event
from openpi.policies.grounding_dataset import visible_pixels
from scripts.preprocess.artifacts import digest, read_json, write_json
from scripts.preprocess.libero.grasp_state import replay


def place_events(meta):
    for event in meta['events']:
        binding, target = event['source_annotation'], event['interaction_target']
        receiver = meta['scene_entities'].get(target.get('entity_id'), {})
        if (event['skill'] == 'Place' and event['boundary_valid'] and event.get('skill_valid', True)
                and binding.get('status') in ('rule_bound', 'manual_bound') and binding['scope']['part'] == 'whole'
                and target['status'] == 'resolved' and receiver.get('kind') in ('object', 'fixture', 'part')
                and binding['instance'] != target['entity_id']):
            yield event


def late_frame(reader, event, contact):
    source = event['source_annotation']['instance']
    target = event['interaction_target']['entity_id']
    if contact['manipulated_object_by_event'].get(event['event_id']) != source:
        return
    start, stop = event['start_frame'], event['stop_frame']
    # Use frame centers in the final quarter and keep all three contact witnesses in this Event.
    first = max(start + 1, start + (3 * (stop - start - 1) + 3) // 4)
    for frame in range(first, stop - 1):
        window = range(frame - 1, frame + 2)
        if not all(planning_event(reader, f) is event and contact['frames'].get(str(f)) for f in window):
            continue
        main, wrist = (reader.rgb(camera, frame) for camera in ('image', 'wrist_image'))
        if main is None or wrist is None:
            continue
        boxes, masks = {}, {}
        for key in (source, target):
            seq = reader.meta['scene_entities'][key]['regions']['image']
            region = reader.region(seq, frame)
            if region is not None and region['bbox'] is not None:
                box = region['bbox']
                if np.all(np.isfinite(box)) and np.all(box[2:] > box[:2]):
                    boxes[key] = box.tolist()
                    mask = visible_pixels(reader, seq, reader.row(seq, frame), main.shape[:2])
                    if mask is not None:
                        masks[key] = mask
        if target in masks and masks[target].any():
            return frame, main, wrist, boxes, masks


def export(index, split, output):
    members = read_json(split)
    train, val = set(members['train']), set(members['val'])
    if train & val:
        raise ValueError('Train and validation parents overlap')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows, selected, eligible = [], set(), set()
    counts = Counter()
    with TemporaryDirectory(dir=output.parent) as staging:
        root = Path(staging) / 'export'
        root.mkdir()
        for entry in sorted(read_json(index)['episodes'], key=lambda e: e['episode_id']):
            reader = EpisodeReader(index.parent / entry['path'])
            meta = reader.meta
            parent = meta['source'].get('parent_trajectory_id') or meta['episode_id']
            if parent not in val:
                continue
            counts['validation_episodes'] += 1
            task = meta['instructions'][0]
            events = list(place_events(meta))
            if not events:
                continue
            eligible.add(task)
            if task in selected:
                continue
            contact = replay(reader.path)
            counts['replayed_episodes'] += 1
            for event in events:
                found = late_frame(reader, event, contact)
                if found is None:
                    continue
                frame, main, wrist, boxes, masks = found
                episode = meta['episode_id']
                key = f'{episode}_{frame}'
                target = event['interaction_target']['entity_id']
                source = event['source_annotation']['instance']
                assets = root / key
                assets.mkdir()
                for camera, rgb in [('main', main), ('wrist', wrist)]:
                    Image.fromarray(rgb).save(assets / f'{camera}.png')
                mask_paths = {}
                for entity, mask in masks.items():
                    path = assets / ('receiver.png' if entity == target else 'source.png')
                    Image.fromarray(mask).save(path)
                    mask_paths[entity] = str(path.relative_to(root))
                witnesses = list(range(frame - 1, frame + 2))
                rows.append(dict(id=key, split='E', scene=episode.rsplit('_r', 1)[0], parent=parent,
                                 episode_id=episode, frame=frame, task=task, case='late_place', skill='Place',
                                 source=source, target=target, image=f'{key}/main.png', wrist_image=f'{key}/wrist.png',
                                 boxes=boxes, entity_masks=mask_paths, current_mask=mask_paths[target],
                                 event_id=event['event_id'], event_interval=[event['start_frame'], event['stop_frame']],
                                 grasp_frames=witnesses, hdf5_rows=[contact['source_to_hdf5_row'][f] for f in witnesses],
                                 provenance=contact['provenance']))
                selected.add(task)
                break
        if not rows:
            raise ValueError('No visible physical receiver with a verified late Place grasp')
        result = dict(protocol='libero40_heldout_late_place', metric_space='source',
                      split_sha256=digest(split), index_sha256=digest(index), rows=rows,
                      selection='First verified frame per held-out task in episode-ID and Event order',
                      counts=dict(counts, eligible_tasks=len(eligible), verified_tasks=len(selected)),
                      tasks_without_verified_frames=sorted(eligible - selected))
        write_json(root / 'manifest.json', result)
        root.rename(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('index', 'split', 'output'):
        parser.add_argument('--' + arg, required=True, type=Path)
    args = parser.parse_args()
    print(export(args.index, args.split, args.output)['counts'])
