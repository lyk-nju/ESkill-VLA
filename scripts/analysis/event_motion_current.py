"""Audit factual LIBERO Event endpoints, Motion slots and legal counterfactual receivers."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
import hashlib
import json
import multiprocessing
from pathlib import Path
import re
import subprocess

import numpy as np
from scipy.ndimage import distance_transform_edt

from openpi.augmentation import GroundingAugmentation
from openpi.policies.eskill_dataset import EpisodeReader, event_motion_targets, planning_event
from openpi.policies.grounding_dataset import (
    GroundingDataset, box_coverage, drawer_transfer, entity_captions, simple_transfer, visible_pixels,
)
from openpi.training.config import TrainConfig, split_episodes
from openpi.transforms import normalized_image_boxes, project_points, resize_image, resize_points

SKILLS = ('Pick', 'Place', 'Open', 'Close', 'Turn', 'Push')
RADII = (0, 5, 10, 15, 20, 30, 40, 60)
QUANTILES = (25, 50, 75, 90, 95)
REGIONS = ('mask', 'box')


def to_list(value):
    return None if value is None else np.asarray(value).astype(float).tolist()


def point_set_distance(point, lo, hi):
    delta = np.maximum(np.maximum(lo - point, point - hi), 0)
    return float(np.linalg.norm(delta, axis=-1).min())


def mask_pixels(mask):
    y, x = np.nonzero(mask > 0)
    return np.stack((x, y), axis=-1)


def region_record(reader, entity, frame, shape, size):
    seq = entity['regions'].get('image', {})
    row = None if not seq or frame is None else reader.row(seq, frame)
    if row is None:
        return None, None, 'missing', None
    reg = reader.region(seq, frame)
    box = reg['bbox'] if reg else None
    geometry = 'native_visible'
    if entity['kind'] == 'region' and not seq.get('visible_masks'):
        geometry = 'projected_region'
        fraction = entity.get('visible_fraction')
        if fraction and reader.array(fraction)[row] <= 0:
            mask = np.zeros(shape, dtype=bool)
        elif reg:
            site = reader.region(seq, frame, include_mask=True)
            mask = site['mask'] if site['mask'] is not None else box_coverage(box, shape)
        else:
            mask = None
    else:
        mask = visible_pixels(reader, seq, row, shape)
    if box is None and mask is not None and mask.any() and entity['kind'] != 'region':
        pixels = mask_pixels(mask)
        box = np.r_[pixels.min(0), pixels.max(0) + 1]
    if box is not None:
        box = normalized_image_boxes(box, shape, size)[0] * size
    if mask is None:
        return None, box, 'missing', geometry
    canvas = resize_image(mask.astype(np.float32), size)
    return canvas, box, 'valid' if canvas.any() else 'empty', geometry


def region_metrics(row, when, point, mask, box):
    for region in REGIONS:
        support = mask_pixels(mask) if region == 'mask' and mask is not None else None
        if region == 'mask':
            exists = support is not None and len(support) > 0
            lo, hi = (support, support + 1) if exists else (None, None)
            center = np.average(support + .5, axis=0,
                                weights=mask[support[:, 1], support[:, 0]]) if exists else None
        else:
            lo, hi = (box[:2], box[2:]) if box is not None else (None, None)
            center = (lo + hi) / 2 if box is not None else None
        row[f'{when}_{region}_center'] = to_list(center)
        if point is not None and center is not None:
            row[f'{when}_{region}_distance'] = point_set_distance(point, lo, hi)
            row[f'{when}_{region}_offset'] = to_list(point - center)


def read_projected_eef(reader, shape):
    seq = reader.meta['effector_pixels']['single']['image']
    coords, valid = resize_points(reader.array(seq['values']), reader.array(seq['valid']), shape, audit.size)
    return {int(frame): xy * audit.size if keep else None
            for frame, xy, keep in zip(reader.array(seq['frame_indices']), coords, valid, strict=True)}


def projection_check(reader, shape):
    meta = reader.meta
    position = meta['effector_positions']['single']
    saved = meta['effector_pixels']['single']['image']
    calibration = meta['cameras']['image']['calibration']
    source_frames = reader.array(position['frame_indices'])
    frames, pos_rows, pixel_rows = np.intersect1d(source_frames, reader.array(saved['frame_indices']),
                                                  assume_unique=True, return_indices=True)
    projected, valid = project_points(
        reader.array(position['values'])[pos_rows], reader.array(calibration['intrinsics']),
        reader.array(calibration['extrinsics']), reader.array(calibration['image_transform']), shape)
    valid &= reader.array(position['valid'])[pos_rows].all(axis=-1)
    stored_valid = reader.array(saved['valid'])[pixel_rows]
    keep = valid & stored_valid
    error = np.linalg.norm(projected[keep] - reader.array(saved['values'])[pixel_rows[keep]], axis=-1)
    return dict(reference=position['reference_point'], compared_frames=len(error), matched_frames=len(frames),
                validity_disagreements=int((valid != stored_valid).sum()),
                max_source_pixel_error=float(error.max()) if len(error) else None)


def event_record(reader, event, cameras, eef, shape, split):
    meta = reader.meta
    scene = meta['scene_entities']
    start, stop, goal = (event[x] for x in ('start_frame', 'stop_frame', 'goal_frame'))
    end_row = np.searchsorted(cameras, stop) - 1
    end = int(cameras[end_row]) if end_row >= 0 and cameras[end_row] >= start else None
    key = (event.get('interaction_target') or {}).get('entity_id')
    binding = (event.get('interaction_target') or {}).get('status') == 'resolved' and key in scene
    entity = scene[key] if binding else None
    start_point, end_point, goal_point = (eef.get(x) if x in cameras else None for x in (start, end, goal))
    row = dict(split=split, episode=meta['episode_id'], event=event['event_id'], skill=event['skill'],
               task=meta['instructions'][0], scene=Path(meta['interaction_annotation']['bddl']).stem.split('_')[0],
               entity=key, category=entity['category'] if entity else None, kind=entity['kind'] if entity else None,
               part=entity['part'] if entity and entity['kind'] == 'part' else None,
               boundary_valid=event['boundary_valid'], skill_valid=event['skill_valid'], binding_valid=binding,
               start=start, stop=stop, end=end, goal=goal, saved_frames=int(np.sum((cameras >= start) & (cameras < stop))),
               start_eef=to_list(start_point), end_eef=to_list(end_point), goal_eef=to_list(goal_point))
    for name, point in (('start', start_point), ('end', end_point), ('goal', goal_point)):
        frame = row[name]
        row[f'{name}_eef_status'] = (
            'no_saved_rgb' if frame is None or frame not in cameras
            else 'missing_projection' if frame not in eef else 'invalid_projection' if point is None else 'valid')
    usable = event['boundary_valid'] and event['skill_valid'] and entity is not None
    for name in ('start', 'end'):
        frame = row[name]
        mask, box, status, geometry = (region_record(reader, entity, frame, shape, audit.size)
                                       if usable and frame in cameras else (None, None, 'missing', None))
        row.update({f'{name}_mask_status': status, f'{name}_geometry': geometry, f'{name}_box': to_list(box)})
        region_metrics(row, name, end_point, mask, box)
    for region in REGIONS:
        begin, finish = (row[f'{name}_{region}_center'] for name in ('start', 'end'))
        row[f'{region}_center_displacement'] = (float(np.linalg.norm(np.subtract(finish, begin)))
                                               if begin is not None and finish is not None else None)
    return row


@lru_cache(maxsize=64)
def initial_relations(path):
    text = Path(path).read_text() if Path(path).is_file() else ''
    return re.findall(r'\(\s*(on|in)\s+([^\s()]+)\s+([^\s()]+)\s*\)',
                      text.partition('(:init')[2].partition('(:goal')[0], re.IGNORECASE)


def relationship(reader, key, alternative):
    scene = reader.meta['scene_entities']
    a, b = (scene[k]['instance'] for k in (key, alternative))
    if a == b:
        return 'same_parent'
    owners = {x.get('site_name'): x['instance'] for x in scene.values() if 'site_name' in x}
    for predicate, item, container in initial_relations(reader.meta['interaction_annotation']['bddl']):
        if {item, owners.get(container, container)} == {a, b}:
            return 'BDDL_initial_' + predicate.lower()
    return 'no_annotated_relation'


def candidate_queries(reader, event, frame, shape, plan, transfer_context):
    """Use the production enumerator with clean, unedited host inputs and train-only goal types."""
    key = plan['entity']
    scene = reader.meta['scene_entities']
    boxes, rows, fractions, locks = {}, {}, {}, {}
    for name, entity in scene.items():
        seq = entity['regions'].get('image')
        if not seq or entity['kind'] not in ('object', 'fixture', 'part', 'region'):
            continue
        row = reader.row(seq, frame)
        if row is None:
            continue
        region = reader.region(seq, frame)
        mask = None if region is not None else visible_pixels(reader, seq, row, shape)
        if region is None and mask is not None and mask.any():
            pixels = mask_pixels(mask)
            region = {'bbox': np.r_[pixels.min(0), pixels.max(0) + 1]}
        if region is None:
            continue
        rows[name], boxes[name] = row, region['bbox']
        if entity.get('visible_fraction'):
            fractions[name] = float(reader.array(entity['visible_fraction'])[row])
        if entity.get('interaction_locked'):
            locks[name] = bool(reader.array(entity['interaction_locked'])[row])
    if key not in boxes:
        return []
    task = plan['task']
    transfer = simple_transfer(task, scene, boxes)
    resolved = transfer or transfer_context
    if resolved is None or resolved[1] != key:
        return []
    def visible(name, *, offset=None):
        return visible_pixels(reader, scene[name]['regions']['image'], rows.get(name), shape)
    # Image values have no effect on the candidate rules with Copy-Paste and Patch Masking disabled.
    sample = audit.dataset._visual_sample(
        reader, frame, event, key, task, transfer, np.zeros((*shape, 3), np.uint8), boxes,
        entity_captions(scene, boxes), None, None, visible, fractions, locks, {'moved_offsets': {}},
        audit.aug, audit.rng, audit.rng, audit.rng, True,
    )
    return sample['review']['paired_queries']


def raster_box(box, size):
    if box is None:
        return None
    t = np.arange(size) + .5
    return ((t >= box[1]) & (t < box[3]))[:, None] & ((t >= box[0]) & (t < box[2]))[None, :]


def overlap_at_radii(a, b):
    if a is None or b is None or not a.any() or not b.any():
        return {}
    d1, d2 = (distance_transform_edt(~x) for x in (a, b))
    metrics = {}
    for radius in RADII:
        first, second = d1 <= radius, d2 <= radius
        inter = np.count_nonzero(first & second)
        area1, area2 = first.sum(), second.sum()
        metrics.update({f'iou_{radius}': float(inter / (area1 + area2 - inter)),
                        f'containment_{radius}': float(inter / min(area1, area2))})
    return metrics


def pair_rows(reader, event, frame, shape, plan, candidates, terminal):
    scene = reader.meta['scene_entities']
    key = plan['entity']
    point = np.asarray(plan['end_eef']) if plan['end_eef'] is not None else None
    correct_mask, correct_box, _, _ = region_record(reader, scene[key], frame, shape, audit.size)
    unique = {(candidate, role): text for text, candidate, role in candidates}
    result = []
    for (candidate, role), text in unique.items():
        row = {k: plan[k] for k in ('split', 'episode', 'event', 'task', 'scene', 'skill', 'category')}
        row.update(frame=frame, at_start=frame == plan['start'], endpoint_eligible=terminal,
                   target=key, candidate=candidate, candidate_category=scene[candidate]['category'],
                   role=role, query=text, relation=relationship(reader, key, candidate))
        if role == 'destination':
            alt_mask, alt_box, _, _ = region_record(reader, scene[candidate], frame, shape, audit.size)
            region_metrics(row, 'target', point, correct_mask, correct_box)
            region_metrics(row, 'candidate', point, alt_mask, alt_box)
            regions = dict(mask=(correct_mask, alt_mask), box=(raster_box(correct_box, audit.size),
                                                               raster_box(alt_box, audit.size)))
            for name, (a, b) in regions.items():
                if a is not None and b is not None:
                    overlap = overlap_at_radii(a > 0, b > 0)
                    row.update({f'{name}_{k}': v for k, v in overlap.items()})
                ca, cb = (row[f'{side}_{name}_center'] for side in ('target', 'candidate'))
                if ca is not None and cb is not None:
                    row[f'{name}_center_distance'] = float(np.linalg.norm(np.subtract(ca, cb)))
        result.append(row)
    return result


def frame_record(reader, row, frame, camera_rows):
    labels, info = event_motion_targets(reader, frame, **audit.motion)
    valid, hold, cross = (labels[k][0] for k in ('trace_valid', 'trace_terminal_hold', 'future_cross_event'))
    sources = info['trace_source_frame_indices'][0]
    end = row['end']
    at_end = valid & (sources == end)
    if at_end.any():
        np.testing.assert_allclose(labels['trace'][0][at_end] * audit.size,
                                   np.tile(row['end_eef'], (at_end.sum(), 1)), atol=1e-5, rtol=0.)
    inside = bool(valid[-1] and row['start'] <= sources[-1] < end)
    cross_trace = valid & (sources >= row['stop'])
    return dict(split=row['split'], episode=row['episode'], event=row['event'], skill=row['skill'], frame=frame,
                remaining_saved_frames=int(camera_rows[end] - camera_rows[frame]), sources=sources.tolist(),
                valid=valid.astype(int).tolist(), hold=hold.astype(int).tolist(),
                cross_observed=cross.astype(int).tolist(), cross_trace=cross_trace.astype(int).tolist(),
                any_trace=bool(valid.any()), endpoint_slots=at_end.astype(int).tolist(),
                endpoint_eligible=bool(at_end.any()), slot8_end=bool(at_end[-1]), slot8_inside=inside,
                slot8_after=bool(cross_trace[-1]), slot8_invalid=not bool(valid[-1]),
                current_mask_status=None, current_box_valid=None, anchor_eef_valid=None)


def audit_episode(job):
    path, split = job
    reader = EpisodeReader(path)
    # Full-scene audits use more arrays per Episode than a single training item.
    reader._raw = lru_cache(maxsize=None)(reader._raw.__wrapped__)
    meta = reader.meta
    cam = meta['cameras']['image']
    shape = (cam['height'], cam['width'])
    cameras = reader.array(cam['frame_indices'])
    saved = cameras[reader.array(cam['valid'])]
    camera_rows = {int(f): i for i, f in enumerate(cameras)}
    eef = read_projected_eef(reader, shape)
    events = [event_record(reader, event, saved, eef, shape, split) for event in meta['events']]
    by_id = {row['event']: row for row in events}
    slot_rows, pairs, invalid = [], [], Counter()
    scene = meta['scene_entities']
    remainder = drawer_transfer(meta['instructions'][0], scene, meta['events'])
    for f in saved[::audit.stride]:
        frame = int(f)
        event = planning_event(reader, frame)
        if event is None:
            invalid['invalid_planning_event'] += 1
            continue
        row = by_id[event['event_id']]
        if not row['skill_valid'] or row['skill'] not in SKILLS:
            invalid['invalid_skill'] += 1
            continue
        slot = frame_record(reader, row, frame, camera_rows)
        slot['anchor_eef_valid'] = eef.get(frame) is not None
        if row['binding_valid']:
            entity = scene[row['entity']]
            mask, box, status, _ = region_record(reader, entity, frame, shape, audit.size)
            slot.update(current_mask_status=status, current_box_valid=box is not None)
        else:
            slot.update(current_mask_status='missing_binding', current_box_valid=False)
        slot_rows.append(slot)
        if row['binding_valid'] and row['skill'] == 'Place':
            candidates = candidate_queries(reader, event, frame, shape, row, remainder)
            if candidates:
                pairs.extend(pair_rows(reader, event, frame, shape, row, candidates, slot['endpoint_eligible']))
    return events, slot_rows, pairs, invalid, meta['source']['revision'], meta['schema_version'], projection_check(reader, shape)


def csv_output(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, list) else v
                             for k, v in row.items()})


def json_output(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


def make_audit(config, jobs, threshold):
    aug = GroundingAugmentation(target_min_visible_fraction=threshold)
    prototypes = {}
    for path, split in jobs:
        if split == 'train':
            meta = json.loads(path.read_text())
            entities = meta['scene_entities']
            annotations = []
            for event in meta['events']:
                binding = event.get('interaction_target') or {}
                entity = entities.get(binding.get('entity_id'), {})
                annotations.append((event['skill'], event['boundary_valid'], event['skill_valid'],
                                    binding.get('status'), entity.get('kind'), entity.get('category')))
            # The destination allowlist in GroundingDataset is a set union over these facts.
            prototypes.setdefault((meta['instructions'][0], tuple(annotations)), meta['episode_id'])
    dataset = GroundingDataset(config.data_index, config, lambda *_: (None, None),
                               episode_ids=set(prototypes.values()), augment=True)
    # Preserve production candidate enumeration but avoid RGB normalization and tokenization.
    dataset.transform.inputs = lambda _: {}
    return argparse.Namespace(dataset=dataset, aug=aug, rng=np.random.default_rng(42), size=config.model.image_size,
                              stride=config.frame_stride, motion=dict(
                                  offsets_frames=config.model.motion_offsets_frames, camera=config.cameras[0],
                                  image_size=config.model.image_size, min_horizon_frames=config.model.motion_min_horizon_frames,
                                  terminal_hold=config.model.motion_terminal_hold))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('examples/libero/train_motion_planner_frozen.json'))
    parser.add_argument('--candidate-config', type=Path, default=Path('examples/libero/train_motion_planner_joint.json'))
    parser.add_argument('--index', type=Path)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    if args.index:
        config.data_index = str(args.index.resolve())
    if args.split:
        config.split_path = str(args.split.resolve())
    global audit
    index = Path(config.data_index)
    splits = split_episodes(index, config.validation_fraction, config.seed, config.split_path)
    membership = {episode: split for split, episodes in splits.items() for episode in episodes}
    joint = TrainConfig.load(args.candidate_config)
    entries = json.loads(index.read_text())['episodes']
    jobs = [(index.parent / e['path'], membership[e['episode_id']]) for e in entries]
    audit = make_audit(config, jobs, joint.grounding_augmentation.target_min_visible_fraction)
    print(f'Scanning {len(jobs)} episodes with {len(audit.dataset.paths)} train allowlist prototypes', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    events, slots, pairs = [], [], []
    invalid, versions = defaultdict(Counter), Counter()
    projections = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as pool:
        for i, (event, slot, pair, bad, revision, schema, projection) in enumerate(
                pool.map(audit_episode, jobs, chunksize=2), 1):
            events.extend(event)
            slots.extend(slot)
            pairs.extend(pair)
            invalid[jobs[i-1][1]].update(bad)
            versions[(revision, schema)] += 1
            projections.append(dict(episode=entries[i-1]['episode_id'], **projection))
            if i % 200 == 0 or i == len(jobs):
                print(f'{i}/{len(jobs)} episodes, {len(pairs)} candidate rows', flush=True)
    provenance = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      script_sha256=sha256(__file__), config_sha256=sha256(args.config),
                      candidate_config_sha256=sha256(args.candidate_config), index=str(index.resolve()),
                      index_sha256=sha256(index), split_sha256=sha256(config.split_path), split_path=config.split_path,
                      data_versions=[dict(revision=r, schema=s, episodes=n) for (r, s), n in versions.items()],
                      data_summary=json.loads((index.parent / 'summary.json').read_text()),
                      split_episodes={k: len(v) for k, v in splits.items()}, timing=audit.motion, frame_stride=audit.stride,
                      min_candidate_visibility=audit.aug.target_min_visible_fraction,
                      train_destination_types={f'{source}|{relation}': sorted(values)
                                               for (source, relation), values in audit.dataset.destination_types.items()},
                      slot_frame_exclusions=invalid, radii_px=RADII,
                      coordinate_units=f'{audit.size}x{audit.size} model canvas pixels; +x right, +y down',
                      projection_check=dict(
                          **{k: sum(p[k] for p in projections)
                             for k in ('compared_frames', 'matched_frames', 'validity_disagreements')},
                          reference_points=sorted({p['reference'] for p in projections}),
                          max_source_pixel_error=max(p['max_source_pixel_error'] or 0 for p in projections)))
    from scripts.analysis.event_motion_current_report import make_report
    make_report(args.output, events, slots, pairs, provenance, audit.motion, index)
    csv_output(args.output / 'events.csv', events)
    csv_output(args.output / 'slots.csv.gz', slots)
    csv_output(args.output / 'candidates.csv.gz', pairs)
    csv_output(args.output / 'projection_check.csv', projections)


if __name__ == '__main__':
    main()
