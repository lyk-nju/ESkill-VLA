"""Compare Motion clocks and measure eligible Place counterfactual supervision offline."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
import multiprocessing
from pathlib import Path
import subprocess

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader, event_motion_targets, planning_event
from scripts.analysis.event_motion_current import csv_output, json_output, mask_pixels, point_set_distance, region_record, sha256

SCHEMES = {
    'uniform40': (5, 10, 15, 20, 25, 30, 35, 40),
    'uniform80': (10, 20, 30, 40, 50, 60, 70, 80),
    'dense_early80': (5, 10, 15, 20, 30, 45, 60, 80),
}
RADIUS = 10
CLOCKS = ('train', 'val', 'all')


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def endpoint_labels(reader, frame, event, final_frame, options):
    labels, info = event_motion_targets(reader, frame, **options)
    valid = labels['trace_valid'][0]
    held = labels['trace_terminal_hold'][0]
    sources = info['trace_source_frame_indices'][0]
    terminal = valid & (sources == final_frame)
    observed = valid & ~held
    cross = valid & (sources >= event['stop_frame'])
    return dict(
        frames=1, endpoint_frames=int(terminal.any()), direct_endpoint_frames=int((terminal & observed).any()),
        last_slot_endpoint_frames=int(terminal[-1]), held_endpoint_frames=int((terminal & held).any()),
        direct_slots=int(observed.sum()), hold_slots=int((valid & held).sum()), invalid_slots=int((~valid).sum()),
        cross_frames=int(cross.any()), cross_direct_frames=int((cross & observed).any()),
        cross_slots=int(cross.sum()), cross_direct_slots=int((cross & observed).sum()),
    )


def measure(reader, entity_key, frame, shape, point):
    entity = reader.meta['scene_entities'][entity_key]
    mask, _, status, _ = region_record(reader, entity, frame, shape, context.size)
    if status != 'valid' or point is None:
        return status, None, None
    pixels = mask_pixels(mask)
    direct = point_set_distance(point, pixels, pixels + 1)
    lo = pixels + context.offset
    return status, direct, point_set_distance(point, lo, lo + 1)


def hit(distance):
    return distance is not None and distance <= RADIUS + 1e-7


def analyse_episode(job):
    path, split = job
    reader = EpisodeReader(path)
    meta = reader.meta
    episode = meta['episode_id']
    parent = meta['source']['parent_trajectory_id']
    stream = meta['cameras']['image']
    shape = stream['height'], stream['width']
    frames = reader.array(stream['frame_indices'])[reader.array(stream['valid'])]
    by_event = context.events[episode]
    counters = defaultdict(Counter)
    candidate_rows, val_rows = [], []
    for frame in map(int, frames[::context.stride]):
        event = planning_event(reader, frame)
        if not event or not event['skill_valid']:
            continue
        event_id, skill = event['event_id'], event['skill']
        record = by_event[event_id]
        final_frame = int(record['end'])
        flags = {}
        any_endpoint = False
        for scheme, options in context.options.items():
            result = endpoint_labels(reader, frame, event, final_frame, options)
            counters[(scheme, 'all')].update(result)
            if skill == 'Place':
                counters[(scheme, 'Place')].update(result)
            ending = bool(result['endpoint_frames'])
            any_endpoint |= ending
            flags[f'{scheme}_endpoint'] = ending
            flags[f'{scheme}_direct_endpoint'] = bool(result['direct_endpoint_frames'])
        if skill != 'Place':
            continue
        alternatives = context.candidates.get(episode, {}).get(frame, ())
        needs_validation = split == 'val' and any_endpoint
        if not alternatives and not needs_validation:
            continue
        point = np.asarray(record['end_eef']) if record['end_eef'] is not None else None
        key = record['entity']
        status, raw_distance, shifted_distance = measure(reader, key, frame, shape, point)
        base = dict(split=split, episode=episode, parent=parent, event=event_id, task=record['task'], frame=frame,
                    event_end=final_frame, target=key, target_visibility=status,
                    target_raw_distance=raw_distance, target_shifted_distance=shifted_distance,
                    target_shifted_hit=hit(shifted_distance), **flags)
        if needs_validation:
            val_rows.append(base)
        for candidate in alternatives:
            if candidate['event'] != event_id or candidate['target'] != key:
                raise ValueError(f'Saved candidate disagrees with the Event at {episode}/{frame}')
            candidate_key = candidate['candidate']
            alt_status, _, alt_shift = measure(reader, candidate_key, frame, shape, point)
            visible = status == alt_status == 'valid'
            geometry_pass = visible and hit(shifted_distance) and alt_shift is not None and not hit(alt_shift)
            candidate_rows.append(dict(
                base, candidate=candidate_key, candidate_visibility=alt_status, candidate_shifted_distance=alt_shift,
                both_visible=visible, candidate_shifted_hit=hit(alt_shift), geometry_pass=geometry_pass,
            ))
    return split, dict(counters), candidate_rows, val_rows


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def motion_summary(counters):
    frames = counters['frames']
    slots = frames * 8
    if sum(counters[k] for k in ('direct_slots', 'hold_slots', 'invalid_slots')) != slots:
        raise ValueError('Direct, held and invalid slots must partition the Motion lattice')
    return dict(counters, slots=slots, **{
        f'{key}_rate': divide(n, slots if key.endswith('_slots') else frames)
        for key,n in counters.items() if key != 'frames'
    })


def cardinalities(rows):
    return dict(
        pairs=len(rows),
        frames=len({(r['episode'],r['frame']) for r in rows}),
        events=len({(r['episode'],r['event']) for r in rows}),
        trajectories=len({r['parent'] for r in rows}),
        tasks=len({r['task'] for r in rows}),
    )


def candidate_summary(rows, scheme, place_frames):
    endpoint = [r for r in rows if r[f'{scheme}_endpoint']]
    visible = [r for r in endpoint if r['both_visible']]
    positive = [r for r in visible if r['target_shifted_hit']]
    selected = [r for r in positive if r['geometry_pass']]
    subsets = dict(all_legal=rows, endpoint=endpoint, endpoint_both_visible=visible, endpoint_positive_hit=positive,
                   excluded_both_hit=[r for r in positive if r['candidate_shifted_hit']], qualified=selected,
                   qualified_direct=[r for r in selected if r[f'{scheme}_direct_endpoint']])
    result = {key: cardinalities(v) for key,v in subsets.items()}
    check = result['qualified']
    # Count a frame once if at least one legal receiver passes, even when several pass.
    result['rates'] = dict(
        pair_among_legal=divide(check['pairs'], result['all_legal']['pairs']),
        pair_among_endpoint_visible=divide(check['pairs'], result['endpoint_both_visible']['pairs']),
        frame_among_legal=divide(check['frames'], result['all_legal']['frames']),
        frame_among_place=divide(check['frames'], place_frames),
    )
    return result


def validation_summary(rows, scheme):
    eligible = [r for r in rows if r[f'{scheme}_endpoint']]
    visible = [r for r in eligible if r['target_visibility'] == 'valid']
    raw_hits = sum(hit(r['target_raw_distance']) for r in visible)
    shifted_hits = sum(r['target_shifted_hit'] for r in visible)
    per_event = defaultdict(list)
    for row in visible:
        per_event[(row['episode'],row['event'])].append(row['target_shifted_hit'])
    direct = [r for r in visible if r[f'{scheme}_direct_endpoint']]
    return dict(eligible=len(eligible), visible=len(visible), invalid_visibility=len(eligible)-len(visible),
                events=len(per_event), trajectories=len({r['parent'] for r in visible}),
                tasks=len({r['task'] for r in visible}), raw_hits=raw_hits, shifted_hits=shifted_hits,
                raw_rate=divide(raw_hits,len(visible)), shifted_rate=divide(shifted_hits,len(visible)),
                shifted_rate_all_eligible=divide(shifted_hits,len(eligible)),
                shifted_rate_event_balanced=float(np.mean([np.mean(v) for v in per_event.values()])),
                direct_visible=len(direct), direct_shifted_hits=sum(r['target_shifted_hit'] for r in direct),
                direct_shifted_rate=divide(sum(r['target_shifted_hit'] for r in direct),len(direct)))


def load_context(path):
    previous = json.loads((path/'summary.json').read_text())
    provenance = previous['provenance']
    index = Path(provenance['index'])
    if sha256(index) != provenance['index_sha256']:
        raise ValueError('Dataset index differs from the input of the previous diagnostic')
    shift, = [r for r in previous['shifts_from_train']
              if r['skill']=='Place' and r['target_type']=='*' and r['region']=='mask']
    events, episodes, candidates = defaultdict(dict), {}, defaultdict(lambda:defaultdict(list))
    for row in read_csv(path/'events.csv'):
        episodes[row['episode']] = row['split']
        row['end_eef'] = json.loads(row['end_eef']) if row['end_eef'] else None
        events[row['episode']][row['event']] = row
    for row in read_csv(path/'candidates.csv.gz'):
        if row['role'] == 'destination':
            candidates[row['episode']][int(row['frame'])].append({
                key: row[key] for key in ('event','target','candidate')
            })
    base = provenance['timing']
    options = {k:dict(base, offsets_frames=v) for k,v in SCHEMES.items()}
    jobs = [(index.parent / row['path'], episodes[row['episode_id']])
            for row in json.loads(index.read_text())['episodes']]
    return argparse.Namespace(previous=previous, shift=shift, offset=np.asarray(shift['median_shift_px']),
                              size=base['image_size'], stride=provenance['frame_stride'], options=options,
                              events=events, candidates=candidates, jobs=jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', default=10, type=int)
    args = parser.parse_args()
    global context
    context = load_context(args.audit)
    totals = defaultdict(Counter)
    candidates, val = [], []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork')) as executor:
        for n, (split, counts, pairs, rows) in enumerate(executor.map(analyse_episode, context.jobs, chunksize=4), 1):
            for (scheme, skill), values in counts.items():
                for s in (split,'all'):
                    totals[(s,scheme,skill)].update(values)
            candidates.extend(pairs)
            val.extend(rows)
            if n % 400 == 0 or n == len(context.jobs):
                print(f'{n}/{len(context.jobs)} Episodes', flush=True)
    motion = {s:{k:{skill:motion_summary(totals[(s,k,skill)]) for skill in ('all','Place')}
                 for k in SCHEMES} for s in CLOCKS}
    for split in CLOCKS:
        for skill in ('all','Place'):
            prior = context.previous['slots'][split][skill]
            actual = motion[split]['uniform40'][skill]
            assert actual['endpoint_frames'] == prior['endpoint_eligibility']['any_slot']
            assert actual['frames'] == prior['frames']
    for split in CLOCKS:
        prior = context.previous['candidate_pairs'][split]['all_frames']
        assert sum(r['split']==split or split=='all' for r in candidates) == prior['same_skill_pairs']
    geometry = {s:{k:candidate_summary([r for r in candidates if s=='all' or r['split']==s],
                                       k, motion[s][k]['Place']['frames']) for k in SCHEMES} for s in CLOCKS}
    val_stats = {k:validation_summary(val,k) for k in SCHEMES}
    for k, row in val_stats.items():
        assert row['eligible'] == motion['val'][k]['Place']['endpoint_frames']
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    validation_start = context.previous['events']['val']['Place']['start_mask']['skill_shift']['shifted']
    result = dict(
        provenance=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                        previous_audit=str(args.audit.resolve()), previous_summary_sha256=sha256(args.audit/'summary.json'),
                        script_sha256=sha256(__file__), timing=context.options, radius_px=RADIUS,
                        place_shift=context.shift),
        motion=motion, candidates=geometry, validation_current_mask=val_stats,
        validation_event_start=dict(valid=validation_start['valid'],
                                    shifted_hits=validation_start['hits'][2],
                                    shifted_rate=validation_start['rates'][2]),
    )
    json_output(output/'summary.json',result)
    csv_output(output/'motion_offsets.csv',[
        dict(split=s, scheme=k, skill=skill, **row)
        for s,t in motion.items() for k,table in t.items() for skill,row in table.items()
    ])
    csv_output(output/'counterfactual_summary.csv',[
        dict(split=s, scheme=k, subset=subset, **row)
        for s,t in geometry.items() for k,table in t.items() for subset,row in table.items() if subset!='rates'
    ])
    csv_output(output/'place_validation_summary.csv',[dict(scheme=k,**row) for k,row in val_stats.items()])
    csv_output(output/'counterfactual_pairs.csv.gz',candidates)
    csv_output(output/'place_validation_frames.csv.gz',val)


if __name__ == '__main__':
    main()
