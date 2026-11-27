"""Summaries and figures for the factual Event Motion/Current audit."""
from collections import Counter, defaultdict
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader, event_motion_targets
from openpi.transforms import resize_image
from scripts.analysis.event_motion_current import (
    QUANTILES, RADII, REGIONS, SKILLS, csv_output, json_output, mask_pixels, point_set_distance, raster_box,
    region_record, sha256,
)

SPLITS = ('all', 'train', 'val')


def cuts(values, percentiles=QUANTILES):
    return {f'p{q}': float(v) for q, v in zip(percentiles, np.percentile(values, percentiles))} if len(values) else {}


def support(values):
    return dict(valid=len(values), mean=float(np.mean(values)) if len(values) else None, **cuts(values))


def coverage(values):
    values = np.asarray(values)
    hits = [int((values <= radius + 1e-7).sum()) for radius in RADII]
    return dict(valid=len(values), hits=hits, rates=[n / len(values) if len(values) else None for n in hits])


def offsets(values):
    a = np.asarray(values)
    if not len(a):
        return dict(valid=0)
    return dict(valid=len(a), mean=a.mean(0).tolist(), median=np.median(a, axis=0).tolist(),
                std=a.std(0).tolist(), above_center=int((a[:,1] < 0).sum()),
                x=cuts(a[:, 0]), y=cuts(a[:, 1]), norm=support(np.linalg.norm(a, axis=1)))


def grouped(events, field):
    groups = defaultdict(list)
    for row in events:
        groups[(row['split'], row['skill'], str(row[field]))].append(row)
    return [{'split': split, 'skill': skill, field: name, 'events': len(rows),
             **{f'{anchor}_{region}': offsets([r[f'{anchor}_{region}_offset'] for r in rows
                                               if f'{anchor}_{region}_offset' in r])
                for anchor in ('start', 'end') for region in REGIONS}}
            for (split, skill, name), rows in sorted(groups.items())]


def region_distances(events, index, size):
    train_offsets = defaultdict(list)
    for row in events:
        if row['split'] != 'train':
            continue
        for region in REGIONS:
            value = row.get(f'start_{region}_offset')
            if value is not None:
                for category in ('*', row['target_type']):
                    train_offsets[(row['skill'], category, region)].append(value)
    shifts = {key: np.median(v, 0) for key, v in train_offsets.items()}
    paths = episode_paths(index)
    for row in events:
        if row['start_eef_status'] == 'no_saved_rgb' or row['end_eef'] is None or not row['binding_valid']:
            continue
        reader = EpisodeReader(paths[row['episode']])
        stream = reader.meta['cameras']['image']
        shape = stream['height'], stream['width']
        entity = reader.meta['scene_entities'][row['entity']]
        mask, box, _, _ = region_record(reader, entity, row['start'], shape, size)
        if mask is not None and mask.any():
            pixels = mask_pixels(mask)
            supports = {'mask': (pixels, pixels + 1)}
        else:
            supports = {}
        if box is not None:
            supports['box'] = (box[:2], box[2:])
        for region, (lo, hi) in supports.items():
            for method, key in (('skill', '*'), ('category', row['target_type'])):
                shift = shifts.get((row['skill'], key, region))
                if shift is not None:
                    row[f'start_{region}_{method}_shift_distance'] = point_set_distance(
                        np.asarray(row['end_eef']), lo + shift, hi + shift)
    return [dict(skill=s, target_type=c, region=r, count=len(train_offsets[s, c, r]), median_shift_px=xy.tolist())
            for (s, c, r), xy in sorted(shifts.items())]


def get(rows, field):
    return [r[field] for r in rows if r.get(field) is not None]


def spatial_summary(rows):
    result = dict(events=len(rows), binding_valid=sum(r['binding_valid'] for r in rows),
                  boundary_valid=sum(r['boundary_valid'] for r in rows), skill_valid=sum(r['skill_valid'] for r in rows),
                  goal_differs_from_last_saved=sum(r['goal'] != r['end'] for r in rows))
    for name in ('start', 'end', 'goal'):
        result[f'{name}_eef_status'] = {k: sum(r[f'{name}_eef_status'] == k for r in rows)
                                        for k in ('valid', 'missing_projection', 'invalid_projection', 'no_saved_rgb')}
    for name in ('start', 'end'):
        result[f'{name}_mask_status'] = dict(Counter(r[f'{name}_mask_status'] for r in rows))
        result[f'{name}_region_kind'] = dict(Counter(r[f'{name}_geometry'] or 'missing' for r in rows))
        boxes = np.asarray(get(rows, f'{name}_box')).reshape(-1, 4)
        result[f'{name}_box_width'] = support(boxes[:, 2] - boxes[:, 0])
        result[f'{name}_box_height'] = support(boxes[:, 3] - boxes[:, 1])
        for region in REGIONS:
            tag = f'{name}_{region}'
            values = get(rows, f'{tag}_distance')
            result[tag] = dict(**coverage(values), invalid=len(rows)-len(values), distance=support(values),
                               offset=offsets(get(rows, f'{tag}_offset')))
            if name == 'start':
                for mode in ('skill', 'category'):
                    matched = [r for r in rows if r.get(f'{tag}_{mode}_shift_distance') is not None]
                    result[tag][f'{mode}_shift'] = dict(
                        paired_raw=coverage(get(matched, f'{tag}_distance')),
                        shifted=coverage(get(matched, f'{tag}_{mode}_shift_distance')))
    for region in REGIONS:
        result[f'{region}_center_displacement'] = support(get(rows, f'{region}_center_displacement'))
    return result


def slot_summary(rows):
    n = len(rows)
    def count(key):
        return sum(bool(r[key]) for r in rows)
    def array(key):
        return np.asarray([r[key] for r in rows], dtype=bool).reshape(n, 8)
    valid, held, cross, cross_trace, endpoint = map(array, ('valid', 'hold', 'cross_observed', 'cross_trace', 'endpoint_slots'))
    real = valid & ~held
    mask = np.asarray([r['current_mask_status'] == 'valid' for r in rows], dtype=bool)
    box = np.asarray([r['current_box_valid'] for r in rows], dtype=bool)
    has_end = endpoint.any(-1)
    eligible = dict(any_slot=int(has_end.sum()), last_slot=count('slot8_end'),
                    any_slot_with_current_mask=int((has_end & mask).sum()),
                    any_slot_with_current_box=int((has_end & box).sum()),
                    any_slot_with_mask_and_anchor_eef=sum(
                        bool(has_end[i] and mask[i] and r['anchor_eef_valid']) for i, r in enumerate(rows)),
                    last_slot_with_current_mask=int((endpoint[:, -1] & mask).sum()),
                    any_real_endpoint=int((endpoint & real).any(-1).sum()),
                    any_held_endpoint=int((endpoint & held).any(-1).sum()))
    remaining = get(rows, 'remaining_saved_frames')
    return dict(frames=n, frames_with_trace=count('any_trace'), frames_without_trace=n-count('any_trace'),
                remaining_frames=support(remaining), remaining_histogram=dict(sorted(Counter(remaining).items())),
                current_mask_status=dict(Counter(r['current_mask_status'] for r in rows)),
                current_box_valid=int(box.sum()), anchor_eef_valid=count('anchor_eef_valid'),
                endpoint_eligibility=eligible,
                endpoint_rates={k: v / n if n else None for k, v in eligible.items()},
                slot8=dict(at_end=count('slot8_end'), inside=count('slot8_inside'), after=count('slot8_after'),
                           invalid=count('slot8_invalid')),
                cross_event=dict(request_frames=int(cross.any(-1).sum()), actual_valid_observed_frames=int((cross & valid).any(-1).sum()),
                                 any_valid_trace_after_event=int(cross_trace.any(-1).sum()),
                                 slot_count=cross.sum(0).tolist(), valid_trace_slot_count=cross_trace.sum(0).tolist()),
                valid_observed=real.sum(0).tolist(), valid_hold=(held & valid).sum(0).tolist(),
                invalid=(~valid).sum(0).tolist(), requested_hold=held.sum(0).tolist(),
                endpoint_slots=endpoint.sum(0).tolist())


def pair_summary(rows):
    dest = [r for r in rows if r['role'] == 'destination']
    sources = [r for r in rows if r['role'] == 'source']
    result = dict(total_queries=len(rows), same_skill_pairs=len(dest), source_change_queries=len(sources),
                  source_change_frames=len({(r['episode'], r['frame']) for r in sources}),
                  same_skill_frames=len({(r['episode'], r['frame']) for r in dest}),
                  same_skill_events=len({(r['episode'], r['event']) for r in dest}),
                  same_skill_related=dict(Counter(r['relation'] for r in dest)),
                  source_change_related=dict(Counter(r['relation'] for r in sources)),
                  classes={f'{a} → {b}': n for (a,b), n in
                           sorted(Counter((r['category'], r['candidate_category']) for r in dest).items())})
    for region in REGIONS:
        prefix = f'{region}_'
        matched = [r for r in dest if r.get('target_' + prefix + 'distance') is not None
                   and r.get('candidate_' + prefix + 'distance') is not None]
        ds = np.array([r['candidate_' + prefix + 'distance'] - r['target_' + prefix + 'distance'] for r in matched])
        result[region] = dict(center_distance=support(get(dest, prefix+'center_distance')),
                              comparable=len(matched), unavailable=len(dest)-len(matched),
                              closer_correct=int((ds > 1e-6).sum()), tied=int((np.abs(ds) <= 1e-6).sum()),
                              closer_candidate=int((ds < -1e-6).sum()), radii={})
        for radius in RADII:
            iou = get(dest, f'{region}_iou_{radius}')
            contain = get(dest, f'{region}_containment_{radius}')
            result[region]['radii'][str(radius)] = dict(iou=support(iou), containment=support(contain),
                                                      overlapping=sum(v > 0 for v in iou),
                                                      overlapping_at_least_half_small=sum(v >= .5 for v in contain),
                                                      both_hit=sum(r['target_'+prefix+'distance'] <= radius
                                                                   and r['candidate_'+prefix+'distance'] <= radius
                                                                   for r in matched))
    return result


def episode_paths(index):
    import json
    return {r['episode_id']: index.parent / r['path'] for r in json.loads(index.read_text())['episodes']}


def interacting_regions(events, index, size):
    """Keep actual annotated relations separate from valid alternative receivers."""
    paths = episode_paths(index)
    records = []
    for row in events:
        if not row['binding_valid'] or row['end'] is None:
            continue
        reader = EpisodeReader(paths[row['episode']])
        scene = reader.meta['scene_entities']
        target = scene[row['entity']]
        event = next(e for e in reader.meta['events'] if e['event_id'] == row['event'])
        relationships = []
        if target['instance'] != row['entity'] and target['instance'] in scene:
            relationships.append((target['instance'], 'same_parent'))
        if row['skill'] == 'Place':
            carried = (event.get('source_annotation') or {}).get('instance')
            relation = re.search(r'\b(inside|into|in|on top of|onto|on)\b', event.get('description') or '',
                                 flags=re.IGNORECASE)
            if carried and carried in scene and carried != row['entity'] and relation:
                name = 'containment_goal' if relation[1].lower() in ('inside', 'into', 'in') else 'support_or_stack_goal'
                relationships.append((carried, name))
        if not relationships:
            continue
        cam = reader.meta['cameras']['image']
        shape = cam['height'], cam['width']
        for when in ('start','end'):
            a, box, _, _ = region_record(reader, target, row[when], shape, size)
            for key, name in relationships:
                b, other_box, _, _ = region_record(reader, scene[key], row[when], shape, size)
                values = dict(split=row['split'], episode=row['episode'], event=row['event'], skill=row['skill'],
                              target=row['entity'], related_entity=key, relation=name, at=when)
                for region, (left,right) in dict(
                    mask=(a,b), box=(raster_box(box, size),raster_box(other_box, size)),
                ).items():
                    if left is not None and right is not None and left.any() and right.any():
                        left, right = left > 0, right > 0
                        inter = (left & right).sum()
                        values[f'{region}_iou'] = float(inter / (left | right).sum())
                        values[f'{region}_contains_smaller'] = float(inter / min(left.sum(), right.sum()))
                records.append(values)
    return records


def fix_scene_and_type(events, pairs, index):
    lookup = {}
    paths = episode_paths(index)
    for row in events:
        key = row['episode']
        if key not in lookup:
            meta = EpisodeReader(paths[key]).meta
            scene_entities = meta['scene_entities']
            stem = Path(meta['interaction_annotation']['bddl']).stem
            scene = re.match(r'(.+?SCENE\d+)_', stem, re.IGNORECASE)
            # Single-suite BDDL names omit a numbered layout. Use the fixture inventory.
            suite = row['episode'].split('_no_noops')[0]
            furniture = Counter(e['category'] for e in scene_entities.values() if e['kind'] == 'fixture')
            layout = scene[1] if scene else suite + '/' + '+'.join(f'{k}*{n}' for k,n in sorted(furniture.items()))
            lookup[key] = layout, scene_entities
        row['scene'], scene_entities = lookup[key]
        entity = scene_entities.get(row['entity'])
        if not entity:
            row['target_type'] = 'unresolved'
        elif row['kind'] == 'region':
            owner = scene_entities[entity['instance']]['category']
            row['target_type'] = owner + '::' + entity['site_name'].removeprefix(entity['instance']+'_')
        elif row['kind'] == 'part':
            row['target_type'] = row['category'] + '::' + row['part']
        else:
            row['target_type'] = row['category']
    event_map = {(r['episode'], r['event']):r for r in events}
    for row in pairs:
        row['scene'] = event_map[(row['episode'], row['event'])]['scene']


def scatter_plot(path, events):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout='constrained')
    for skill, ax in zip(SKILLS, axes.flat):
        rows = [r for r in events if r['skill'] == skill]
        for when, color, marker in (('start', '#176d9b', 'o'), ('end', '#c87c2f', 'x')):
            xy = np.asarray(get(rows, f'{when}_box_offset'))
            if len(xy):
                ax.scatter(*xy.T, marker=marker, s=9, alpha=.33, c=color, label=f'{when} bbox center (n={len(xy)})')
        ax.axvline(0, color='.6', lw=.8)
        ax.axhline(0, color='.6', lw=.8)
        ax.set_title(skill)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=8, loc='upper left')
    axes.flat[0].invert_yaxis()
    fig.supxlabel('EEF endpoint − Current center, Δx (model px)')
    fig.supylabel('Δy (model px, down positive)')
    fig.savefig(path, dpi=155)
    plt.close(fig)


def curves_plot(path, stats):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True, layout='constrained')
    for skill, ax in zip(SKILLS, axes.flat):
        record = stats[skill]
        for region, color in (('mask', '#176d9b'), ('box', '#c87c2f')):
            start = record[f'start_{region}']
            for legend, series, style in (
                ('start', start, '-'), ('start + train median', start['skill_shift']['shifted'], '--'),
                ('end', record[f'end_{region}'], ':'),
            ):
                if series['valid']:
                    ax.plot(RADII, np.array(series['rates'])*100, style, color=color,
                            marker='.' if legend == 'start' else '', label=f'{region} {legend} (n={series["valid"]})')
        ax.set(ylim=(0, 101), title=skill)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc='lower right')
    fig.supxlabel('Tolerance radius on 224×224 canvas (px)')
    fig.supylabel('Fraction of valid Event endpoints (%)')
    fig.savefig(path, dpi=155)
    plt.close(fig)


def remaining_plot(path, stats):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True, layout='constrained')
    for skill, ax in zip(SKILLS, axes.flat):
        row = stats[skill]
        if row['frames']:
            x = np.array(list(row['remaining_histogram']))
            y = np.array(list(row['remaining_histogram'].values())).cumsum() / row['frames']
            ax.step(x, 100*y, where='post', color='#176d9b')
            ax.set_title(f'{skill}  |  endpoint eligible {100*row["endpoint_rates"]["any_slot"]:.1f}%')
        else:
            ax.set_title(f'{skill}  |  no samples')
        for offset in (10, 40):
            ax.axvline(offset, color='.6', lw=.8, linestyle='--')
        ax.grid(alpha=.2)
    fig.supxlabel('Saved camera frames from anchor to actual Event end')
    fig.supylabel('Training anchors at or below this length (%)')
    fig.savefig(path, dpi=155)
    plt.close(fig)


def candidate_plot(path, candidates):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for region, ax in zip(REGIONS, axes):
        records = candidates[region]['radii']
        n = candidates[region]['radii']['0']['iou']['valid']
        ax.set_title(f'{region.upper()} (n={n})')
        if n:
            ax.plot(RADII, [records[str(r)]['iou']['p50'] for r in RADII], '-o', label='Median IoU')
            ax.plot(RADII, [records[str(r)]['containment']['p50'] for r in RADII], '--o', label='Median intersection / smaller')
            ax.legend()
        else:
            ax.text(.5, .5, 'No same-skill pairs', transform=ax.transAxes, ha='center')
        ax.set(ylim=(0, 1.02), xlabel='Tolerance radius (model px)')
        ax.grid(alpha=.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_observation(ax, reader, entity, frame, motion):
    size = motion['image_size']
    rgb = reader.rgb('image', frame)
    shape = rgb.shape[:2]
    ax.imshow(resize_image(rgb, size), extent=(0,size,size,0))
    mask, box, status, _ = region_record(reader, entity, frame, shape, size)
    if status == 'valid':
        tint = np.zeros((size, size, 4))
        tint[:, :, :3] = (1, .2, .3)
        tint[:, :, 3] = .34 * mask
        ax.imshow(tint, extent=(0,size,size,0))
    if box is not None:
        from matplotlib.patches import Rectangle
        lo, hi = box[:2], box[2:]
        ax.add_patch(Rectangle(lo, *(hi-lo), fill=False, edgecolor='#ff5979', linewidth=1.5))
    ax.set(xlim=(0,size), ylim=(size,0), xticks=[], yticks=[])


def examples(output, events, index, motion):
    paths = episode_paths(index)
    records = []
    size = motion['image_size']
    for skill in SKILLS:
        rows = [r for r in events if r['skill'] == skill and r.get('start_mask_distance') is not None
                and r['start_eef'] is not None and r['split'] == 'val']
        if len(rows) < 2:
            rows = [r for r in events if r['skill'] == skill and r.get('start_mask_distance') is not None
                    and r['start_eef'] is not None]
        if not rows:
            continue
        median = np.median(get(rows, 'start_mask_distance'))
        typical = min(rows, key=lambda r: (abs(r['start_mask_distance']-median), r['episode']))
        far = max(rows, key=lambda r: r['start_mask_distance'])
        if far is typical and len(rows) > 1:
            far = next(r for r in rows if r is not typical)
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout='constrained')
        fig.suptitle(f'{skill}  |  factual Current and EEF endpoint', fontsize=15)
        for axs, row, kind in zip(axes, (typical, far), ('median', 'largest')):
            reader = EpisodeReader(paths[row['episode']])
            entity = reader.meta['scene_entities'][row['entity']]
            start, end = row['start'], row['end']
            labels, info = event_motion_targets(reader, start, **motion)
            trace = labels['trace'][0] * size
            valid, hold = (labels[k][0] for k in ('trace_valid', 'trace_terminal_hold'))
            sources = info['trace_source_frame_indices'][0]
            records.append(dict(skill=skill, case=kind, episode=row['episode'], split=row['split'], event=row['event'],
                                frames=[start,end], current_target=row['entity'], source_frames=sources.tolist(),
                                xy=trace.tolist(), valid=valid.tolist(), hold=hold.tolist(),
                                endpoint_mask_distance_px=row['start_mask_distance']))
            for ax, frame in zip(axs, (start, end)):
                plot_observation(ax, reader, entity, frame, motion)
                ax.scatter(*row['end_eef'], marker='*', color='#fff595', edgecolor='black', s=122, zorder=5)
            axs[0].set_title(f'{kind} | source {start} | start mask distance {row["start_mask_distance"]:.1f}px', fontsize=10)
            axs[1].set_title(f'Event end | source {end} | {row["split"]}', fontsize=10)
            axs[1].set_xlabel(f'{row["episode"]} / {row["event"]}', fontsize=9)
            axs[0].scatter(*row['start_eef'], marker='o', color='white', edgecolor='black', s=48, zorder=5)
            points = np.r_[np.asarray(row['start_eef'])[None], np.where(valid[:, None], trace, np.nan)]
            axs[0].plot(*points.T, color='#52dfea', linewidth=1.6)
            for source in np.unique(sources[valid]):
                which = np.flatnonzero(valid & (sources == source))
                xy = trace[which[0]]
                keep = bool(hold[which].all())
                axs[0].scatter(*xy, marker='s' if keep else 'o', s=27, color='#d1a3ff' if keep else '#52dfea', zorder=4)
                axs[0].annotate(','.join(str(i+1) for i in which), xy, xytext=(4,4), textcoords='offset points',
                                color='white', fontsize=7, bbox=dict(facecolor='#15252f', alpha=.65,pad=1,edgecolor='none'))
        fig.supxlabel('Pink: Current region / bbox     White: anchor EEF     Cyan: observed slots\n'
                      'Purple: repeated terminal labels     Yellow star: actual Event endpoint', fontsize=10)
        fig.savefig(output / f'{skill.lower()}.png', dpi=150)
        plt.close(fig)
    return records


def make_report(output, events, slots, pairs, provenance, motion, index):
    fix_scene_and_type(events, pairs, index)
    shifts = region_distances(events, index, motion['image_size'])
    result = dict(provenance={**provenance, 'report_script_sha256': sha256(__file__)}, shifts_from_train=shifts)
    result['events'] = {split: {skill: spatial_summary(
        [r for r in events if (split == 'all' or r['split'] == split) and (skill == 'all' or r['skill'] == skill)])
        for skill in (*SKILLS, 'all')} for split in SPLITS}
    result['slots'] = {split: {skill: slot_summary(
        [r for r in slots if (split == 'all' or r['split'] == split) and (skill == 'all' or r['skill'] == skill)])
        for skill in (*SKILLS, 'all')} for split in SPLITS}
    result['candidate_pairs'] = {}
    overlap_rows = []
    for split in SPLITS:
        rows = [r for r in pairs if split == 'all' or r['split'] == split]
        views = {
            'all_frames': rows,
            'event_start': [r for r in rows if r['at_start']],
            'endpoint_eligible': [r for r in rows if r['endpoint_eligible']],
        }
        by_relation = defaultdict(list)
        for row in rows:
            by_relation[row['relation']].append(row)
        views.update({f'relation/{name}': group for name, group in by_relation.items()})
        result['candidate_pairs'][split] = {name: pair_summary(group) for name, group in views.items()}
        for scope, table in result['candidate_pairs'][split].items():
            for region in REGIONS:
                for radius, record in table[region]['radii'].items():
                    overlap_rows.append(dict(
                        split=split, scope=scope, region=region, radius_px=int(radius),
                        iou_valid=record['iou']['valid'], iou_p50=record['iou'].get('p50'),
                        containment_p50=record['containment'].get('p50'),
                        overlapping=record['overlapping'], both_hit=record['both_hit'],
                        nearer_correct=table[region]['closer_correct'], tied=table[region]['tied'],
                        nearer_candidate=table[region]['closer_candidate']))
    csv_output(output/'candidate_overlap.csv', overlap_rows)
    related = interacting_regions(events, index, motion['image_size'])
    csv_output(output/'related_regions.csv', related)
    relation_groups = defaultdict(list)
    for row in related:
        relation_groups[(row['relation'], row['at'])].append(row)
    result['related_regions_outside_candidates'] = [
        dict(relation=relation, anchor=anchor, n=len(group),
             **{f'{region}_{metric}': support(get(group, f'{region}_{metric}'))
                for region in REGIONS for metric in ('iou', 'contains_smaller')})
        for (relation, anchor), group in sorted(relation_groups.items())
    ]
    result['offset_strata'] = {field: grouped(events, field) for field in ('task', 'target_type', 'scene')}
    rows = []
    coverage_rows = []
    for split in SPLITS:
        for skill in (*SKILLS, 'all'):
            e, s = result['events'][split][skill], result['slots'][split][skill]
            rows.append(dict(split=split, skill=skill, events=e['events'], frames=s['frames'],
                             **{f'{a}_{r}_{metric}': value
                                for a in ('start','end') for r in REGIONS
                                for metric, value in [
                                    ('valid', e[f'{a}_{r}']['valid']), ('p50', e[f'{a}_{r}']['distance'].get('p50')),
                                    ('hit0', e[f'{a}_{r}']['rates'][0]), ('hit20', e[f'{a}_{r}']['rates'][4]),
                                ]}, **s['endpoint_eligibility']))
            for anchor in ('start', 'end'):
                for region in REGIONS:
                    original = e[f'{anchor}_{region}']
                    methods = {'raw': original}
                    if anchor == 'start':
                        for mode in ('skill','category'):
                            comparison = original[f'{mode}_shift']
                            methods[f'{mode}_shift'] = comparison['shifted']
                            methods[f'raw_paired_for_{mode}_shift'] = comparison['paired_raw']
                    for method, curve in methods.items():
                        coverage_rows.extend(dict(split=split, skill=skill, anchor=anchor, region=region, method=method,
                                                  radius_px=radius, valid=curve['valid'], hits=hits, rate=rate)
                                             for radius, hits, rate in zip(RADII,curve['hits'],curve['rates']))
    csv_output(output/'skill_summary.csv', rows)
    csv_output(output/'coverage.csv', coverage_rows)
    json_output(output/'summary.json', result)
    visual = output/'figures'
    visual.mkdir(exist_ok=True)
    scatter_plot(visual/'offsets.png', events)
    curves_plot(visual/'coverage_all.png', result['events']['all'])
    curves_plot(visual/'coverage_validation.png', result['events']['val'])
    remaining_plot(visual/'remaining_training.png', result['slots']['train'])
    candidate_plot(visual/'candidate_overlap.png', result['candidate_pairs']['all']['all_frames'])
    json_output(visual/'examples.json', examples(visual, events, index, motion))
