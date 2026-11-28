"""Single-query text relabeling of simple LIBERO transfers using offline contact windows."""
from dataclasses import dataclass
import hashlib
import json

# Whole-instance destinations in the first Object/Goal release. Drawers require an open-state label.
PLACEMENT = {
    'basket': 'in', 'akita_black_bowl': 'in', 'plate': 'on',
    'flat_stove': 'on', 'wooden_cabinet': 'on top of', 'wine_rack': 'on',
}


def load_windows(index):
    path = index.with_name('counterfactual_states.json')
    if not path.is_file():
        raise FileNotFoundError(f'{path}: run scripts.preprocess.libero.counterfactual_states or sync the sidecar')
    data = json.loads(path.read_text())
    if data['version'] not in (2, 3) or data['index_sha256'] != hashlib.sha256(index.read_bytes()).hexdigest():
        raise ValueError(f'{path}: counterfactual windows belong to another dataset index or schema')
    return data['episodes']


def validate_windows(row, meta):
    if row is None:
        return
    events = meta['events']
    if ([e['skill'] for e in events] != ['Pick', 'Place'] or row['task'] != meta['instructions'][0]
            or [e['interaction_target']['entity_id'] for e in events] != [row['source'], row['destination']]):
        raise ValueError(f"{meta['episode_id']}: saved counterfactual task or bindings changed")
    def validate_ranges(name, intervals, event):
        last = event['start_frame']
        for start, stop in intervals:
            if not last <= start < stop <= event['stop_frame']:
                raise ValueError(f"{meta['episode_id']}: invalid {name} interval")
            last = stop

    validate_ranges('pre_grasp', row['pre_grasp'], events[0])
    validate_ranges('holding', row['holding'], events[1])
    episode = dict(start_frame=0, stop_frame=meta.get('frame_count', events[-1]['stop_frame']))
    for field, bounds, kinds in (('sources_touching_destination', events[0], ('object',)),
                                 ('protected', episode, ('object',)),
                                 ('source_touching_receivers', episode, ('object', 'fixture'))):
        for entity, intervals in row.get(field, {}).items():
            if meta['scene_entities'].get(entity, {}).get('kind') not in kinds:
                raise ValueError(f"{meta['episode_id']}: unknown contact object {entity}")
            validate_ranges(entity, intervals, bounds)


def transfer_state(row, event, frame):
    if row and event:
        field = {'Pick': 'pre_grasp', 'Place': 'holding'}.get(event['skill'])
        if field and any(lo <= frame < hi for lo, hi in row[field]):
            return field
    return None


def contains_frame(intervals, frame):
    return any(start <= frame < stop for start, stop in intervals)


def protected_entities(row, events, frame, state):
    protected = {key for key, ranges in (row or {}).get('protected', {}).items() if contains_frame(ranges, frame)}
    if row and state != 'pre_grasp':
        protected.add(row['source'])
        if state != 'holding':
            protected.add(row['destination'])
    # Preserve already completed placements in longer demonstrations.
    for event in events:
        if event['stop_frame'] > frame or event['skill'] != 'Place' or not event.get('boundary_valid', True):
            continue
        bound = event.get('interaction_target') or {}
        if bound.get('status', 'resolved') == 'resolved':
            protected.update(filter(None, (bound.get('entity_id'), (event.get('source_annotation') or {}).get('instance'))))
    return protected


def relation_key(preposition):
    """Use the same placement vocabulary as the unseen-composition split."""
    if preposition in ('in', 'into'):
        return 'in'
    if preposition in ('on', 'onto', 'on top of'):
        return 'on'
    return None


def compatible(source, relation, destination):
    """Check known receiving roles and clear conflicts, independent of observed task combinations."""
    expected = PLACEMENT.get(destination)
    if expected is None or relation_key(relation) != relation_key(expected):
        return False
    if destination == 'wine_rack':
        return source in ('wine_bottle', 'ketchup', 'bbq_sauce', 'salad_dressing')
    return not (destination == 'akita_black_bowl' and source in ('basket', 'plate'))


@dataclass(frozen=True)
class TransferContext:
    row: dict
    state: str
    frame: int
    transfer: tuple
    current: str


def candidates(context, scene, names, appearances, locked, boxes, moved, *,
               min_visible, gripper_in_view, held_visible, held_out, reasons):
    """Generate state-preserving tasks and their Current entities on the edited host."""
    if context is None:
        reasons['state_or_binding'] = 1
        return []
    row, frame = context.row, context.frame
    source, dest, match = context.transfer
    picking = context.state == 'pre_grasp'
    if source not in names or dest not in names:
        reasons['ambiguous_reference'] = 1
        return []
    if not gripper_in_view or (not picking and not held_visible):
        reasons['interaction_evidence'] = 1
        return []

    def reject(reason):
        reasons[reason] = reasons.get(reason, 0) + 1

    def separated(a, b):
        box_a, box_b = boxes[a], boxes[b]
        return any(box_a[i + 2] <= box_b[i] or box_b[i + 2] <= box_a[i] for i in (0, 1))

    def check(src, dst, changed, current, prep, kind):
        if changed not in names:
            return 'ambiguous_reference'
        if min(appearances.get(current, 0.), appearances.get(changed, 0.)) < min_visible:
            return 'insufficient_visibility'
        if scene[src]['kind'] != 'object' or not compatible(scene[src]['category'], prep, scene[dst]['category']):
            return 'affordance_conflict'
        if kind == 'source' and locked.get(src, True):
            return 'interaction_lock'
        if kind == 'source' and contains_frame(row['sources_touching_destination'].get(src, ()), frame):
            return 'recorded_destination_contact'
        if kind != 'source' and contains_frame(row.get('source_touching_receivers', {}).get(dst, ()), frame):
            return 'recorded_destination_contact'
        # Once either endpoint moves, separation needs current evidence from the edited scene.
        if (src in moved or dst in moved) and (src not in boxes or dst not in boxes or not separated(src, dst)):
            return 'edited_relation_uncertain'
        if (scene[src]['category'], relation_key(prep), scene[dst]['category']) in held_out:
            return 'held_out'
        return None

    def offer(other, kind, result):
        if kind == 'source':
            src, dst, current, prep = other, dest, other, match['preposition'].lower()
        else:
            src, dst = source, other
            current = context.current if kind == 'future_goal' else other
            prep = PLACEMENT.get(scene[other]['category'], '')
        reason = check(src, dst, other, current, prep, kind)
        if reason:
            reject(reason)
            return
        if kind == 'source':
            start, stop = match.span('source')
            text = match.string[:start] + names[other] + match.string[stop:]
        else:
            start, stop = match.span('preposition')
            noun, end = match.span('destination')
            text = match.string[:start] + prep + match.string[stop:noun] + names[other] + match.string[end:]
        result.append((text, current, kind))

    others = [key for key, entity in scene.items()
              if key not in (source, dest) and entity['kind'] in ('object', 'fixture')]
    options = []
    if picking:
        for key in others:
            offer(key, 'source', options)
        if options:
            return options
    kind = 'future_goal' if picking else 'destination'
    for key in others:
        offer(key, kind, options)
    return options
