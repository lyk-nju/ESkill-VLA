"""Single-query text relabeling of simple LIBERO transfers using offline contact windows."""
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
    if data['version'] != 2 or data['index_sha256'] != hashlib.sha256(index.read_bytes()).hexdigest():
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
    for entity, intervals in row['sources_touching_destination'].items():
        if meta['scene_entities'].get(entity, {}).get('kind') != 'object':
            raise ValueError(f"{meta['episode_id']}: unknown contact object {entity}")
        validate_ranges(entity, intervals, events[0])


def transfer_state(row, event, frame):
    if row and event:
        field = {'Pick': 'pre_grasp', 'Place': 'holding'}.get(event['skill'])
        if field and any(lo <= frame < hi for lo, hi in row[field]):
            return field
    return None


def compatible(source, destination):
    """Coarse LIBERO roles: an open receptacle, a support surface, or a rack for bottles."""
    if destination not in PLACEMENT:
        return False
    if destination == 'wine_rack':
        return source in ('wine_bottle', 'ketchup', 'bbq_sauce', 'salad_dressing')
    return not (destination == 'akita_black_bowl' and source in ('basket', 'plate'))


def candidates(state, row, frame, transfer, key, scene, captions, appearances, locked, *,
               min_visible, held_visible, held_out):
    if not state or not transfer or transfer[:2] != (row['source'], row['destination']):
        return []
    source, dest, match = transfer
    picking = state == 'pre_grasp'
    if key != (source if picking else dest) or (not picking and not held_visible):
        return []
    at_destination = row['sources_touching_destination'] if picking else {}
    choices = []
    for other, fraction in appearances.items():
        caption = captions.get(other)
        if (not caption or 'in the image' in caption or fraction < min_visible or other in (source, dest)
                or scene[other]['kind'] not in ('object', 'fixture')):
            continue
        if picking and (scene[other]['kind'] != 'object' or locked.get(other, True)
                        or any(lo <= frame < hi for lo, hi in at_destination.get(other, ()))):
            continue
        src = scene[other if picking else source]['category']
        dst = scene[dest if picking else other]['category']
        if not compatible(src, dst):
            continue
        prep = match['preposition'].lower() if picking else PLACEMENT[dst]
        relation = 'in' if prep in ('in', 'into') else 'on'
        if (src, relation, dst) in held_out:
            continue
        if picking:
            lo, hi = match.span('source')
            text = match.string[:lo] + caption + match.string[hi:]
            choices.append((text, other, 'source'))
        else:
            # Replace both spans, keeping the original verb, article and punctuation.
            lo, hi = match.span('preposition')
            noun, end = match.span('destination')
            text = match.string[:lo] + prep + match.string[hi:noun] + caption + match.string[end:]
            choices.append((text, other, 'destination'))
    return choices
