"""Text / scene / context binding. Asset names live in the catalog, never episode rules."""
import json
from pathlib import Path
import re

from .bind_instances import normal, catalog, resolve_geometry

CATALOG_PATH = Path(__file__).with_name('binding_catalog_v2.json')
CONFIG = json.loads(CATALOG_PATH.read_text())
VERBS = {'pick up': 'Pick', 'pick': 'Pick', 'place': 'Place', 'put': 'Place',
         'hold': 'Hold', 'open': 'Open', 'close': 'Close', 'push': 'Push',
         'turn on': 'Turn', 'turn off': 'Turn', 'turn': 'Turn'}
ACTION = '|'.join(sorted(VERBS, key=len, reverse=True))
RELATION = r'on top of|next to|between|to left of|to right of|left of|right of|inside|into|onto|under|behind|near|from|on|in|to'
PART_ACTIONS = {'Open', 'Close', 'Turn'}


def words(text):
    return tuple(w[:-1] if w.endswith('s') and not w.endswith('ss') else w
                 for w in normal(text).split() if w not in {'and', 'both'})


def noun_match(query, label):
    q, label = words(query), words(label)
    return bool(q and label and q[-1] == label[-1] and set(q) <= set(label))


def describe(text, config=CONFIG):
    """Separate a part from its owner, retaining every modifier of either noun."""
    text = normal(text)
    parts = {p for specs in config['parts'].values() for p in specs}
    parts |= {p.split()[-1] for p in parts}
    for part in sorted(parts, key=len, reverse=True):
        if text == part:
            return {'name': None, 'part': part}
        if text.startswith(part + ' of '):
            return {'name': text[len(part)+4:], 'part': part}
        if text.endswith(' ' + part):
            return {'name': text[:-(len(part)+1)], 'part': part}
    return {'name': text, 'part': None}


def parse_action(text, config=CONFIG):
    text = normal(text)
    match = re.match(rf'^({ACTION})\s+(.+)$', text)
    if not match:
        return None
    split = re.split(rf'\s+((?:{RELATION})\s+.+)$', match[2], maxsplit=1)
    return {'action': VERBS[match[1]], 'verb': match[1],
            'object': describe(split[0], config), 'location': split[1] if len(split)>1 else None}


def task_clauses(task, config=CONFIG):
    return [p for clause in re.split(rf'\s+(?:and|then)\s+(?=(?:{ACTION})\b)', normal(task))
            if (p := parse_action(clause, config)) is not None]


class SceneIndex:
    """Small LIBERO adapter: categories, ownership and named-region semantics."""
    def __init__(self, scene, config=CONFIG):
        self.scene, self.config, self.types = scene, config, catalog(scene)
        self.labels = {name: [normal(kind), *config['aliases'].get(kind, [])]
                       for name, kind in self.types.items()}

    def entities(self, phrase):
        if not phrase:
            return set(self.types)
        q = words(phrase)
        exact = {name for name, labels in self.labels.items()
                 if any(q == words(label) or (q and words(label) and q[-1] == words(label)[-1]
                        and set(q) == set(words(label))) for label in labels)}
        if exact:
            return exact
        return {name for name, labels in self.labels.items() if any(noun_match(phrase, label) for label in labels)}

    def parts(self, instance):
        family = self.config['asset_families'].get(self.types[instance])
        return self.config['parts'].get(family, {})

    def targets(self, descriptor, action=None):
        result = set()
        for instance in self.entities(descriptor['name']):
            part = descriptor['part']
            if not part and action not in PART_ACTIONS:
                result.add((instance, 'whole'))
                continue
            options = self.parts(instance)
            for label, spec in options.items():
                if action is not None and action not in spec['actions']:
                    continue
                if part and noun_match(part, label):
                    result.add((instance, label))
                elif not part and (spec.get('default') or not any(v.get('default') for v in options.values())):
                    result.add((instance, label))
        return result

    def scope(self, target):
        instance, part = target
        return {'part': part, 'body_suffix': None if part == 'whole' else self.parts(instance)[part]['body']}

    def references(self, phrase):
        desc = describe(phrase, self.config)
        result = set()
        for instance, part in self.targets(desc):
            if part == 'whole':
                result.add(instance)
                result.update(n for n, r in self.scene['regions'].items() if r['target'] == instance)
            else:
                suffix = self.parts(instance)[part].get('region')
                if suffix and instance + '_' + suffix in self.scene['regions']:
                    result.add(instance + '_' + suffix)
        return result

    def _region_signature(self, name, region):
        """Decode source naming convention once; the query operates on semantic roles."""
        if 'semantic' in region:
            return region['semantic']
        tail = normal(name.removeprefix(region['target']+'_').removesuffix('_region'))
        if tail.startswith('next to '):
            return {'kind': 'next_to', 'references': [tail[8:]]}
        if tail.startswith('between '):
            tokens = tail[8:].split()
            for split in range(1, len(tokens)):
                refs = [' '.join(tokens[:split]), ' '.join(tokens[split:])]
                if all(len(self.entities(ref)) == 1 for ref in refs):
                    return {'kind': 'between', 'references': refs}
        for position in ['center', 'front', 'back']:
            if tail == position or (tail.endswith(' '+position)
                    and region['target'] in self.entities(tail[:-(len(position)+1)])):
                return {'kind': position, 'owner': region['target']}
        return None

    def relation(self, location, predicates=None, time_scope='initial'):
        match = re.fullmatch(rf'({RELATION})\s+(.+)', normal(location))
        if not match:
            return None
        relation, reference = match.groups()
        relation = {'on top of':'on', 'onto':'on', 'inside':'in', 'into':'in', 'next to':'next_to'}.get(relation, relation)
        refs, allowed = set(), {'on', 'in'}
        named = re.fullmatch(r'(.+) (center|front|back)', reference) if relation == 'from' else None
        if relation in {'on','in','from'} and not named:
            refs = self.references(reference)
            allowed = {relation} if relation != 'from' else {'on','in'}
            if not refs:
                return None
        elif relation in {'next_to','between'} or named:
            query_refs = reference.split(' and ') if relation == 'between' else [reference]
            expected_count = 2 if relation == 'between' else 1
            if not named and (len(query_refs) != expected_count or any(len(self.entities(r)) != 1 for r in query_refs)):
                return None
            for name, region in self.scene['regions'].items():
                signature = self._region_signature(name, region)
                if not signature:
                    continue
                if named:
                    if signature['kind'] == named[2] and signature.get('owner') in self.entities(named[1]):
                        refs.add(name)
                elif signature['kind'] == relation:
                    stored = signature.get('references', [])
                    if (len(stored) == expected_count and all(len(self.entities(r)) == 1 for r in stored)
                            and {next(iter(self.entities(r))) for r in stored}
                            == {next(iter(self.entities(r))) for r in query_refs}):
                        refs.add(name)
            if not refs:
                return None
        else:
            return None
        source = self.scene['initial_state'] if predicates is None else predicates
        matched = [p for p in source if len(p)==3 and p[0] in allowed and p[2] in refs]
        return {'method':'scene_relation', 'location':location, 'time_scope':time_scope,
                'regions_or_entities':sorted(refs), 'predicates':matched, 'candidates':sorted({p[1] for p in matched})}


def reference_descriptor(parsed, config=CONFIG):
    if not parsed['location']:
        return None
    match = re.fullmatch(rf'({RELATION})\s+(.+)', parsed['location'])
    return describe(match[2], config) if match else None


def bind_events(scene, events, task, config=CONFIG, motion=None, current_relations=None):
    """Return identity proposals and independent skill checks, preserving source fields."""
    index, clauses = SceneIndex(scene, config), task_clauses(task, config)
    task_parts = set()
    for clause in clauses:
        for desc in [clause['object'], reference_descriptor(clause, config)]:
            if desc and desc['part']:
                task_parts |= index.targets(desc)
    rows, carry, previous_rows = [], {}, {}
    for event in events:
        arm = event['arm']
        previous = previous_rows.get(arm)
        continuous = previous is not None and previous['stop_frame'] == event['start_frame']
        held = carry.pop(arm, None)
        if held and held['stop_frame'] != event['start_frame']:
            held = None
        row = {**event, 'instance':None, 'scope':None, 'status':'unresolved', 'candidates':[],
               'candidate_targets':[], 'evidence':[], 'geometry':{'status':'not_checked'},
               'motion':{'status':'not_measured'}, 'skill_check':'unknown', 'manual_review':'pending'}
        parsed = parse_action(event['description'], config)
        row['parsed'] = parsed
        previous_rows[arm] = row
        rows.append(row)
        if parsed is None:
            row['reason'] = 'unsupported_event_language'
            continue
        action, desc = parsed['action'], parsed['object']
        row['skill_check'] = 'consistent' if action.lower() == event['skill'].lower() else 'conflict'
        pool = index.targets(desc, action)
        pronoun = desc['name'] in {'it','object'} and desc['part'] is None
        if pronoun and action in {'Place','Hold'} and held:
            pool = {held['target']}
            row['evidence'].append({'method':'same_arm_pronoun','source_event':held['source_id']})
        if pronoun and action in PART_ACTIONS:
            pool = previous.get('_reference_targets', set()) if continuous else set()
        initial = pool.copy()
        row['candidates'] = sorted({t[0] for t in initial})
        row['evidence'].append({'method':'category_attributes_part','descriptor':desc,
                                'targets':[list(t) for t in sorted(pool)]})
        problem = None

        # Task spatial descriptors specify initial identity, never an Event ordinal.
        constraints = []
        if action in {'Pick','Place','Hold','Push'} and pool:
            nouns = {words(label)[-1] for t in pool for label in index.labels[t[0]]}
            for clause in clauses:
                q = clause['object']['name']
                if clause['action'] != 'Pick' or not q or not clause['location'] or words(q)[-1] not in nouns:
                    continue
                c = index.relation(clause['location'])
                if c is None:
                    problem = 'unsupported_task_spatial_relation'
                else:
                    c['candidates'] = sorted(set(c['candidates']) & index.entities(q))
                    constraints.append(c)
            if constraints:
                row['evidence'].extend(constraints)
                allowed = {n for c in constraints for n in c['candidates']}
                pool = {t for t in pool if t[0] in allowed}

        if desc['part'] and pool:
            related = {t for t in task_parts if any(t[0] == p[0] and t[1].split()[-1] == p[1].split()[-1] for p in pool)}
            if related:
                pool &= related
                row['evidence'].append({'method':'explicit_task_part','targets':[list(t) for t in sorted(related)]})
            if continuous and previous.get('_reference_targets'):
                refs = previous['_reference_targets']
                compatible = pool & refs
                if compatible:
                    pool = compatible
                    row['evidence'].append({'method':'previous_reference','source_event':previous['source_id']})

        if action == 'Pick' and parsed['location']:
            predicates = (current_relations or {}).get(event['source_id'])
            if event['start_frame'] > 0 and predicates is None:
                problem = 'current_spatial_evidence_unavailable'
            else:
                c = index.relation(parsed['location'], predicates, 'current' if predicates is not None else 'initial')
                if c is None:
                    problem = 'unsupported_event_spatial_relation'
                else:
                    pool = {t for t in pool if t[0] in c['candidates']}
                    row['evidence'].append(c)

        ref = reference_descriptor(parsed, config)
        refs = index.targets(ref) if ref else set()
        if ref and ref['part'] and task_parts & refs:
            refs &= task_parts
        row['_reference_targets'] = refs
        row['reference_candidates'] = sorted({t[0] for t in refs})
        if action in {'Place','Hold'} and held:
            if held['target'] not in pool:
                problem = 'continuation_conflict'
                pool = set()
            else:
                pool = {held['target']}
                row['evidence'].append({'method':'same_arm_continuation','source_event':held['source_id'],
                                        'previous_event_motion_supported':held['motion_supported']})

        row['candidate_targets'] = [{'instance':t[0], 'scope':index.scope(t)} for t in sorted(pool)]
        measured = (motion or {}).get(event['source_id'])
        if measured is not None:
            row['motion'] = measured
            supported = {(m['instance'],m['part']) for m in measured['candidates'] if m['supported']}
            agreed = pool & supported
            if not problem and len(pool)>1 and len(agreed)==1 and len(supported)==1:
                pool = agreed
                row['evidence'].append({'method':'contact_and_motion','target':list(next(iter(pool)))})
            elif not problem and len(pool)==1 and supported and not agreed:
                problem = 'motion_disagrees_with_identity'

        row['remaining_candidates'] = sorted({t[0] for t in pool})
        if problem:
            row['reason'] = problem
            row['status'] = 'conflict' if 'conflict' in problem or 'disagrees' in problem else 'unresolved'
        elif not pool:
            row['reason'] = 'no_consistent_candidate'
            row['status'] = 'conflict' if initial else 'unresolved'
        elif len(pool)>1:
            row.update(status='ambiguous', reason='multiple_targets')
        else:
            target = next(iter(pool))
            row.update(status='rule_bound', instance=target[0], scope=index.scope(target))
            if action in {'Pick','Hold'}:
                supported = measured is not None and any(m['supported'] and (m['instance'],m['part']) == target for m in measured['candidates'])
                carry[arm] = {'target':target, 'stop_frame':event['stop_frame'], 'source_id':event['source_id'],
                              'motion_supported':supported}
    for row in rows:
        row.pop('_reference_targets', None)
    return rows
