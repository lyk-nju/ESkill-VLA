"""Augmentation phase 1: one full task with local edits and Event target supervision.

Conventional batch geometry/color follows in augmentation.augment_training_batch.
"""
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from openpi.augmentation import COPY_MIN_VISIBLE_FRACTION
from openpi.policies.eskill_dataset import EpisodeReader
from openpi.transforms import (ModelTransforms, mirror_eligible, normalized_image_boxes, resize_image, role_canvas,
                               source_pixel_geometry, spatial_augmentation_eligible)


# Names used in full LIBERO instructions, in addition to the display captions.
TRANSFER_ALIASES = {
    'black bowl': 'bowl',
    'flat stove': 'stove',
    'wine rack': 'rack',
    'wooden cabinet': 'cabinet',
    'white cabinet': 'cabinet',
}
DISPLAY_NAMES = {'akita_black_bowl': 'black bowl', 'porcelain_mug': 'white mug',
                 'white_yellow_mug': 'yellow and white mug', 'glazed_rim_porcelain_ramekin': 'ramekin'}
SPATIAL_ALIASES = {'cookies': 'cookie box', 'black_book': 'book', 'desk_caddy': 'caddy'}

def object_name(entity):
    category = entity['category']
    return DISPLAY_NAMES.get(category, category.replace('_', ' '))


def entity_captions(entities, boxes):
    """Natural descriptions, with image-relative qualifiers for same-category instances."""
    result, groups = {}, {}
    for key in boxes:
        e = entities[key]
        noun = object_name(e)
        if e['kind'] in ('object', 'fixture'):
            result[key] = noun
            groups.setdefault(noun, []).append(key)
    for noun, keys in groups.items():
        if len(keys) < 2:
            continue
        keys.sort(key=lambda k: float(boxes[k][0]+boxes[k][2]))
        centers = [float(boxes[k][0]+boxes[k][2]) for k in keys]
        if min(np.diff(centers)) < 2:
            for key in keys:
                result.pop(key)  # No instance-number pseudo-language for overlapping indistinguishable objects.
            continue
        for i, key in enumerate(keys):
            position = 'leftmost' if i == 0 else 'rightmost' if i == len(keys)-1 else f'number {i+1} from the left'
            result[key] = f'{position} {noun} in the image'
    # Parts/regions inherit the already disambiguated parent, never an instance-number caption.
    for key in boxes:
        e = entities[key]
        if e['kind'] not in ('part', 'region'):
            continue
        owner = result.get(e['instance'])
        parent = entities[e['instance']]
        # Background surfaces have no object bbox, but may own a genuine projected goal region.
        if (owner is None and e['kind'] == 'region' and parent['kind'] == 'background'
                and sum(v['category'] == parent['category'] for v in entities.values() if v['kind'] == 'background') == 1):
            owner = parent['category'].replace('_', ' ')
        if owner is None:
            continue
        if e['kind'] == 'part':
            result[key] = f'{e["part"].replace("_", " ")} of the {owner}'
        else:
            tail = e['site_name'].removeprefix(e['instance']+'_').removesuffix('_region')
            result[key] = (f'{tail[:-8].replace("_", " ")} compartment of the {owner}' if tail.endswith('_contain') else
                           f'{tail.replace("_", " ")} region of the {owner}')
    return result


def simple_transfer(task, entities, boxes):
    """Resolve a single source/destination noun pair; relational/multi-object tasks stay original."""
    match = re.fullmatch(
        r'(?:pick up|put|place) (?:the )?(?P<source>.+?)'
        r'(?P<relation>(?: and (?:put|place) it)? (?:on top of|in|into|on|onto) (?:the )?)'
        r'(?P<destination>.+?)[.!]?', task.strip(), re.IGNORECASE)
    if not match:
        return None
    captions = entity_captions(entities, boxes)
    def resolve(noun):
        matches = []
        for key, caption in captions.items():
            if entities[key]['kind'] not in ('object', 'fixture'):
                continue
            names = {caption, caption.removesuffix(' box')}
            if caption in TRANSFER_ALIASES:
                names.add(TRANSFER_ALIASES[caption])
            if noun.lower() in names:
                matches.append(key)
        return matches[0] if len(matches) == 1 else None
    source, destination = resolve(match['source']), resolve(match['destination'])
    if source is None or destination is None or source == destination:
        return None
    return source, destination, match


def drawer_transfer(task, entities, events):
    """Bind the put clause of LIBERO-Goal's open-then-place task to its Events."""
    match = re.fullmatch(r'open the top drawer and put the (?P<source>bowl) inside[.!]?',
                         task.strip(), re.IGNORECASE)
    if not match or [event['skill'] for event in events] != ['Open', 'Pick', 'Place']:
        return None
    opened, source, destination = (e.get('interaction_target', {}).get('entity_id') for e in events)
    drawer = entities.get(destination, {})
    if (opened != destination or drawer.get('part') != 'top drawer' or drawer.get('category') != 'wooden_cabinet'
            or entities.get(source, {}).get('category') != 'akita_black_bowl'):
        return None
    return source, destination, match


def task_has_spatial_relation(task):
    """Detect when absolute or relative entity positions help define the task."""
    spatial = re.search(r'\b(left|right|leftmost|rightmost|front|back|center|middle|top|bottom|beside|behind|'
                        r'between|next to|closest|nearest)\b', task, re.IGNORECASE)
    first_clause = re.split(r'\band (?:place|put)\b',task,maxsplit=1,flags=re.IGNORECASE)[0]
    source_relation = re.search(r'\b(pick up|grasp|lift)\b.+?\b(on|in|from)\b', first_clause, re.IGNORECASE)
    return bool(spatial or source_relation)


def independent_event_copy_target(task, entities, events, event):
    """Find a named Current that is independent of the task's spatial references."""
    if not event or event['skill'] not in ('Pick', 'Place'):
        return None
    current = event['interaction_target']['entity_id']
    if entities.get(current, {}).get('kind') != 'object':
        return None

    names = {}
    for key, entity in entities.items():
        if entity['kind'] not in ('object', 'fixture'):
            continue
        caption = object_name(entity)
        names[key] = {caption, caption.removesuffix(' box'), TRANSFER_ALIASES.get(caption, caption),
                      SPATIAL_ALIASES.get(entity['category'], caption)}

    def mentions_current(text):
        return any(re.search(r'\b' + re.escape(name) + r'\b', text, re.IGNORECASE)
                   for name in names[current])

    if event['skill'] == 'Place':
        match = re.fullmatch(
            r'pick up (?P<pick>.+?) and (?:place|put) it (?:into|onto|in|on) (?:the )?'
            r'(?P<destination>.+?)[.!]?', task.strip(), re.IGNORECASE,
        )
        if (match and [e['skill'] for e in events] == ['Pick', 'Place']
                and not mentions_current(match['pick'])
                and [key for key, aliases in names.items() if match['destination'].lower() in aliases] == [current]):
            return current
        return None

    # Named pick objects are independent of position words attached to their destinations.
    clauses = re.split(r'\s+and\s+(?=(?:put|place)\s)', task.strip(), flags=re.IGNORECASE)
    sources = []
    for clause in clauses:
        match = re.fullmatch(
            r'(?:put|place) (?:the )?(?P<source>.+?) (?:to the (?:left|right) of|onto|into|on|in) '
            r'(?P<destination>.+?)[.!]?', clause, re.IGNORECASE,
        )
        if not match or mentions_current(match['destination']):
            return None
        resolved = [key for key, aliases in names.items() if match['source'].lower() in aliases]
        if len(resolved) != 1:
            return None
        sources.append(resolved[0])
    return current if sources.count(current) == 1 else None


def spatial_copy_protection(task, entities, events, transfer, *, event=None):
    """Protect reference entities and annotated regions in spatial tasks."""
    if not task_has_spatial_relation(task):
        return set()
    if transfer and 'on top of' in transfer[2]['relation'].lower():
        # A simple placement on top of a named surface does not specify the source location.
        return {transfer[1]}

    protected = set()
    for annotation in events:
        target = (annotation.get('interaction_target') or {}).get('entity_id')
        manipulated = (annotation.get('source_annotation') or {}).get('instance')
        for key in (target, manipulated):
            if key in entities:
                protected.add(key)
                parent = entities[key].get('instance')
                if parent in entities and entities[parent]['kind'] != 'background':
                    protected.add(parent)

    # Some references never become Event targets, such as the cookie box in "bowl next to the cookie box".
    for key, entity in entities.items():
        if entity['kind'] not in ('object', 'fixture'):
            continue
        caption = object_name(entity)
        names = {caption, TRANSFER_ALIASES.get(caption, caption), SPATIAL_ALIASES.get(entity['category'], caption)}
        if any(re.search(r'\b' + re.escape(name) + r'\b', task, re.IGNORECASE) for name in names):
            protected.add(key)
    # Lock same-category instances too, since moving one can change which is leftmost or nearest.
    categories = {entities[key]['category'] for key in protected if entities[key]['kind'] == 'object'}
    protected.update(key for key, entity in entities.items() if entity['kind'] == 'object'
                     and entity['category'] in categories)
    current = independent_event_copy_target(task, entities, events, event)
    if current is not None:
        protected.discard(current)
    return protected


def visible_pixels(reader, sequence, row, shape):
    ref = sequence.get('visible_masks')
    valid = sequence.get('visible_masks_valid')
    if row is None or ref is None or (valid is not None and not reader.array(valid)[row]):
        return None
    value = reader._raw(ref['uri'], ref['key'])[row]
    if ref.get('encoding') == 'packbits':
        value = np.unpackbits(value, count=shape[0] * shape[1], bitorder='big')
    return value.reshape(shape).astype(bool)


def check_stage_v_visible_masks(reader, camera, frames):
    """Require visibility labels for physical entities with boxes on sampled frames."""
    for key, entity in reader.meta['scene_entities'].items():
        sequence = entity['regions'].get(camera)
        if entity['kind'] not in ('object', 'fixture', 'part') or not sequence or not sequence.get('boxes_xyxy'):
            continue
        indices = reader.array(sequence['frame_indices'])
        required = reader.array(sequence['boxes_valid']) & np.isin(indices, frames)
        if not required.any():
            continue
        error = (f"{reader.meta['episode_id']}/{camera}/{key}: Visual Primitive requires a valid visible mask "
                 "for every sampled frame with a valid physical bbox. Regenerate or sync the visibility labels.")
        try:
            ref = sequence['visible_masks']
            masks = reader._raw(ref['uri'], ref['key'])
            valid_ref = sequence.get('visible_masks_valid')
            valid = reader.array(valid_ref) if valid_ref is not None else np.ones_like(required)
            if masks.ndim < 2 or len(masks) != len(indices) or valid.shape != indices.shape:
                raise ValueError('Mask rows do not match the entity frames')
        except (KeyError, TypeError, OSError, ValueError) as exc:
            raise ValueError(error) from exc
        missing = required & ~valid.astype(bool)
        if missing.any():
            frame = indices[np.flatnonzero(missing)[0]]
            raise ValueError(f'{error} First invalid frame: {frame}.')


def box_coverage(box, shape):
    """Intersection area of a pixel square with a source-image xyxy box."""
    if box is None:
        return np.zeros(shape, np.float32)
    y, x = (np.arange(n, dtype=np.float32) for n in shape)
    dx = (np.minimum(x + 1, box[2]) - np.maximum(x, box[0])).clip(0, 1)
    dy = (np.minimum(y + 1, box[3]) - np.maximum(y, box[1])).clip(0, 1)
    return dy[:, None] * dx


class GroundingDataset:
    def __init__(self, index, config, tokenizer, *, episode_ids=None, augment=True):
        self.config, self.tokenizer, self.augment = config, tokenizer, augment
        self.transform = ModelTransforms(config.model, {}, tokenizer)
        index = Path(index)
        self.paths = [index.parent / e['path'] for e in json.loads(index.read_text())['episodes']
                      if episode_ids is None or e['episode_id'] in episode_ids]
        self.samples = []
        self.visual_primitive = config.training_stage == 'visual_primitive'
        self.destination_types = {}
        for i in range(len(self.paths)):
            reader = self.reader(i)
            if self.visual_primitive and augment:
                # Observed source/relation/destination types permit conservative goal substitutions.
                events = reader.meta['events']
                if [e['skill'] for e in events] == ['Pick', 'Place'] and all(
                        e.get('boundary_valid', True) and e.get('skill_valid', True) and
                        (e.get('interaction_target') or {}).get('status', 'resolved') == 'resolved' for e in events):
                    scene = reader.meta['scene_entities']
                    source, dest = [scene.get((e.get('interaction_target') or {}).get('entity_id')) for e in events]
                    relations = re.findall(r'\b(on top of|on|onto|in|into)\b', reader.meta['instructions'][0].lower())
                    if source and dest and dest['kind'] in ('object', 'fixture') and relations:
                        relation = 'in' if relations[-1] in ('into', 'in') else 'on'
                        self.destination_types.setdefault((source['category'], relation), set()).add(dest['category'])
            if not reader.meta.get('scene_entities'):
                raise ValueError('Grounding training needs scene_entities and interaction_target annotations')
            if config.model.grounding_subject and not self.visual_primitive and not reader.meta.get('effector_regions', {}).get('single', {}).get(config.cameras[0]):
                raise ValueError('Subject grounding needs saved gripper effector_regions for the main camera')
            aug = config.grounding_augmentation
            if augment and aug.target_swap_prob:
                for key, entity in reader.meta['scene_entities'].items():
                    sequence = entity['regions'].get(config.cameras[0])
                    if entity['kind'] != 'object' or sequence is None:
                        continue
                    error = (f'{reader.meta["episode_id"]}: counterfactual target swapping requires '
                             f'per-frame boolean interaction_locked for {key}; run '
                             'python -m scripts.preprocess.libero.interaction_locks '
                             f'--index {index} --output <new-report-directory>')
                    try:
                        locked = reader.array(entity['interaction_locked'])
                    except (KeyError, OSError, ValueError) as exc:
                        raise ValueError(error) from exc
                    if locked.dtype.kind != 'b' or locked.shape != reader.array(sequence['frame_indices']).shape:
                        raise ValueError(error)
            if augment and (aug.copy_prob or aug.patch_prob) and 'online_grounding' not in reader.meta:
                raise ValueError('Run interaction_review --stage augment to prepare visible masks and editing metadata')
            if augment and aug.copy_prob:
                robot = reader.meta['scene_entities'].get('robot::single', {}).get('regions', {}).get(config.cameras[0], {})
                if not robot.get('masks') or not robot.get('masks_valid') or not robot.get('frame_indices'):
                    raise ValueError(f'{reader.meta["episode_id"]}: Copy-Paste requires robot::single masks for '
                                     f'{config.cameras[0]}; regenerate scene regions with protect_robot=true')
                frames = reader.array(reader.meta['cameras'][config.cameras[0]]['frame_indices'])
                mask_valid = reader.array(robot['masks_valid'])
                masks = reader._raw(robot['masks']['uri'], robot['masks']['key'])
                if (not np.array_equal(reader.array(robot['frame_indices']), frames)
                        or mask_valid.shape != frames.shape or not mask_valid.all() or len(masks) != len(frames)):
                    raise ValueError(f'{reader.meta["episode_id"]}: Copy-Paste requires valid robot::single masks '
                                     'for every main-camera frame')
            stream = reader.meta['cameras'][config.cameras[0]]
            frames = reader.array(stream['frame_indices'])
            if self.visual_primitive:
                frames = frames[reader.array(stream['valid'])]
                check_stage_v_visible_masks(reader, config.cameras[0], frames[::config.frame_stride])
            self.samples.extend((i, int(f)) for f in frames[::config.frame_stride])

    @lru_cache(maxsize=8)
    def reader(self, index):
        return EpisodeReader(self.paths[index])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index, *, review=False):
        from openpi.augmentation import (GroundingAugmentation, copy_objects, mask_grounding_patches,
                                        augmentation_invalidates)
        # Keys come from the batch sampler: replayable across workers and mid-epoch resumes.
        index, epoch, draw = index if isinstance(index, tuple) else (index, 0, index)
        episode, frame = self.samples[index]
        reader = self.reader(episode)
        entities, meta = reader.meta['scene_entities'], reader.meta
        camera = self.config.cameras[0]
        original = reader.rgb(camera, frame)
        h, w = original.shape[:2]
        seed = int.from_bytes(hashlib.sha256(f'{self.config.seed}:{meta["episode_id"]}:{frame}:{epoch}:{draw}'.encode()).digest()[:8], 'little')
        copy_rng, patch_rng, query_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        if self.visual_primitive:
            # The same host frame in an epoch has one augmented RGB across all query draws.
            host = f'{self.config.seed}:{meta["episode_id"]}:{frame}:{epoch}'.encode()
            host_seed = int.from_bytes(hashlib.sha256(host).digest()[:8], 'little')
            copy_rng, patch_rng, geometry_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(host_seed).spawn(3)]
        aug = self.config.grounding_augmentation if self.augment else GroundingAugmentation()
        boxes, rows, visible_fractions, locked = {}, {}, {}, {}
        for key, entity in entities.items():
            seq = entity['regions'].get(camera)
            region = reader.region(seq, frame)
            if self.visual_primitive and region is None and seq and seq.get('visible_masks'):
                # A visible segmentation may still supply the target when its full box is unavailable.
                pixels = visible_pixels(reader, seq, reader.row(seq, frame), (h, w))
                if pixels is not None and pixels.any():
                    y, x = np.nonzero(pixels)
                    region = {'bbox': np.array([x.min(), y.min(), x.max()+1, y.max()+1], np.float32)}
            if region and region['bbox'] is not None and entity['kind'] in ('object','fixture','part','region'):
                boxes[key] = region['bbox']
                rows[key] = reader.row(seq, frame)
                if entity.get('visible_fraction'):
                    visible_fractions[key] = float(reader.array(entity['visible_fraction'])[rows[key]])
                if entity.get('interaction_locked'):
                    locked[key] = bool(reader.array(entity['interaction_locked'])[rows[key]])
        original_boxes = boxes
        event = reader.event_at('single', frame)
        real = (event.get('interaction_target') or {}).get('entity_id') if event else None
        task = meta['instructions'][0]
        transfer = simple_transfer(task, entities, boxes)
        task_protected = (spatial_copy_protection(task, entities, meta['events'], transfer, event=event)
                          if aug.copy_prob else set())
        if aug.copy_prob and real in boxes and entities[real]['kind'] == 'region':
            # Preserve the labeled region and keep a movable parent attached to it.
            task_protected.update((real, entities[real]['instance']))
        gripper = reader.region(meta.get('effector_regions', {}).get('single', {}).get(camera), frame)
        gripper_box = gripper['bbox'] if gripper else None
        pad = aug.copy_gripper_margin_px
        gripper_near = None if gripper_box is None else gripper_box + [-pad, -pad, pad, pad]
        # Copy uses the current box margin. Stored locks still serve text selection.
        near_entities = {
            key for key, box in boxes.items() if entities[key]['kind'] in ('object', 'part')
            and np.all(box[:2] < gripper_near[2:]) and np.all(gripper_near[:2] < box[2:])
        } if aug.copy_prob and gripper_near is not None else set()
        copy_candidates = [
            key for key in boxes if entities[key]['kind'] == 'object'
            and key not in near_entities and key not in task_protected
            and not (self.visual_primitive and locked.get(key, False))
            and visible_fractions.get(key, 0.) >= np.float32(COPY_MIN_VISIBLE_FRACTION)
        ]
        release_pick_source = bool(
            not self.visual_primitive and aug.copy_prob and event and event['skill'] == 'Pick'
            and frame < event.get('copy_precontact_stop_frame', event['start_frame'])
            and real in near_entities and entities[real]['kind'] == 'object'
            and real not in task_protected
            and visible_fractions.get(real, 0.) >= np.float32(COPY_MIN_VISIBLE_FRACTION))
        if release_pick_source:
            copy_candidates.append(real)
        contact = None
        if 'online_grounding' in meta:
            # Supplemental metadata is dense in source-frame order.
            contact = reader.array(meta['online_grounding']['interaction_boxes'])[frame]

        def visible(key, *, offset=None):
            sequence = entities[key]['regions'][camera]
            mask = visible_pixels(reader, sequence, rows.get(key), (h, w))
            if mask is not None and offset is not None:
                dx, dy = offset
                mask = np.roll(mask, (dy, dx), (0, 1))
            return mask

        def robot_mask():
            return reader.region(entities['robot::single']['regions'][camera], frame, include_mask=True)['mask']

        # Protect actual components independently so the empty corners of their union stay editable.
        physical = {k:v for k,v in boxes.items() if entities[k]['kind'] != 'region'}
        task_boxes = tuple(box for key, box in boxes.items() if key in task_protected)
        interaction_boxes = tuple(box for key, box in physical.items() if key in near_entities
                                  and not (release_pick_source and key == real))
        edited, updated, info = copy_objects(original, physical, copy_candidates, visible, robot_mask, copy_rng, aug,
                                             priority_target=real, protected_boxes=(*interaction_boxes, *task_boxes),
                                             gripper_box=gripper_near, release_priority_source=release_pick_source)
        moved_offsets = info['moved_offsets']
        boxes = dict(boxes, **updated)
        captions = entity_captions(entities, boxes)  # Recompute spatial qualifiers AFTER moving objects.
        if self.visual_primitive:
            return self._visual_sample(reader, frame, event, real, task, transfer, edited, boxes, captions,
                                       gripper_box, visible, visible_fractions, locked, info, aug,
                                       patch_rng, query_rng, geometry_rng, review)
        key = real if real in boxes else None
        text = task
        # Localization, full-task swapping and pixel editing have separate criteria.
        locatable_objects = [k for k in captions if entities[k]['kind'] == 'object'
                             and visible_fractions.get(k, 0.) >= aug.target_min_visible_fraction]
        swap_candidates = []
        text_transfer = transfer or drawer_transfer(task, entities, meta['events'])
        if text_transfer and event and event['skill'] == 'Place':
            source, destination, match = text_transfer
            if real == destination:
                # A different requested source becomes the next object to pick up.
                swap_candidates = [k for k in locatable_objects if k not in (source, destination)
                                   and 'in the image' not in captions[k] and not locked.get(k, True)]
        def swapped_task(candidate):
            start, end = match.span('source')
            return match.string[:start] + captions[candidate] + match.string[end:]

        if swap_candidates and query_rng.random() < aug.target_swap_prob:
            key = swap_candidates[int(query_rng.integers(len(swap_candidates)))]
            text = swapped_task(key)
        region = {'bbox': boxes[key]} if key else None
        canvas, weight, valid = role_canvas(edited, region, self.config.model.canvas_size)
        subject_canvas, subject_weight, subject_valid = role_canvas(edited, gripper, self.config.model.canvas_size)
        selected_visible = visible(key, offset=moved_offsets.get(key)) if key else None
        masked, drop = mask_grounding_patches(edited, contact, patch_rng, aug, self.config.model.image_size,
                                              target_box=boxes[key] if key else None, gripper_box=gripper_box)
        invalidated = augmentation_invalidates(selected_visible, drop)
        valid = valid and not invalidated

        def inputs(rgb, prompt):
            return self.transform.inputs(dict(images=[rgb], image_mask=np.ones(1, bool), prompt=prompt,
                                               arm_presence=np.array([True, False])))
        roles = [(canvas, weight, valid, boxes.get(key))]
        if self.config.model.grounding_subject:
            roles.append((subject_canvas, subject_weight, subject_valid, gripper_box))
        canvases = np.full((2,len(roles),3,*canvas.shape[:2]), -1., np.float32)
        weights = np.zeros((2,len(roles),*weight.shape), np.float32)
        role_valid = np.zeros((2,len(roles)), bool)
        # Normalized edge coordinates on the main input, before batch augmentation.
        core_boxes = np.zeros((2,len(roles),4), np.float32)
        for i,(rgb,foreground,available,box) in enumerate(roles):
            canvases[0,i] = rgb.transpose(2,0,1).astype(np.float32)/127.5-1
            weights[0,i] = foreground
            role_valid[0,i] = available
            if box is not None:
                core_boxes[0,i] = normalized_image_boxes(box, original.shape, self.config.model.image_size)[0]
        info.update(target_swapped=bool(key and key != real), target_moved=key in moved_offsets,
                    locatable_object_count=len(locatable_objects), swap_candidate_count=len(swap_candidates),
                    copy_candidate_count=len(copy_candidates), copy_task_protected=sorted(task_protected),
                    copy_precontact_candidate=release_pick_source,
                    missing_interaction_lock_count=sum(k not in locked for k in locatable_objects),
                    transfer_resolved=text_transfer is not None,
                    target_visible_fraction=visible_fractions.get(key),
                    patch_fraction=float(drop.mean()), augmentation_invalidated=invalidated,
                    native_invisible=bool(selected_visible is not None and not selected_visible.any()),
                    current_valid=bool(valid))
        if self.config.model.grounding_subject:
            info['subject_valid'] = bool(subject_valid)
        result = dict(inputs=inputs(masked, text), targets=dict(role_images=canvases, role_weights=weights,
                      role_valid=role_valid, role_core_boxes=core_boxes),
                      metadata=dict(episode_id=meta['episode_id'], frame=frame, entity_id=key or '',
                                    task=text, source_task=task, real_event_target=key == real,
                                    mirror_eligible=mirror_eligible(text),
                                    spatial_augmentation_eligible=spatial_augmentation_eligible(text),
                                    augmentation=info))
        if self.config.model.entity_contrast_enabled:
            candidates = []
            for candidate, box in boxes.items():
                entity = entities[candidate]
                if (candidate not in captions or entity['kind'] == 'region'
                        or not visible_fractions.get(candidate, 0.) >= .8):
                    continue
                if candidate != key and key and (entity.get('instance') == key
                                                  or entities[key].get('instance') == candidate
                                                  or np.array_equal(box, boxes[key])):
                    continue
                if candidate not in moved_offsets and entity.get('instance') in moved_offsets:
                    continue
                x0, y0 = np.floor(box[:2]).astype(int)
                x1, y1 = np.ceil(box[2:]).astype(int)
                if drop[y0:y1, x0:x1].any():
                    pixels = visible(candidate, offset=moved_offsets.get(candidate))
                    if pixels is None:
                        continue
                    fraction = visible_fractions[candidate] * (pixels & ~drop).sum() / max(1, pixels.sum())
                    if fraction < .8:
                        continue
                candidates.append(candidate)
            entity_boxes = normalized_image_boxes([boxes[k] for k in candidates], original.shape, self.config.model.image_size)
            result['targets'].update(entity_boxes=entity_boxes,
                                     positive_entity=candidates.index(key) if key in candidates else -1)
        if review:
            result['review'] = dict(original=original, edited=edited, masked=masked, canvas=canvas, drop=drop,
                                    boxes=boxes, original_boxes=original_boxes, target=key,
                                    visible=selected_visible, interaction_box=contact,
                                    gripper_box=gripper_box, subject_canvas=subject_canvas)
            if not self.augment and swap_candidates:
                alternative = sorted(swap_candidates)[int(query_rng.integers(len(swap_candidates)))]
                result['review']['paired_query'] = (swapped_task(alternative), alternative)
        return result

    def _visual_sample(self, reader, frame, event, real, task, transfer, edited, boxes, captions,
                       gripper_box, visible, fractions, locked, info, aug, patch_rng, query_rng,
                       geometry_rng, review):
        from openpi.augmentation import mask_grounding_patches

        meta, size = reader.meta, self.config.model.image_size
        scene, shape = meta['scene_entities'], edited.shape[:2]
        # Decide from the original task before selecting a query or a target.
        spatial_patch_skip = bool(aug.patch_prob and task_has_spatial_relation(task))
        if spatial_patch_skip:
            rgb, drop = edited, np.zeros(shape, dtype=bool)
        else:
            rgb, drop = mask_grounding_patches(edited, None, patch_rng, aug, size, gripper_box=gripper_box)
        offsets = info['moved_offsets']
        masks, appearances = {}, {}
        for entity in boxes:
            if scene[entity]['kind'] == 'region' or scene[entity].get('instance') in offsets and entity not in offsets:
                continue
            mask = visible(entity, offset=offsets.get(entity))
            if mask is None:
                continue
            kept = mask & ~drop
            masks[entity] = kept
            area = mask.sum()
            if area and kept.any():
                appearances[entity] = fractions.get(entity, 0.) * kept.sum() / area

        binding = (event.get('interaction_target') or {}) if event else {}
        competing = any(e is not event and e['arm'] == 'single' and e['start_frame'] <= frame < e['stop_frame']
                        and (e.get('interaction_target') or {}).get('entity_id') != real for e in meta['events'])
        key = (real if event and event.get('boundary_valid', True) and event.get('skill_valid', True)
               and not competing and binding.get('status', 'resolved') == 'resolved' and real in boxes else None)
        text = task
        alternatives = []
        transfer = transfer or drawer_transfer(task, scene, meta['events'])
        if key and transfer and event['skill'] == 'Place':
            source, dest, match = transfer
            if key == dest:
                relation = ('in' if 'relation' not in match.groupdict() or
                            re.search(r'\b(in|into)\b', match['relation']) else 'on')
                allowed_goals = self.destination_types.get((scene[source]['category'], relation), ())
                for candidate, fraction in appearances.items():
                    caption = captions.get(candidate)
                    if (fraction < aug.target_min_visible_fraction or candidate in (source, dest)
                            or not caption or 'in the image' in caption):
                        continue
                    roles = []
                    if scene[candidate]['kind'] == 'object' and not locked.get(candidate, True):
                        roles.append('source')
                    if ('destination' in match.groupdict()
                            and scene[candidate]['kind'] in ('object','fixture')
                            and scene[candidate]['category'] in allowed_goals):
                        roles.append('destination')
                    for field in roles:
                        lo, hi = match.span(field)
                        alternatives.append((match.string[:lo] + caption + match.string[hi:], candidate, field))
        swap = None
        if alternatives and query_rng.random() < aug.target_swap_prob:
            text, key, swap = alternatives[int(query_rng.integers(len(alternatives)))]

        current = masks.get(key)
        if key and scene[key]['kind'] == 'region':
            # Projected task sites have geometry, but no material surface with a visibility mask.
            seq = scene[key]['regions']['image']
            if seq.get('visible_masks'):
                current = visible(key)
            elif fractions.get(key, 1.) > 0:
                site = reader.region(seq, frame, include_mask=True)
                current = site['mask'] if site and site['mask'] is not None else box_coverage(boxes[key], shape)
            if current is not None:
                current = current * ~drop
        subject_seq = meta.get('effector_regions', {}).get('single', {}).get('image')
        subject = None
        if subject_seq and subject_seq.get('visible_masks'):
            subject = visible_pixels(reader, subject_seq, reader.row(subject_seq, frame), shape)
        else:
            subject = box_coverage(gripper_box, shape)
        def canvas(value):
            if value is None:
                return np.zeros((size, size), np.float32)
            return resize_image(value.astype(np.float32), size)

        candidates = [k for k, fraction in appearances.items()
                      if fraction >= .8 and k in captions and (k == key or not key or (
                          scene[k].get('instance') != key and scene[key].get('instance') != k
                          and not np.array_equal(boxes[k], boxes[key])))]
        def norm_box(box):
            return normalized_image_boxes([box], shape, size)[0] if box is not None else np.zeros(4, np.float32)

        targets = dict(current_mask=canvas(current), subject_mask=canvas(subject * ~drop if subject is not None else None),
                       current_box=norm_box(boxes.get(key)), subject_box=norm_box(gripper_box),
                       center_px_scale=source_pixel_geometry(shape, size)[0].astype(np.float32),
                       entity_boxes=normalized_image_boxes([boxes[k] for k in candidates], shape, size),
                       positive_entity=candidates.index(key) if key in candidates else -1)
        info.update(target_swapped=bool(swap), target_moved=key in offsets,
                    patch_fraction=float(drop.mean()), patch_skipped_for_spatial_relation=spatial_patch_skip,
                    current_valid=bool(targets['current_mask'].any()),
                    subject_valid=bool(targets['subject_mask'].any()))
        # Mark the observations on either side of an internal Event transition.
        neighbors = (reader.event_at('single', frame + d) for d in (-1, 1))
        boundary = competing or bool(event and any(e and e is not event for e in neighbors))
        metadata = dict(episode_id=meta['episode_id'], frame=frame, entity_id=key or '',
                        parent_trajectory_id=meta.get('source', {}).get('parent_trajectory_id', meta['episode_id']),
                        task=text, source_task=task, real_event_target=key is not None and not swap,
                        counterfactual=swap or 'none', event_boundary=boundary, augmentation=info,
                        # Paired prompts use the host geometry and the same phase-2 random numbers.
                        mirror_eligible=mirror_eligible(task), spatial_augmentation_eligible=spatial_augmentation_eligible(task),
                        image_draws=geometry_rng.random(11, dtype=np.float32).tolist())
        result = dict(inputs=self.transform.inputs(dict(images=[rgb], image_mask=np.ones(1, bool), prompt=text)),
                      targets=targets, metadata=metadata)
        if review:
            result['review'] = dict(masked=rgb, boxes=boxes, drop=drop, target=key, paired_queries=alternatives,
                                    candidate_entities=candidates)
        return result


def collate_grounding(samples):
    import torch
    from torch.utils.data._utils.collate import default_collate
    inputs = default_collate([s['inputs'] for s in samples])
    def trim(value):
        length = int(value['token_mask'].sum(-1).max())
        for key in ('tokens','token_mask'):
            value[key] = value[key][:,:length]
    trim(inputs)
    rows = [dict(s['targets']) for s in samples]
    if 'entity_boxes' in rows[0]:
        boxes = [torch.as_tensor(row.pop('entity_boxes')) for row in rows]
        width = max(1, max(len(value) for value in boxes))
        padded = torch.zeros(len(rows), width, 4)
        for i, value in enumerate(boxes):
            padded[i, :len(value)] = value
        targets = default_collate(rows)
        targets['entity_boxes'] = padded
        targets['entity_valid'] = torch.arange(width)[None] < torch.tensor([len(v) for v in boxes])[:, None]
    else:
        targets = default_collate(rows)
    return dict(inputs=inputs, targets=targets,
                metadata=[s['metadata'] for s in samples])
