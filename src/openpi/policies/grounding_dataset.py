"""Augmentation phase 1: one full task with local edits and Event target supervision.

Conventional batch geometry/color follows in augmentation.augment_training_batch.
"""
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from openpi.transforms import ModelTransforms, role_canvas, mirror_eligible, resize_geometry


def entity_captions(entities, boxes):
    """Natural descriptions, with image-relative qualifiers for same-category instances."""
    aliases = {'akita_black_bowl': 'black bowl', 'porcelain_mug': 'white mug',
               'white_yellow_mug': 'yellow and white mug', 'glazed_rim_porcelain_ramekin': 'ramekin'}
    result, groups = {}, {}
    for key in boxes:
        e = entities[key]
        noun = aliases.get(e['category'], e['category'].replace('_', ' '))
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
        r'(?P<prefix>(?:pick up|put|place) (?:the )?)(?P<source>.+?)'
        r'(?P<relation>(?: and (?:put|place) it)? (?:in|into|on|onto) (?:the )?)'
        r'(?P<destination>.+?)(?P<punctuation>[.!]?)', task.strip(), re.IGNORECASE)
    if not match:
        return None
    captions = entity_captions(entities, boxes)
    def resolve(noun):
        matches = []
        for key, caption in captions.items():
            if entities[key]['kind'] not in ('object', 'fixture'):
                continue
            names = {caption, caption.removesuffix(' box')}
            if caption == 'black bowl':
                names.add('bowl')
            if noun.lower() in names:
                matches.append(key)
        return matches[0] if len(matches) == 1 else None
    source, destination = resolve(match['source']), resolve(match['destination'])
    if source is None or destination is None or source == destination:
        return None
    return source, destination, match


def copy_safe_for_grounding(task):
    """Small conservative guard for spatial referring expressions, independent of task parsing."""
    spatial = re.search(r'\b(left|right|leftmost|rightmost|front|back|center|middle|top|bottom|beside|behind|'
                        r'between|next to|closest|nearest)\b', task, re.IGNORECASE)
    first_clause = re.split(r'\band (?:place|put)\b',task,maxsplit=1,flags=re.IGNORECASE)[0]
    source_relation = re.search(r'\b(pick up|grasp|lift)\b.+?\b(on|in|from)\b', first_clause, re.IGNORECASE)
    return not (spatial or source_relation)


class GroundingDataset:
    def __init__(self, index, config, tokenizer, *, episode_ids=None, augment=True):
        self.config, self.tokenizer, self.augment = config, tokenizer, augment
        self.transform = ModelTransforms(config.model, {}, tokenizer)
        index = Path(index)
        self.paths = [index.parent / e['path'] for e in json.loads(index.read_text())['episodes']
                      if episode_ids is None or e['episode_id'] in episode_ids]
        self.samples = []
        for i in range(len(self.paths)):
            reader = self.reader(i)
            if not reader.meta.get('scene_entities'):
                raise ValueError('Grounding training needs scene_entities and interaction_target annotations')
            if config.model.grounding_subject and not reader.meta.get('effector_regions', {}).get('single', {}).get(config.cameras[0]):
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
            self.samples.extend((i, int(f)) for f in reader.array(reader.meta['cameras'][config.cameras[0]]['frame_indices'])[::config.frame_stride])

    @lru_cache(maxsize=8)
    def reader(self, index):
        return EpisodeReader(self.paths[index])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.sample(index)

    def sample(self, index, *, review=False):
        from openpi.augmentation import (GroundingAugmentation, copy_object, mask_grounding_patches,
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
        aug = self.config.grounding_augmentation if self.augment else GroundingAugmentation()
        boxes, rows, copy_candidates, visible_fractions, locked = {}, {}, [], {}, {}
        for key, entity in entities.items():
            seq = entity['regions'].get(camera)
            region = reader.region(seq, frame)
            if region and region['bbox'] is not None and entity['kind'] in ('object','fixture','part','region'):
                boxes[key] = region['bbox']
                rows[key] = reader.row(seq, frame)
                if entity.get('visible_fraction'):
                    visible_fractions[key] = float(reader.array(entity['visible_fraction'])[rows[key]])
                if entity.get('copy_eligible') and reader.array(entity['copy_eligible'])[rows[key]]:
                    copy_candidates.append(key)
                if entity.get('interaction_locked'):
                    locked[key] = bool(reader.array(entity['interaction_locked'])[rows[key]])
        original_boxes = boxes
        event = reader.event_at('single', frame)
        real = event['interaction_target']['entity_id'] if event else None
        # A supervised virtual region stays attached to its parent. Keep that
        # parent still rather than leave the region's label at its old position.
        if real in boxes and entities[real]['kind'] == 'region':
            parent = entities[real]['instance']
            copy_candidates = [key for key in copy_candidates if key != parent]
        task = meta['instructions'][0]
        transfer = simple_transfer(task, entities, boxes)

        def visible(key):
            ref = entities[key]['regions'][camera].get('visible_masks')
            if ref is None:
                return None
            bits = reader._raw(ref['uri'], ref['key'])[rows[key]]
            return np.unpackbits(bits, count=h*w, bitorder='big').reshape(h,w).astype(bool)

        def robot_mask():
            return reader.region(entities['robot::single']['regions'][camera], frame, include_mask=True)['mask']

        # Virtual regions are not physical obstacles; eligibility is saved geometry, not query identity.
        physical = {k:v for k,v in boxes.items() if entities[k]['kind'] != 'region'}
        allow_copy = copy_safe_for_grounding(task)
        copy_aug = aug if allow_copy else replace(aug, copy_prob=0.)
        edited, updated, info = copy_object(original, physical, copy_candidates, visible, robot_mask, copy_rng, copy_aug,
                                            task_target=real)
        if not allow_copy:
            info['copy_reason'] = 'spatial_reference'
        boxes = dict(boxes, **updated)
        captions = entity_captions(entities, boxes)  # Recompute spatial qualifiers AFTER moving objects.
        key = real if real in boxes else None
        text = task
        # Localization, full-task swapping and pixel editing have separate criteria.
        locatable_objects = [k for k in captions if entities[k]['kind'] == 'object'
                             and visible_fractions.get(k, 0.) >= aug.target_min_visible_fraction]
        # This works in Pick and Place: changing the requested source to an unheld
        # object makes that object the next target, regardless of the original Event.
        swap_candidates = []
        if transfer and event and event['skill'] in ('Pick', 'Place'):
            source, destination, match = transfer
            expected = source if event['skill'] == 'Pick' else destination
            if real == expected:
                swap_candidates = [k for k in locatable_objects if k not in (source, destination)
                                   and 'in the image' not in captions[k] and not locked.get(k, True)]
        if swap_candidates and query_rng.random() < aug.target_swap_prob:
            key = swap_candidates[int(query_rng.integers(len(swap_candidates)))]
            text = f"{match['prefix']}{captions[key]}{match['relation']}{match['destination']}{match['punctuation']}"
        region = {'bbox': boxes[key]} if key else None
        canvas, weight, valid = role_canvas(edited, region, self.config.model.canvas_size)
        gripper = reader.region(meta.get('effector_regions', {}).get('single', {}).get(camera), frame)
        gripper_box = gripper['bbox'] if gripper else None
        subject_canvas, subject_weight, subject_valid = role_canvas(edited, gripper, self.config.model.canvas_size)
        selected_visible = visible(key) if key else None
        if key and info['copy_applied'] and key == info['moved_entity']:
            dx,dy = info['offset']
            selected_visible = np.roll(selected_visible, (dy,dx), (0,1))
        contact = None
        if 'online_grounding' in meta:
            # Supplemental metadata is dense in source-frame order.
            contact = reader.array(meta['online_grounding']['interaction_boxes'])[frame]
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
        # Unexpanded full bboxes, in normalized main-input coordinates. Only used
        # to reject targets cropped out by phase 2; not supplied to the model.
        core_boxes = np.zeros((2,len(roles),4), np.float32)
        nh, nw, top, left = resize_geometry(original.shape, self.config.model.image_size)
        for i,(rgb,foreground,available,box) in enumerate(roles):
            canvases[0,i] = rgb.transpose(2,0,1).astype(np.float32)/127.5-1
            weights[0,i] = foreground
            role_valid[0,i] = available
            if box is not None:
                core_boxes[0,i] = (np.asarray(box) * [nw/w,nh/h,nw/w,nh/h]
                                   + [left,top,left,top]) / self.config.model.image_size
        info.update(target_swapped=bool(key and key != real), target_moved=bool(key and key == info['moved_entity']),
                    locatable_object_count=len(locatable_objects), swap_candidate_count=len(swap_candidates),
                    copy_candidate_count=len(copy_candidates), copy_allowed=allow_copy,
                    missing_interaction_lock_count=sum(k not in locked for k in locatable_objects),
                    transfer_resolved=transfer is not None,
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
                                    spatial_augmentation_eligible=mirror_eligible(text),
                                    augmentation=info))
        if review:
            result['review'] = dict(original=original, edited=edited, masked=masked, canvas=canvas, drop=drop,
                                    boxes=boxes, original_boxes=original_boxes, target=key,
                                    visible=selected_visible, interaction_box=contact,
                                    gripper_box=gripper_box, subject_canvas=subject_canvas)
        return result


def collate_grounding(samples):
    from torch.utils.data._utils.collate import default_collate
    inputs = default_collate([s['inputs'] for s in samples])
    def trim(value):
        length = int(value['token_mask'].sum(-1).max())
        for key in ('tokens','token_mask'):
            value[key] = value[key][:,:length]
    trim(inputs)
    targets = default_collate([s['targets'] for s in samples])
    return dict(inputs=inputs, targets=targets,
                metadata=[s['metadata'] for s in samples])
