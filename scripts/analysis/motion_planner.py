"""Inspect continuous event predictions or isolate the inputs of a fixed Skill Planner."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.models.visual_primitive import patch_coverage
from openpi.policies.eskill_dataset import EpisodeReader, event_motion_targets, planning_event
from openpi.policies.grounding_dataset import box_coverage, visible_pixels
from openpi.policies.motion_planner import create_motion_planner_policy
from openpi.transforms import resize_image


def entity_coverage(reader, entity, frame, size, side):
    """Current patch coverage using the training mask convention, including projected sites."""
    item = reader.meta['scene_entities'][entity]
    seq = item['regions'].get('image')
    if not seq:
        return None
    stream = reader.meta['cameras']['image']
    shape = (stream['height'], stream['width'])
    row = reader.row(seq, frame)
    if row is None:
        return None
    if seq.get('visible_masks'):
        mask = visible_pixels(reader, seq, row, shape)
    elif item['kind'] == 'region':
        if item.get('visible_fraction') and not reader.array(item['visible_fraction'])[row] > 0:
            return None
        region = reader.region(seq, frame, include_mask=True)
        if region is None:
            return None
        mask = region['mask'] if region['mask'] is not None else box_coverage(region['bbox'], shape)
    else:
        return None
    if mask is None or not mask.any():
        return None
    resized = resize_image(mask.astype(np.float32), size)
    return patch_coverage(torch.from_numpy(resized)[None], side)


def motion_labels(reader, frame, cfg):
    if planning_event(reader, frame) is None:
        slots = len(cfg.motion_offsets_frames)
        return np.zeros((slots, 2), np.float32), np.zeros(slots, bool), np.zeros(slots, bool)
    labels, _ = event_motion_targets(reader, frame,
                                    offsets_frames=cfg.motion_offsets_frames, image_size=cfg.image_size,
                                    min_horizon_frames=cfg.motion_min_horizon_frames,
                                    terminal_hold=cfg.motion_terminal_hold)
    return labels['trace'][0], labels['trace_valid'][0], labels['trace_terminal_hold'][0]


def tensor_rms(value):
    return value.float().square().mean().sqrt().item()


def receiver_event(reader, event):
    """Follow only a contiguous, annotated single-arm Pick → Place pair."""
    if event is None or not event.get('skill_valid', True):
        return None
    if event['skill'] == 'Place':
        return event
    if event['skill'] == 'Pick':
        next_event = planning_event(reader, event['stop_frame'])
        if (next_event and next_event['start_frame'] == event['stop_frame']
                and next_event['skill'] == 'Place' and next_event.get('skill_valid', True)):
            return next_event
    return None


def bound_entity(event):
    binding = (event or {}).get('interaction_target') or {}
    return binding.get('entity_id') if binding.get('status') == 'resolved' else None


def plan_pair(policy, rgb, tasks, *, regions, observed, other_skill=None):
    """Both prompts see identical RGB. Controlled interventions keep the A route fixed."""
    a, b = [policy.features(rgb, t) for t in tasks]
    route = a['skill_logits'].argmax(-1)
    base = policy.read_plan(a, skill_id=route)
    variants = {'task_a': base, 'task_b': policy.read_plan(b),
                'prior_only': policy.read_plan(a, current_prior=b['current_map'], skill_id=route),
                'hidden_only': policy.read_plan(dict(a, visual_hidden=b['visual_hidden']), skill_id=route),
                'query_only': policy.read_plan(dict(a, motion_hidden=b['motion_hidden']), skill_id=route),
                'remove_specialists': policy.read_plan(a, skill_id=route, remove_experts=True)}
    if other_skill is not None:
        variants['change_expert'] = policy.read_plan(a, skill_id=torch.full_like(route, other_skill))
    for name, region in zip(('a', 'b'), regions, strict=True):
        if region is not None:
            variants['annotation_' + name] = policy.read_plan(
                a, current_prior=(region / region.sum()).to(route.device), skill_id=route)

    centers, direction = None, None
    if all(r is not None for r in regions):
        side = int(regions[0].numel() ** .5)
        axis = (torch.arange(side) + .5) / side
        grid = torch.cartesian_prod(axis, axis).flip(-1)
        centers = torch.stack([(r.reshape(-1) / r.sum()) @ grid for r in regions])
        delta = centers[1] - centers[0]
        if delta.norm() > 1e-6:
            direction = delta / delta.norm()
    slots = np.flatnonzero(observed)
    base_trace = base['trace'][0].detach().float().cpu()
    output = {}
    for name, plan in variants.items():
        trace = plan['trace'][0].detach().float().cpu()
        projection, distances = None, None
        size = policy.model.config.image_size
        if len(slots) and centers is not None:
            distances = ((trace[slots[-1]] - centers).norm(dim=1) * size).tolist()
            if direction is not None:
                projection = ((trace - base_trace)[slots] @ direction).mean().item() * size
        output[name] = dict(skill_id=plan['skill_id'].item(),
                            first_readout_delta_rms=tensor_rms(plan['first_readout'] - base['first_readout']),
                            motion_delta_rms=tensor_rms(plan['motion_tokens'] - base['motion_tokens']),
                            trace=trace.tolist(), last_observed_slot_distance_to_a_b_px=distances,
                            toward_b_px_on_observed_slots=projection)
    return {'tasks': tasks, 'observed_reference_slots': slots.tolist(),
            'current_mass': [[(features['current_map'].float().cpu() * r).sum().item()
                              if r is not None else None for r in regions] for features in (a, b)],
            'conditions': output}


def episode_rows(policy, reader, frames, grasp):
    cfg = policy.model.config
    task = reader.meta['instructions'][0]
    for frame in frames:
        rgb = reader.rgb('image', frame)
        if rgb is None:
            continue
        pred = policy.infer(rgb, task)
        event = planning_event(reader, frame)
        receiver = receiver_event(reader, event)
        current_key, receiver_key = bound_entity(event), bound_entity(receiver)
        weights = pred['current_map'].float().cpu()
        def mass(key):
            if not key:
                return None
            region = entity_coverage(reader, key, frame, cfg.image_size, int(weights.numel() ** .5))
            return None if region is None else (weights * region).sum().item()

        label, valid, held = motion_labels(reader, frame, cfg)
        trace = pred['trace'][0].float().cpu().numpy()
        error = np.linalg.norm(trace - label, axis=-1) * cfg.image_size
        probs = pred['skill_probabilities'][0].cpu().tolist()
        yield dict(frame=frame, event_id=event['event_id'] if event else None,
                   annotated_skill=event['skill'] if event and event.get('skill_valid', True) else None,
                   skill_probability=dict(zip(cfg.skill_names[:len(probs)], probs, strict=True)),
                   current_entity=current_key, current_mass=mass(current_key),
                   receiver=receiver_key, receiver_mass=mass(receiver_key),
                   simulator_grasp=grasp.get(str(frame)), trace=trace.tolist(),
                   observed_error_px=error[valid & ~held].tolist(), hold_error_px=error[valid & held].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episode', required=True, type=Path, help='grounding episode.json')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--pretrained-path')
    commands = parser.add_subparsers(dest='mode', required=True)
    episode = commands.add_parser('episode')
    episode.add_argument('--grasp-state', type=Path, help='optional simulator ground truth JSON')
    pair = commands.add_parser('pair')
    pair.add_argument('--frame', type=int, required=True)
    pair.add_argument('--task-a', required=True)
    pair.add_argument('--task-b', required=True)
    pair.add_argument('--entity-a', required=True)
    pair.add_argument('--entity-b', required=True)
    pair.add_argument('--other-skill', type=int, choices=range(6))
    args = parser.parse_args()
    reader = EpisodeReader(args.episode)
    policy = create_motion_planner_policy(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer_path,
                                          pretrained_path=args.pretrained_path)
    cfg = policy.model.config
    result = dict(episode_id=reader.meta['episode_id'], checkpoint=str(args.checkpoint), mode=args.mode)
    if args.mode == 'episode':
        grasp = {}
        if args.grasp_state:
            data = json.loads(args.grasp_state.read_text())
            if data['episode_id'] != result['episode_id'] or not data['source']:
                parser.error('grasp annotations must identify the simulator source and match the episode')
            grasp = data['frames']
            if any(type(value) is not bool for value in grasp.values()):
                parser.error('grasp states must be booleans indexed by source frame')
            result['grasp_source'] = data['source']
        frames = reader.array(reader.meta['cameras']['image']['frame_indices'])
        result['frames'] = list(episode_rows(policy, reader, map(int, frames), grasp))
    else:
        rgb = reader.rgb('image', args.frame)
        if rgb is None:
            parser.error('frame must have a valid main camera observation')
        regions = [entity_coverage(reader, key, args.frame, cfg.image_size, cfg.image_size // cfg.patch_size)
                   for key in (args.entity_a, args.entity_b)]
        _, valid, held = motion_labels(reader, args.frame, cfg)
        result.update(frame=args.frame, entities=[args.entity_a, args.entity_b],
                      **plan_pair(policy, rgb, [args.task_a, args.task_b], regions=regions,
                                  observed=valid & ~held, other_skill=args.other_skill))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
