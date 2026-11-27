"""Inspect continuous event predictions or isolate the inputs of a fixed Skill Planner."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from openpi.models.visual_primitive import patch_coverage
from openpi.policies.eskill_dataset import EpisodeReader, event_boundary_targets, event_motion_targets, planning_event
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


def plan_pair(policy, rgb, tasks, *, regions, observed, other_skill=None, target_centers=None):
    """Both prompts see identical RGB. Controlled interventions keep the A route fixed."""
    a, b = [policy.features(rgb, t) for t in tasks]
    route = a['skill_logits'].argmax(-1)
    base = policy.read_plan(a, skill_id=route)
    variants = {'task_a': base, 'task_b': policy.read_plan(b),
                'prior_only': policy.read_plan(a, current_prior=b['current_map'], skill_id=route),
                'hidden_only': policy.read_plan(dict(a, visual_hidden=b['visual_hidden']), skill_id=route),
                'query_only': policy.read_plan(dict(a, motion_hidden=b['motion_hidden']), skill_id=route),
                'remove_specialists': policy.read_plan(a, skill_id=route, remove_experts=True)}
    size = policy.model.config.image_size
    has_boundaries = 'boundary_latents' in base
    if has_boundaries:
        donor = variants['task_b']
        variants['end_swap'] = policy.read_plan(a, skill_id=route, end_latent=donor['first_readout'][:, 1:2])
        end_direction = (donor['end'] - base['end'])[0].float().cpu() * size
    if other_skill is not None:
        variants['change_expert'] = policy.read_plan(a, skill_id=torch.full_like(route, other_skill))
    for name, region in zip(('a', 'b'), regions, strict=True):
        if region is not None:
            variants['annotation_' + name] = policy.read_plan(
                a, current_prior=(region / region.sum()).to(route.device), skill_id=route)

    centers, direction = None, None
    if target_centers is not None:
        centers = torch.as_tensor(target_centers, dtype=torch.float32).reshape(2, 2)
    elif all(r is not None for r in regions):
        side = int(regions[0].numel() ** .5)
        axis = (torch.arange(side) + .5) / side
        grid = torch.cartesian_prod(axis, axis).flip(-1)
        centers = torch.stack([(r.reshape(-1) / r.sum()) @ grid for r in regions])
    if centers is not None:
        delta = centers[1] - centers[0]
        if delta.norm() > 1e-6:
            direction = delta / delta.norm()
    slots = np.flatnonzero(observed)
    base_trace = base['trace'][0].detach().float().cpu()
    output = {}
    for name, plan in variants.items():
        trace = plan['trace'][0].detach().float().cpu()
        projection, distances = None, None
        if len(slots) and centers is not None:
            distances = ((trace[slots[-1]] - centers).norm(dim=1) * size).tolist()
            if direction is not None:
                projection = ((trace - base_trace)[slots] @ direction).mean().item() * size
        output[name] = dict(skill_id=plan['skill_id'].item(),
                            first_readout_delta_rms=tensor_rms(plan['first_readout'] - base['first_readout']),
                            motion_delta_rms=tensor_rms(plan['motion_tokens'] - base['motion_tokens']),
                            trace=trace.tolist(), last_observed_slot_distance_to_a_b_px=distances,
                            toward_b_px_on_observed_slots=projection)
        delta = (trace - base_trace) * size
        output[name]['trace_displacement_px'] = delta.norm(dim=-1).tolist()
        output[name]['last4_toward_b_px'] = (delta[-4:] @ direction).mean().item() if direction is not None else None
        if has_boundaries:
            start_delta = (plan['start'] - base['start'])[0].float().cpu() * size
            end_delta = (plan['end'] - base['end'])[0].float().cpu() * size
            output[name].update(
                start=plan['start'][0].tolist(), end=plan['end'][0].tolist(),
                start_displacement_px=start_delta.norm().item(), end_displacement_px=end_delta.norm().item(),
                end_toward_b_px=(end_delta @ direction).item() if direction is not None else None)
    if has_boundaries:
        # The normalized response is unstable for almost coincident predicted endpoints.
        swap = output['end_swap']
        shifted = (variants['end_swap']['trace'][0].float().cpu() - base_trace) * size
        separation = end_direction.norm().item()
        swap.update(swap_source='pre_temporal_end_slot', donor_end_distance_px=separation, min_end_distance_px=1.,
                    end_direction_response=(shifted @ end_direction / end_direction.square().sum()).tolist()
                    if separation >= 1. else None)
    return {'tasks': tasks, 'planner_version': policy.model.config.planner_version,
            'observed_reference_slots': slots.tolist(),
            'same_predicted_skill': variants['task_a']['skill_id'].item() == variants['task_b']['skill_id'].item(),
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
        views = [rgb, reader.rgb('wrist_image', frame)] if len(getattr(policy, 'cameras', ('image',))) == 2 else rgb
        pred = policy.infer(views, task)
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
        row = dict(frame=frame, event_id=event['event_id'] if event else None,
                   annotated_skill=event['skill'] if event and event.get('skill_valid', True) else None,
                   skill_probability=dict(zip(cfg.skill_names[:len(probs)], probs, strict=True)),
                   current_entity=current_key, current_mass=mass(current_key),
                   receiver=receiver_key, receiver_mass=mass(receiver_key),
                   simulator_grasp=grasp.get(str(frame)), trace=trace.tolist(),
                   observed_error_px=error[valid & ~held].tolist(), hold_error_px=error[valid & held].tolist())
        row['observed_error_by_slot_px'] = [float(v) if ok else None for v, ok in zip(error, valid & ~held)]
        if 'start' in pred:
            xy, good, sources = event_boundary_targets(reader, frame, event, image_size=cfg.image_size)
            for j, key in enumerate(('start', 'end')):
                point = pred[key][0].float().cpu().numpy()
                row[key] = dict(predicted=point.tolist(), source_frame=int(sources[j]) if sources[j] >= 0 else None,
                                target=xy[j].tolist() if good[j] else None,
                                error_px=float(np.linalg.norm(point - xy[j]) * cfg.image_size) if good[j] else None)
        yield row


def native_report(policy, index, manifest):
    """Score exactly the saved observations specified by the V1 validation protocol."""
    paths = {item['episode_id']: index.parent / item['path']
             for item in json.loads(index.read_text())['episodes']}
    manifest_bytes = manifest.read_bytes()
    requested = json.loads(manifest_bytes)['rows']
    if not requested:
        raise ValueError('The native validation manifest contains no observations')
    grouped = defaultdict(list)
    for pos, row in enumerate(requested):
        grouped[row['episode_id']].append((pos, row['frame']))
    rows = [None] * len(requested)
    offsets = policy.model.config.motion_offsets_frames
    names = ['observed_error_px', 'hold_error_px', 'skill_accuracy']
    names.extend(f'observed_offset_{offset}_px' for offset in offsets)
    if policy.model.config.planner_has_boundaries:
        names.extend(('start_error_px', 'end_error_px'))
    values = {key: [] for key in names}
    for episode_id, observations in grouped.items():
        reader = EpisodeReader(paths[episode_id])
        for pos, frame in observations:
            row = next(episode_rows(policy, reader, [frame], {}), None)
            if row is None or row['event_id'] is None:
                raise ValueError(f'Invalid locked observation {episode_id} frame {frame}')
            rows[pos] = dict(episode_id=episode_id, **row)
            for key in ('observed_error_px', 'hold_error_px'):
                values[key].extend(row[key])
            for offset, value in zip(offsets, row['observed_error_by_slot_px'], strict=True):
                if value is not None:
                    values[f'observed_offset_{offset}_px'].append(value)
            for role in ('start', 'end'):
                if role in row and row[role]['error_px'] is not None:
                    values[role + '_error_px'].append(row[role]['error_px'])
            if row['annotated_skill'] is not None:
                predicted = max(row['skill_probability'], key=row['skill_probability'].get)
                values['skill_accuracy'].append(predicted == row['annotated_skill'])
    return dict(planner_version=policy.model.config.planner_version,
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), model_size=policy.model.config.image_size,
                episodes=len(grouped), events=len({(r['episode_id'], r['event_id']) for r in rows}), frames=len(rows),
                metrics={k: dict(n=len(v), mean=float(np.mean(v)) if v else None) for k, v in values.items()},
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episode', type=Path, help='grounding episode.json for episode or pair')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--pretrained-path')
    commands = parser.add_subparsers(dest='mode', required=True)
    native = commands.add_parser('native', help='score a locked list of native validation frames')
    native.add_argument('--index', type=Path, required=True, help='grounding index.json')
    native.add_argument('--manifest', type=Path, required=True, help='JSON rows with episode_id and source frame')
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
    if args.mode != 'native' and args.episode is None:
        parser.error('--episode is required for episode and pair')
    policy = create_motion_planner_policy(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer_path,
                                          pretrained_path=args.pretrained_path)
    cfg = policy.model.config
    result = dict(checkpoint=str(args.checkpoint), mode=args.mode, planner_version=cfg.planner_version)
    if args.mode != 'native':
        reader = EpisodeReader(args.episode)
        result['episode_id'] = reader.meta['episode_id']
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
    elif args.mode == 'native':
        result.update(native_report(policy, args.index, args.manifest))
    else:
        rgb = reader.rgb('image', args.frame)
        if rgb is None:
            parser.error('frame must have a valid main camera observation')
        regions = [entity_coverage(reader, key, args.frame, cfg.image_size, cfg.image_size // cfg.patch_size)
                   for key in (args.entity_a, args.entity_b)]
        if len(policy.cameras) == 2:
            rgb = [rgb, reader.rgb('wrist_image', args.frame)]
        _, valid, held = motion_labels(reader, args.frame, cfg)
        result.update(frame=args.frame, entities=[args.entity_a, args.entity_b],
                      **plan_pair(policy, rgb, [args.task_a, args.task_b], regions=regions,
                                  observed=valid & ~held, other_skill=args.other_skill))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
