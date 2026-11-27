"""Global diagnostic aggregation and the five W&B namespaces."""
import torch
import torch.distributed as dist


# Dashboard summaries only; full diagnostics remain in local JSONL.
AUX_METRICS = {
    **{f'action_{role}_attention_mass': 'action_attention_count' for role in ('main', 'wrist', 'text', 'motion', 'self')},
    'grounding_main_attention_mass': 'grounding_main_attention_mass_count',
    'grounding_wrist_attention_mass': 'grounding_wrist_attention_mass_count',
    'grounding_wrist_present': 'grounding_wrist_present_count',
    'grounding_bbox_iou': 'grounding_bbox_iou_count',
    'grounding_bbox_center_px': 'grounding_bbox_center_px_count',
    'grounding_bbox_area_ratio': 'grounding_bbox_area_ratio_count',
    'grounding_subject_bbox_iou': 'grounding_subject_bbox_iou_count',
    'grounding_subject_bbox_center_px': 'grounding_subject_bbox_center_px_count',
    'grounding_gac_recall': 'grounding_gac_recall_count',
    'current_skill_accuracy': 'current_skill_count',
    'motion_l2_px': 'motion_count',
    'motion_start_l2_px': 'motion_start_count',
    'motion_end_l2_px': 'motion_end_count',
    'gripper_accuracy': 'gripper_accuracy_count',
    'gripper_transition_accuracy': 'gripper_transition_accuracy_count',
}


def motion_gradient_balance(model, losses, loss_weights=None):
    """Weighted FM/auxiliary gradients on shared Motion weights, before clipping.

    Reuse the actual training graph. Explicitly average vectors across ranks: averaging
    norms or cosines would miss cancellation and give a different global diagnostic.
    autograd.grad leaves optimizer .grad buffers untouched; retain the graph for backward.
    """
    parameters = [p for name, p in (model.planner if hasattr(model, 'planner') else model.motion).named_parameters()
                  if p.requires_grad and not name.startswith(('trace_head.', 'gripper_head.'))]
    if not parameters:
        raise ValueError('Motion diagnostics require trainable shared Motion parameters')

    def vector(loss):
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        flat = torch.cat([(gradient.detach().float() if gradient is not None else torch.zeros_like(parameter).float()).flatten()
                          for parameter, gradient in zip(parameters, gradients, strict=True)])
        if dist.is_initialized():
            dist.all_reduce(flat)
            flat /= dist.get_world_size()
        return flat

    weights = loss_weights if loss_weights is not None else {key: 1. for key in losses}
    fm = vector(weights['action'] * losses['action'])
    auxiliary = vector(sum(weights[key] * losses[key]
                           for key in losses if key != 'action'))
    values = torch.stack((fm.norm(), auxiliary.norm(), fm.dot(auxiliary)))
    if not torch.isfinite(values).all():
        raise FloatingPointError('Nonfinite Motion gradient diagnostics')
    fm_norm, aux_norm, dot = values.cpu().tolist()
    return dict(fm_norm=fm_norm, aux_norm=aux_norm,
                fm_aux_ratio=fm_norm / aux_norm if aux_norm > 0 else None,
                cosine=max(-1., min(1., dot / (fm_norm * aux_norm))) if fm_norm > 0 and aux_norm > 0 else None)


@torch.no_grad()
def expert_gradient_norms(model):
    """Post-clipping expert gradients for local diagnostics, never W&B curves."""
    if hasattr(model, 'planner'):
        results = {}
        for i, layer in enumerate(model.planner.layers):
            for k, module in enumerate(layer.experts):
                grads = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
                # None distinguishes globally unused parameters from a real zero gradient.
                results[f'layer_{i}_{model.config.skill_names[k].lower()}'] = (
                    float(torch.stack(grads).norm()) if grads else None)
        return results
    skill = model.skill
    experts = [('shared', skill.shared), *(
        (model.config.skill_names[index].lower(), expert)
        for index, expert in zip(model.config.active_skill_ids, skill.experts, strict=True))]
    result = {}
    for name, module in experts:
        norms = [parameter.grad.float().norm() for parameter in module.parameters() if parameter.grad is not None]
        result[name] = float(torch.stack(norms).norm()) if norms else 0.
    return result


def reduce_diagnostics(statistics):
    if not statistics:
        return {}
    keys = sorted(statistics)
    totals = torch.stack([statistics[key].detach().double() for key in keys])
    if dist.is_initialized():
        dist.all_reduce(totals)
    rows = totals.cpu().tolist()
    result = {key: numerator / max(count, 1) for key, (numerator, count) in zip(keys, rows)}
    # Each layer sees the same valid action queries. Average their mass for the dashboard.
    action_mass = {}
    for key, (numerator, count) in zip(keys, rows, strict=True):
        if key.startswith('action_layer_'):
            role = key.split('_')[-2]
            total = action_mass.setdefault(role, [0., 0.])
            total[0] += numerator
            total[1] += count
            if role == 'main':
                result['action_attention_count'] = int(count)
        elif key == 'current_skill_accuracy':
            result['current_skill_count'] = int(count)
        elif key.startswith('planner_route_'):
            result[key + '_count'] = int(numerator)
        elif key.startswith(('grounding_', 'skill_')):
            result[key + '_count'] = int(count)
        elif key.startswith('motion_') and key.endswith('_l2_px'):
            result[key.removesuffix('_l2_px') + '_count'] = int(count)
        elif key.startswith('gripper_'):
            result[key + '_count'] = int(count)
    for role, (numerator, count) in action_mass.items():
        result[f'action_{role}_attention_mass'] = numerator / max(count, 1)
    return result


def wrist_sample_statistics(samples):
    totals = dict.fromkeys(('paired', 'copy_unverified', 'random_dropout', 'missing'), 0)
    groups = {}
    pairing = {group: dict(available=0, paired=0) for group in ('with_copy', 'without_copy')}
    for sample in samples:
        group, status = sample['wrist_group'], sample['wrist_status']
        counts = groups.setdefault(group, {})
        counts[status] = counts.get(status, 0) + 1
        totals[status] += 1
        entry = pairing['with_copy' if group in ('copy', 'copy_and_text_swap') else 'without_copy']
        entry['available'] += status != 'missing'
        entry['paired'] += status == 'paired'
    for entry in pairing.values():
        entry['rate'] = entry['paired'] / entry['available'] if entry['available'] else None
    totals['paired_text_swaps'] = groups.get('text_swap', {}).get('paired', 0)
    return dict(wrist_samples=totals, wrist_by_augmentation=groups, wrist_pairing=pairing)


def visual_lora_gradient_norms(model, losses):
    """First-update calibration on the language projections selected for LoRA."""
    params = [p for name, p in model.named_parameters()
              if '.language_model.layers.' in name and name.endswith(('.lora_A', '.lora_B')) and p.requires_grad]
    result = {}
    for name, loss in losses.items():
        gradients = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        vector = torch.cat([(g.detach().float() if g is not None else torch.zeros_like(p).float()).flatten()
                            for p, g in zip(params, gradients, strict=True)])
        if dist.is_initialized():
            dist.all_reduce(vector)
            vector /= dist.get_world_size()
        result[name] = vector.norm().item()
    return result


def loss_metrics(record):
    metrics = {'loss': record['loss']}
    for source, name in (('action', 'action_loss'), ('skill', 'skill_loss'), ('trace', 'traj_loss'),
                         ('current', 'current_object_loss'), ('future_object', 'future_object_loss'),
                         ('subject', 'subject_grounding_loss'), ('bbox', 'bbox_loss'), ('gac', 'gac_loss'),
                         ('current_align', 'current_align_loss'), ('gripper', 'gripper_loss')):
        if source in record:
            metrics[name] = record[source]
    return metrics


def phase_metrics(phase, losses, auxiliary=None):
    metrics = {f'{phase}/{key}': value for key, value in loss_metrics(losses).items()}
    detail = {**losses, **(auxiliary or {})}
    metrics.update({f'aux_{phase}/{key}': detail[key] for key, count in AUX_METRICS.items()
                    if key in detail and detail.get(count, 1) > 0})
    return metrics


def wandb_metrics(record):
    metrics = phase_metrics('train', record, record.get('aux'))
    metrics.update({'chart/' + key: record[key] for key in ('learning_rate', 'grad_norm', 'step')})
    metrics['chart/time_per_step'] = record['seconds']
    return metrics
