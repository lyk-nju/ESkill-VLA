"""Global diagnostic aggregation and the five W&B namespaces."""
import torch
import torch.distributed as dist


# Dashboard summaries only; full diagnostics remain in local JSONL.
AUX_METRICS = {
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
    parameters = [p for name, p in model.motion.named_parameters()
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
                           for key in ('skill', 'current', 'future_object', 'trace', 'gripper')))
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
    if getattr(model.config, 'current_grounding', None) == 'motion_planner':
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
    for key, (numerator, count) in zip(keys, rows, strict=True):
        if key == 'current_skill_accuracy':
            result['current_skill_count'] = int(count)
        elif key.startswith('planner_route_'):
            result[key + '_count'] = int(numerator)
        elif key.startswith(('grounding_', 'skill_')):
            result[key + '_count'] = int(count)
        elif key.startswith('motion_') and key.endswith('_l2_px'):
            result[key.removesuffix('_l2_px') + '_count'] = int(count)
        elif key.startswith('gripper_'):
            result[key + '_count'] = int(count)
    return result


def visual_lora_gradient_norms(model, losses):
    """First-update calibration of each unweighted loss on the shared Q adapters."""
    params = [p for name, p in model.named_parameters()
              if '.q_proj.lora_' in name and p.requires_grad]
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
