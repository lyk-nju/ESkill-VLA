"""Global diagnostic aggregation and the five W&B namespaces."""
import torch
import torch.distributed as dist


# Dashboard summaries only; full diagnostics remain in local JSONL.
AUX_METRICS = {
    'current_skill_accuracy': 'current_skill_count',
    'motion_l2_px': 'motion_count',
    'gripper_accuracy': 'gripper_accuracy_count',
    'gripper_transition_accuracy': 'gripper_transition_accuracy_count',
}


@torch.no_grad()
def expert_gradient_norms(model):
    """Post-clipping expert gradients for local diagnostics, never W&B curves."""
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
    counts = dict(zip(keys, (row[1] for row in rows)))
    if 'current_skill_accuracy' in counts:
        result['current_skill_count'] = int(counts['current_skill_accuracy'])
    for key, count in counts.items():
        if key.startswith('grounding_'):
            result[key + '_count'] = int(count)
        elif key == 'motion_l2_px' or (key.startswith('motion_offset_') and key.endswith('_l2_px')):
            result[key.removesuffix('_l2_px') + '_count'] = int(count)
        elif key.startswith('gripper_'):
            result[key + '_count'] = int(count)
    return result


def loss_metrics(record):
    metrics = {'loss': record['loss']}
    for source, name in (('action', 'action_loss'), ('skill', 'skill_loss'), ('trace', 'traj_loss'),
                         ('current', 'current_object_loss'), ('future_object', 'future_object_loss'),
                         ('gripper', 'gripper_loss')):
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
