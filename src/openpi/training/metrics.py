"""Global diagnostic aggregation and the five W&B namespaces."""
import torch
import torch.distributed as dist


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
    for prefix in ('current_skill', 'future_skill'):
        key = prefix + '_accuracy'
        if key in counts:
            result[prefix + '_count'] = int(counts[key])
    for key, count in counts.items():
        if key.startswith('grounding_'):
            result[key + '_count'] = int(count)
        elif key.endswith('_l2_px') and (key in ('motion_l2_px', 'motion_observed_l2_px',
                                                'motion_padded_l2_px', 'motion_unclassified_l2_px')
                                        or key.startswith('motion_offset_')):
            result[key.removesuffix('_l2_px') + '_count'] = int(count)
    return result


def loss_metrics(record):
    metrics = {'loss': record['loss']}
    for source, name in (('action', 'action_loss'), ('grounding', 'grounding_loss'), ('trace', 'traj_loss')):
        if source in record:
            metrics[name] = record[source]
    if 'skill' in record or 'future' in record:
        metrics['skill_loss'] = record.get('skill', 0.) + record.get('future', 0.)
    return metrics


def phase_metrics(phase, losses, auxiliary=None):
    metrics = {f'{phase}/{key}': value for key, value in loss_metrics(losses).items()}
    detail = {key: value for key, value in losses.items() if key.startswith('grounding_')}
    detail.update(auxiliary or {})
    for source, name in (('skill', 'current_skill_loss'), ('future', 'future_skill_loss')):
        if source in losses:
            detail[name] = losses[source]
    metrics.update({f'{phase}/aux/{key}': value for key, value in detail.items()})
    return metrics


def wandb_metrics(record):
    metrics = phase_metrics('train', record, record.get('aux'))
    metrics.update({'chart/' + key: record[key] for key in ('learning_rate', 'grad_norm', 'step')})
    metrics['chart/time_per_step'] = record['seconds']
    return metrics
