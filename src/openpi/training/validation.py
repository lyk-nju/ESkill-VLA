"""Fixed-subset validation with global valid-item weighting and isolated RNG."""
import torch
import torch.distributed as dist

from openpi.models.model import to_device
from openpi.training.metrics import reduce_diagnostics


@torch.no_grad()
def validation_losses(model, loader, config, *, include_aux=False):
    keys = ['action']
    if config.model_type == 'eskill':
        keys += ['skill', 'current', 'future_object', 'trace', 'gripper']
    totals = torch.zeros(2, len(keys), dtype=torch.float64, device=config.device)
    diagnostic_totals = {}
    was_training = model.training
    devices = [torch.device(config.device).index or 0] if config.device.startswith('cuda') else []
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(config.validation_seed + (dist.get_rank() if dist.is_initialized() else 0))
            for batch in loader:
                batch = to_device(batch, config.device)
                inputs, targets = batch['inputs'], batch['targets']
                with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16,
                                    enabled=config.precision == 'bfloat16'):
                    losses = model(inputs, targets)
                if include_aux:
                    for key, pair in getattr(model, 'diagnostic_statistics', {}).items():
                        if key not in diagnostic_totals:
                            diagnostic_totals[key] = pair.detach().double().clone()
                        else:
                            diagnostic_totals[key] += pair.detach().double()
                counts = [targets['action_valid'].sum() if config.model.mask_action_loss
                          else targets['actions'].new_tensor(targets['actions'].numel())]
                values = [losses['action']]
                if config.model_type == 'eskill':
                    # Counts come from the model's effective masks (active skills, padding,
                    # current projection and per-role validity), exactly matching its loss.
                    counts = [model.loss_counts[key] for key in keys]
                    values = [losses[key] for key in keys]
                counts = torch.stack(counts).double()
                if dist.is_initialized():
                    dist.all_reduce(counts)
                totals[0] += torch.stack(values).double() * counts
                totals[1] += counts
        if dist.is_initialized():
            dist.all_reduce(totals[0])
            totals[0] /= dist.get_world_size()
        if not torch.isfinite(totals).all():
            raise FloatingPointError('Nonfinite validation loss')
        mean = (totals[0] / totals[1].clamp_min(1)).tolist()
        result = dict(zip(keys, mean))
        result['loss'] = sum(config.loss_weights[key] * result[key] for key in config.loss_weights if key in result)
        if include_aux:
            result['aux'] = reduce_diagnostics(diagnostic_totals)
        return result
    finally:
        model.train(was_training)
