"""Detached diagnostic sums/counts; never used as model conditions or losses."""
import torch


@torch.no_grad()
def mean_statistics(values, valid):
    return torch.stack((torch.where(valid, values.detach(), 0).sum(dtype=torch.float64),
                        valid.sum(dtype=torch.float64)))


@torch.no_grad()
def prediction_statistics(prediction, trace, inputs, targets, config):
    result = {}
    presence = inputs['arm_presence']
    for prefix, logits, labels, valid in (
        ('current_skill', prediction['skill_logits'], targets['skill_labels'], targets['skill_valid'] & presence),
        ('future_skill', prediction['future_skill_logits'], targets['future_skill_labels'],
         targets['future_skill_valid'] & presence[:, :, None]),
    ):
        predicted = logits.argmax(-1)
        result[prefix + '_accuracy'] = mean_statistics(predicted == labels, valid)
        counts = torch.bincount(predicted[valid], minlength=len(config.skill_names)).double()
        total = valid.sum(dtype=torch.float64)
        for i, name in enumerate(config.skill_names):
            result[f'{prefix}_pred_{name.lower()}'] = torch.stack((counts[i], total))

    valid = targets['trace_valid'] & presence[:, :, None]
    difference = torch.where(valid[..., None], trace.detach() - targets['trace'].float(), 0)
    uv = torch.linalg.vector_norm(difference, dim=-1)
    pixels = uv * config.image_size
    observed = valid & targets.get('future_observed', torch.zeros_like(valid))
    padded = valid & targets.get('future_padded', torch.zeros_like(valid))
    unknown = valid & ~(observed | padded)
    result['motion_l2_uv'] = mean_statistics(uv, valid)
    for prefix, mask in (('motion', valid), ('motion_observed', observed),
                         ('motion_padded', padded), ('motion_unclassified', unknown)):
        result[prefix + '_l2_px'] = mean_statistics(pixels, mask)
        if prefix == 'motion_unclassified':
            continue
        for i, offset in enumerate(config.motion_offsets):
            result[f'{prefix}_offset_{offset}_l2_px'] = mean_statistics(pixels[:, :, i], mask[:, :, i])
    return result
