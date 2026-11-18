"""Single-target box regression, spatial feature sampling and within-image entity contrast."""
from math import isqrt

import torch
from torch.nn import functional as F

from openpi.models.diagnostics import mean_statistics
from openpi.models.model import masked_mean


def corners(box):
    center, size = box.split(2, dim=-1)
    return torch.cat((center - size / 2, center + size / 2), dim=-1)


def box_area(box):
    return (box[..., 2:] - box[..., :2]).clamp_min(0).prod(-1)


def box_loss(predicted, target, valid, image_size, *, metric_prefix='grounding_bbox'):
    """TransVG reduction: sum L1 over four normalized coordinates, then average valid boxes."""
    lo, hi = target.split(2, dim=-1)
    target_cxcywh = torch.cat(((lo + hi) / 2, hi - lo), -1)
    pred = corners(predicted)
    overlap = (torch.minimum(pred[..., 2:], hi) - torch.maximum(pred[..., :2], lo)).clamp_min(0).prod(-1)
    union = box_area(pred) + box_area(target) - overlap
    enclosing = (torch.maximum(pred[..., 2:], hi) - torch.minimum(pred[..., :2], lo)).prod(-1)
    iou = overlap / union.clamp_min(1e-9)
    giou = iou - (enclosing - union) / enclosing.clamp_min(1e-9)
    l1 = (predicted - target_cxcywh).abs().sum(-1)
    # Report the screen-clipped prediction used by the online policy.
    clipped = pred.clamp(0, 1)
    pred_area = box_area(clipped)
    screen_iou = overlap / (pred_area + box_area(target) - overlap).clamp_min(1e-9)
    center_error = torch.linalg.vector_norm((clipped[..., :2] + clipped[..., 2:] - lo - hi) / 2, dim=-1)
    metrics = {f'{metric_prefix}_iou': screen_iou, f'{metric_prefix}_center_px': center_error * image_size,
               f'{metric_prefix}_area_ratio': pred_area / box_area(target).clamp_min(1e-9)}
    return masked_mean(l1 + 1 - giou, valid), {k: mean_statistics(v, valid) for k, v in metrics.items()}


def read_regions(visual, boxes):
    """Read a 4x4 grid per normalized xyxy box. RGB coordinates refer to pixel edges."""
    b, patches, width = visual.shape
    side = isqrt(patches)
    feature = visual.transpose(1, 2).reshape(b, width, side, side)
    axis = (torch.arange(4, device=boxes.device, dtype=torch.float32) + .5) / 4
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    unit = torch.stack((x, y), dim=-1).reshape(16, 2)
    locations = boxes[..., None, :2] + unit * (boxes[..., None, 2:] - boxes[..., None, :2])
    grid = (2 * locations - 1).reshape(b, -1, 16, 2)
    tokens = F.grid_sample(feature.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return tokens.permute(0, 2, 3, 1), locations


def entity_contrast(query, visual, projection, targets, current_valid):
    """Positive and distractors come from this image, including on a counterfactual draw."""
    tokens, _ = read_regions(visual, targets['entity_boxes'])
    keys = F.normalize(projection(tokens.mean(2)).float(), dim=-1)
    similarity = (F.normalize(query.float(), dim=-1)[:, None] * keys).sum(-1) / .07
    eligible = targets['entity_valid']
    logits = similarity.masked_fill(~eligible, -1e4)
    positive = targets['positive_entity']
    index = positive.clamp_min(0)
    valid = current_valid & (positive >= 0) & eligible.gather(1, index[:, None])[:, 0] & (eligible.sum(1) >= 2)
    loss = F.cross_entropy(logits, index, reduction='none')
    return masked_mean(loss, valid), valid.sum(), mean_statistics((logits.argmax(1) == positive).float(), valid)
