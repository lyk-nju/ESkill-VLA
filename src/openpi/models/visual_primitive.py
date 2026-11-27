"""Single Current from native language attention and independent gripper localization."""
from math import isqrt

import torch
from torch import nn
from torch.nn import functional as F

from .diagnostics import HIT_RADIUS_PX, mean_statistics, spatial_centers
from .model import CameraLayout, masked_mean


def patch_coverage(mask, side):
    """Fractional area of a pixel mask on the RGB-aligned patch grid."""
    return F.adaptive_avg_pool2d(mask[:, None].float(), (side, side)).flatten(1)


def box_coverage(boxes, side):
    """Exact fraction of each square patch covered by a normalized xyxy rectangle."""
    left = torch.arange(side, dtype=torch.float32, device=boxes.device) / side
    right = left + 1 / side
    lo, hi = boxes.float().unsqueeze(-1).split(2, -2)
    overlaps = (torch.minimum(hi, right) - torch.maximum(lo, left)).clamp_min(0) * side
    x, y = overlaps.unbind(-2)
    return (y.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)


def spatial_kl(maps, coverage):
    """One KL per map and sample. An empty target gets zero loss and zero count."""
    mass = coverage.sum(-1, keepdim=True)
    target = coverage / mass.clamp_min(1e-20)
    target = target.unsqueeze(1).expand_as(maps)
    return F.kl_div(maps.clamp_min(1e-30).log(), target, reduction='none').sum(-1), mass[:, 0] > 0


def center_statistics(metrics, role, spatial, targets, visible):
    lo, hi = targets[f'{role}_box'].split(2, dim=-1)
    center = (lo + hi) * .5
    valid = visible & (hi > lo).all(-1)
    has_peak = spatial.amax(-1) > 0
    for readout, point in zip(('peak3', 'argmax'), spatial_centers(spatial), strict=True):
        error = ((point - center) * targets['center_px_scale']).norm(dim=-1)
        key = f'grounding_{role}_{readout}'
        metrics[f'{key}_error_px'] = mean_statistics(error, valid & has_peak)
        metrics[f'{key}_hit10'] = mean_statistics((error <= HIT_RADIUS_PX) & has_peak, valid)


class CurrentAttention(nn.Module):
    """Native text-to-image KLAL readout shared with the motion planner."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._init_heads(config)

    def _init_heads(self, config):
        width = config.vlm_width
        if config.learned_alpha:
            self.alpha_scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            nn.init.zeros_(self.alpha_scorer[-1].weight)
            nn.init.zeros_(self.alpha_scorer[-1].bias)

    def read_current(self, text, text_mask, attention):
        with torch.autocast(text.device.type, enabled=False):
            scores = (self.alpha_scorer(text.float()).squeeze(-1) if self.config.learned_alpha
                      else torch.zeros_like(text_mask, dtype=torch.float32))
            alpha = scores.masked_fill(~text_mask, -torch.inf).softmax(-1)
            query = (alpha.unsqueeze(-1) * text.float()).sum(1)
            mass = attention.sum(-1)
            # Normalize within each token/head first. Low visual mass must not reweight alpha.
            conditional = attention / mass.clamp_min(1e-30).unsqueeze(-1)
            layers = (conditional.mean(2) * alpha[:, None, :, None]).sum(2)
        return dict(current_map=layers.mean(1), layer_maps=layers, alpha=alpha,
                    current_query=query, visual_attention_mass=mass)

    def forward(self, prefix, text_mask):
        layout = CameraLayout.from_prefix(self.config, prefix, text_mask)
        patches, visual_end = layout.patches, layout.visuals
        text = prefix.context[:, layout.text]
        # Native softmax already included all available keys. Condition on the main grid only.
        native = prefix.text_image_attention
        main = native[..., :patches]
        prediction = self.read_current(text, text_mask, main)
        prediction.update(native_attention=main, text_mask=text_mask)
        if visual_end > patches:
            prediction.update(wrist_attention_mass=native[..., patches:].sum(-1),
                              wrist_valid=prefix.mask[:, patches])
        return prediction

    def losses(self, prediction, targets):
        maps = prediction['layer_maps']
        coverage = patch_coverage(targets['current_mask'], isqrt(maps.size(-1)))
        kl, valid = spatial_kl(maps, coverage)
        metrics = {}
        with torch.no_grad():
            alpha = prediction['alpha'][:, None]
            rows = prediction['text_mask'].any(-1)
            main = (prediction['visual_attention_mass'].mean(2) * alpha).sum(-1)
            gt_mass = (maps * coverage[:, None]).sum(-1)
            metrics['grounding_main_attention_mass'] = mean_statistics(main.mean(1), rows)
            metrics['grounding_current_gt_mass'] = mean_statistics(gt_mass.mean(1), valid)
            pooled_kl, _ = spatial_kl(prediction['current_map'][:, None], coverage)
            metrics['grounding_current_map_kl'] = mean_statistics(pooled_kl[:, 0], valid)
            if 'wrist_valid' in prediction:
                paired = prediction['wrist_valid']
                wrist = (prediction['wrist_attention_mass'].mean(2) * alpha).sum(-1)
                metrics['grounding_wrist_present'] = mean_statistics(paired.float(), rows)
                metrics['grounding_wrist_attention_mass'] = mean_statistics(wrist.mean(1), rows)
                metrics['grounding_paired_wrist_mass'] = mean_statistics(wrist.mean(1), rows & paired)
                metrics['grounding_paired_main_mass'] = mean_statistics(main.mean(1), rows & paired)
            for i, layer in enumerate(self.config.attention_layers):
                root = f'grounding_layer_{layer}_'
                metrics[root + 'kl'] = mean_statistics(kl[:, i], valid)
                metrics[root + 'gt_mass'] = mean_statistics(gt_mass[:, i], valid)
                metrics[root + 'visual_mass'] = mean_statistics(main[:, i], rows)
                if 'wrist_valid' in prediction:
                    metrics[root + 'wrist_mass'] = mean_statistics(wrist[:, i], rows)
            center_statistics(metrics, 'current', prediction['current_map'], targets, valid)
        return {'current_align': masked_mean(kl.mean(1), valid)}, {'current_align': valid.sum()}, metrics
