"""Action attention on the annotated main grid, with native mass preserved per head and step."""
from functools import partial
from math import isqrt, sqrt

import torch
from torch import nn

from openpi.models.diagnostics import mean_statistics
from openpi.models.gemma import rotary
from openpi.models.model import ActionCondition


def trajectory_prior(points, patches, *, legacy_mean=False):
    """One-patch Gaussian of distance to the nearest valid predicted point on the main grid."""
    side = isqrt(patches)
    axis = torch.arange(side, device=points.device, dtype=torch.float32) + .5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack((x, y), -1).reshape(1, 1, patches, 2)
    valid = torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= 1).all(-1)
    xy = points.float().masked_fill(~valid[..., None], 0) * side
    scores = -.5 * (centers - xy[:, :, None]).square().sum(-1)
    if legacy_mean:
        kernels = scores.softmax(-1)
        return (kernels * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
    closest = scores.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
    has_point = valid.any(-1, keepdim=True)
    return closest.masked_fill(~has_point, 0).softmax(-1) * has_point


def guided_probabilities(q, k, mask, prior=None):
    """The prior sums to its mixing weight; empty priors leave the native distribution intact."""
    with torch.autocast(q.device.type, enabled=False):
        grouped = q.float().unflatten(1, (k.size(1), -1))
        scores = (grouped @ k.float()[:, :, None].transpose(-1, -2)).flatten(1, 2) / sqrt(q.size(-1))
        native = scores.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
        if prior is None:
            return native
        n = prior.shape[-1]
        main = native[..., :n]
        prior = prior[:, None, None, :]
        mixed = (1 - prior.sum(-1, keepdim=True)) * main + main.sum(-1, keepdim=True) * prior
        return torch.cat((mixed, native[..., n:]), -1)


def guided_attention(q, k, v, mask, *, prior):
    p = guided_probabilities(q, k, mask, prior)
    output = p.to(v.dtype).unflatten(1, (v.size(1), -1)) @ v[:, :, None]
    return output.flatten(1, 2).transpose(1, 2).flatten(2)


@torch.no_grad()
def record_action_mass(layer, q, k, mask, *, statistics, prefix_sizes, valid_steps):
    name = f'action_layer_{layer:02d}_'
    if name + 'main_mass' in statistics:
        return  # Activation checkpointing may replay this layer during backward.
    # Both guidance priors redistribute mass within main only, so native modality totals are exact.
    weights = guided_probabilities(q, k, mask).mean(1)
    groups = weights.split((*prefix_sizes, q.size(2)), dim=-1)
    for role, probs in zip(('main', 'wrist', 'text', 'motion', 'self'), groups, strict=True):
        statistics[name + role + '_mass'] = mean_statistics(probs.sum(-1), valid_steps)


class ActionGuidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.motion_kv = nn.ModuleList(
            nn.Linear(config.hidden_dim, 2 * config.kv_heads * config.head_dim)
            for _ in range(config.depth) if config.action_motion_kv)

    def forward(self, prefix, plan, current, *, statistics=None, valid_steps=None):
        cfg = self.config
        prefix_length = prefix.keys[0].size(2)
        visuals = prefix.visual_tokens.size(1)
        position_mask = prefix.mask[:, :prefix_length]
        keys, values = prefix.keys, prefix.values
        if self.motion_kv:
            motion = plan['event_motion_latents']
            position_mask = torch.cat((position_mask, position_mask.new_ones(motion.shape[:2])), dim=1)
            positions = position_mask.long().cumsum(-1)[:, -motion.size(1):] - 1
            keys, values = [], []
            for key, value, project in zip(prefix.keys, prefix.values, self.motion_kv, strict=True):
                kv = project(motion).reshape(*motion.shape[:2], 2, cfg.kv_heads, cfg.head_dim)
                k = rotary(kv[:, :, 0].transpose(1, 2), positions)
                keys.append(torch.cat((key, k.to(key.dtype)), 2))
                values.append(torch.cat((value, kv[:, :, 1].transpose(1, 2).to(value.dtype)), 2))
        mask = position_mask
        if not cfg.action_text_kv:
            mask = position_mask.clone()
            mask[:, visuals:prefix_length] = False
        attention = observer = None
        if cfg.action_current_ratio or cfg.action_motion_ratio:
            prior = cfg.action_current_ratio * current.float() / current.sum(-1, keepdim=True).clamp_min(1e-30)
            if cfg.action_motion_ratio:
                points = plan['trace']
                legacy = cfg.action_motion_prior == 'legacy_mean_8'
                if not legacy:
                    points = torch.cat((plan['start'][:, None], plan['end'][:, None], points), dim=1)
                prior = prior + cfg.action_motion_ratio * trajectory_prior(points, current.size(-1), legacy_mean=legacy)
            attention = partial(guided_attention, prior=prior)
        if statistics is not None:
            main = current.size(-1)
            sizes = (main, visuals - main, prefix_length - visuals, mask.size(1) - prefix_length)
            observer = partial(record_action_mass, statistics=statistics, prefix_sizes=sizes, valid_steps=valid_steps)
        return ActionCondition(keys, values, mask, position_mask=position_mask, attention=attention,
                               observe_attention=observer)
