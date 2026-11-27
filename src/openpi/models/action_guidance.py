"""Action attention on the annotated main grid, with native mass preserved per head and step."""
from functools import partial
from math import isqrt, sqrt

import torch
from torch import nn

from openpi.models.gemma import rotary
from openpi.models.model import ActionCondition


def trajectory_prior(points, patches):
    """Equal mixture of one-patch-standard-deviation kernels at the eight predicted future points."""
    side = isqrt(patches)
    axis = torch.arange(side, device=points.device, dtype=torch.float32) + .5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    centers = torch.stack((x, y), -1).reshape(1, 1, patches, 2)
    valid = torch.isfinite(points).all(-1) & (points >= 0).all(-1) & (points <= 1).all(-1)
    xy = points.float().masked_fill(~valid[..., None], 0) * side
    kernels = (-.5 * (centers - xy[:, :, None]).square().sum(-1)).softmax(-1)
    return (kernels * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def guided_probabilities(q, k, mask, prior):
    """The prior sums to its mixing weight; empty priors leave the native distribution intact."""
    with torch.autocast(q.device.type, enabled=False):
        grouped = q.float().unflatten(1, (k.size(1), -1))
        scores = (grouped @ k.float()[:, :, None].transpose(-1, -2)).flatten(1, 2) / sqrt(q.size(-1))
        native = scores.masked_fill(~mask[:, None, None], -torch.inf).softmax(-1)
        n = prior.shape[-1]
        main = native[..., :n]
        prior = prior[:, None, None, :]
        mixed = (1 - prior.sum(-1, keepdim=True)) * main + main.sum(-1, keepdim=True) * prior
        return torch.cat((mixed, native[..., n:]), -1)


def guided_attention(q, k, v, mask, *, prior):
    p = guided_probabilities(q, k, mask, prior)
    output = p.to(v.dtype).unflatten(1, (v.size(1), -1)) @ v[:, :, None]
    return output.flatten(1, 2).transpose(1, 2).flatten(2)


class ActionGuidance(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.motion_kv = nn.ModuleList(
            nn.Linear(config.hidden_dim, 2 * config.kv_heads * config.head_dim)
            for _ in range(config.depth) if config.action_motion_kv)

    def forward(self, prefix, plan, current):
        cfg = self.config
        mask = prefix.mask[:, :prefix.keys[0].size(2)]
        keys, values = prefix.keys, prefix.values
        if self.motion_kv:
            motion = plan['event_motion_latents']
            mask = torch.cat((mask, mask.new_ones(motion.shape[:2])), dim=1)
            positions = mask.long().cumsum(-1)[:, -motion.size(1):] - 1
            keys, values = [], []
            for key, value, project in zip(prefix.keys, prefix.values, self.motion_kv, strict=True):
                kv = project(motion).reshape(*motion.shape[:2], 2, cfg.kv_heads, cfg.head_dim)
                k = rotary(kv[:, :, 0].transpose(1, 2), positions)
                keys.append(torch.cat((key, k.to(key.dtype)), 2))
                values.append(torch.cat((value, kv[:, :, 1].transpose(1, 2).to(value.dtype)), 2))
        attention = None
        if cfg.action_current_ratio or cfg.action_motion_ratio:
            prior = cfg.action_current_ratio * current.float() / current.sum(-1, keepdim=True).clamp_min(1e-30)
            if cfg.action_motion_ratio:
                prior = prior + cfg.action_motion_ratio * trajectory_prior(plan['trace'], current.size(-1))
            attention = partial(guided_attention, prior=prior)
        return ActionCondition(keys, values, mask, attention=attention)
