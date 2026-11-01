"""Three role queries and training-only VAE latent denoising."""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.model import masked_mean
from openpi.models.pi0 import sine_position
from openpi.models.skill import QueryDecoder


class Grounding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(3, config.role_queries, config.hidden_dim) * .02)
        self.decoder = QueryDecoder(config)

    def current(self, context, mask, arms):
        b, _, d = context.shape
        query = self.queries[None, :2] + arms[:, None, None]
        query = query.unsqueeze(0).expand(b, -1, -1, -1, -1).flatten(0, 2)
        value = self.decoder(query, context.repeat_interleave(4, 0), mask.repeat_interleave(4, 0))
        return value.reshape(b, 2, 2, -1, d)

    def goal(self, context, mask, arms, current, skill):
        b, n, d = context.shape
        memory = torch.cat((context[:, None].expand(-1, 2, -1, -1),
                            current.flatten(2, 3), skill[:, :, None]), 2).flatten(0, 1)
        memory_mask = torch.cat((mask[:, None].expand(-1, 2, -1),
                                torch.ones(b, 2, current.shape[2] * current.shape[3] + 1,
                                           dtype=torch.bool, device=mask.device)), -1).flatten(0, 1)
        query = (self.queries[2] + arms[:, None]).unsqueeze(0).expand(b, -1, -1, -1).flatten(0, 1)
        goal = self.decoder(query, memory, memory_mask).reshape(b, 2, 1, -1, d)
        return torch.cat((current, goal), 2)


class VisualSupervision(nn.Module):
    def __init__(self, config, vae):
        super().__init__()
        self.config = config
        self.vae = vae.requires_grad_(False).eval()
        channels = vae.config.latent_channels
        d = config.hidden_dim
        self.noisy_in = nn.Linear(channels, d)
        self.noise_out = nn.Linear(d, channels)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.roles = nn.Embedding(3, d)
        self.decoder = QueryDecoder(config)

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def forward(self, anchors, targets, arms):
        valid = targets["role_valid"].flatten()
        if not valid.any():
            return anchors.sum() * 0
        images = targets["role_images"].flatten(0, 2)[valid]
        with torch.no_grad():
            latent = self.vae.encode(images.to(next(self.vae.parameters()).dtype)).latent_dist.mode()
            shift = getattr(self.vae.config, "shift_factor", None) or 0.
            latent = (latent - shift) * self.vae.config.scaling_factor
        latent = latent.float()
        b, channels, height, width = latent.shape
        d = anchors.shape[-1]
        time = torch.rand(b, device=latent.device) * .998 + .001
        alpha = torch.cos((time + .008) / 1.008 * torch.pi / 2).square()
        noise = torch.randn_like(latent)
        noisy = alpha[:, None, None, None].sqrt() * latent + (1 - alpha[:, None, None, None]).sqrt() * noise
        y, x = torch.meshgrid(torch.linspace(0, 1, height, device=latent.device),
                              torch.linspace(0, 1, width, device=latent.device), indexing="ij")
        spatial = torch.cat((sine_position(x.flatten(), d // 2),
                             sine_position(y.flatten(), d // 2)), -1)
        role_ids = torch.arange(3, device=latent.device)[None, None].expand(anchors.shape[0], 2, 3).flatten()[valid]
        arm_ids = torch.arange(2, device=latent.device)[None, :, None].expand(anchors.shape[0], 2, 3).flatten()[valid]
        query = self.noisy_in(noisy.flatten(2).transpose(1, 2))
        query = query + spatial + (self.time(sine_position(time, d)) + self.roles(role_ids) + arms[arm_ids])[:, None]
        context = anchors.flatten(0, 2)[valid]
        hidden = self.decoder(query, context, torch.ones(context.shape[:2], dtype=torch.bool, device=context.device))
        predicted = self.noise_out(hidden).transpose(1, 2).reshape(b, channels, height, width)
        foreground = F.interpolate(targets["role_weights"].flatten(0, 2)[valid, None],
                                   size=(height, width), mode="area")[:, 0]
        weights = self.config.background_weight + (
            self.config.foreground_weight - self.config.background_weight) * foreground
        error = (predicted.float() - noise).square().mean(1)
        per_item = (error * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-6)
        # Equal weight per role; each role averages valid batch/arm entries.
        return sum(masked_mean(per_item, role_ids == role) for role in range(3))
