"""Three role queries and training-only regression to frozen VAE latents."""
from math import isqrt
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.model import masked_mean
from openpi.models.diagnostics import mean_statistics
from openpi.models.skill import QueryDecoder


def load_vae(path):
    """Load local original-FLUX or Diffusers weights strictly, with FP32 execution."""
    from diffusers import AutoencoderKL
    from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
    from safetensors.torch import load_file

    path = Path(path)
    config = AutoencoderKL.load_config(str(path), local_files_only=True)
    weights = load_file(str(path / 'diffusion_pytorch_model.safetensors'), device='cpu')
    if 'encoder.down.0.block.0.norm1.weight' in weights:
        weights = convert_ldm_vae_checkpoint(weights, config)
    vae = AutoencoderKL.from_config(config).float()
    vae.load_state_dict(weights, strict=True)
    return vae.requires_grad_(False).eval()


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
        self.vae = vae.float().requires_grad_(False).eval()
        self.channels = vae.config.latent_channels
        self.grid = isqrt(config.role_queries)
        factor = 2 ** (len(vae.config.block_out_channels) - 1)
        if self.grid ** 2 != config.role_queries or config.canvas_size % (factor * self.grid):
            raise ValueError('Role queries must form a square grid dividing the VAE latent canvas')
        if not 0 < config.background_weight < config.foreground_weight:
            raise ValueError('Visual loss requires foreground_weight > background_weight > 0')
        self.latent_size = config.canvas_size // factor
        self.patch_size = self.latent_size // self.grid
        d = config.hidden_dim
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, self.channels * self.patch_size ** 2))
        self.metrics = {}
        self.diagnostic_statistics = {}

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()
        return self

    def predict_latents(self, anchors):
        """Row-major token grid -> channel-first latent image; accepts leading batch/arm/role axes."""
        g, p = self.grid, self.patch_size
        blocks = self.head(anchors).reshape(-1, g, g, self.channels, p, p)
        return blocks.permute(0, 3, 1, 4, 2, 5).reshape(
            *anchors.shape[:-2], self.channels, self.latent_size, self.latent_size)

    @torch.no_grad()
    def encode_targets(self, images):
        # VAE stays FP32 even inside the training loop's BF16 autocast.
        with torch.autocast(device_type=images.device.type, enabled=False):
            raw = self.vae.encode(images.float()).latent_dist.mode()
            shift = getattr(self.vae.config, 'shift_factor', None) or 0.
            return (raw - shift) * self.vae.config.scaling_factor

    def forward(self, anchors, targets):
        valid = targets["role_valid"].flatten()
        predicted = self.predict_latents(anchors.flatten(0, 2)[valid]).float()
        latent = (self.encode_targets(targets['role_images'].flatten(0, 2)[valid])
                  if valid.any() else torch.empty_like(predicted))
        if predicted.shape != latent.shape:
            raise ValueError('Role canvas and projected latent dimensions differ')
        foreground = F.interpolate(targets["role_weights"].flatten(0, 2)[valid, None],
                                   size=(self.latent_size, self.latent_size), mode="area")[:, 0].float()
        weights = self.config.background_weight + (
            self.config.foreground_weight - self.config.background_weight) * foreground
        error = (predicted - latent).square().mean(1)
        per_item = (error * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-6)
        role_ids = torch.arange(3, device=anchors.device)[None, None].expand(anchors.shape[:3]).flatten()[valid]
        losses, self.metrics, self.diagnostic_statistics = [], {}, {}
        # Empty ranks execute the same role reductions; an empty MLP keeps zero gradients connected.
        for role, name in enumerate(('ee', 'obj', 'goal')):
            selected = role_ids == role
            loss = masked_mean(per_item, selected)
            losses.append(loss)
            self.metrics[f'grounding_{name}'] = loss.detach()
            self.diagnostic_statistics[f'grounding_{name}'] = mean_statistics(per_item, selected)
            for area, mask in (('fg', foreground), ('bg', 1 - foreground)):
                count = mask.sum((1, 2))
                value = (error.detach() * mask).sum((1, 2)) / count.clamp_min(1e-6)
                self.diagnostic_statistics[f'grounding_{name}_{area}'] = mean_statistics(value, selected & (count > 0))
        return sum(losses)
