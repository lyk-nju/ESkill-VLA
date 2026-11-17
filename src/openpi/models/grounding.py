"""Parallel task grounding, legacy action grounding, and frozen-VAE supervision."""
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


class VisualSupervision(nn.Module):
    def __init__(self, config, vae):
        super().__init__()
        self.config = config
        self.role_names = ('current', 'future_object')
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

    def forward(self, anchors, targets, *, role_names=None):
        role_names = self.role_names if role_names is None else role_names
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
        role_ids = torch.arange(len(role_names), device=anchors.device)[None, None].expand(anchors.shape[:3]).flatten()[valid]
        losses, self.metrics, self.diagnostic_statistics = [], {}, {}
        # Empty ranks execute the same role reductions; an empty MLP keeps zero gradients connected.
        for role, name in enumerate(role_names):
            selected = role_ids == role
            loss = masked_mean(per_item, selected)
            losses.append(loss)
            self.metrics[f'grounding_{name}'] = loss.detach()
            self.diagnostic_statistics[f'grounding_{name}'] = mean_statistics(per_item, selected)
            for area, mask in (('fg', foreground), ('bg', 1 - foreground)):
                count = mask.sum((1, 2))
                value = (error.detach() * mask).sum((1, 2)) / count.clamp_min(1e-6)
                self.diagnostic_statistics[f'grounding_{name}_{area}'] = mean_statistics(value, selected & (count > 0))
        return dict(zip(role_names, losses, strict=True))


class Grounding(nn.Module):
    """Text-conditioned Current, then Current-conditioned Event-end Future (v2/v3).

    v0 retains the original joint decoder and parameter names for old checkpoints.
    Language is taken from the VLM's fused text-token positions, not a parsed noun.
    """
    def __init__(self, config):
        super().__init__()
        self.version = config.visual_primitive_version
        self.queries = nn.Parameter(torch.randn(2, config.role_queries, config.hidden_dim) * .02)
        self.decoder = QueryDecoder(config)
        if self.version != 'v0':
            self.text_attention = nn.MultiheadAttention(
                config.hidden_dim, config.auxiliary_heads, batch_first=True)
        if self.version in ('v2', 'v3'):
            self.future_decoder = QueryDecoder(config)
    def forward(self, context, mask, arms, *, text_context=None, text_mask=None):
        b, _, d = context.shape
        queries = (self.queries[None] + arms[:, None, None])[None].expand(b, -1, -1, -1, -1)
        memory = context.repeat_interleave(2, 0)
        memory_mask = mask.repeat_interleave(2, 0)

        if self.version == 'v0':
            # Exact old joint query graph; no new parameters are created in v0.
            return self.decoder(queries.flatten(0, 1).flatten(1, 2), memory, memory_mask).reshape(
                b, 2, 2, -1, d)

        if text_context is None or text_mask is None:
            raise ValueError('Text-conditioned grounding requires VLM text tokens and their mask')
        text_memory = text_context.repeat_interleave(2, 0)
        text_memory_mask = text_mask.repeat_interleave(2, 0)
        current = queries[:, :, 0].flatten(0, 1)
        future = queries[:, :, 1].flatten(0, 1)
        current = current + self.text_attention(
            current, text_memory, text_memory, key_padding_mask=~text_memory_mask,
            need_weights=False)[0]

        if self.version == 'v1':
            # Only Current's initialization differs; retain the old joint role decoder.
            both = torch.stack((current, future), dim=1).flatten(1, 2)
            return self.decoder(both, memory, memory_mask).reshape(b, 2, 2, -1, d)

        # Current cannot look at Future queries. Future gets Current plus the
        # observed scene, so both object identity and intended end state are available.
        current = self.decoder(current, memory, memory_mask)
        future = future + self.text_attention(
            future, text_memory, text_memory, key_padding_mask=~text_memory_mask,
            need_weights=False)[0]
        future = future + current.mean(dim=1, keepdim=True)
        future_memory = torch.cat((current, memory), dim=1)
        current_mask = torch.ones(current.shape[:2], dtype=torch.bool, device=current.device)
        future_mask = torch.cat((current_mask, memory_mask), dim=1)
        future = self.future_decoder(future, future_memory, future_mask)
        return torch.stack((current, future), dim=1).reshape(b, 2, 2, -1, d)


class TaskGrounding(nn.Module):
    """Single-arm parallel readers: pure task-conditioned Current and visual Subject."""
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.visual_projection = nn.Linear(config.vlm_width, d)
        self.text_projection = nn.Linear(config.vlm_width, d)
        self.current_queries = nn.Parameter(torch.randn(config.role_queries, d) * .02)
        self.subject_queries = nn.Parameter(torch.randn(config.role_queries, d) * .02)
        self.text_attention = nn.MultiheadAttention(d, config.auxiliary_heads, batch_first=True)
        self.current_decoder = QueryDecoder(config)
        self.subject_decoder = QueryDecoder(config)

    def forward(self, visual, visual_mask, text, text_mask):
        memory = self.visual_projection(visual)
        text = self.text_projection(text)
        current = self.current_queries[None].expand(len(memory), -1, -1)
        current = current + self.text_attention(current, text, text,
            key_padding_mask=~text_mask, need_weights=False)[0]
        subject = self.subject_queries[None].expand(len(memory), -1, -1)
        current = self.current_decoder(current, memory, visual_mask)
        subject = self.subject_decoder(subject, memory, visual_mask)
        # Keep the common two-slot label layout; the second arm is absent, not decoded.
        return {name: torch.stack((value, torch.zeros_like(value)), dim=1)
                for name, value in (('current', current), ('subject', subject))}
