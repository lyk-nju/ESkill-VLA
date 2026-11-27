"""Read selected PaliGemma text rows while leaving π0.5's SDPA forward unchanged."""

from dataclasses import dataclass
import math
from threading import RLock
from unittest.mock import patch

import torch

import openpi.models.pi0 as pi0


_CAPTURE_LOCK = RLock()


@dataclass(frozen=True)
class NativeAttention:
    spatial: torch.Tensor       # [batch, method, layer, head, visual key slot]
    visual_mass: torch.Tensor   # [batch, method, layer, head], full softmax
    valid: torch.Tensor         # [batch, method], true iff the method has query tokens
    layers: tuple[int, ...]
    visual_keys: torch.Tensor   # [batch, visual key slot], false for masked camera views

    def region_mass(self, patch_weights):
        """Raw and spatial-conditional mass for fractional target/patch overlaps."""
        weights = torch.as_tensor(
            patch_weights, device=self.spatial.device, dtype=self.spatial.dtype)
        if weights.shape != self.visual_keys.shape:
            raise ValueError("Expected one target overlap weight for each image key slot")
        conditional = (self.spatial * weights[:, None, None, None, :]).sum(-1)
        return conditional * self.visual_mass, conditional

    def normalized_entropy(self):
        """Spatial entropy in [0, 1], normalized by the number of valid image keys."""
        p = self.spatial
        entropy = -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
        normalizer = self.visual_keys.sum(-1).clamp_min(2).float().log()
        return (entropy / normalizer[:, None, None, None]).clamp(0, 1)

    def patch_centers(self, height, width):
        """Return soft centroids in unit-square (x, y) coordinates for one image view."""
        if self.spatial.shape[-1] != height * width:
            raise ValueError("Center calculation expects the patch grid of one image view")
        axis_x = (torch.arange(width, device=self.spatial.device, dtype=torch.float32) + .5) / width
        axis_y = (torch.arange(height, device=self.spatial.device, dtype=torch.float32) + .5) / height
        y, x = torch.meshgrid(axis_y, axis_x, indexing="ij")
        coordinates = torch.stack((x.flatten(), y.flatten()), dim=-1)
        with torch.autocast(device_type=self.spatial.device.type, enabled=False):
            centers = self.spatial @ coordinates
        return centers.masked_fill(self.visual_mass[..., None] == 0, float("nan"))


def selected_probabilities(q, k, allowed_keys, rows):
    """Full-key softmax for selected post-RoPE query rows, without V or a square matrix.

    Gemma uses the default scaled_dot_product_attention scale and repeats each
    KV head consecutively. Grouping queries implements that same ordering
    without copying the keys. This function is differentiable with respect to
    Q/K; the E0 capture wrapper freezes inference independently.
    """
    batch, heads, length, width = q.shape
    if k.ndim != 4 or k.shape[0] != batch or k.shape[-1] != width or heads % k.shape[1]:
        raise ValueError("Q/K shapes are incompatible with grouped query attention")
    if rows.ndim != 1 or rows.dtype != torch.long:
        raise ValueError("Expected a vector of sequence row indices")
    kv_heads, keys = k.shape[1:3]
    if allowed_keys.dtype != torch.bool:
        raise ValueError("The native model must provide a Boolean attention mask")
    if allowed_keys.shape == (batch, keys):
        mask = allowed_keys[:, None, None, :]
    elif allowed_keys.shape == (batch, length, keys):
        mask = allowed_keys.index_select(1, rows)[:, None, :, :]
    else:
        raise ValueError("Unexpected native attention mask shape")
    with torch.autocast(device_type=q.device.type, enabled=False):
        grouped = q.index_select(2, rows).float().reshape(batch, kv_heads, heads // kv_heads,
                                                        len(rows), width)
        logits = grouped @ k.float().transpose(-1, -2).unsqueeze(2)
        logits = logits.flatten(1, 2) * (width ** -.5)
        # This is the all-keys denominator, including every valid text token.
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = torch.where(mask.any(-1, keepdim=True), logits, 0)
        return logits.softmax(-1).masked_fill(~mask, 0)


@torch.no_grad()
def capture_native_attention(model, inputs, query_mask, *, layers=None):
    """Observe E0 attention during one encode_observation call.

    query_mask has shape [batch, method, padded text tokens]. Empty methods
    remain unavailable. Each method first averages the full-softmax image
    probability over its text rows; conditional spatial maps divide by that
    method/head's resulting visual mass. All layers are zero indexed.
    """
    if model.training:
        raise ValueError("Native grounding probe requires evaluation mode")
    text_mask = inputs["token_mask"]
    if text_mask.ndim != 2 or text_mask.dtype != torch.bool:
        raise ValueError("Expected a Boolean [batch, text] model token mask")
    batch, length = text_mask.shape
    selected = torch.as_tensor(query_mask, device=text_mask.device)
    if (selected.dtype != torch.bool or selected.ndim != 3 or selected.shape[0] != batch
            or selected.shape[1] == 0 or selected.shape[2] != length):
        raise ValueError("Query mask must be Boolean [batch, method, text]")
    if (selected & ~text_mask[:, None]).any():
        raise ValueError("Probe cannot query an invalid or padded text token")
    vision_mask = inputs["image_mask"]
    if vision_mask.ndim != 2 or vision_mask.shape[0] != batch or vision_mask.dtype != torch.bool:
        raise ValueError("Expected a Boolean [batch, camera] model image mask")
    if not vision_mask.any(-1).all():
        raise ValueError("Each input must contain a valid camera view")

    num_layers = len(model.paligemma_with_expert.paligemma.language_model.layers)
    requested = tuple(range(num_layers)) if layers is None else tuple(layers)
    if (not requested or len(set(requested)) != len(requested)
            or any(not isinstance(i, int) or not 0 <= i < num_layers for i in requested)):
        raise ValueError("Expected distinct zero-based PaliGemma layer indices")
    requested_set = set(requested)
    token_rows = torch.where(selected.any(dim=(0, 1)))[0]
    weights = selected.index_select(2, token_rows).float()
    counts = weights.sum(-1, keepdim=True)
    weights = weights / counts.clamp_min(1)
    valid = counts[..., 0] > 0
    maps = {}
    visual_slots = None
    visual_keys = None
    calls = 0

    # Native order is established by checking encode_observation's own key mask
    # at the point where projected, rotated Q/K enter SDPA. No inference module,
    # model parameters, projections, or logits are changed.
    with _CAPTURE_LOCK:
        native = pi0.attend

        def observe(q, k, v, mask):
            nonlocal calls, visual_slots, visual_keys
            layer = calls
            calls += 1
            if visual_slots is None:
                visual_slots = k.shape[2] - length
                views = vision_mask.shape[1]
                if visual_slots <= 0 or views == 0 or visual_slots % views:
                    raise RuntimeError("Unexpected image/text token layout")
                visual_keys = vision_mask.repeat_interleave(visual_slots // views, dim=1)
                if mask.ndim != 2 or not torch.equal(mask, torch.cat((visual_keys, text_mask), dim=1)):
                    raise RuntimeError("Native prefix ordering or mask differs from image-then-text")
            if layer in requested_set:
                if len(token_rows):
                    rows = token_rows + visual_slots
                    raw = selected_probabilities(q, k, mask, rows)[..., :visual_slots]
                    with torch.autocast(device_type=q.device.type, enabled=False):
                        maps[layer] = torch.einsum("bmq,bhqp->bmhp", weights, raw)
                else:
                    maps[layer] = q.new_zeros((batch, selected.shape[1], q.shape[1],
                                                visual_slots), dtype=torch.float32)
            return native(q, k, v, mask)

        with patch.object(pi0, "attend", observe):
            prefix = model.encode_observation(inputs)

    if calls != num_layers or visual_keys is None or not torch.equal(
            prefix.visual_mask[:, :visual_slots], visual_keys):
        raise RuntimeError("Capture did not observe the expected PaliGemma prefix layers")
    raw = torch.stack([maps[layer] for layer in requested], dim=2)
    mass = raw.sum(-1)
    spatial = raw / torch.where(mass > 0, mass, 1)[..., None]
    return NativeAttention(spatial, mass, valid, requested, visual_keys)
