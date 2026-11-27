"""Tensor contracts shared by training and policy inference."""
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass
class Prefix:
    context: Tensor
    mask: Tensor
    visual_mask: Tensor
    keys: list[Tensor]
    values: list[Tensor]
    # Projected visual patches before language-model fusion; optional for legacy callers.
    visual_tokens: Tensor | None = None
    text_image_attention: Tensor | None = None  # [B, selected layers, heads, text, image patches]; full-key denominator.


@dataclass(frozen=True)
class CameraLayout:
    """Contiguous main, optional auxiliary views, task, then optional planner queries."""
    patches: int
    visuals: int
    words: int

    @classmethod
    def from_prefix(cls, config, prefix, text_mask):
        patches = (config.image_size // config.patch_size) ** 2
        return cls(patches, prefix.visual_tokens.size(1), text_mask.size(1))

    @property
    def main(self):
        return slice(0, self.patches)

    @property
    def text(self):
        return slice(self.visuals, self.prefix_length)

    @property
    def prefix_length(self):
        return self.visuals + self.words


@dataclass
class ActionCondition:
    keys: list[Tensor]
    values: list[Tensor]
    mask: Tensor
    # Attention visibility can differ from the original valid positions (Action patch masking).
    position_mask: Tensor | None = None
    attention: Callable | None = None
    observe_attention: Callable | None = None  # Receives layer, Q, K and the actual key visibility.


def to_device(value, device):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Global valid-item mean after DDP averages gradients (and logged losses)."""
    valid = valid.to(values.dtype).expand_as(values)
    count = valid.sum(dtype=torch.float32)
    world_size = 1
    if dist.is_initialized():
        dist.all_reduce(count)
        world_size = dist.get_world_size()
    return (torch.where(valid.bool(), values, 0) * valid).sum() * world_size / count.clamp_min(1)
