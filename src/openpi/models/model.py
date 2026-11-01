"""Tensor contracts shared by training and policy inference."""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Prefix:
    context: Tensor
    mask: Tensor
    visual_mask: Tensor
    keys: list[Tensor]
    values: list[Tensor]


@dataclass
class ActionCondition:
    keys: list[Tensor]
    values: list[Tensor]
    mask: Tensor
    step_skill: Tensor


def to_device(value, device):
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    valid = valid.to(values.dtype).expand_as(values)
    return (torch.where(valid.bool(), values, 0) * valid).sum() / valid.sum().clamp_min(1)
