"""LoRA for the two Gemma streams; base parameter names stay checkpoint-compatible."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, layer, rank, alpha):
        super().__init__()
        self.weight, self.bias = layer.weight, layer.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, layer.in_features, device=layer.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(layer.out_features, rank, device=layer.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value):
        base = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(value.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (update * self.scale).to(base.dtype)


def apply_lora(model, rank=16, alpha=16):
    for name, parameter in model.named_parameters():
        if name in model._base_keys:
            parameter.requires_grad_(False)
    targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    streams = (model.paligemma_with_expert.paligemma.language_model,
               model.paligemma_with_expert.gemma_expert.model)
    count = 0
    for stream in streams:
        for layer in stream.layers:
            for parent in layer.modules():
                for name, child in list(parent.named_children()):
                    if name in targets and isinstance(child, nn.Linear):
                        setattr(parent, name, LoRALinear(child, rank, alpha))
                        count += 1
    return count
