"""LoRA for the two Gemma streams; base parameter names stay checkpoint-compatible."""
import math
from dataclasses import dataclass

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
        # Autocast already aligns Linear inputs and weights; avoid BF16 -> FP32 -> BF16 copies.
        lora_input = value if torch.is_autocast_enabled(value.device.type) else value.to(self.lora_A.dtype)
        update = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
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


def apply_grounding_lora(model, rank, alpha):
    """Install adapters only on the selected native language query projections."""
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    for i in model.config.attention_layers:
        attention = layers[i].self_attn
        attention.q_proj = LoRALinear(attention.q_proj, rank, alpha)


@dataclass(frozen=True)
class VLMFinetune:
    """PaliGemma language updates, independent of the layers observed by KLAL."""
    mode: str = 'frozen'
    layers: tuple[int, ...] | str = 'all'
    target_modules: tuple[str, ...] = ('q_proj',)
    lora_rank: int = 16
    lora_alpha: float = 16.
    freeze_vision_tower: bool = True
    freeze_image_projector: bool = True

    def __post_init__(self):
        if self.mode not in ('frozen', 'lora', 'full'):
            raise ValueError('vlm_finetune.mode must be frozen, lora or full')
        if self.layers != 'all':
            if isinstance(self.layers, str) or not self.layers:
                raise ValueError('layers must be all or a nonempty list')
            object.__setattr__(self, 'layers', tuple(self.layers))
        object.__setattr__(self, 'target_modules', tuple(self.target_modules))
        allowed = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
        if self.mode == 'lora' and (not self.target_modules or set(self.target_modules) - allowed
                or self.lora_rank < 1 or not math.isfinite(self.lora_alpha) or self.lora_alpha <= 0):
            raise ValueError('Invalid VLM LoRA targets, rank or alpha')

    def select(self, model):
        vlm = model.paligemma_with_expert.paligemma
        language = vlm.language_model
        indices = range(len(language.layers)) if self.layers == 'all' else self.layers
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(language.layers) for i in indices):
            raise ValueError('Invalid VLM layer indices')
        vlm.requires_grad_(False)
        vlm.vision_tower.requires_grad_(not self.freeze_vision_tower)
        vlm.multi_modal_projector.requires_grad_(not self.freeze_image_projector)
        if not self.freeze_vision_tower and model.config.gradient_checkpointing:
            vlm.vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.mode == 'full' and self.layers == 'all':
            language.requires_grad_(True)
        for i in indices:
            layer = language.layers[i]
            if self.mode == 'full':
                layer.requires_grad_(True)
            elif self.mode == 'lora':
                for parent in list(layer.modules()):
                    for name, child in list(parent.named_children()):
                        if name in self.target_modules:
                            if not isinstance(child, LoRALinear):
                                child = LoRALinear(child, self.lora_rank, self.lora_alpha)
                                setattr(parent, name, child)
                            child.lora_A.requires_grad_(True)
                            child.lora_B.requires_grad_(True)
