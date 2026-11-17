"""π0.5's paired Gemma streams, using local AdaRMSNorm and explicit KV access.

Adapted from Physical Intelligence's OpenPI (Apache-2.0). See THIRD_PARTY_NOTICES.
No patches to the installed Transformers package are required.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import GemmaConfig, GemmaForCausalLM, PaliGemmaConfig, PaliGemmaForConditionalGeneration


class RMSNorm(nn.Module):
    def __init__(self, width, *, adaptive=False, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.dense = nn.Linear(width, 3 * width) if adaptive else None
        if adaptive:
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            with torch.no_grad():
                self.dense.bias[2 * width:].fill_(1)
        else:
            self.weight = nn.Parameter(torch.zeros(width))

    def forward(self, x, cond=None):
        norm = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if self.dense is None:
            return (norm * (1 + self.weight.float())).to(x.dtype), None
        scale, shift, gate = self.dense(cond).unsqueeze(1).chunk(3, -1)
        return (norm * (1 + scale.float()) + shift.float()).to(x.dtype), gate.to(x.dtype)


def rotary(x, positions):
    half = x.shape[-1] // 2
    inv = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions.float()[:, None, :, None] * inv
    sin, cos = angles.sin().to(x.dtype), angles.cos().to(x.dtype)
    first, second = x.chunk(2, -1)
    return torch.cat((first * cos - second * sin, second * cos + first * sin), -1)


def qkv(layer, hidden, heads, kv_heads, head_dim, positions):
    shape = (*hidden.shape[:2], -1, head_dim)
    q = layer.self_attn.q_proj(hidden).view(shape).transpose(1, 2)
    k = layer.self_attn.k_proj(hidden).view(shape).transpose(1, 2)
    v = layer.self_attn.v_proj(hidden).view(shape).transpose(1, 2)
    return rotary(q, positions), rotary(k, positions), v


def attend(q, k, v, mask):
    repeats = q.shape[1] // k.shape[1]
    # π0.5 has one KV head: expanding shares storage instead of copying it eight times.
    k = k[:, :, None].expand(-1, -1, repeats, -1, -1).flatten(1, 2)
    v = v[:, :, None].expand(-1, -1, repeats, -1, -1).flatten(1, 2)
    attention_mask = mask[:, None, None, :] if mask.ndim == 2 else mask[:, None]
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return output.transpose(1, 2).flatten(2)


def residual_layer(layer, hidden, attention, gate, cond=None):
    delta = layer.self_attn.o_proj(attention)
    hidden = hidden + (delta if gate is None else delta * gate)
    norm, gate = layer.post_attention_layernorm(hidden, cond)
    delta = layer.mlp(norm)
    return hidden + (delta if gate is None else delta * gate)


class PaliGemmaWithExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        text = dict(hidden_size=config.vlm_width, intermediate_size=config.vlm_mlp,
                    num_hidden_layers=config.depth, num_attention_heads=config.heads,
                    num_key_value_heads=config.kv_heads, head_dim=config.head_dim,
                    vocab_size=config.vocab_size, hidden_activation="gelu_pytorch_tanh")
        vision = dict(hidden_size=config.vision_width, intermediate_size=config.vision_mlp,
                      num_hidden_layers=config.vision_depth, num_attention_heads=config.vision_heads,
                      image_size=config.image_size, patch_size=config.patch_size, vision_use_head=False)
        pali = PaliGemmaConfig(text_config=text, vision_config=vision,
                              projection_dim=config.vlm_width, hidden_size=config.vlm_width,
                              vocab_size=config.vocab_size, image_token_index=config.vocab_size - 1)
        self.paligemma = PaliGemmaForConditionalGeneration(pali)
        expert = GemmaConfig(**{**text, "hidden_size": config.expert_width,
                                "intermediate_size": config.expert_mlp})
        self.gemma_expert = GemmaForCausalLM(expert)
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.lm_head = None
        for model, width, adaptive in (
            (self.paligemma.language_model, config.vlm_width, False),
            (self.gemma_expert.model, config.expert_width, True),
        ):
            for layer in model.layers:
                layer.input_layernorm = RMSNorm(width, adaptive=adaptive)
                layer.post_attention_layernorm = RMSNorm(width, adaptive=adaptive)
            model.norm = RMSNorm(width, adaptive=adaptive)

    def embed_images(self, images):
        # OpenPI uses the projector output without HF's extra 1/sqrt(width).
        tower = self.paligemma.vision_tower
        value = tower(pixel_values=images.to(next(tower.parameters()).dtype)).last_hidden_state
        return self.paligemma.multi_modal_projector(value)

    def embed_text(self, tokens):
        result = self.paligemma.language_model.embed_tokens(tokens)
        return result * math.sqrt(result.shape[-1])
