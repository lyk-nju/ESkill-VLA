"""Causal target-text completion using the existing PaliGemma language head.

The observed image/task prefix is bidirectional. Completion tokens attend to that
prefix and preceding completion tokens only; no Event or target image enters it.
"""
import torch
from torch.utils.checkpoint import checkpoint

from openpi.models.gemma import attend, qkv, residual_layer


def completion_logits(model, prefix, previous, previous_valid):
    """Logits for token 0..L; previous contains teacher tokens 0..L-1, without BOS."""
    positions = torch.arange(prefix.mask.shape[1], device=prefix.mask.device)
    last = torch.where(prefix.mask, positions, -1).max(-1).values
    first = prefix.context[torch.arange(len(last), device=last.device), last][:, None]
    length = previous.shape[1]
    hidden = model.paligemma_with_expert.embed_text(previous)
    if length:
        positions = prefix.mask.sum(-1, keepdim=True) + torch.arange(length, device=hidden.device)[None]
        causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
        mask = torch.cat((prefix.mask[:, None].expand(-1, length, -1),
                          causal[None] & previous_valid[:, None]), -1)
        for i, layer in enumerate(model.paligemma_with_expert.paligemma.language_model.layers):
            def apply_layer(x, layer=layer, i=i):
                norm, gate = layer.input_layernorm(x)
                q, k, v = qkv(layer, norm, model.config.heads, model.config.kv_heads, model.config.head_dim, positions)
                k, v = torch.cat((prefix.keys[i], k), 2), torch.cat((prefix.values[i], v), 2)
                return residual_layer(layer, x, attend(q, k, v, mask), gate)
            if model.config.gradient_checkpointing and model.training and hidden.requires_grad:
                hidden = checkpoint(apply_layer, hidden, use_reentrant=False)
            else:
                hidden = apply_layer(hidden)
        hidden, _ = model.paligemma_with_expert.paligemma.language_model.norm(hidden)
    states = torch.cat((first, hidden), 1)
    return model.paligemma_with_expert.paligemma.lm_head(states)


@torch.no_grad()
def generate_target_text(model, inputs, *, eos_id, max_tokens=32):
    """Greedy completion. Return termination separately; token confidence is not presence confidence."""
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    prefix = model.encode_observation(inputs)
    tokens = inputs['tokens'].new_empty((len(prefix.mask), 0))
    valid = torch.empty_like(tokens, dtype=torch.bool)
    done = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    for _ in range(max_tokens):
        logits = completion_logits(model, prefix, tokens, valid)[:, -1]
        next_token = logits.argmax(-1)
        next_token = torch.where(done, eos_id, next_token)
        tokens = torch.cat((tokens, next_token[:, None]), -1)
        valid = torch.cat((valid, ~done[:, None]), -1)
        done |= next_token == eos_id
        if done.all():
            break
    return dict(tokens=tokens, valid=valid, terminated=done)
