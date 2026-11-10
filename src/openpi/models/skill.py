"""Continuous visual skills with semantic routing and shared/specialized experts."""
import torch
from torch import nn
from torch.nn import functional as F


class QueryLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, query, context, context_mask, query_mask=None, film=None):
        norm = self.norms[0](query)
        if film is not None:
            scale, shift = film
            norm = norm * scale + shift
        query = query + self.self_attention(
            norm, norm, norm, key_padding_mask=None if query_mask is None else ~query_mask,
            need_weights=False)[0]
        query = query + self.cross_attention(
            self.norms[1](query), context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        return query + self.ffn(self.norms[2](query))


class QueryDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            QueryLayer(config.hidden_dim, config.auxiliary_heads) for _ in range(config.auxiliary_depth))

    def forward(self, query, context, context_mask, query_mask=None):
        for layer in self.layers:
            query = layer(query, context, context_mask, query_mask)
        return query


class SkillExpert(nn.Module):
    """Small residual SwiGLU stack, returning its learned residual contribution."""
    def __init__(self, config):
        super().__init__()
        d, m = config.skill_dim, config.skill_mlp
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(config.skill_depth))
        self.gates = nn.ModuleList(nn.Linear(d, m, bias=False) for _ in self.norms)
        self.up = nn.ModuleList(nn.Linear(d, m, bias=False) for _ in self.norms)
        self.down = nn.ModuleList(nn.Linear(m, d, bias=False) for _ in self.norms)
        self.scale = config.skill_residual_scale

    def forward(self, hidden):
        value = hidden
        for norm, gate, up, down in zip(self.norms, self.gates, self.up, self.down, strict=True):
            x = norm(value)
            value = value + self.scale * down(F.silu(gate(x)) * up(x))
        return value - hidden


class Skill(nn.Module):
    """Continuous visual query with CE-trained semantic routing and shared + Top-1 experts."""
    def __init__(self, config):
        super().__init__()
        from dataclasses import replace
        d = config.skill_dim
        self.context_projection = nn.Linear(config.vlm_width, d)
        self.arm_embedding = nn.Embedding(2, d)
        self.query = nn.Parameter(torch.randn(1, d) * .02)
        self.decoder = QueryDecoder(replace(config, hidden_dim=d))
        self.classifier = nn.Linear(d, len(config.active_skill_ids))
        self.shared = SkillExpert(config)
        self.experts = nn.ModuleList(SkillExpert(config) for _ in config.active_skill_ids)
        self.norm = nn.LayerNorm(d)
        self.to_motion = nn.Linear(d, config.hidden_dim)
        self.register_buffer('active_skill_ids', torch.tensor(config.active_skill_ids, dtype=torch.long))

    def forward(self, context, mask, presence):
        b = len(context)
        memory = self.context_projection(context).repeat_interleave(2, 0)
        query = (self.query + self.arm_embedding.weight)[None].expand(b, -1, -1).reshape(b * 2, 1, -1)
        hidden = self.decoder(query, memory, mask.repeat_interleave(2, 0))[:, 0]
        logits = self.classifier(hidden)
        route = logits.argmax(-1)
        selected = torch.zeros_like(hidden)
        # Empty slices retain zero-gradient edges for unused experts, keeping DDP reductions aligned.
        # Only the routed examples execute each specialist's nonempty FFN workload.
        for index, expert in enumerate(self.experts):
            chosen = (route == index) & presence.flatten()
            selected = selected.index_add(0, chosen.nonzero().flatten(), expert(hidden[chosen]))
        prior = self.norm(hidden + self.shared(hidden) + selected).reshape(b, 2, -1)
        prior = torch.where(presence[..., None], prior, 0)
        return logits.reshape(b, 2, -1), prior, self.to_motion(prior)

    def labels_to_active(self, labels):
        equal = labels[..., None] == self.active_skill_ids
        return equal.long().argmax(-1), equal.any(-1)
