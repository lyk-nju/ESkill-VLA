"""Current semantic selection and event-conditioned future skill prediction."""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.pi0 import sine_position


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


class Skill(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, k = config.hidden_dim, len(config.skill_names)
        self.current_query = nn.Parameter(torch.randn(1, d) * .02)
        self.current_decoder = QueryDecoder(config)
        self.semantic_projection = nn.Linear(d, config.vlm_width)
        self.register_buffer("text_prototypes", torch.zeros(k, config.vlm_width))
        self.table = nn.Embedding(k, d)
        self.future_query = nn.Parameter(torch.randn(1, d) * .02)
        self.future_decoder = QueryDecoder(config)
        self.future_head = nn.Linear(d, k)
        self.horizon = config.action_horizon

    def current(self, context, mask, arms):
        b, n, d = context.shape
        query = (self.current_query + arms).unsqueeze(0).expand(b, -1, -1).reshape(b * 2, 1, d)
        hidden = self.current_decoder(query, context.repeat_interleave(2, 0), mask.repeat_interleave(2, 0))
        semantic = F.normalize(self.semantic_projection(hidden[:, 0]).float(), dim=-1)
        logits = (semantic @ self.text_prototypes.float().T / .07).reshape(b, 2, -1)
        return logits, self.table(logits.argmax(-1))

    def future(self, context, mask, arms, presence):
        b, _, d = context.shape
        positions = sine_position(torch.arange(self.horizon, device=context.device), d)
        query = self.future_query + positions[None] + arms[:, None]
        query = query.unsqueeze(0).expand(b, -1, -1, -1)
        query_mask = presence[:, :, None].expand(b, 2, self.horizon).flatten(1)
        hidden = self.future_decoder(query.flatten(1, 2), context, mask, query_mask)
        logits = self.future_head(hidden).reshape(b, 2, self.horizon, -1)
        vectors = self.table(logits.argmax(-1)) * presence[:, :, None, None]
        return logits, vectors
