"""Fixed-frame Event-guided motion tokens with path and gripper readouts."""
import torch
from torch import nn

from openpi.models.skill import QueryLayer


class Motion(nn.Module):
    """Fixed-frame Event-guided plan; path and gripper are optional auxiliary readouts."""
    def __init__(self, config):
        super().__init__()
        from openpi.models.pi0 import sine_position
        d = config.hidden_dim
        self.queries = nn.Parameter(torch.randn(config.motion_queries, d) * .02)
        self.register_buffer('offset_embedding', sine_position(torch.tensor(config.motion_offsets_frames), d))
        self.layers = nn.ModuleList(QueryLayer(d, config.auxiliary_heads) for _ in range(config.auxiliary_depth))
        self.film = nn.ModuleList(nn.Linear(d, 2 * d) for _ in self.layers)
        for film in self.film:
            nn.init.normal_(film.weight, std=.01)
            nn.init.zeros_(film.bias)
        self.trace_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.gripper_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, context, mask, skill, arms, presence):
        b, _, d = context.shape
        count = len(self.queries)
        query = self.queries + self.offset_embedding.to(self.queries.dtype) + arms[:, None]
        query = query[None].expand(b, -1, -1, -1).flatten(1, 2)
        query_mask = presence[:, :, None].expand(b, 2, count).flatten(1)
        for layer, film in zip(self.layers, self.film, strict=True):
            scale, shift = film(skill)[:, :, None].expand(b, 2, count, 2 * d).flatten(1, 2).chunk(2, -1)
            query = layer(query, context, mask, query_mask, (1 + scale, shift))
        return query.reshape(b, 2, count, d)
