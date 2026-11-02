"""Time-ordered motion tokens and a shared per-token 2D trace head."""
import torch
from torch import nn

from openpi.models.skill import QueryLayer


class Motion(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.queries = nn.Parameter(torch.randn(config.motion_queries, d) * .02)
        self.layers = nn.ModuleList(QueryLayer(d, config.auxiliary_heads) for _ in range(config.auxiliary_depth))
        self.film = nn.ModuleList(nn.Linear(d, 2 * d) for _ in self.layers)
        for film in self.film:
            nn.init.zeros_(film.weight)
            with torch.no_grad():
                film.bias[:d].fill_(1)
                film.bias[d:].zero_()
        self.trace_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def forward(self, context, mask, skill, arms, presence):
        b, _, d = context.shape
        count = len(self.queries)
        # Query j is supervised at (j + 1) * motion_stride, independently of Event length.
        query = self.queries + arms[:, None]
        query = query[None].expand(b, -1, -1, -1).flatten(1, 2)
        query_mask = presence[:, :, None].expand(b, 2, count).flatten(1)
        for layer, film in zip(self.layers, self.film, strict=True):
            scale, shift = film(skill)[:, :, None].expand(b, 2, count, 2 * d).flatten(1, 2).chunk(2, -1)
            query = layer(query, context, mask, query_mask, (scale, shift))
        latent = query.reshape(b, 2, count, d)
        return latent
