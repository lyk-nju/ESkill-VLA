"""Joint-arm spatial traces conditioned on each arm's current skill."""
import torch
from torch import nn

from openpi.models.pi0 import sine_position
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
        self.trace_head = nn.Linear(d, 3)

    def forward(self, context, mask, skill, arms, presence):
        b, _, d = context.shape
        count = len(self.queries)
        progress = sine_position(torch.linspace(0, 1, count, device=context.device), d)
        query = self.queries + progress + arms[:, None]
        query = query[None].expand(b, -1, -1, -1).flatten(1, 2)
        query_mask = presence[:, :, None].expand(b, 2, count).flatten(1)
        for layer, film in zip(self.layers, self.film, strict=True):
            scale, shift = film(skill)[:, :, None].expand(b, 2, count, 2 * d).flatten(1, 2).chunk(2, -1)
            query = layer(query, context, mask, query_mask, (scale, shift))
        latent = query.reshape(b, 2, count, d)
        return latent, self.trace_head(latent)
