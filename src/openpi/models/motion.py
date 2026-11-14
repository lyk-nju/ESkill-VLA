"""Fixed-frame Event-guided motion tokens with optional goal-first visual bottleneck."""
import torch
from torch import nn

from openpi.models.skill import QueryDecoder, QueryLayer


class Motion(nn.Module):
    """Visual motion prior; v3 never consumes the language-fused global VLM context."""
    def __init__(self, config):
        super().__init__()
        from openpi.models.pi0 import sine_position
        d = config.hidden_dim
        self.goal_first = config.visual_primitive_version == 'v3'
        self.queries = nn.Parameter(torch.randn(config.motion_queries, d) * .02)
        self.register_buffer('offset_embedding', sine_position(torch.tensor(config.motion_offsets_frames), d))
        self.layers = nn.ModuleList(QueryLayer(d, config.auxiliary_heads) for _ in range(config.auxiliary_depth))
        self.film = nn.ModuleList(nn.Linear(d, 2 * d) for _ in self.layers)
        for film in self.film:
            nn.init.normal_(film.weight, std=.01)
            nn.init.zeros_(film.bias)
        if self.goal_first:
            # The first decoder can only read same-arm Current, Future and Skill.
            self.goal_decoder = QueryDecoder(config)
            # Vision tower patches still carry image-grid positional embeddings.
            self.spatial_projection = nn.Linear(config.vlm_width, d)
        self.trace_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.gripper_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, context, mask, skill, arms, presence, *, grounding=None, spatial=None, spatial_mask=None):
        b, _, d = skill.shape
        count = len(self.queries)
        query = self.queries + self.offset_embedding.to(self.queries.dtype) + arms[:, None]
        query = query[None].expand(b, -1, -1, -1).flatten(1, 2)
        query_mask = presence[:, :, None].expand(b, 2, count).flatten(1)

        if self.goal_first:
            if grounding is None or spatial is None or spatial_mask is None:
                raise ValueError('v3 Motion requires predicted Current/Future and pre-fusion visual patches')
            if grounding.ndim != 5 or grounding.shape[:3] != (b, 2, 2) or grounding.shape[-1] != d:
                raise ValueError('Grounding must have [batch, arm, current/future, query, hidden] axes')
            goal = torch.cat((grounding.flatten(2, 3), skill[:, :, None]), dim=2)
            goal = goal.flatten(0, 1)
            goal_mask = torch.ones(goal.shape[:2], dtype=torch.bool, device=goal.device)
            # Per-arm object/goal selection precedes spatial refinement. Absent
            # arms are kept finite here; the joint spatial stage masks them.
            query = self.goal_decoder(query.reshape(b * 2, count, d), goal, goal_mask)
            query = query.reshape(b, 2 * count, d)
            context = self.spatial_projection(spatial.to(self.spatial_projection.weight.dtype))
            mask = spatial_mask
        for layer, film in zip(self.layers, self.film, strict=True):
            scale, shift = film(skill)[:, :, None].expand(b, 2, count, 2 * d).flatten(1, 2).chunk(2, -1)
            query = layer(query, context, mask, query_mask, (1 + scale, shift))
        return query.reshape(b, 2, count, d)
