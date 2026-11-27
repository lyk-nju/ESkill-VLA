"""Historical planner architectures compared against the joint ten-slot planner."""
import torch
from torch import nn

from openpi.models.motion_planner import MotionPlanner, PlannerLayer
from openpi.models.pi0 import sine_position


class EarlyBoundaryPlanner(MotionPlanner):
    joint_slots = False

    def temporal_slots(self, layer, readout):
        kv = layer.temporal_norm(readout)
        return readout[:, 2:] + layer.temporal(kv[:, 2:], kv, kv, need_weights=False)[0]

    def decode_slots(self, motion, readout, boundaries):
        xy = self.trace_head(boundaries).float()
        return dict(start=xy[:, 0], end=xy[:, 1], boundary_latents=boundaries, first_readout=readout,
                    trace=self.trace_head(motion).float(), motion_tokens=motion, motion_latents=motion)


class EightSlotPlanner(MotionPlanner):
    with_boundaries = False
    joint_slots = False

    def offsets(self, config):
        return torch.tensor(config.motion_offsets_frames)

    def build_reader(self, config):
        self.register_buffer('planner_time', sine_position(self.offsets(config), config.hidden_dim), persistent=False)
        self.layers = nn.ModuleList([PlannerLayer(config, config.vlm_width), PlannerLayer(config, config.hidden_dim)])

    def _plan(self, hidden, memory, mask, prior, dispatch, remove_experts, end_latent, diagnostics):
        if end_latent is not None:
            raise ValueError('End Latent Swap requires an event-boundary planner')
        first, second = self.layers
        readout = first.cross(first.cross_norm(hidden), memory, mask, prior, self.config.planner_prior_ratio)
        x = first.temporal_step(readout + self.planner_time)
        if not remove_experts:
            x = first.expert_step(x, dispatch)
        x = second.temporal_step(x)
        x = x + second.cross(second.cross_norm(x), memory, mask)
        if not remove_experts:
            x = second.expert_step(x, dispatch)
        return dict(trace=self.trace_head(x).float(), motion_tokens=x, first_readout=readout)
