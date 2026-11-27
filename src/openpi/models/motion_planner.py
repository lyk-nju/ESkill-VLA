"""Event-routed 2D motion planning from a native Current prior and fused visual patches."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.diagnostics import HIT_RADIUS_PX, mean_statistics, spatial_centers
from openpi.models.model import masked_mean
from openpi.models.pi0 import sine_position
from openpi.models.visual_primitive import box_coverage, patch_coverage, spatial_kl


def rms(x):
    return x.float().square().mean().sqrt()


def normalize_prior(prior, valid):
    """Convert a KLAL map into a masked FP32 probability distribution for the planner."""
    prior = prior.float().masked_fill(~valid, 0)
    if not valid.any(-1).all() or not torch.isfinite(prior).all() or (prior < 0).any():
        raise ValueError('Planner priors need finite nonnegative mass and at least one valid visual patch')
    mass = prior.sum(-1, keepdim=True)
    uniform = valid.float() / valid.sum(-1, keepdim=True)
    return torch.where(mass > 1e-8, prior / mass.clamp_min(1e-8), uniform)


def query_attention_statistics(q, k, valid, patches, prefix_length, hidden):
    """First-layer native Motion rows for initialization diagnostics only."""
    with torch.autocast(q.device.type, enabled=False):
        keys = k.repeat_interleave(q.size(1) // k.size(1), dim=1)
        scores = (q[:, :, prefix_length:].float() @ keys.float().transpose(-1, -2)) / math.sqrt(q.size(-1))
        p = scores.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
    return dict(first_layer_motion_rms=float(rms(hidden)),
                first_layer_image_mass=float(p[..., :patches].sum(-1).mean()),
                first_layer_text_mass=float(p[..., patches:prefix_length].sum(-1).mean()),
                first_layer_motion_mass=float(p[..., prefix_length:].sum(-1).mean()),
                first_layer_attention_entropy=float(-(p * p.clamp_min(1e-30).log()).sum(-1).mean()))


class CrossAttention(nn.Module):
    def __init__(self, query_dim, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(query_dim, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def heads_of(self, x):
        return x.unflatten(-1, (self.heads, -1)).transpose(1, 2)

    def forward(self, query, memory, valid, prior=None, ratio=0.):
        q, k, v = (self.heads_of(project(x)) for project, x in (
            (self.q, query), (self.k, memory), (self.v, memory)))
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :], dropout_p=0.)
        if ratio:
            with torch.autocast(v.device.type, enabled=False):
                guided = torch.einsum('bp,bhpd->bhd', prior, v.float())
            output = (1 - ratio) * output + ratio * guided.unsqueeze(2).to(output.dtype)
        return self.out(output.transpose(1, 2).flatten(2))


class Expert(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate = nn.Linear(width, 4 * width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PlannerLayer(nn.Module):
    def __init__(self, config, query_dim):
        super().__init__()
        width, heads = config.hidden_dim, config.auxiliary_heads
        self.cross_norm = nn.LayerNorm(query_dim)
        self.cross = CrossAttention(query_dim, width, heads)
        self.temporal_norm = nn.LayerNorm(width)
        self.temporal = nn.MultiheadAttention(width, heads, batch_first=True)
        self.expert_norm = nn.LayerNorm(width)
        self.experts = nn.ModuleList(Expert(width) for _ in config.active_skill_ids)

    def temporal_step(self, x):
        norm = self.temporal_norm(x)
        return x + self.temporal(norm, norm, norm, need_weights=False)[0]

    def expert_step(self, x, route):
        norm = self.expert_norm(x)
        output = x
        for index, expert in enumerate(self.experts):
            selected = (route == index).nonzero(as_tuple=True)[0]
            if selected.numel():
                update = expert(norm.index_select(0, selected)).to(x.dtype)
                output = output.index_add(0, selected, update)
        return output


class MotionPlanner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        w, d = config.vlm_width, config.hidden_dim
        offsets = torch.tensor(config.motion_offsets_frames)
        self.with_boundaries = config.planner_has_boundaries
        self.joint_slots = config.planner_version == 'v2_joint'
        if self.with_boundaries:
            # [Start, End, +5, ..., +40]. End has a native sequence position but no time PE.
            offsets = torch.cat((offsets.new_zeros(2), offsets))
        self.native_queries = nn.Parameter(torch.randn(len(offsets), w) * .02)
        self.register_buffer('native_time', sine_position(offsets, w), persistent=False)
        self.register_buffer('query_time_scale', torch.tensor(.02))
        if self.with_boundaries:
            self.native_time[1].zero_()
        else:
            self.register_buffer('planner_time', sine_position(offsets, d), persistent=False)
        side = config.image_size // config.patch_size
        # Wavelengths from two patches to twice the image extent, using patch centers.
        axis = sine_position(torch.arange(side) + .5, d // 2, min_period=2., max_period=2. * side)
        y, x = torch.meshgrid(torch.arange(side), torch.arange(side), indexing='ij')
        self.register_buffer('xy', torch.cat((axis[x], axis[y]), dim=-1).reshape(side * side, d),
                             persistent=False)
        self.project = nn.Linear(w, d)
        self.memory_norm = nn.LayerNorm(d)
        self.router = nn.Sequential(nn.LayerNorm(w), nn.Linear(w, len(config.active_skill_ids)))
        if self.with_boundaries:
            self.motion_project = nn.Linear(w, d)
            self.read_norm = nn.LayerNorm(d)
            self.grounded_cross = CrossAttention(d, d, config.auxiliary_heads)
            self.layers = nn.ModuleList([PlannerLayer(config, d)])
        else:
            self.layers = nn.ModuleList([PlannerLayer(config, w), PlannerLayer(config, d)])
        self.trace_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))

    def query_inputs(self):
        return self.native_queries + self.query_time_scale * self.native_time

    @torch.no_grad()
    def set_query_rms(self, target):
        # Equal RMS contributions from the seeded random vectors and fixed time features.
        fixed = self.native_time.any(dim=-1)
        self.query_time_scale.copy_(rms(self.native_queries[fixed]) / rms(self.native_time[fixed]))
        factor = target / rms(self.query_inputs()[fixed])
        self.native_queries.mul_(factor)
        self.query_time_scale.mul_(factor)
        if self.with_boundaries:
            end = self.native_queries[1]
            end.mul_(rms(self.query_inputs()[fixed]) / rms(end))

    def forward(self, features, *, current_prior=None, skill_id=None, remove_experts=False, active=None,
                end_latent=None, diagnostics=None):
        mask = features['visual_mask']
        prior = normalize_prior(features['current_map'] if current_prior is None else current_prior, mask)
        memory = self.memory_norm(self.project(features['visual_hidden']) + self.xy)
        route = features['skill_logits'].argmax(-1) if skill_id is None else skill_id
        selected = route if active is None else route.masked_fill(~active, -1)
        if self.with_boundaries:
            result = self._boundary_plan(features['motion_hidden'], memory, mask, prior, selected,
                                         remove_experts, end_latent, diagnostics)
            return dict(**result, skill_id=route)
        if end_latent is not None:
            raise ValueError('End Latent Swap requires an event-boundary planner')
        first, second = self.layers
        readout = first.cross(first.cross_norm(features['motion_hidden']), memory, mask,
                              prior, self.config.planner_prior_ratio)
        x = first.temporal_step(readout + self.planner_time)
        if not remove_experts:
            x = first.expert_step(x, selected)
        x = second.temporal_step(x)
        x = x + second.cross(second.cross_norm(x), memory, mask)
        if not remove_experts:
            x = second.expert_step(x, selected)
        return dict(trace=self.trace_head(x).float(), motion_tokens=x, first_readout=readout, skill_id=route)

    def _boundary_plan(self, hidden, memory, mask, prior, dispatch, remove_experts, end_latent, diagnostics):
        residual = self.motion_project(hidden)
        guided = self.grounded_cross(self.read_norm(residual), memory, mask, prior, self.config.planner_prior_ratio)
        grounded = residual + guided
        boundaries, times = grounded[:, :2], grounded[:, 2:]
        if end_latent is not None:
            if end_latent.shape != boundaries[:, 1:2].shape:
                raise ValueError('End intervention expects one pre-norm [batch, 1, hidden_dim] latent')
            boundaries = torch.cat((boundaries[:, :1], end_latent), dim=1)
        layer = self.layers[0]
        readout = torch.cat((boundaries, times), dim=1)
        if self.joint_slots:
            motion = layer.temporal_step(readout)
        else:
            kv = layer.temporal_norm(readout)
            motion = times + layer.temporal(kv[:, 2:], kv, kv, need_weights=False)[0]
        motion = motion + layer.cross(layer.cross_norm(motion), memory, mask)
        if not remove_experts:
            motion = layer.expert_step(motion, dispatch)
        if diagnostics is not None:
            self._boundary_statistics(diagnostics, residual, guided, memory, prior, readout)
        if self.joint_slots:
            xy = self.trace_head(motion).float()
            return dict(start=xy[:, 0], end=xy[:, 1], trace=xy[:, 2:], first_readout=readout,
                        event_motion_latents=motion, boundary_latents=motion[:, :2],
                        motion_tokens=motion[:, 2:], motion_latents=motion[:, 2:])
        xy = self.trace_head(boundaries).float()
        return dict(start=xy[:, 0], end=xy[:, 1], boundary_latents=boundaries, first_readout=readout,
                    trace=self.trace_head(motion).float(), motion_tokens=motion, motion_latents=motion)

    @torch.no_grad()
    def _boundary_statistics(self, stats, residual, guided, memory, prior, readout):
        # Isolate the Current contribution after W_O, excluding the shared output bias.
        cross = self.grounded_cross
        with torch.autocast(memory.device.type, enabled=False):
            values = F.linear(memory.float(), cross.v.weight.float(), cross.v.bias.float())
            pooled = torch.einsum('bp,bpd->bd', prior, values)
            current = self.config.planner_prior_ratio * F.linear(pooled, cross.out.weight.float())
        valid = torch.ones(len(memory), dtype=torch.bool, device=memory.device)
        for role, slots in (('start', slice(0, 1)), ('end', slice(1, 2)), ('future', slice(2, None))):
            denom = residual[:, slots].float().square().mean((1, 2)).sqrt().clamp_min(1e-8)
            g = guided[:, slots].float().square().mean((1, 2)).sqrt()
            stats[f'planner_{role}_readout_to_residual_rms'] = mean_statistics(g / denom, valid)
            stats[f'planner_{role}_prior_to_residual_rms'] = mean_statistics(current.square().mean(1).sqrt() / denom, valid)
        layer = self.layers[0]
        kv = layer.temporal_norm(readout)
        queries = kv if self.joint_slots else kv[:, 2:]
        _, mass = layer.temporal(queries, kv, kv, need_weights=True)
        rows = [f'offset_{offset}' for offset in self.config.motion_offsets_frames]
        if self.joint_slots:
            rows = ['start', 'end', *rows]
        for j, row in enumerate(rows):
            for role, value in (('start', mass[:, j, 0]), ('end', mass[:, j, 1]), ('future', mass[:, j, 2:].sum(-1))):
                stats[f'planner_{row}_{role}_mass'] = mean_statistics(value.float(), valid)

    def supervision(self, features, targets, *, router_only=False, diagnostics=False):
        logits = features['skill_logits']
        labels, valid = targets['skill_labels'], targets['skill_valid']
        ce = F.cross_entropy(logits.float(), labels.masked_fill(~valid, 0), reduction='none')
        losses = dict(skill=masked_mean(ce, valid))
        counts = dict(skill=valid.sum())
        route = logits.argmax(-1)
        correct = route == labels
        stats = dict(current_skill_accuracy=mean_statistics(correct, valid))
        for k in self.config.active_skill_ids:
            name = self.config.skill_names[k].lower()
            stats[f'skill_{name}_accuracy'] = mean_statistics(correct, valid & (labels == k))
        if router_only:
            return losses, counts, stats

        xy_valid = targets['trace_valid']
        # Dispatch only samples with supervision downstream of the skill expert.
        active = xy_valid.any(-1)
        if self.joint_slots:
            active = active | targets['boundary_valid'].any(-1)
        for k in self.config.active_skill_ids:
            name = self.config.skill_names[k].lower()
            stats[f'planner_route_{name}'] = mean_statistics(route == k, active)
        plan = self(features, active=active, diagnostics=stats if diagnostics else None)
        trace = plan['trace']
        xy = targets['trace'].masked_fill(~xy_valid[..., None], 0)
        pred, label, valid_points = trace, xy, xy_valid
        if self.with_boundaries:
            boundary_valid = targets['boundary_valid']
            boundary_gt = targets['boundary_xy'].masked_fill(~boundary_valid[..., None], 0)
            boundary_pred = torch.stack((plan['start'], plan['end']), dim=1)
            pred = torch.cat((boundary_pred, trace), dim=1)
            label = torch.cat((boundary_gt, xy), dim=1)
            valid_points = torch.cat((boundary_valid, xy_valid), dim=1)
        error = F.smooth_l1_loss(pred, label, reduction='none', beta=self.config.trace_beta).mean(-1)
        maps = features['layer_maps']
        coverage = patch_coverage(targets['current_mask'], math.isqrt(maps.size(-1)))
        kl, visible = spatial_kl(maps, coverage)
        losses.update(current_align=masked_mean(kl.mean(1), visible), trace=masked_mean(error, valid_points))
        counts.update(current_align=visible.sum(), trace=valid_points.sum())
        with torch.no_grad():
            held = targets['trace_terminal_hold']
            scale = targets['center_px_scale']
            # Motion uses model canvas pixels; Current follows the historical source-pixel protocol.
            distance = (trace - xy).norm(dim=-1) * self.config.image_size
            if self.with_boundaries:
                boundary_distance = (boundary_pred - boundary_gt).norm(dim=-1) * self.config.image_size
                for j, role in enumerate(('start', 'end')):
                    stats[f'motion_{role}_l2_px'] = mean_statistics(boundary_distance[:, j], boundary_valid[:, j])
                for j, offset in enumerate(self.config.motion_offsets_frames):
                    stats[f'motion_offset_{offset}_l2_px'] = mean_statistics(distance[:, j], xy_valid[:, j] & ~held[:, j])
            point, _ = spatial_centers(features['current_map'])
            bounds = targets['current_box'].reshape(-1, 2, 2)
            center_error = ((point - bounds.mean(1)) * scale).norm(dim=-1)
            center_valid = visible & (bounds[:, 1] > bounds[:, 0]).all(1)
            hit = center_error <= HIT_RADIUS_PX
            stats.update(motion_l2_px=mean_statistics(distance, xy_valid),
                         motion_observed_l2_px=mean_statistics(distance, xy_valid & ~held),
                         motion_terminal_hold_l2_px=mean_statistics(distance, xy_valid & held),
                         grounding_current_peak3_hit10=mean_statistics(hit, center_valid),
                         grounding_current_peak3_error_px=mean_statistics(center_error, center_valid),
                         event_joint_accuracy=mean_statistics(correct & hit, valid & center_valid))
        return losses, counts, stats

    @torch.no_grad()
    def memory_statistics(self, features, targets=None):
        content = self.project(features['visual_hidden']).float()
        xy = self.xy.float().reshape(math.isqrt(len(self.xy)), -1, self.xy.shape[-1])
        stats = dict(projected_visual_rms=float(rms(content)), xy_rms=float(rms(xy)),
                     xy_neighbor_l2=float((xy[:, 1:] - xy[:, :-1]).norm(dim=-1).mean()),
                     xy_corner_l2=float((xy[-1, -1] - xy[0, 0]).norm()),
                     xy_spatial_variance=float(xy.flatten(0, 1).var(0, unbiased=False).mean()))
        if targets is not None and 'entity_boxes' in targets:
            valid = targets['entity_valid']
            boxes = targets['entity_boxes']
            # Compare actual first-layer Values of the annotated target and distinct scene entities.
            positive = targets['positive_entity']
            keep = (positive >= 0) & (valid.sum(-1) >= 2)
            if keep.any():
                idx = positive[keep]
                area = box_coverage(boxes[keep], xy.size(0))
                weights = area / area.sum(-1, keepdim=True).clamp_min(1e-20)
                cross = self.grounded_cross if self.with_boundaries else self.layers[0].cross
                values = cross.v(self.memory_norm(content + self.xy))[keep]
                pooled = weights.to(values.dtype) @ values
                rows = torch.arange(len(idx), device=idx.device)
                other = valid[keep] & (boxes[keep] != boxes[keep][rows, idx][:, None]).any(-1)
                other &= (area.sum(-1) > 0) & (area[rows, idx].sum(-1) > 0)[:, None]
                if other.any():
                    stats['target_other_pairs'] = int(other.sum())
                    stats['target_other_value_l2'] = float((pooled - pooled[rows, idx][:, None]).float()
                                                          .norm(dim=-1)[other].mean())
        return stats
