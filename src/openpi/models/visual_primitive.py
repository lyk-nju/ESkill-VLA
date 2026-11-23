"""Single Current from native language attention and independent gripper localization."""
from math import isqrt, sqrt

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.diagnostics import HIT_RADIUS_PX, mean_statistics, spatial_centers
from openpi.models.grounding_bbox import read_regions
from openpi.models.model import masked_mean
from openpi.models.skill import QueryDecoder


def patch_coverage(mask, side):
    """Fractional area of a pixel mask on the RGB-aligned patch grid."""
    return F.adaptive_avg_pool2d(mask[:, None].float(), (side, side)).flatten(1)


def box_coverage(boxes, side):
    """Exact fraction of each square patch covered by a normalized xyxy rectangle."""
    left = torch.arange(side, dtype=torch.float32, device=boxes.device) / side
    right = left + 1 / side
    lo, hi = boxes.float().unsqueeze(-1).split(2, -2)
    overlaps = (torch.minimum(hi, right) - torch.maximum(lo, left)).clamp_min(0) * side
    x, y = overlaps.unbind(-2)
    return (y.unsqueeze(-1) * x.unsqueeze(-2)).flatten(-2)


def spatial_kl(maps, coverage):
    """One KL per map and sample. An empty target gets zero loss and zero count."""
    mass = coverage.sum(-1, keepdim=True)
    target = coverage / mass.clamp_min(1e-20)
    target = target.unsqueeze(1).expand_as(maps)
    return F.kl_div(maps.clamp_min(1e-30).log(), target, reduction='none').sum(-1), mass[:, 0] > 0


class VisualPrimitive(nn.Module):
    """Shared text aggregation for semantic contrast and multilayer Current maps."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, d = config.vlm_width, config.hidden_dim
        self.gac_query = nn.Linear(width, d)
        self.gac_key = nn.Linear(width, d)
        self.subject_queries = nn.Parameter(torch.randn(config.role_queries, d) * .02)
        self.subject_projection = nn.Linear(width, d)
        self.subject_decoder = QueryDecoder(config)
        self.subject_spatial = nn.Linear(d, d)
        if config.learned_alpha:
            self.alpha_scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
            nn.init.zeros_(self.alpha_scorer[-1].weight)
            nn.init.zeros_(self.alpha_scorer[-1].bias)

    def forward(self, prefix, text_mask):
        visual = prefix.visual_tokens
        patches = visual.size(1)
        text = prefix.context[:, patches:]
        attention = prefix.text_image_attention
        with torch.autocast(text.device.type, enabled=False):
            scores = (self.alpha_scorer(text.float()).squeeze(-1) if self.config.learned_alpha
                      else torch.zeros_like(text_mask, dtype=torch.float32))
            alpha = scores.masked_fill(~text_mask, -torch.inf).softmax(-1)
            query = (alpha.unsqueeze(-1) * text.float()).sum(1)
            mass = attention.sum(-1)
            # Normalize within each token/head first. Low visual mass must not reweight alpha.
            conditional = attention / mass.clamp_min(1e-30).unsqueeze(-1)
            layers = (conditional.mean(2) * alpha[:, None, :, None]).sum(2)
        # Subject sees one image-only memory; new projections and queries train from scratch.
        memory = self.subject_projection(visual.detach())
        queries = self.subject_queries.unsqueeze(0).expand(len(memory), -1, -1)
        subject = self.subject_decoder(queries, memory, prefix.visual_mask[:, :patches])
        spatial_query = self.subject_spatial(subject.mean(1))
        with torch.autocast(text.device.type, enabled=False):
            scores = (memory.float() * spatial_query.float().unsqueeze(1)).sum(-1) / sqrt(memory.size(-1))
            subject_map = scores.softmax(-1)
        return dict(current_map=layers.mean(1), subject_map=subject_map, visual_tokens=visual,
                    vlm_context=prefix.context, text_hidden=text, text_mask=text_mask, native_attention=attention,
                    alpha=alpha, current_query=query, visual_attention_mass=mass, layer_maps=layers)

    def contrast(self, prediction, targets):
        tokens, _ = read_regions(prediction['visual_tokens'].detach(), targets['entity_boxes'])
        key = F.normalize(self.gac_key(tokens.mean(2)).float(), dim=-1)
        query = F.normalize(self.gac_query(prediction['current_query']).float(), dim=-1)
        similarity = (query[:, None] * key).sum(-1)
        eligible = targets['entity_valid']
        positive = targets['positive_entity']
        index = positive.clamp_min(0)[:, None]
        valid = (positive >= 0) & eligible.gather(1, index)[:, 0] & (eligible.sum(1) > 1)
        logits = (similarity / self.config.gac_temperature).masked_fill(~eligible, -1e9)
        ce = F.cross_entropy(logits, index[:, 0], reduction='none')
        negative = eligible.scatter(1, index, False)
        margin = similarity.gather(1, index)[:, 0] - similarity.masked_fill(~negative, -2.).max(1).values
        metrics = dict(grounding_gac_recall=mean_statistics((logits.argmax(1) == positive).float(), valid),
                       grounding_gac_margin=mean_statistics(margin, valid))
        return masked_mean(ce, valid), valid.sum(), metrics

    def losses(self, prediction, targets):
        maps = prediction['layer_maps']
        # The collator pads empty candidate lists. Direct callers can supply a zero-width batch too.
        if targets['entity_boxes'].size(1) == 0:
            targets = dict(targets, entity_boxes=maps.new_zeros(len(maps), 1, 4),
                           entity_valid=torch.zeros(len(maps), 1, device=maps.device, dtype=torch.bool),
                           positive_entity=torch.full_like(targets['positive_entity'], -1))
        side = isqrt(maps.size(-1))
        coverage = patch_coverage(targets['current_mask'], side)
        gripper = patch_coverage(targets['subject_mask'], side)
        use_allocation = self.config.current_alignment_loss == 'allocation'
        kl, valid = spatial_kl(maps.detach() if use_allocation else maps, coverage)
        subject_kl, subject_valid = spatial_kl(prediction['subject_map'][:, None], gripper)
        gac, gac_count, metrics = self.contrast(prediction, targets)
        align, align_valid = kl, valid
        if use_allocation:
            # A³-VLA §3.3, using the same visual-only layer maps as KL.
            region = coverage > self.config.allocation_min_patch_overlap
            align_valid = region.any(-1)
            roi_mass = (maps * region[:, None]).sum(-1)
            align = (self.config.allocation_min_mass - roi_mass).relu().square()
            metrics['grounding_allocation_roi_mass'] = mean_statistics(roi_mass.mean(1), align_valid)
            metrics['grounding_allocation_satisfied'] = mean_statistics((align == 0).float().mean(1), align_valid)
        losses = dict(gac=gac, current_align=masked_mean(align.mean(1), align_valid),
                      subject=masked_mean(subject_kl[:, 0], subject_valid))
        counts = dict(gac=gac_count, current_align=align_valid.sum(), subject=subject_valid.sum())
        with torch.no_grad():
            self.map_statistics(metrics, prediction, targets, coverage, gripper, kl, valid, subject_valid)
        return losses, counts, metrics

    def map_statistics(self, metrics, pred, targets, coverage, gripper, kl, valid, subject_valid):
        side = isqrt(coverage.size(-1))
        maps = torch.cat((pred['layer_maps'], pred['current_map'].unsqueeze(1)), 1)
        boxes = targets['entity_boxes']
        index = torch.arange(boxes.size(1), device=boxes.device)
        negative = targets['entity_valid'] & (index != targets['positive_entity'][:, None])
        same_box = torch.isclose(boxes, targets['current_box'][:, None]).all(-1)
        negative = negative & ~same_box
        with torch.autocast(maps.device.type, enabled=False):
            other = maps @ box_coverage(boxes, side).transpose(1, 2)
        strongest = other.masked_fill(~negative[:, None], 0.).amax(-1)
        gt_mass = (maps * coverage[:, None]).sum(-1)
        alpha = pred['alpha']
        visual_mass = (pred['visual_attention_mass'].mean(2) * alpha[:, None]).sum(-1)
        for i, layer in enumerate(self.config.attention_layers):
            root = f'grounding_layer_{layer}_'
            metrics[root + 'kl'] = mean_statistics(kl[:, i], valid)
            metrics[root + 'gt_mass'] = mean_statistics(gt_mass[:, i], valid)
            metrics[root + 'distractor_mass'] = mean_statistics(strongest[:, i], valid & negative.any(1))
            metrics[root + 'visual_mass'] = mean_statistics(visual_mass[:, i], pred['text_mask'].any(1))
        metrics['grounding_current_gt_mass'] = mean_statistics(gt_mass[:, -1], valid)
        final_kl, _ = spatial_kl(pred['current_map'][:, None], coverage)
        metrics['grounding_current_map_kl'] = mean_statistics(final_kl[:, 0], valid)
        metrics['grounding_current_distractor_mass'] = mean_statistics(strongest[:, -1], valid & negative.any(1))
        metrics['grounding_subject_gt_mass'] = mean_statistics((pred['subject_map'] * gripper).sum(-1), subject_valid)
        metrics['grounding_current_peak'] = mean_statistics(pred['current_map'].amax(-1), valid)
        for role, visible in (('current', valid), ('subject', subject_valid)):
            box = targets[f'{role}_box']
            lo, hi = box.split(2, dim=-1)
            center = (lo + hi) * .5
            center_valid = visible & (hi > lo).all(-1)
            spatial = pred[f'{role}_map']
            has_peak = spatial.amax(-1) > 0
            for readout, point in zip(('peak3', 'argmax'), spatial_centers(spatial), strict=True):
                error = ((point - center) * targets['center_px_scale']).norm(dim=-1)
                key = f'grounding_{role}_{readout}'
                metrics[f'{key}_error_px'] = mean_statistics(error, center_valid & has_peak)
                metrics[f'{key}_hit10'] = mean_statistics((error <= HIT_RADIUS_PX) & has_peak, center_valid)
        entropy = -(alpha * alpha.clamp_min(1e-30).log()).sum(-1)
        metrics['grounding_alpha_entropy'] = mean_statistics(entropy, pred['text_mask'].any(1))
