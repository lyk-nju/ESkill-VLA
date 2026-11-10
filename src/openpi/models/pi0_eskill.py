"""Continuous Skill -> object-conditioned Motion -> Action."""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.grounding import Grounding, VisualSupervision
from openpi.models.gemma import rotary
from openpi.models.model import ActionCondition, masked_mean
from openpi.models.motion import Motion
from openpi.models.pi0 import Pi0
from openpi.models.skill import Skill


class Pi0ESkill(Pi0):
    """Skill -> object-conditioned local Motion -> Action; only Motion adds Action K/V."""
    def __init__(self, config, *, vae=None):
        super().__init__(config)
        d = config.hidden_dim
        self.context_projection = nn.Linear(config.vlm_width, d)
        self.arm_embedding = nn.Embedding(2, d)
        self.state_projection = nn.Linear(config.arm_state_dim, d)
        self.start_projection = nn.Linear(3, d)
        self.skill = Skill(config)
        self.grounding = Grounding(config)
        self.motion = Motion(config)
        self.auxiliary_kv = nn.ModuleList(
            nn.Linear(d, 2 * config.kv_heads * config.head_dim) for _ in range(config.depth))
        self.visual_supervision = VisualSupervision(config, vae) if vae is not None else None
        self.diagnostic_statistics = {}
        self.loss_counts = {}
        self.training_stage = 'joint'

    def configure_stage(self, stage):
        self.training_stage = stage
        if stage == 'joint':
            return
        self.requires_grad_(False)
        for name in ('context_projection', 'arm_embedding', 'state_projection', 'start_projection',
                     'skill', 'grounding', 'motion'):
            getattr(self, name).requires_grad_(True)
        self.visual_supervision.head.requires_grad_(True)
        if stage == 'stage2':
            self.paligemma_with_expert.gemma_expert.requires_grad_(True)
            for name in ('action_in_proj', 'action_out_proj', 'time_mlp_in', 'time_mlp_out', 'auxiliary_kv'):
                getattr(self, name).requires_grad_(True)

    @staticmethod
    def classification_loss(logits, labels, valid):
        error = F.cross_entropy(logits.float().flatten(0, -2), labels.clamp_min(0).flatten(), reduction="none")
        return masked_mean(error.reshape_as(labels), valid)

    def predict_condition(self, inputs, *, with_action=True):
        prefix = self.encode_observation(inputs)
        context = self.context_projection(prefix.context)
        arms, presence = self.arm_embedding.weight, inputs['arm_presence']
        logits, prior, skill = self.skill(prefix.context, prefix.mask, presence)
        grounding = self.grounding(context, prefix.mask, arms)
        # A mirrored view keeps its 2D start but receives no physical body-state feature.
        state_mask = inputs['arm_state_mask'] & presence
        state = torch.where(state_mask[..., None], inputs['arm_state'], 0)
        state_feature = torch.where(state_mask[..., None], self.state_projection(state), 0)
        start_valid = inputs['effector_start_valid'] & presence
        start = torch.where(start_valid[..., None], inputs['effector_start'], 0)
        start_condition = torch.cat((start, start_valid[..., None].to(start.dtype)), -1)
        body = state_feature + self.start_projection(start_condition) + arms
        role_mask = presence[:, :, None, None].expand(grounding.shape[:-1]).flatten(1)
        memory = torch.cat((context, grounding.flatten(1, 3), body), 1)
        memory_mask = torch.cat((prefix.mask, role_mask, presence), -1)
        latent = self.motion(memory, memory_mask, skill, arms, presence)
        condition = self.action_condition(prefix, latent, presence, inputs) if with_action else None
        prediction = dict(skill_logits=logits, skill_ids=self.skill.active_skill_ids[logits.argmax(-1)],
                          skill_prior=prior, grounding=grounding, motion=latent, effector_start=start)
        return condition, prediction

    def action_condition(self, prefix, motion, presence, inputs):
        auxiliary = motion.flatten(1, 2)
        auxiliary_mask = presence[:, :, None].expand(motion.shape[:-1]).flatten(1)
        position_mask = torch.cat((prefix.mask, auxiliary_mask), -1)
        positions = (position_mask.long().cumsum(-1) - 1).clamp_min(0)[:, prefix.mask.shape[1]:]
        visible = self.action_visibility(prefix, inputs)
        keys, values = [], []
        for i, project in enumerate(self.auxiliary_kv):
            kv = project(auxiliary).reshape(*auxiliary.shape[:2], 2, self.config.kv_heads, self.config.head_dim)
            key = rotary(kv[:, :, 0].transpose(1, 2), positions)
            keys.append(torch.cat((prefix.keys[i], key.to(prefix.keys[i].dtype)), 2))
            values.append(torch.cat((prefix.values[i], kv[:, :, 1].transpose(1, 2).to(prefix.values[i].dtype)), 2))
        return ActionCondition(keys, values, torch.cat((visible, auxiliary_mask), -1), position_mask)

    def action_visibility(self, prefix, inputs):
        """One visibility draw per condition, reused by every layer and FM iteration."""
        visible = prefix.mask.clone()
        if not self.training:
            return visible
        views = inputs['images'].shape[1]
        probabilities = self.config.action_patch_mask_prob
        if len(probabilities) == 1:
            probabilities = probabilities * views
        if len(probabilities) < views:
            raise ValueError('Provide an Action patch mask probability for every camera')
        if not any(probabilities[:views]):
            return visible
        patches = (prefix.mask.shape[1] - inputs['tokens'].shape[1]) // views
        keep_full = torch.rand(len(visible), device=visible.device) < self.config.action_unmasked_prob
        for view, probability in enumerate(probabilities[:views]):
            keep = torch.rand(len(visible), patches, device=visible.device) >= probability
            visible[:, view * patches:(view + 1) * patches] &= keep | keep_full[:, None]
        return visible

    def read_motion(self, prediction):
        trace = prediction['effector_start'][:, :, None] + self.motion.trace_head(prediction['motion']).float()
        gripper = self.motion.gripper_head(prediction['motion']).squeeze(-1).float()
        return trace, gripper

    def forward(self, inputs, targets, *, noise=None, time=None):
        condition, prediction = self.predict_condition(inputs, with_action=self.training_stage != 'stage1')
        if self.visual_supervision is None:
            raise RuntimeError('ESkill training needs a frozen VAE; inference does not.')
        presence = inputs['arm_presence']
        labels, supported = self.skill.labels_to_active(targets['skill_labels'])
        skill_valid = targets['skill_valid'] & presence & supported
        role_valid = targets['role_valid'] & presence[:, :, None]
        visual = self.visual_supervision(prediction['grounding'], dict(targets, role_valid=role_valid))
        trace, gripper = self.read_motion(prediction)
        padded = targets.get('terminal_padded_mask', targets.get('future_padded', torch.zeros_like(targets['trace_valid'])))
        trace_valid = (targets['trace_valid'] & presence[:, :, None]
                       & inputs['effector_start_valid'][:, :, None] & ~padded)
        gripper_valid = targets['gripper_valid'] & presence[:, :, None] & ~padded
        xy_target = torch.where(trace_valid[..., None], targets['trace'].float(), 0)
        grip_target = torch.where(gripper_valid, targets['gripper'].float(), 0)
        xy_error = F.smooth_l1_loss(trace, xy_target, reduction='none', beta=self.config.trace_beta).mean(-1)
        grip_error = F.binary_cross_entropy_with_logits(gripper, grip_target, reduction='none')
        losses = dict(skill=Pi0ESkill.classification_loss(prediction['skill_logits'], labels, skill_valid),
                      **visual, trace=masked_mean(xy_error, trace_valid), gripper=masked_mean(grip_error, gripper_valid))
        self.loss_counts = dict(
            skill=skill_valid.sum(), current=role_valid[..., 0].sum(),
            future_object=role_valid[..., 1].sum(), trace=trace_valid.sum(), gripper=gripper_valid.sum())
        if self.training_stage != 'stage1':
            losses['action'] = self.flow_loss(targets['actions'], targets['action_valid'], condition, noise=noise, time=time)
            self.loss_counts['action'] = (targets['action_valid'].sum() if self.config.mask_action_loss
                                         else targets['actions'].new_tensor(targets['actions'].numel()))
        self.diagnostic_statistics = self.statistics(prediction, trace, gripper, inputs, targets,
                                                      skill_valid, trace_valid, gripper_valid, grip_error)
        self.diagnostic_statistics.update(self.visual_supervision.diagnostic_statistics)
        return losses

    @torch.no_grad()
    def statistics(self, prediction, trace, gripper, inputs, targets, skill_valid,
                   trace_valid, gripper_valid, grip_error):
        from openpi.models.diagnostics import mean_statistics
        presence = inputs['arm_presence']
        predicted = prediction['skill_ids']
        distance = torch.linalg.vector_norm(trace - torch.where(trace_valid[..., None], targets['trace'], 0), dim=-1)
        grip_correct = (gripper >= 0) == (targets['gripper'] >= .5)
        transition = torch.zeros_like(gripper_valid)
        transition[..., 1:] = (gripper_valid[..., 1:] & gripper_valid[..., :-1]
                               & (targets['gripper'][..., 1:] != targets['gripper'][..., :-1]))
        transition = targets.get('gripper_transition', transition) & gripper_valid
        result = dict(current_skill_accuracy=mean_statistics(predicted == targets['skill_labels'], skill_valid),
                      motion_l2_px=mean_statistics(distance * self.config.image_size, trace_valid),
                      gripper_accuracy=mean_statistics(grip_correct, gripper_valid),
                      gripper_transition_accuracy=mean_statistics(grip_correct, transition),
                      gripper_bce=mean_statistics(grip_error, gripper_valid))
        for skill_id in self.config.active_skill_ids:
            name = self.config.skill_names[skill_id].lower()
            result[f'current_skill_pred_{name}'] = mean_statistics((predicted == skill_id).float(), presence)
            result[f'current_skill_label_{name}'] = mean_statistics(
                (targets['skill_labels'] == skill_id).float(), skill_valid)
            selected = skill_valid & (targets['skill_labels'] == skill_id)
            result[f'skill_{name}_accuracy'] = mean_statistics(predicted == skill_id, selected)
            result[f'skill_{name}_motion_l2_px'] = mean_statistics(
                distance * self.config.image_size, trace_valid & selected[..., None])
            result[f'skill_{name}_gripper_accuracy'] = mean_statistics(
                grip_correct, gripper_valid & selected[..., None])
        for i, frames in enumerate(self.config.motion_offsets_frames):
            result[f'motion_offset_{frames}f_l2_px'] = mean_statistics(
                distance[:, :, i] * self.config.image_size, trace_valid[:, :, i])
            result[f'gripper_offset_{frames}f_accuracy'] = mean_statistics(
                grip_correct[:, :, i], gripper_valid[:, :, i])
            result[f'motion_offset_{frames}f_coverage'] = mean_statistics(trace_valid[:, :, i].float(), presence)
            result[f'gripper_offset_{frames}f_coverage'] = mean_statistics(gripper_valid[:, :, i].float(), presence)
        return result
