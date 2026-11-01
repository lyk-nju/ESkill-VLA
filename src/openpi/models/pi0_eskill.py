"""Current event -> future skill sequence -> time-aligned joint actions."""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.grounding import Grounding, VisualSupervision
from openpi.models.model import masked_mean
from openpi.models.motion import Motion
from openpi.models.pi0 import Pi0
from openpi.models.skill import Skill


class Pi0ESkill(Pi0):
    def __init__(self, config, *, vae=None):
        super().__init__(config)
        d = config.hidden_dim
        self.context_projection = nn.Linear(config.vlm_width, d)
        self.arm_embedding = nn.Embedding(2, d)
        self.type_embedding = nn.Embedding(5, d)
        self.state_projection = nn.Linear(config.arm_state_dim, d)
        self.skill = Skill(config)
        self.grounding = Grounding(config)
        self.motion = Motion(config)
        self.step_projection = nn.Linear(2 * d, config.expert_width)
        self.auxiliary_kv = nn.ModuleList(
            nn.Linear(d, 2 * config.kv_heads * config.head_dim) for _ in range(config.depth))
        self.visual_supervision = VisualSupervision(config, vae) if vae is not None else None

    def predict_condition(self, inputs):
        prefix = self.encode_observation(inputs)
        context = self.context_projection(prefix.context)
        arms, presence = self.arm_embedding.weight, inputs["arm_presence"]
        b, _, d = context.shape
        logits, skill = self.skill.current(context, prefix.mask, arms)
        current = self.grounding.current(context, prefix.mask, arms)
        grounding = self.grounding.goal(context, prefix.mask, arms, current, skill)
        body = self.state_projection(inputs["arm_state"]) + arms
        role_mask = presence[:, :, None, None].expand(grounding.shape[:-1]).flatten(1)
        motion_context = torch.cat((context, grounding.flatten(1, 3), body), 1)
        motion_mask = torch.cat((prefix.visual_mask, role_mask, presence), -1)
        latent, trace = self.motion(motion_context, motion_mask, skill, arms, presence)

        # Explicit arm/type tags distinguish current Event information from per-step skills.
        auxiliary = torch.cat((
            skill[:, :, None] + self.type_embedding.weight[0],
            (grounding + self.type_embedding.weight[1:4][None, None, :, None]).flatten(2, 3),
            latent + self.type_embedding.weight[4]), 2) + arms[None, :, None]
        auxiliary_mask = presence[:, :, None].expand(auxiliary.shape[:-1]).flatten(1)
        auxiliary = auxiliary.flatten(1, 2)
        future_context = torch.cat((context, auxiliary, body), 1)
        future_mask = torch.cat((prefix.mask, auxiliary_mask, presence), -1)
        future_logits, future_vectors = self.skill.future(future_context, future_mask, arms, presence)
        step_skill = self.step_projection(future_vectors.transpose(1, 2).flatten(2))
        condition = self.action_condition(prefix, auxiliary, auxiliary_mask, step_skill)
        predictions = dict(skill_logits=logits, future_skill_logits=future_logits,
                           grounding=grounding, motion=latent, trace=trace)
        return condition, predictions

    @staticmethod
    def classification_loss(logits, labels, valid):
        error = F.cross_entropy(logits.float().flatten(0, -2), labels.clamp_min(0).flatten(), reduction="none")
        return masked_mean(error.reshape_as(labels), valid)

    def forward(self, inputs, targets, *, noise=None, time=None):
        condition, prediction = self.predict_condition(inputs)
        presence = inputs["arm_presence"]
        if self.visual_supervision is None:
            raise RuntimeError("Training needs a frozen VAE; inference does not.")
        losses = dict(
            action=self.flow_loss(targets["actions"], targets["action_valid"], condition, noise=noise, time=time),
            skill=self.classification_loss(prediction["skill_logits"], targets["skill_labels"],
                                           targets["skill_valid"] & presence),
            future=self.classification_loss(prediction["future_skill_logits"], targets["future_skill_labels"],
                                            targets["future_skill_valid"] & presence[:, :, None]),
        )
        visual_targets = {**targets, "role_valid": targets["role_valid"] & presence[:, :, None]}
        losses["grounding"] = self.visual_supervision(prediction["grounding"], visual_targets, self.arm_embedding.weight)
        trace_error = F.smooth_l1_loss(
            prediction["trace"].float() / self.config.trace_scale,
            targets["trace"] / self.config.trace_scale, reduction="none", beta=self.config.trace_beta).mean(-1)
        losses["trace"] = masked_mean(trace_error, targets["trace_valid"] & presence[:, :, None])
        return losses

    @torch.no_grad()
    def sample_actions(self, inputs, *, noise=None, steps=10):
        condition, prediction = self.predict_condition(inputs)
        return self.sample_from_condition(condition, noise=noise, steps=steps), prediction
