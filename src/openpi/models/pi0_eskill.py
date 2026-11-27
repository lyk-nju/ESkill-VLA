"""See and plan once per observation and reuse the condition throughout flow integration."""
import torch

from openpi.models.action_guidance import ActionGuidance
from openpi.models.planning import PlanningModel


class Pi0ESkill(PlanningModel):
    def __init__(self, config, *, vae=None):
        super().__init__(config)
        if config.action_execution:
            self.supervise_trajectory = True
            self.execution = ActionGuidance(config)
            self.configure_stage(config.current_grounding)

    def configure_stage(self, stage):
        super().configure_stage(stage)
        if self.config.action_execution:
            self.execution.requires_grad_(True)
            for module in (self.paligemma_with_expert.gemma_expert, self.action_in_proj, self.action_out_proj,
                           self.time_mlp_in, self.time_mlp_out):
                module.requires_grad_(self.config.train_action_expert)

    def predict_condition(self, inputs):
        if not self.config.action_execution:
            raise ValueError('Action inference needs an action_execution checkpoint')
        return self.condition_from_features(self.plan_features(inputs, cache_prefix=True))

    def condition_from_features(self, features, *, current_prior=None, skill_id=None, motion_plan=None):
        """Reuse an observation for paired Current, skill or final-plan interventions."""
        current = features['current_map'] if current_prior is None else current_prior
        if motion_plan is None:
            motion_plan = self.plan_from_features(features, current_prior=current, skill_id=skill_id)
        elif skill_id is not None:
            raise ValueError('A donated plan already contains its selected expert')
        condition = self.execution(features['prefix'], motion_plan, current)
        return condition, dict(**motion_plan, current_map=current, skill_logits=features['skill_logits'])

    def forward(self, inputs, targets, *, noise=None, time=None, router_only=False, planner_diagnostics=False,
                action_diagnostics=False):
        if not self.config.action_execution:
            return super().forward(inputs, targets, router_only=router_only, planner_diagnostics=planner_diagnostics)
        features = self.plan_features(inputs, cache_prefix=True)
        supervised = targets['trace_valid'].any(-1) | targets['boundary_valid'].any(-1)
        rows = targets['action_sample_valid']
        plan = self.planner(features, active=(supervised | rows) if self.supervise_trajectory else rows)
        losses, self.loss_counts, self.diagnostic_statistics = self.planner.supervision(features, targets, plan=plan)
        valid = targets['action_valid']
        elements = (valid if self.config.mask_action_loss else torch.ones_like(valid)) & rows[:, None, None]
        condition = self.execution(features['prefix'], plan, features['current_map'],
                                   statistics=self.diagnostic_statistics if action_diagnostics else None,
                                   valid_steps=elements.any(-1) if action_diagnostics else None)
        losses['action'] = self.flow_loss(targets['actions'], valid, condition, noise=noise, time=time, sample_valid=rows)
        self.loss_counts['action'] = elements.sum()
        return losses
