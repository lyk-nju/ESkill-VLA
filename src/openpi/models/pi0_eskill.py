"""Continuous Skill -> object-conditioned Motion -> Action."""
import torch
from torch import nn
from torch.nn import functional as F

from openpi.models.grounding import Grounding, TaskGrounding, VisualSupervision
from openpi.models.grounding_bbox import box_loss, corners, entity_contrast, read_regions
from openpi.models.gemma import rotary
from openpi.models.model import ActionCondition, masked_mean
from openpi.models.motion import Motion
from openpi.models.motion_planner import MotionPlanner, rms
from openpi.models.pi0 import Pi0
from openpi.models.skill import Skill
from openpi.models.lora import LoRALinear
from openpi.models.visual_primitive import CurrentAttention, VisualPrimitive


class Pi0ESkill(Pi0):
    """Skill -> object-conditioned local Motion -> Action; only Motion adds Action K/V."""
    def __init__(self, config, *, vae=None):
        super().__init__(config, action_expert=not config.motion_planner)
        d = config.hidden_dim
        if config.motion_planner:
            self.grounding = CurrentAttention(config)
            self.planner = MotionPlanner(config)
        elif config.visual_primitive:
            self.grounding = VisualPrimitive(config)
        elif config.grounding_subject:
            self.grounding = TaskGrounding(config)
        else:
            self.context_projection = nn.Linear(config.vlm_width, d)
            self.arm_embedding = nn.Embedding(2, d)
            self.skill = Skill(config)
            self.grounding = Grounding(config)
            self.motion = Motion(config)
            self.auxiliary_kv = nn.ModuleList(
                nn.Linear(d, 2 * config.kv_heads * config.head_dim) for _ in range(config.depth))
        self.visual_supervision = VisualSupervision(config, vae) if vae is not None else None
        self.diagnostic_statistics = {}
        self.loss_counts = {}
        self.training_stage = 'grounding' if config.grounding_subject else 'joint'
        self._joint_trainable_names = None
        if config.motion_planner:
            self.configure_stage('motion_planner')
        elif config.visual_primitive:
            self.configure_stage('visual_primitive')

    def configure_stage(self, stage):
        """Restore the initial joint scope or select a stage; rebuild optimizer/DDP after switching."""
        if self.config.motion_planner:
            if stage != 'motion_planner':
                raise ValueError('Motion Planner requires the motion_planner stage')
            self.training_stage = stage
            self.requires_grad_(False)
            self.planner.requires_grad_(True)
            self.grounding.requires_grad_(self.config.planner_joint)
            for i in self.config.attention_layers:
                projection = self.paligemma_with_expert.paligemma.language_model.layers[i].self_attn.q_proj
                if isinstance(projection, LoRALinear):
                    projection.lora_A.requires_grad_(self.config.planner_joint)
                    projection.lora_B.requires_grad_(self.config.planner_joint)
            return
        if self.config.visual_primitive:
            if stage != 'visual_primitive':
                raise ValueError('Visual Primitive requires the visual_primitive stage')
            self.training_stage = stage
            self.requires_grad_(False)
            self.grounding.requires_grad_(True)
            for i in self.config.attention_layers:
                q_proj = self.paligemma_with_expert.paligemma.language_model.layers[i].self_attn.q_proj
                if isinstance(q_proj, LoRALinear):
                    q_proj.lora_A.requires_grad_(True)
                    q_proj.lora_B.requires_grad_(True)
            return
        if (stage == 'grounding') != self.config.grounding_subject:
            raise ValueError('The grounding stage requires the Task Grounding architecture')
        if stage not in ('joint', 'stage1', 'stage2', 'grounding'):
            raise ValueError('Unknown training stage')
        if stage != 'joint' and self.visual_supervision is None and not self.config.subject_bbox:
            raise ValueError('Stage training requires visual supervision')
        if self._joint_trainable_names is None:
            # Capture after optional LoRA installation, before any stage freezes parameters.
            self._joint_trainable_names = {name for name, p in self.named_parameters() if p.requires_grad}
        self.training_stage = stage
        if stage == 'joint':
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name in self._joint_trainable_names)
            return
        self.requires_grad_(False)
        if stage == 'grounding':
            self.grounding.requires_grad_(True)
            if self.visual_supervision is not None:
                self.visual_supervision.head.requires_grad_(True)
            return
        for name in ('context_projection', 'arm_embedding',
                     'skill', 'grounding', 'motion'):
            getattr(self, name).requires_grad_(True)
        self.visual_supervision.head.requires_grad_(True)
        if stage == 'stage2':
            if not self.config.freeze_vlm:
                self.paligemma_with_expert.paligemma.requires_grad_(True)
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
        grounding = self.grounding(
            context, prefix.mask, arms,
            text_context=context[:, -inputs['tokens'].shape[1]:], text_mask=inputs['token_mask'])
        if self.config.visual_primitive_version == 'v3':
            # The motion decoder sees only visual primitive + unfused vision patches.
            # The first camera defines the supervised 2D trajectory's image frame.
            patches_per_view = prefix.visual_tokens.shape[1] // inputs['images'].shape[1]
            spatial = prefix.visual_tokens[:, :patches_per_view]
            spatial_mask = prefix.visual_mask[:, :patches_per_view]
            if not spatial_mask.any(-1).all():
                raise ValueError('v3 Motion requires a valid primary camera')
            latent = self.motion(
                None, None, skill, arms, presence, grounding=grounding,
                spatial=spatial, spatial_mask=spatial_mask)
        else:
            role_mask = presence[:, :, None, None].expand(grounding.shape[:-1]).flatten(1)
            memory = torch.cat((context, grounding.flatten(1, 3)), 1)
            memory_mask = torch.cat((prefix.mask, role_mask), -1)
            latent = self.motion(memory, memory_mask, skill, arms, presence)
        condition = self.action_condition(prefix, latent, presence, inputs) if with_action else None
        prediction = dict(skill_logits=logits, skill_ids=self.skill.active_skill_ids[logits.argmax(-1)],
                          skill_prior=prior, grounding=grounding, motion=latent)
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
        trace = self.motion.trace_head(prediction['motion']).float()
        gripper = self.motion.gripper_head(prediction['motion']).squeeze(-1).float()
        return trace, gripper

    def plan_features(self, inputs, *, query_diagnostics=None, router_only=False, attention_diagnostics=False):
        if inputs['images'].size(1) != 1 or not inputs['image_mask'].all() or not inputs['token_mask'].any(1).all():
            raise ValueError('Motion Planner needs one valid main camera and a nonempty task per sample')
        prefix = self.encode_observation(inputs, attention_layers=self.config.attention_layers,
                                         motion_queries=None if router_only else self.planner.query_inputs(),
                                         query_diagnostics=query_diagnostics)
        n = prefix.visual_tokens.size(1)
        end = n + inputs['tokens'].size(1)
        current = self.grounding.read_current(prefix.context[:, n:end], inputs['token_mask'],
                                              prefix.text_image_attention)
        if attention_diagnostics:
            current.update(text_mask=inputs['token_mask'], native_attention=prefix.text_image_attention)
        return dict(**current, visual_hidden=prefix.context[:, :n], motion_hidden=prefix.context[:, end:],
                    visual_mask=prefix.visual_mask[:, :n], skill_logits=self.planner.router(current['current_query']))

    def plan_from_features(self, features, *, current_prior=None, skill_id=None, remove_experts=False, end_latent=None):
        return self.planner(features, current_prior=current_prior, skill_id=skill_id,
                            remove_experts=remove_experts, end_latent=end_latent)

    def predict_plan(self, inputs, **intervention):
        features = self.plan_features(inputs)
        return dict(**features, **self.plan_from_features(features, **intervention))

    @torch.no_grad()
    def calibrate_motion_queries(self, batches, *, targets=None):
        '''Initialize from valid embedded text RMS. Report query/PE and visual scales before training.'''
        if callable(batches):
            replay = batches
        else:
            # The trainer provides a factory. Direct callers can pass a single batch or small list.
            rows = [batches] if isinstance(batches, dict) else batches
            if iter(rows) is rows:
                raise ValueError('Calibration requires two passes; provide a minibatch factory or sequence')
            replay = lambda: iter(rows)
        moments = {}
        backbone = self.paligemma_with_expert
        sample_count = 0
        for batch in replay():
            inputs = batch.get('inputs', batch)
            sample_count += len(inputs['tokens'])
            text = backbone.embed_text(inputs['tokens'])[inputs['token_mask']]
            image = backbone.embed_images(inputs['images'].flatten(0, 1))
            for name, tensor in (('text', text), ('visual', image)):
                value = tensor.float()
                accum = moments.setdefault(name, [0., 0., 0])
                accum[0] += value.sum(dtype=torch.float64).item()
                accum[1] += value.square().sum(dtype=torch.float64).item()
                accum[2] += value.numel()
        if not sample_count or not moments['text'][2]:
            raise ValueError('Calibrate on at least one valid training instruction')
        report = {'calibration_samples': sample_count}
        for name, (total, square, count) in moments.items():
            mean = total / count
            report[name + '_rms'] = (square / count) ** .5
            report[name + '_mean'] = mean
            report[name + '_std'] = max(0., square / count - mean ** 2) ** .5
        self.planner.set_query_rms(report['text_rms'])
        fixed = self.planner.native_time.any(dim=-1)
        raw_time = self.planner.native_time[fixed]
        scaled_time = raw_time * self.planner.query_time_scale
        learned_rms = rms(self.planner.native_queries[fixed])
        query_inputs = self.planner.query_inputs()
        report.update(native_query_rms=float(learned_rms), native_time_rms=float(rms(scaled_time)),
                      native_time_raw_rms=float(rms(raw_time)), query_to_time_rms=float(learned_rms / rms(scaled_time)),
                      motion_input_rms=float(rms(query_inputs)))
        if self.planner.with_boundaries:
            report.update(end_input_rms=float(rms(query_inputs[1])), fixed_input_rms=float(rms(query_inputs[fixed])))
        totals, pairs, checked = {}, 0, 0
        for batch in replay():
            inputs = batch.get('inputs', batch)
            count = len(inputs['tokens'])
            stats = {}
            features = self.plan_features(inputs, query_diagnostics=stats)
            stats.update(self.planner.memory_statistics(features, batch.get('targets', targets)))
            pair_count = stats.pop('target_other_pairs', 0)
            for name, value in stats.items():
                weight = pair_count if name == 'target_other_value_l2' else count
                contribution = value ** 2 if name.endswith('_rms') else value
                totals[name] = totals.get(name, 0.) + weight * contribution
            checked += count
            pairs += pair_count
        if checked != sample_count:
            raise ValueError('Calibration data must be replayable for both passes')
        for name, total in totals.items():
            weight = pairs if name == 'target_other_value_l2' else checked
            mean = total / weight
            report[name] = mean ** .5 if name.endswith('_rms') else mean
        report.update(target_other_pairs=pairs, diagnostic_samples=checked,
                      xy_to_content_rms=report['xy_rms'] / max(report['projected_visual_rms'], 1e-20))
        return report

    def forward(self, inputs, targets, *, noise=None, time=None, router_only=False, planner_diagnostics=False):
        if self.training_stage == 'motion_planner':
            features = self.plan_features(inputs, router_only=router_only)
            losses, self.loss_counts, self.diagnostic_statistics = self.planner.supervision(
                features, targets, router_only=router_only, diagnostics=planner_diagnostics)
            return losses
        if self.training_stage == 'visual_primitive':
            prediction = self.predict_grounding(inputs)
            losses, self.loss_counts, self.diagnostic_statistics = self.grounding.losses(prediction, targets)
            return losses
        if self.training_stage == 'grounding':
            return self.forward_grounding(inputs, targets)
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
        terminal_hold = targets.get('trace_terminal_hold', torch.zeros_like(padded))
        trace_valid = targets['trace_valid'] & presence[:, :, None] & (~padded | terminal_hold)
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

    def predict_current(self, inputs):
        return self.predict_grounding(inputs)['current_map' if self.config.visual_primitive else 'current']

    def predict_grounding(self, inputs):
        if self.config.visual_primitive:
            if inputs['images'].size(1) != 1 or not inputs['image_mask'].all() or not inputs['token_mask'].any(-1).all():
                raise ValueError('Visual Primitive needs one valid main image and a nonempty task per observation')
            prefix = self.encode_observation(inputs, attention_layers=self.config.attention_layers)
            return self.grounding(prefix, inputs['token_mask'])
        prediction, visual = self._grounding_features(inputs)
        if self.config.current_bbox:
            roles = [('current', self.grounding.bbox_head, self.grounding.current_projection)]
            if self.config.subject_bbox:
                roles.append(('subject', self.grounding.subject_bbox_head, self.grounding.subject_projection))
            boxes = torch.stack([corners(head(prediction[name][:, 0].mean(1)).float().sigmoid())
                                 for name, head, _ in roles], dim=1).clamp(0, 1)
            tokens, xy = read_regions(visual, boxes)
            for i, (name, _, project) in enumerate(roles):
                local = project(tokens[:, i])
                prediction[name] = torch.stack((local, torch.zeros_like(local)), 1)
                prefix = '' if name == 'current' else 'subject_'
                prediction[prefix + 'bbox_xyxy'] = boxes[:, i]
                prediction[prefix + 'region_xy'] = xy[:, i]
        return prediction

    def _grounding_features(self, inputs):
        """One frozen image/task pass: fused Current and image-only Subject."""
        if not self.config.grounding_subject:
            raise ValueError('Task Grounding requires grounding_subject=true')
        if inputs['images'].shape[1] != 1 or not inputs['image_mask'].all():
            raise ValueError('Task Grounding requires one valid main camera')
        if not inputs['arm_presence'][:, 0].all() or inputs['arm_presence'][:, 1].any():
            raise ValueError('Task Grounding currently supports the single-arm slot only')
        if not inputs['token_mask'].any(-1).all():
            raise ValueError('Task Grounding requires a nonempty task')
        # These encoders are fixed feature extractors, including the original image projector.
        with torch.no_grad():
            prefix = self.encode_observation(inputs)
        patches = prefix.visual_tokens.shape[1]
        return self.grounding(prefix.context[:, :patches], prefix.mask[:, :patches],
                              prefix.context[:, patches:], inputs['token_mask'],
                              subject_visual=prefix.visual_tokens), prefix.visual_tokens

    def forward_grounding(self, inputs, targets):
        """Current localization and independently supervised gripper localization."""
        prediction, visual = self._grounding_features(inputs)
        valid = targets['role_valid'] & inputs['arm_presence'][:, :, None]
        if self.config.current_bbox:
            query = prediction['current'][:, 0].mean(1)
            box = self.grounding.bbox_head(query).float().sigmoid()
            current_valid = valid[:, 0, 0]
            loss, metrics = box_loss(box, targets['role_core_boxes'][:, 0, 0], current_valid, self.config.image_size)
            losses = {'bbox': loss}
            if self.config.subject_bbox:
                subject_box = self.grounding.subject_bbox_head(prediction['subject'][:, 0].mean(1)).float().sigmoid()
                losses['subject'], subject_metrics = box_loss(subject_box, targets['role_core_boxes'][:, 0, 1],
                    valid[:, 0, 1], self.config.image_size, metric_prefix='grounding_subject_bbox')
            else:
                subject = {key: targets[key][:, :, 1:2] for key in ('role_images', 'role_weights')}
                subject['role_valid'] = valid[:, :, 1:2]
                losses.update(self.visual_supervision(prediction['subject'].unsqueeze(2), subject, role_names=('subject',)))
                subject_metrics = self.visual_supervision.diagnostic_statistics
            self.loss_counts = dict(subject=valid[:, 0, 1].sum(), bbox=current_valid.sum())
            self.diagnostic_statistics = {**subject_metrics, **metrics}
        else:
            roles = ('current', 'subject')
            anchors = torch.stack([prediction[name] for name in roles], dim=2)
            losses = self.visual_supervision(anchors, dict(targets, role_valid=valid), role_names=roles)
            self.loss_counts = {name: valid[..., i].sum() for i, name in enumerate(roles)}
            self.diagnostic_statistics = dict(self.visual_supervision.diagnostic_statistics)
        if self.config.entity_contrast_enabled:
            # Preserve the original BBox+GAC projection for its existing checkpoints.
            projection = self.grounding.current_projection if self.config.current_bbox else self.grounding.entity_projection
            losses['gac'], self.loss_counts['gac'], recall = entity_contrast(
                prediction['current'][:, 0].mean(1), visual, projection, targets, valid[:, 0, 0])
            self.diagnostic_statistics['grounding_gac_recall'] = recall
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
        held = targets.get('trace_terminal_hold', torch.zeros_like(trace_valid))
        result['motion_observed_l2_px'] = mean_statistics(distance * self.config.image_size, trace_valid & ~held)
        result['motion_terminal_hold_l2_px'] = mean_statistics(distance * self.config.image_size, trace_valid & held)
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
