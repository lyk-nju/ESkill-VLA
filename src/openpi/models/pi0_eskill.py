"""Visual research models for Current and Subject grounding."""
import torch
from openpi.models.grounding import TaskGrounding, VisualSupervision
from openpi.models.grounding_bbox import box_loss, corners, entity_contrast, read_regions
from openpi.models.lora import LoRALinear
from openpi.models.pi0 import Pi0
from openpi.models.visual_primitive import CurrentAttention
from openpi.models.visual_comparisons import VisualPrimitive


class Pi0ESkill(Pi0):
    def __init__(self, config, *, vae=None):
        super().__init__(config, action_expert=False)
        if config.visual_primitive:
            self.grounding = CurrentAttention(config) if config.current_only else VisualPrimitive(config)
        else:
            self.grounding = TaskGrounding(config)
        self.visual_supervision = VisualSupervision(config, vae) if vae is not None else None
        self.diagnostic_statistics, self.loss_counts = {}, {}
        self.configure_stage('visual_primitive' if config.visual_primitive else 'grounding')

    def configure_stage(self, stage):
        if stage != ('visual_primitive' if self.config.visual_primitive else 'grounding'):
            raise ValueError('Stage must match the selected visual method')
        self.training_stage = stage
        self.requires_grad_(False)
        self.grounding.requires_grad_(True)
        if self.visual_supervision is not None:
            self.visual_supervision.head.requires_grad_(True)
        layers = self.paligemma_with_expert.paligemma.language_model.layers
        if self.config.visual_primitive:
            for i in self.config.finetune_vlm_layers:
                layers[i].requires_grad_(True)
            for i in self.config.attention_layers:
                proj = layers[i].self_attn.q_proj
                if isinstance(proj, LoRALinear):
                    proj.lora_A.requires_grad_(True)
                    proj.lora_B.requires_grad_(True)

    def forward(self, inputs, targets, **kwargs):
        if self.config.visual_primitive:
            losses, self.loss_counts, self.diagnostic_statistics = self.grounding.losses(self.predict_grounding(inputs), targets)
            return losses
        return self.forward_grounding(inputs, targets)

    def predict_current(self, inputs):
        return self.predict_grounding(inputs)['current_map' if self.config.visual_primitive else 'current']

    def predict_grounding(self, inputs):
        if self.config.visual_primitive:
            views = inputs['images'].size(1)
            if (views not in ((1, 2) if self.config.current_only else (1,))
                    or not inputs['image_mask'][:, 0].all() or not inputs['token_mask'].any(-1).all()):
                raise ValueError('Visual Primitive requires a valid main image and task; only Current-only supports a wrist view')
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
        """One image/task pass supplies fused Current patches and image-only Subject patches."""
        if not self.config.grounding_subject:
            raise ValueError('Task Grounding requires grounding_subject=true')
        if inputs['images'].shape[1] != 1 or not inputs['image_mask'].all():
            raise ValueError('Task Grounding requires one valid main camera')
        if not inputs['arm_presence'][:, 0].all() or inputs['arm_presence'][:, 1].any():
            raise ValueError('Task Grounding currently supports the single-arm slot only')
        if not inputs['token_mask'].any(-1).all():
            raise ValueError('Task Grounding requires a nonempty task')
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
