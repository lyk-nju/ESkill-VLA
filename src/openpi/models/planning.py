"""Native Current KLAL, event motion planning and π0.5 execution."""
import torch

from openpi.models.lora import LoRALinear
from openpi.models.model import CameraLayout
from openpi.models.motion_planner import rms
from openpi.models.research import build_planner
from openpi.models.pi0 import Pi0
from openpi.models.visual_primitive import CurrentAttention


class PlanningModel(Pi0):
    def __init__(self, config, *, vae=None):
        super().__init__(config, action_expert=config.action_execution)
        self.grounding = CurrentAttention(config)
        if config.motion_planner:
            self.planner = build_planner(config)
        self.loss_counts, self.diagnostic_statistics = {}, {}
        PlanningModel.configure_stage(self, config.current_grounding)

    def configure_stage(self, stage):
        if stage != self.config.current_grounding:
            raise ValueError(f'This architecture uses the {self.config.current_grounding} stage')
        self.training_stage = stage
        self.requires_grad_(False)
        cfg = self.config
        tune_current = cfg.visual_primitive or cfg.planner_joint or (cfg.action_execution and cfg.train_current)
        self.grounding.requires_grad_(tune_current)
        if cfg.motion_planner:
            self.planner.requires_grad_(not cfg.action_execution or cfg.train_planner)
        language = self.paligemma_with_expert.paligemma.language_model
        if cfg.visual_primitive:
            for index in cfg.finetune_vlm_layers:
                language.layers[index].requires_grad_(True)
        for index in cfg.attention_layers:
            proj = language.layers[index].self_attn.q_proj
            if isinstance(proj, LoRALinear):
                proj.lora_A.requires_grad_(tune_current)
                proj.lora_B.requires_grad_(tune_current)

    def predict_grounding(self, inputs):
        if not inputs['image_mask'][:, 0].all() or not inputs['token_mask'].any(-1).all():
            raise ValueError('Current KLAL requires a valid main image and task')
        prefix = self.encode_observation(inputs, attention_layers=self.config.attention_layers)
        return self.grounding(prefix, inputs['token_mask'])

    def predict_current(self, inputs):
        return self.predict_grounding(inputs)['current_map']

    def plan_features(self, inputs, *, query_diagnostics=None, router_only=False, attention_diagnostics=False, cache_prefix=False):
        if not inputs['image_mask'][:, 0].all() or not inputs['token_mask'].any(1).all():
            raise ValueError('Motion Planner needs a valid main camera and a nonempty task per sample')
        prefix = self.encode_observation(inputs, attention_layers=self.config.attention_layers,
                                         motion_queries=None if router_only else self.planner.query_inputs(),
                                         query_diagnostics=query_diagnostics, cache_prefix=cache_prefix)
        layout = CameraLayout.from_prefix(self.config, prefix, inputs['token_mask'])
        n = layout.visuals
        end = n + inputs['tokens'].size(1)
        current = self.grounding.read_current(prefix.context[:, layout.text], inputs['token_mask'],
                                              prefix.text_image_attention[..., layout.main])
        if attention_diagnostics:
            current.update(text_mask=inputs['token_mask'], native_attention=prefix.text_image_attention[..., layout.main])
        if cache_prefix:
            current['prefix'] = prefix
        return dict(**current, visual_hidden=prefix.context[:, layout.main], motion_hidden=prefix.context[:, end:],
                    visual_mask=prefix.visual_mask[:, layout.main], skill_logits=self.planner.router(current['current_query']))

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
            image = backbone.embed_images(inputs['images'][inputs['image_mask']])
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

    def forward(self, inputs, targets, *, router_only=False, planner_diagnostics=False, **kwargs):
        if self.config.visual_primitive:
            losses, self.loss_counts, self.diagnostic_statistics = self.grounding.losses(self.predict_grounding(inputs), targets)
        else:
            features = self.plan_features(inputs, router_only=router_only)
            losses, self.loss_counts, self.diagnostic_statistics = self.planner.supervision(
                features, targets, router_only=router_only, diagnostics=planner_diagnostics)
        return losses
