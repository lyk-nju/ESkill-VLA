"""Current observation and task to main-view grounding, skill and a 2D plan."""
import numpy as np
import torch

from openpi.policies.libero_policy import camera_inputs
from openpi.models.model import to_device
from openpi.policies.policy_config import load_train_config
from openpi.training.checkpoints import check_resources, load_weights
from openpi.training.config import create_model
from openpi.transforms import ModelTransforms, TokenizePrompt


class MotionPlannerPolicy:
    def __init__(self, model, tokenizer, precision, cameras=('image',)):
        self.cameras = tuple(cameras)
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.precision = precision
        self.transform = ModelTransforms(model.config, {}, tokenizer)

    def features(self, rgb, task, *, attention_diagnostics=False):
        views = [rgb] if isinstance(rgb, np.ndarray) else list(rgb)
        if len(views) != len(self.cameras):
            raise ValueError(f'Expected camera order {self.cameras}; use None for a missing wrist')
        inputs = self.transform.inputs(dict(**camera_inputs(views), prompt=task))
        n = int(inputs['token_mask'].sum())
        inputs['tokens'] = inputs['tokens'][:n]
        inputs['token_mask'] = inputs['token_mask'][:n]
        batch = to_device({key: torch.as_tensor(val)[None] for key, val in inputs.items()}, self.device)
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                    enabled=self.precision == 'bfloat16'):
            return self.model.plan_features(batch, attention_diagnostics=attention_diagnostics)

    def read_plan(self, features, **intervention):
        with torch.inference_mode(), torch.autocast(self.device.type, dtype=torch.bfloat16,
                                                    enabled=self.precision == 'bfloat16'):
            return self.model.plan_from_features(features, **intervention)

    def infer(self, rgb, task):
        features = self.features(rgb, task)
        return {**self.read_plan(features), 'current_map': features['current_map'],
                'skill_probabilities': features['skill_logits'].float().softmax(-1)}


def create_motion_planner_policy(checkpoint, *, device='cpu', tokenizer_path=None, pretrained_path=None):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = load_train_config(saved, device=device, tokenizer_path=tokenizer_path)
    if config.training_stage != 'motion_planner':
        raise ValueError('Expected a Motion Planner checkpoint')
    if pretrained_path is not None:
        config.pretrained_path = str(pretrained_path)
    compact = saved.get('weights_kind') == 'planner_adapter'
    if compact:
        check_resources(saved, config, fields=('pretrained_path', 'tokenizer_path'), purpose='load Motion Planner')
    model = create_model(config, training=False, initialize=compact)
    load_weights(model, saved['model'], adapter_only=compact)
    tokenizer = TokenizePrompt(config.tokenizer_path, model.config.max_token_len)
    return MotionPlannerPolicy(model, tokenizer, config.precision, config.cameras)
