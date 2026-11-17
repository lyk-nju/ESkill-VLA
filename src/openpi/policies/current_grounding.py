"""RGB/full task -> parallel Current and Subject. No Event, scene IDs or boxes at inference."""
import numpy as np
import torch

from openpi.models.model import to_device
from openpi.transforms import ModelTransforms, TokenizePrompt


class CurrentGroundingPolicy:
    def __init__(self, model, tokenizer_path, *, precision='float32'):
        self.model = model.eval()
        self.tokenizer = TokenizePrompt(tokenizer_path, model.config.max_token_len)
        self.transform = ModelTransforms(model.config, {}, self.tokenizer)
        self.precision = precision

    def _inputs(self, images, prompt):
        if len(images) != 1:
            raise ValueError('Current Grounding expects only the main image, as during training')
        data = self.transform.inputs(dict(images=images, image_mask=np.ones(len(images), bool), prompt=prompt,
                                           arm_presence=np.array([True, False])))
        return to_device({k: torch.from_numpy(v)[None] for k, v in data.items()}, next(self.model.parameters()).device)

    @torch.no_grad()
    def infer(self, images, task):
        """Direct task-conditioned localization, without intermediate text or oracle labels."""
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.precision == 'bfloat16'):
            prediction = self.model.predict_grounding(self._inputs(images, task))
        return dict(task=task, status='predicted', grounding=prediction['current'],
                    subject_grounding=prediction['subject'], target_presence_confidence=None)


def create_current_grounding_policy(checkpoint, *, device='cpu', tokenizer_path=None):
    from openpi.policies.policy_config import load_train_config
    from openpi.training.config import create_model
    from openpi.training.checkpoints import load_weights
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = load_train_config(saved, device=device, tokenizer_path=tokenizer_path)
    if config.training_stage != 'grounding' or not config.model.grounding_subject:
        raise ValueError('This policy requires a task-conditioned Current/Subject checkpoint')
    model = create_model(config, training=False, initialize=False)
    load_weights(model, {k: v for k, v in saved['model'].items() if not k.startswith('visual_supervision.')})
    return CurrentGroundingPolicy(model, config.tokenizer_path, precision=config.precision)
