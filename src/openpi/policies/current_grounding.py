"""RGB/task -> generated target text -> Current. No Event, scene IDs or boxes at inference."""
import numpy as np
import torch

from openpi.models.model import to_device
from openpi.models.target_text import generate_target_text
from openpi.policies.grounding_dataset import task_prompt
from openpi.transforms import ModelTransforms, TokenizePrompt


class CurrentGroundingPolicy:
    def __init__(self, model, tokenizer_path, *, max_target_tokens=32, precision='float32'):
        self.model = model.eval()
        self.tokenizer = TokenizePrompt(tokenizer_path, model.config.max_token_len)
        self.transform = ModelTransforms(model.config, {}, self.tokenizer)
        self.max_target_tokens = max_target_tokens
        self.precision = precision

    def _inputs(self, images, prompt):
        if len(images) != 1:
            raise ValueError('Current Grounding expects only the main image, as during training')
        data = self.transform.inputs(dict(images=images, image_mask=np.ones(len(images), bool), prompt=prompt,
                                           arm_presence=np.array([True, False])))
        return to_device({k: torch.from_numpy(v)[None] for k, v in data.items()}, next(self.model.parameters()).device)

    @torch.no_grad()
    def infer(self, images, task, *, target_text=None):
        """Explicit text is a localization diagnostic; ordinary inference generates it from RGB/task."""
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.precision == 'bfloat16'):
            return self._infer(images, task, target_text=target_text)

    def _infer(self, images, task, *, target_text=None):
        supplied = target_text is not None
        if not supplied:
            sp = self.tokenizer.tokenizer
            generated = generate_target_text(self.model, self._inputs(images, task_prompt(task)),
                                             eos_id=sp.eos_id(), max_tokens=self.max_target_tokens)
            tokens = generated['tokens'][0][generated['valid'][0]].tolist()
            if tokens and tokens[-1] == sp.eos_id():
                tokens.pop()
            target_text = sp.decode(tokens).strip()
            if not generated['terminated'][0] or not target_text:
                return dict(target_text=target_text, status='incomplete_target_text', grounding=None,
                            target_text_source='predicted')
        prediction = self.model.predict_grounding(self._inputs(images, target_text))
        return dict(target_text=target_text, target_text_source='provided' if supplied else 'predicted',
                    status='predicted', grounding=prediction['current'], subject_grounding=prediction.get('subject'),
                    target_presence_confidence=None)


def create_current_grounding_policy(checkpoint, *, device='cpu', tokenizer_path=None):
    from openpi.policies.policy_config import load_train_config
    from openpi.training.config import create_model
    from openpi.training.checkpoints import load_weights
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = load_train_config(saved, device=device, tokenizer_path=tokenizer_path)
    if not config.grounding_target_text:
        raise ValueError('This checkpoint has no trained target-text objective; use explicit-text diagnostics instead')
    model = create_model(config, training=False, initialize=False)
    load_weights(model, {k: v for k, v in saved['model'].items() if not k.startswith('visual_supervision.')})
    return CurrentGroundingPolicy(model, config.tokenizer_path, max_target_tokens=config.grounding_max_target_tokens,
                                  precision=config.precision)
