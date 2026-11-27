"""RGB/full task -> parallel Current and Subject. No Event, scene IDs or boxes at inference."""
import numpy as np
import torch

from openpi.policies.libero_policy import camera_inputs
from openpi.models.model import to_device
from openpi.transforms import ModelTransforms, TokenizePrompt


class CurrentGroundingPolicy:
    def __init__(self, model, tokenizer_path, *, precision='float32', cameras=('image',)):
        self.model = model.eval()
        self.tokenizer = TokenizePrompt(tokenizer_path, model.config.max_token_len)
        self.transform = ModelTransforms(model.config, {}, self.tokenizer)
        self.precision = precision
        self.cameras = tuple(cameras)

    def _inputs(self, images, prompt):
        if len(images) != len(self.cameras) or images[0] is None:
            raise ValueError(f'Expected images in checkpoint camera order {self.cameras}, with a valid main image')
        data = self.transform.inputs(dict(**camera_inputs(images), prompt=prompt,
                                           arm_presence=np.array([True, False])))
        if self.model.training_stage == 'visual_primitive':
            data = {k: data[k] for k in ('images', 'image_mask', 'tokens', 'token_mask')}
            length = int(data['token_mask'].sum())
            data['tokens'], data['token_mask'] = data['tokens'][:length], data['token_mask'][:length]
        return to_device({k: torch.from_numpy(v)[None] for k, v in data.items()}, next(self.model.parameters()).device)

    @torch.no_grad()
    def infer(self, images, task):
        """Direct task-conditioned localization, without intermediate text or oracle labels."""
        device = next(self.model.parameters()).device
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=self.precision == 'bfloat16'):
            prediction = self.model.predict_grounding(self._inputs(images, task))
        if self.model.training_stage == 'visual_primitive':
            return dict(prediction, task=task, status='predicted')
        result = dict(task=task, status='predicted', grounding=prediction['current'],
                      subject_grounding=prediction['subject'], target_presence_confidence=None)
        if 'bbox_xyxy' in prediction:
            result.update(bbox_xyxy=prediction['bbox_xyxy'], region_xy=prediction['region_xy'])
        if 'subject_bbox_xyxy' in prediction:
            result.update(subject_bbox_xyxy=prediction['subject_bbox_xyxy'], subject_region_xy=prediction['subject_region_xy'])
        return result


def create_current_grounding_policy(checkpoint, *, device='cpu', tokenizer_path=None, pretrained_path=None):
    from openpi.policies.policy_config import load_train_config
    from openpi.training.config import create_model
    from openpi.training.checkpoints import check_resources, load_weights
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = load_train_config(saved, device=device, tokenizer_path=tokenizer_path)
    if config.training_stage not in ('grounding', 'visual_primitive'):
        raise ValueError('This policy requires a task-conditioned grounding checkpoint')
    compact = saved.get('weights_kind') == 'visual_adapter'
    if pretrained_path is not None:
        config.pretrained_path = str(pretrained_path)
    if compact:
        check_resources(saved, config, fields=('pretrained_path', 'tokenizer_path'), purpose='load Visual Primitive')
    model = create_model(config, training=False, initialize=compact)
    load_weights(model, {k: v for k, v in saved['model'].items() if not k.startswith('visual_supervision.')},
                 adapter_only=compact)
    return CurrentGroundingPolicy(model, config.tokenizer_path, precision=config.precision, cameras=config.cameras)
