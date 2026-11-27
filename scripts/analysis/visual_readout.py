"""Decode the stored visual head with its matched VAE for historical comparisons."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from openpi.models.grounding import VisualSupervision, load_vae
from openpi.training.checkpoints import check_resources

class VisualReadout:
    """Load only a checkpoint's saved head and the matching frozen VAE, never a random head."""
    def __init__(self, checkpoint, config, device, vae_path=None):
        self.module, self.status = None, 'VISUAL HEAD NOT SAVED'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        prefix = 'visual_supervision.head.'
        weights = {key[len(prefix):]: value for key, value in saved['model'].items() if key.startswith(prefix)}
        if not weights:
            return
        path = vae_path or config.vae_path
        if not path or not Path(path).exists():
            self.status = 'VAE UNAVAILABLE; supply --vae-path'
            return
        check_resources(saved, replace(config, vae_path=str(path)), fields=('vae_path',),
                        purpose='decode saved Grounding predictions')
        self.module = VisualSupervision(config.model, load_vae(path)).to(device).eval()
        self.module.head.load_state_dict(weights, strict=True)
        self.status = 'SAVED HEAD / MATCHED FROZEN VAE'

    @torch.no_grad()
    def decode(self, prediction, targets, presence, precision):
        if self.module is None:
            return None, None
        visual = self.module
        device = prediction['grounding'].device
        present = torch.as_tensor(presence, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
            predicted = visual.predict_latents(prediction['grounding'][0, present]).flatten(0, 1)
        reference = None
        if targets is not None:
            images = torch.as_tensor(targets['role_images'][presence], device=device).flatten(0, 1)
            reference = visual.encode_targets(images)
        with torch.autocast(device.type, enabled=False):
            z = predicted.float() if reference is None else torch.cat((predicted.float(), reference), 0)
            raw = z / visual.vae.config.scaling_factor + (getattr(visual.vae.config, 'shift_factor', None) or 0.)
            decoded = visual.vae.decode(raw).sample
        rgb = ((decoded.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        prediction_rgb = np.zeros((2, 2, *rgb.shape[1:]), np.uint8)
        reference_rgb = None if reference is None else np.zeros_like(prediction_rgb)
        n = int(presence.sum()) * 2
        prediction_rgb[presence] = rgb[:n].reshape(-1, 2, *rgb.shape[1:])
        if reference is not None:
            reference_rgb[presence] = rgb[n:].reshape(-1, 2, *rgb.shape[1:])
        return prediction_rgb, reference_rgb
