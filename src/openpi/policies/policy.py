"""Online inference uses current observations only; no Event labels or future RGB."""
import numpy as np
import torch

from openpi.models.model import to_device


class Policy:
    def __init__(self, model, input_adapter, output_adapter, transforms, *, device, precision="float32"):
        if precision not in ("float32", "bfloat16"):
            raise ValueError("Inference precision must be float32 or bfloat16")
        self.model = model.eval()
        self.precision = precision
        self.input_adapter, self.output_adapter = input_adapter, output_adapter
        self.transforms, self.device = transforms, device

    @torch.no_grad()
    def infer(self, observation, *, noise=None, steps=10):
        observation = dict(observation)
        observation.setdefault("state_valid", np.ones_like(observation["state"], dtype=bool))
        data = self.transforms.inputs(self.input_adapter(observation))
        inputs = to_device({key: torch.as_tensor(value)[None] for key, value in data.items()}, self.device)
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device)
            if noise.ndim == 2:
                noise = noise[None]
        with torch.autocast(device_type=torch.device(self.device).type, dtype=torch.bfloat16,
                            enabled=self.precision == "bfloat16"):
            actions, prediction = self.model.sample_actions(inputs, noise=noise, steps=steps)
        result = dict(actions=self.output_adapter(actions[0].cpu().numpy()))
        if "skill_logits" not in prediction:
            return result
        presence = data["arm_presence"]
        current = prediction["skill_logits"][0].argmax(-1).cpu().numpy()
        current = np.asarray(self.model.config.active_skill_ids, dtype=np.int64)[current]
        result.update(current_skills=np.where(presence, current, -1),
                      skill_names=self.model.config.skill_names)
        return result
