"""PyTorch π0.5 backbone and cached flow-matching action expert."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from openpi.models.gemma import PaliGemmaWithExpert, attend, qkv, residual_layer
from openpi.models.model import ActionCondition, Prefix


def sine_position(value, width, min_period=1., max_period=10000.):
    fractions = torch.linspace(0, 1, width // 2, device=value.device)
    period = min_period * (max_period / min_period) ** fractions
    angle = value.float()[..., None] * 2 * torch.pi / period
    return torch.cat((angle.sin(), angle.cos()), -1)


def augment_images(images):
    """OpenPI-style SFT augmentation: geometry on main view, color on both views."""
    views = []
    for camera in range(images.shape[1]):
        value = images[:, camera] / 2 + .5
        if camera == 0:
            h, w = value.shape[-2:]
            ch, cw = int(h * .95), int(w * .95)
            y = int(torch.randint(h - ch + 1, (), device=value.device))
            x = int(torch.randint(w - cw + 1, (), device=value.device))
            value = F.interpolate(value[:, :, y:y + ch, x:x + cw], (h, w), mode="bilinear", align_corners=False)
            angle = (torch.rand((), device=value.device) * 10 - 5) * torch.pi / 180
            grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h, device=value.device),
                                          torch.linspace(-1, 1, w, device=value.device), indexing="ij")
            grid = torch.stack((grid_x * angle.cos() - grid_y * angle.sin(),
                                grid_x * angle.sin() + grid_y * angle.cos()), -1)
            value = F.grid_sample(value, grid[None].expand(len(value), -1, -1, -1), align_corners=False)
        value = value * (.7 + torch.rand((), device=value.device) * .6)
        mean = value.mean((1, 2, 3), keepdim=True)
        value = (value - mean) * (.6 + torch.rand((), device=value.device) * .8) + mean
        gray = value.mean(1, keepdim=True)
        value = gray + (value - gray) * (.5 + torch.rand((), device=value.device))
        views.append(value.clamp(0, 1) * 2 - 1)
    return torch.stack(views, 1)


class Pi0(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.paligemma_with_expert = PaliGemmaWithExpert(config)
        self.action_in_proj = nn.Linear(config.action_dim, config.expert_width)
        self.action_out_proj = nn.Linear(config.expert_width, config.action_dim)
        self.time_mlp_in = nn.Linear(config.expert_width, config.expert_width)
        self.time_mlp_out = nn.Linear(config.expert_width, config.expert_width)
        self._base_keys = set(self.state_dict())
        if config.freeze_vlm:
            self.paligemma_with_expert.paligemma.requires_grad_(False)
        elif config.gradient_checkpointing:
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})

    def load_base_weights(self, path):
        from safetensors.torch import load_file
        path = Path(path)
        state = load_file(str(path / "model.safetensors" if path.is_dir() else path))
        converted = {}
        prefix = "paligemma_with_expert.paligemma."
        for key, value in state.items():
            for name in ("language_model.", "vision_tower.", "multi_modal_projector."):
                if key.startswith(prefix + name):
                    key = prefix + "model." + key[len(prefix):]
                    break
            converted[key] = value
        embedding = prefix + "model.language_model.embed_tokens.weight"
        head = prefix + "lm_head.weight"
        if embedding in converted and head not in converted:
            converted[head] = converted[embedding]
        unused = {"paligemma_with_expert.gemma_expert.lm_head.weight",
                  "paligemma_with_expert.gemma_expert.model.embed_tokens.weight"}
        unexpected = set(converted) - self._base_keys - unused
        missing = self._base_keys - set(converted)
        if missing or unexpected:
            raise ValueError(f"Not a matching π0.5 checkpoint: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        self.load_state_dict({k: v for k, v in converted.items() if k in self._base_keys}, strict=False)

    def _text_prefix(self, hidden, mask, visual_mask):
        keys, values = [], []
        positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
        for layer in self.paligemma_with_expert.paligemma.language_model.layers:
            def apply_layer(x, layer=layer):
                norm, gate = layer.input_layernorm(x)
                q, k, v = qkv(layer, norm, self.config.heads, self.config.kv_heads,
                              self.config.head_dim, positions)
                return residual_layer(layer, x, attend(q, k, v, mask), gate), k, v
            if self.config.gradient_checkpointing and self.training and (
                    hidden.requires_grad or any(p.requires_grad for p in layer.parameters())):
                hidden, k, v = checkpoint(apply_layer, hidden, use_reentrant=False)
            else:
                hidden, k, v = apply_layer(hidden)
            keys.append(k)
            values.append(v)
        hidden, _ = self.paligemma_with_expert.paligemma.language_model.norm(hidden)
        return Prefix(hidden, mask, visual_mask, keys, values)

    def encode_observation(self, inputs):
        images = inputs["images"]
        b, views = images.shape[:2]
        image_tokens = self.paligemma_with_expert.embed_images(images.flatten(0, 1))
        patches = image_tokens.shape[1]
        image_tokens = image_tokens.reshape(b, views * patches, -1)
        image_mask = inputs["image_mask"].repeat_interleave(patches, -1)
        text = self.paligemma_with_expert.embed_text(inputs["tokens"])
        mask = torch.cat((image_mask, inputs["token_mask"]), -1)
        visual = torch.cat((image_mask, torch.zeros_like(inputs["token_mask"])), -1)
        return self._text_prefix(torch.cat((image_tokens, text), 1), mask, visual)

    def predict_condition(self, inputs):
        """Original π0.5 conditioning, without any ESkill modules."""
        prefix = self.encode_observation(inputs)
        return ActionCondition(prefix.keys, prefix.values, prefix.mask), {}

    def forward(self, inputs, targets, *, noise=None, time=None):
        if self.training and self.config.augment_images:
            inputs = dict(inputs, images=augment_images(inputs["images"]))
        condition, _ = self.predict_condition(inputs)
        return {"action": self.flow_loss(targets["actions"], targets["action_valid"], condition,
                                         noise=noise, time=time)}

    @torch.no_grad()
    def sample_actions(self, inputs, *, noise=None, steps=10):
        condition, prediction = self.predict_condition(inputs)
        return self.sample_from_condition(condition, noise=noise, steps=steps), prediction

    def velocity(self, noisy_actions, time, condition: ActionCondition):
        dtype = self.action_in_proj.weight.dtype
        hidden = self.action_in_proj(noisy_actions.to(dtype))
        time_emb = sine_position(time, self.config.expert_width, 4e-3, 4.)
        time_emb = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(time_emb.to(dtype)))))
        n = hidden.shape[1]
        position_mask = condition.mask if condition.position_mask is None else condition.position_mask
        positions = position_mask.sum(-1, keepdim=True) + torch.arange(n, device=hidden.device)[None]
        mask = torch.cat((condition.mask, torch.ones(hidden.shape[:2], dtype=torch.bool, device=hidden.device)), -1)
        for i, layer in enumerate(self.paligemma_with_expert.gemma_expert.model.layers):
            def apply_layer(x, layer=layer, i=i):
                norm, gate = layer.input_layernorm(x, time_emb)
                q, k, v = qkv(layer, norm, self.config.heads, self.config.kv_heads,
                              self.config.head_dim, positions)
                k = torch.cat((condition.keys[i].to(k.dtype), k), 2)
                v = torch.cat((condition.values[i].to(v.dtype), v), 2)
                return residual_layer(layer, x, attend(q, k, v, mask), gate, time_emb)
            hidden = (checkpoint(apply_layer, hidden, use_reentrant=False)
                      if self.config.gradient_checkpointing and self.training else apply_layer(hidden))
        hidden, _ = self.paligemma_with_expert.gemma_expert.model.norm(hidden, time_emb)
        return self.action_out_proj(hidden).float()

    def flow_loss(self, actions, valid, condition, *, noise=None, time=None):
        from openpi.models.model import masked_mean
        noise = torch.randn_like(actions) if noise is None else noise
        if time is None:
            time = torch.distributions.Beta(1.5, 1.).sample((len(actions),)).to(actions.device) * .999 + .001
        u = time[:, None, None]
        prediction = self.velocity(u * noise + (1 - u) * actions, time, condition)
        loss = (prediction - (noise - actions)).square()
        return masked_mean(loss, valid) if self.config.mask_action_loss else loss.mean()

    @torch.no_grad()
    def sample_from_condition(self, condition, *, noise=None, steps=10):
        if steps < 1:
            raise ValueError("Flow sampler needs at least one step")
        b = condition.mask.shape[0]
        shape = (b, self.config.action_horizon, self.config.action_dim)
        value = torch.randn(shape, device=condition.mask.device) if noise is None else noise.clone()
        for step in range(steps):
            time = torch.full((b,), 1 - step / steps, device=value.device)
            value = value - self.velocity(value, time, condition) / steps
        return value
