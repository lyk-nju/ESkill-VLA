"""Dropping right-padding must preserve all five losses and trainable gradients."""
import torch
from unittest import mock

from openpi.models import pi0
from openpi.models.lora import LoRALinear, apply_lora
from openpi.training.data_loader import collate_samples
from test_eskill_model import batch, model


def test_trimmed_batch_preserves_losses_and_gradients(model):
    inputs, targets = batch(model.config)
    inputs['token_mask'][0, 2:] = False
    inputs['token_mask'][1, 4:] = False
    samples = [dict(inputs={k:v[i] for k,v in inputs.items()},
                    targets={k:v[i] for k,v in targets.items()}, metadata={}) for i in range(2)]
    trimmed = collate_samples(samples)
    assert trimmed['inputs']['tokens'].shape == (2, 4)
    torch.testing.assert_close(trimmed['inputs']['token_mask'], inputs['token_mask'][:, :4])
    noise = torch.randn_like(targets['actions'])
    time = torch.full((2,), .5)
    full_losses = model(inputs, targets, noise=noise, time=time)
    sum(full_losses.values()).backward()
    gradients = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    losses = model(trimmed['inputs'], trimmed['targets'], noise=noise, time=time)
    sum(losses.values()).backward()
    for key in full_losses:
        torch.testing.assert_close(losses[key], full_losses[key], rtol=1e-5, atol=1e-6)
    for key, parameter in model.named_parameters():
        if key in gradients:
            torch.testing.assert_close(parameter.grad, gradients[key], rtol=1e-4, atol=1e-6)


def test_copy_elision_preserves_five_losses_and_gradients(model):
    from torch.nn import functional as F

    def copied_attention(q, k, v, mask):
        repeats = q.shape[1] // k.shape[1]
        return F.scaled_dot_product_attention(
            q, k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1),
            attn_mask=mask[:, None, None]).transpose(1, 2).flatten(2)

    def cast_lora(self, value):
        base = F.linear(value, self.weight, self.bias)
        update = F.linear(F.linear(value.to(self.lora_A.dtype), self.lora_A), self.lora_B)
        return base + (update * self.scale).to(base.dtype)

    apply_lora(model, rank=2, alpha=2)
    for name, parameter in model.named_parameters():
        if name.endswith('lora_B'):
            torch.nn.init.normal_(parameter, std=.01)
    inputs, targets = batch(model.config)
    noise, time = torch.randn_like(targets['actions']), torch.full((2,), .5)
    with mock.patch.object(pi0, 'attend', copied_attention), mock.patch.object(LoRALinear, 'forward', cast_lora):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            expected = model(inputs, targets, noise=noise, time=time)
        sum(expected.values()).backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model(inputs, targets, noise=noise, time=time)
    sum(actual.values()).backward()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], rtol=1e-4, atol=1e-6, msg=name)
