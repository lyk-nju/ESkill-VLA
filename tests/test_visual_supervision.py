"""Spatial assembly, masked latent targets, FP32 encoding and strict VAE loading."""
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')
from torch import nn
from diffusers import AutoencoderKL
from safetensors.torch import load_file, save_file

from openpi.models.grounding import VisualSupervision, load_vae
from openpi.models.pi0_config import ESkillConfig


class RecordingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(latent_channels=1, block_out_channels=[1],
                                      scaling_factor=.5, shift_factor=.25)
        self.calls = 0

    def encode(self, images):
        assert images.dtype == torch.float32 and self.weight.dtype == torch.float32
        assert not torch.is_autocast_enabled(images.device.type)
        assert not torch.is_grad_enabled() and not self.training
        self.calls += 1
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda: images[:, :1] * self.weight))


def supervision():
    config = ESkillConfig.tiny(hidden_dim=4, canvas_size=4, role_queries=4)
    return VisualSupervision(config, RecordingVAE())


def test_patches_keep_row_major_positions_and_all_leading_axes():
    model = supervision()
    model.head = nn.Identity()
    anchors = torch.arange(16.).reshape(4, 4).expand(2, 2, 2, -1, -1)
    expected = torch.tensor([[0., 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    actual = model.predict_latents(anchors)
    assert actual.shape == (2, 2, 2, 1, 4, 4)
    torch.testing.assert_close(actual, expected.expand_as(actual))


def test_loss_equals_independent_per_role_weighted_target_mean_and_ignores_invalid_nan():
    model = supervision().train()
    nn.init.zeros_(model.head[-1].weight)
    nn.init.zeros_(model.head[-1].bias)
    anchors = torch.randn(2, 2, 2, 4, 4, requires_grad=True)
    images = torch.linspace(-1, 1, 2 * 2 * 2 * 3 * 4 * 4).reshape(2, 2, 2, 3, 4, 4)
    region = torch.zeros(2, 2, 2, 4, 4)
    region[..., :2, :] = 1
    valid = torch.tensor([[[1, 1], [0, 1]], [[1, 0], [0, 1]]], dtype=torch.bool)
    images[~valid], region[~valid] = float('nan'), float('nan')
    target = dict(role_images=images, role_weights=region, role_valid=valid)
    expected = {}
    for role, name in enumerate(('current', 'future_object')):
        values = []
        for batch in range(2):
            for arm in range(2):
                if valid[batch, arm, role]:
                    latent = (images[batch, arm, role, 0] - .25) * .5
                    weights = .1 + .9 * region[batch, arm, role]
                    values.append((latent.square() * weights).sum() / weights.sum())
        expected[name] = torch.stack(values).mean()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        losses = model(anchors, target)
    assert set(losses) == {'current', 'future_object'}
    for name, value in losses.items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, expected[name])
        torch.testing.assert_close(model.metrics['grounding_' + name], value.detach())
    loss = sum(losses.values())
    assert all(not value.requires_grad for value in model.metrics.values())
    loss.backward()
    assert torch.isfinite(anchors.grad).all()
    assert not anchors.grad[~valid].any()
    assert model.vae.weight.grad is None


def test_all_missing_targets_skip_vae_but_keep_zero_gradients_connected():
    model = supervision()
    anchors = torch.randn(1, 2, 2, 4, 4, requires_grad=True)
    target = dict(role_images=torch.full((1, 2, 2, 3, 4, 4), float('nan')),
                  role_weights=torch.full((1, 2, 2, 4, 4), float('nan')),
                  role_valid=torch.zeros(1, 2, 2, dtype=torch.bool))
    loss = sum(model(anchors, target).values())
    assert loss == 0 and model.vae.calls == 0
    assert all(value == 0 for value in model.metrics.values())
    loss.backward()
    assert anchors.grad is not None and not anchors.grad.any()
    assert all(p.grad is not None and not p.grad.any() for p in model.head.parameters())


def test_loader_accepts_diffusers_layout_and_rejects_missing_parameters(tmp_path):
    vae = AutoencoderKL(down_block_types=('DownEncoderBlock2D',), up_block_types=('UpDecoderBlock2D',),
                        block_out_channels=(8,), layers_per_block=1, norm_num_groups=4, latent_channels=4)
    vae.save_pretrained(tmp_path)
    loaded = load_vae(tmp_path)
    assert not loaded.training
    assert all(not p.requires_grad and p.dtype == torch.float32 for p in loaded.parameters())
    for name, value in vae.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
    path = tmp_path / 'diffusion_pytorch_model.safetensors'
    weights = load_file(path)
    del weights['encoder.conv_in.weight']
    save_file(weights, path)
    with pytest.raises(RuntimeError, match='encoder.conv_in.weight'):
        load_vae(tmp_path)


def test_non_square_queries_fail_before_training():
    with pytest.raises(ValueError, match='square grid'):
        VisualSupervision(ESkillConfig.tiny(role_queries=2), RecordingVAE())
