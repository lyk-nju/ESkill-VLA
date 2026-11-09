"""Reject unsupported ESkill precision before model/data/distributed initialization."""
import pytest
torch = pytest.importorskip('torch')

from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig, create_model
from scripts.train import train
from test_eskill_model import batch


@pytest.mark.parametrize('entry', [create_model, train])
def test_bf16_eskill_without_autocast_fails_before_loading_assets(entry, tmp_path, monkeypatch):
    config = TrainConfig('missing', 'missing', str(tmp_path / 'output'), 'missing',
                         backbone_dtype='bfloat16', precision='float32', device='cuda')
    monkeypatch.setenv('WORLD_SIZE', '4')
    with pytest.raises(ValueError, match="requires precision='bfloat16'"):
        entry(config)
    assert not (tmp_path / 'output').exists()


@pytest.mark.parametrize('backbone,precision', [
    ('float32', 'float32'), ('float32', 'bfloat16'), ('bfloat16', 'bfloat16'),
])
def test_supported_eskill_precision_combinations_forward_and_backward(backbone, precision):
    torch.set_num_threads(2)
    config = TrainConfig('unused', 'unused', 'unused', 'unused',
                         model=ESkillConfig.tiny(vocab_size=128), device='cpu', smoke=True,
                         backbone_dtype=backbone, precision=precision)
    model = create_model(config, initialize=False)
    inputs, targets = batch(model.config)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=precision == 'bfloat16'):
        losses = model(inputs, targets)
        loss = sum(losses.values())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.context_projection.weight.grad is not None
    assert torch.isfinite(model.context_projection.weight.grad).all()


@pytest.mark.parametrize('precision,expected_dtype', [('float32', torch.float32), ('bfloat16', torch.bfloat16)])
def test_policy_compute_precision_is_independent_of_fp32_storage(precision, expected_dtype):
    import numpy as np
    from types import SimpleNamespace
    from openpi.policies.policy import Policy
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)
        def sample_actions(self, inputs, **kwargs):
            output = self.layer(inputs['value'])
            assert output.dtype == expected_dtype
            assert self.layer.weight.dtype == torch.float32
            return output.float()[:, None, :], {}
    policy = Policy(Model(), lambda data: data, lambda data: data,
                    SimpleNamespace(inputs=lambda data: {'value': np.ones(2, np.float32)}),
                    device='cpu', precision=precision)
    assert np.isfinite(policy.infer({'state': np.zeros(2)})['actions']).all()


@pytest.mark.parametrize('model_type', ['pi05', 'eskill'])
def test_inference_model_construction_does_not_import_diffusers(model_type, monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == 'diffusers' or name.startswith('diffusers.'):
            raise AssertionError('Inference must not import the VAE dependency')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    config = TrainConfig('unused', 'unused', 'unused', 'unused', model_type=model_type,
                         model=ESkillConfig.tiny(vocab_size=128), device='cpu', smoke=True)
    model = create_model(config, training=False, initialize=False)
    assert getattr(model, 'visual_supervision', None) is None
