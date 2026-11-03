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
