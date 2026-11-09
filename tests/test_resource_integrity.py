"""Weight identity covers tensor values and VAE configuration, not only shapes."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from openpi.training.checkpoints import check_resources, resource_identity, _safetensors_identity


def test_same_header_different_weights_rejected(tmp_path):
    path = tmp_path / 'model.safetensors'
    save_file({'weight': torch.zeros(4)}, path)
    original_header = _safetensors_identity(path)
    saved = {'resources': {'pretrained_path': resource_identity('pretrained_path', path)}}
    save_file({'weight': torch.ones(4)}, path)
    assert _safetensors_identity(path) == original_header
    with pytest.raises(ValueError, match='pretrained_path content differs'):
        check_resources(saved, SimpleNamespace(pretrained_path=str(path)), fields=('pretrained_path',))


def test_vae_configuration_changes_identity(tmp_path):
    save_file({'weight': torch.zeros(4)}, tmp_path / 'diffusion_pytorch_model.safetensors')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'scaling_factor': 1., 'shift_factor': 0.}))
    original = resource_identity('vae_path', tmp_path)
    config.write_text(json.dumps({'scaling_factor': 2., 'shift_factor': 0.}))
    assert resource_identity('vae_path', tmp_path) != original


def test_legacy_header_check_is_explicitly_reported(tmp_path):
    path = tmp_path / 'model.safetensors'
    save_file({'weight': torch.zeros(4)}, path)
    saved = {'resources': {'pretrained_path': _safetensors_identity(path)}}
    with pytest.warns(RuntimeWarning, match='original weight values'):
        check_resources(saved, SimpleNamespace(pretrained_path=str(path)), fields=('pretrained_path',))


def test_lora_policy_checks_base_before_model_construction(tmp_path, monkeypatch):
    from openpi.policies import policy_config
    from openpi.training.config import TrainConfig
    path = tmp_path / 'model.safetensors'
    save_file({'weight': torch.zeros(4)}, path)
    config = TrainConfig('unused', 'unused', 'unused', 'unused', pretrained_path=str(path), lora_rank=2)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save(dict(config=config.to_dict(), weights_kind='lora',
                    resources={'pretrained_path': resource_identity('pretrained_path', path)}), checkpoint)
    save_file({'weight': torch.ones(4)}, path)
    def unexpected(*args, **kwargs):
        pytest.fail('Model constructed before checking base identity')
    monkeypatch.setattr(policy_config, 'create_model', unexpected)
    with pytest.raises(ValueError, match='pretrained_path content differs'):
        policy_config.create_trained_policy(checkpoint)
