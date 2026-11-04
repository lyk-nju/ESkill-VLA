"""Validation is count-weighted, reproducible and does not disturb training."""
from types import SimpleNamespace
import json
from datetime import timedelta
from unittest import mock

import pytest
torch = pytest.importorskip('torch')
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from openpi.models.model import masked_mean
from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig
from openpi.training.data_loader import create_data_loader
from openpi.training.validation import validation_losses


class LossModel(nn.Module):
    def __init__(self, eskill=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.eskill = eskill
        self.visual_supervision = SimpleNamespace(metrics={})

    def forward(self, inputs, targets):
        assert not self.training and not torch.is_grad_enabled()
        torch.rand(1)  # Flow matching consumes random noise/time, even during validation.
        result = dict(action=masked_mean(self.weight * targets['actions'], targets['action_valid']))
        if self.eskill:
            result.update(skill=self.weight * 0, future=self.weight * 0, trace=self.weight * 0)
            roles = []
            for r, name in enumerate(('ee', 'obj', 'goal')):
                value = masked_mean(targets['role_error'][:, :, r], targets['role_valid'][:, :, r])
                self.visual_supervision.metrics['grounding_' + name] = value
                roles.append(value)
            result['grounding'] = sum(roles)
        return result


def config(eskill=False):
    return TrainConfig('unused', 'unused', 'unused', 'unused', device='cpu', precision='float32',
                       model_type='eskill' if eskill else 'pi05', model=ESkillConfig.tiny(), num_workers=0)


def test_validation_weights_valid_items_instead_of_batch_means_and_restores_rng():
    batches = []
    for value, count in [(9., 1), (1., 9)]:
        batches.append(dict(inputs={}, targets=dict(actions=torch.full((1, 1, 9), value),
                                                    action_valid=(torch.arange(9) < count)[None, None])))
    model = LossModel().train()
    rng = torch.get_rng_state().clone()
    result = validation_losses(model, batches, config())
    assert result['action'] == pytest.approx(1.8)
    assert result['loss'] == pytest.approx(1.8)
    assert model.training and model.weight.grad is None
    assert torch.equal(rng, torch.get_rng_state())
    assert validation_losses(model, batches, config()) == result


def test_visual_roles_are_normalized_separately_across_validation_batches():
    batches = []
    for i, valid in enumerate(([[1, 0, 1], [0, 0, 0]], [[1, 1, 0], [1, 1, 0]])):
        targets = dict(actions=torch.ones(1, 1, 1), action_valid=torch.ones(1, 1, 1, dtype=torch.bool),
                       skill_valid=torch.zeros(1, 2, dtype=torch.bool),
                       future_skill_valid=torch.zeros(1, 2, 1, dtype=torch.bool),
                       trace_valid=torch.zeros(1, 2, 1, dtype=torch.bool),
                       role_valid=torch.tensor([valid], dtype=torch.bool),
                       role_error=torch.tensor([[[9. if i == 0 else 1., 2., 4.]] * 2]))
        batches.append(dict(inputs=dict(arm_presence=torch.ones(1, 2, dtype=torch.bool)), targets=targets))
    result = validation_losses(LossModel(True), batches, config(True))
    assert result['grounding'] == pytest.approx(29 / 3)
    assert result['loss'] == pytest.approx(1 + .1 * 29 / 3)


def test_validation_auxiliary_statistics_are_accumulated_before_averaging():
    class DiagnosticModel(LossModel):
        def forward(self, inputs, targets):
            result = super().forward(inputs, targets)
            valid = targets['action_valid']
            correct = (targets['actions'] == 9) & valid
            self.diagnostic_statistics = dict(current_skill_accuracy=torch.stack((correct.sum(), valid.sum())))
            return result
    batches = [dict(inputs={}, targets=dict(actions=torch.full((1, 1, 9), value),
                                           action_valid=(torch.arange(9) < count)[None, None]))
               for value, count in ((9., 1), (1., 9))]
    result = validation_losses(DiagnosticModel(), batches, config(), include_aux=True)
    assert result['aux']['current_skill_accuracy'] == pytest.approx(.1)
    assert result['aux']['current_skill_count'] == 10


def test_validation_sampler_has_fixed_unique_indices_across_ranks(monkeypatch, tmp_path):
    import openpi.training.data_loader as module
    monkeypatch.setattr(module, 'create_dataset', lambda *args: list(range(10)))
    monkeypatch.setattr(module.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(module.dist, 'get_world_size', lambda: 3)
    cfg = config()
    manifest = tmp_path / 'split.json'
    manifest.write_text(json.dumps(dict(seed=42, validation_fraction=.1, train=[], val=[])))
    cfg.split_path = str(manifest)
    cfg.batch_size, cfg.val_max_batches = 2, 4
    selected = []
    for rank in range(3):
        monkeypatch.setattr(module.dist, 'get_rank', lambda rank=rank: rank)
        loader = create_data_loader(cfg, 'val')
        selected.extend(loader.dataset.indices[i] for i in loader.batch_sampler.sampler)
    assert len(selected) == len(set(selected)) == 9
    other = create_data_loader(cfg, 'val')
    assert loader.dataset.indices == other.dataset.indices
    cfg.seed = 123  # Training randomness can change without changing the fixed validation subset.
    assert create_data_loader(cfg, 'val').dataset.indices == other.dataset.indices


def validation_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        valid = torch.zeros(2, 1, 9, dtype=torch.bool)
        valid[0, 0, 0], valid[1] = True, True
        actions = torch.tensor([9., 1.])[:, None, None].expand(2, 1, 9)
        local = [dict(inputs={}, targets=dict(actions=actions[rank:rank+1], action_valid=valid[rank:rank+1]))]
        result = validation_losses(LossModel(), local, config())
        assert result['action'] == pytest.approx(1.8)
        # The independently evaluated global batch has the same loss on each rank.
        with mock.patch('openpi.training.validation.dist.is_initialized', return_value=False):
            expected = validation_losses(LossModel(), [dict(inputs={}, targets=dict(actions=actions, action_valid=valid))], config())
        assert result == expected
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='Requires Gloo')
def test_validation_ddp_matches_the_global_valid_item_mean(tmp_path):
    mp.spawn(validation_worker, args=((tmp_path/'rendezvous').as_uri(),), nprocs=2, join=True)
