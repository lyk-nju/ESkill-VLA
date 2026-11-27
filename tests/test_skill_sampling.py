"""Skill sampling follows observation/Event pairing, train splits and resumable rank sharding."""
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.policies.eskill_dataset import EpisodeDataset, EpisodeReader
from openpi.training.config import TrainConfig
from openpi.training.data_loader import (ResumableBatchSampler, SkillWeightedSampler, TransformedDataset,
                                         create_data_loader, skill_sampling_weights)
from scripts.preprocess.export import write_index
from test_training_data import saved_episode


def saved_data(tmp_path, stride=1):
    def all_pick(meta, arrays):
        meta['events'][1]['skill'] = 'Pick'
    def all_push(meta, arrays):
        for event in meta['events']:
            event['skill'] = 'Push'
    index = tmp_path / 'index.json'
    write_index(index, [saved_episode(tmp_path, 'one'), saved_episode(tmp_path, 'two', edit=all_pick),
                       saved_episode(tmp_path, 'val', edit=all_push)])
    return EpisodeDataset(index, action_horizon=4, auxiliary=False,
                          episode_ids={'one', 'two'}, frame_stride=stride)


@pytest.mark.parametrize('stride,pick,place', [(1, 9, 3), (2, 5, 1)])
def test_weights_use_train_rows_and_paired_observations_without_rgb(tmp_path, stride, pick, place):
    data = saved_data(tmp_path, stride)
    with mock.patch.object(EpisodeReader, 'rgb', side_effect=AssertionError('Sampler decoded RGB')):
        weights, report = skill_sampling_weights(data)
    assert report['skills']['pick']['count'] == pick
    assert report['skills']['place']['count'] == place
    assert report['skills']['push']['count'] == 0  # Validation episodes are not counted.
    assert weights[-1] == 1  # All-Pick episode.
    assert weights[2 if stride == 1 else 1] == 1  # Action frame 1 still observes frame 0 (Pick).
    assert weights[3 if stride == 1 else 2] == pytest.approx(np.sqrt(pick / place))
    assert report['samples'] == pick + place


def test_cap_missing_labels_and_two_arm_samples():
    events = [dict(skill=s, boundary_valid=True, skill_valid=True)
              for s in ('Pick', 'Push', 'Hold', 'unknown')]
    events += [dict(skill='Push', boundary_valid=False, skill_valid=True)]
    # Nine common arms, one rare arm, then unsupported/missing/invalid annotations.
    left = np.array([0] * 9 + [2, -1, 4])
    right = np.array([-1] * 8 + [1, 3, -1, -1])
    reader = SimpleNamespace(meta=dict(actions={'conditioning_image_indices': {'image': np.arange(12)}},
        cameras={'image': {'frame_indices': np.arange(12)}}, events=events, arm_ids=['left', 'right'],
        event_index={'left': left, 'right': right}), array=lambda ref: ref)
    data = SimpleNamespace(samples=[(0, i) for i in range(12)], skills={}, active_skill_ids=set(range(6)),
                           camera='image', reader=lambda _: reader)
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.samples)
    weights, report = skill_sampling_weights(Dataset(**vars(data)))
    assert weights.tolist() == [1.] * 8 + [3.] + [1.] * 3
    assert report['samples_without_skill'] == 3
    assert report['skills']['push']['weight'] == 3
    assert report['skills']['pick']['weight'] == 1


def test_global_draw_is_sharded_and_resume_replays_remaining_batches():
    weights = torch.arange(1, 24, dtype=torch.double)
    samplers = [SkillWeightedSampler(weights, seed=42, num_replicas=4, rank=r) for r in range(4)]
    state = torch.random.get_rng_state().clone()
    local = [list(s) for s in samplers]
    assert [len(s) for s in samplers] == [6] * 4
    reconstructed = torch.tensor(list(zip(*local))).flatten()
    expected = torch.multinomial(weights, 24, replacement=True, generator=torch.Generator().manual_seed(42))
    torch.testing.assert_close(reconstructed, expected)
    torch.testing.assert_close(state, torch.random.get_rng_state())
    assert local[0] != local[1]
    assert list(samplers[0]) == local[0]
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != local[0]
    batches = ResumableBatchSampler(samplers[0], 2, drop_last=False)
    complete = list(batches)
    batches.start_batch = 1
    assert list(batches) == complete[1:]
    assert len(batches) == len(complete) - 1


@pytest.mark.parametrize('world', [1, 2, 8])
def test_loader_enables_weighting_only_for_training(tmp_path, monkeypatch, world):
    data = saved_data(tmp_path)
    wrapped = TransformedDataset(data, lambda x: x)
    monkeypatch.setattr('openpi.training.data_loader.create_dataset', lambda *args: wrapped)
    monkeypatch.setattr('openpi.training.data_loader.dist.is_initialized', lambda: world > 1)
    monkeypatch.setattr('openpi.training.data_loader.dist.get_world_size', lambda: world)
    monkeypatch.setattr('openpi.training.data_loader.dist.get_rank', lambda: world - 1)
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused', num_workers=0,
                      model_type='pi05', device='cpu', batch_size=2, val_max_batches=2, skill_sampling=True)
    train = create_data_loader(cfg)
    assert isinstance(train.batch_sampler.sampler, SkillWeightedSampler)
    assert train.batch_sampler.sampler.summary['draws_per_epoch'] == (len(data) + world - 1) // world * world
    monkeypatch.setattr('openpi.training.data_loader.skill_sampling_weights',
                        lambda *args: pytest.fail('Validation or uniform sampling computed skill weights'))
    val = create_data_loader(cfg, 'val')
    uniform = create_data_loader(replace(cfg, skill_sampling=False))
    uniform_val = create_data_loader(replace(cfg, skill_sampling=False), 'val')
    assert not isinstance(uniform.batch_sampler.sampler, SkillWeightedSampler)
    assert val.dataset.indices == uniform_val.dataset.indices
    assert list(val.batch_sampler.sampler) == list(uniform_val.batch_sampler.sampler)
