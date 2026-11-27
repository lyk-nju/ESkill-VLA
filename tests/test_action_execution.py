from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from openpi.models import research
from openpi.models.action_guidance import guided_probabilities, trajectory_prior
from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig, create_model
from openpi.training.checkpoints import save_checkpoint


def test_priors_redistribute_only_native_main_camera_mass():
    torch.manual_seed(9)
    q, k = torch.randn(2, 4, 3, 8), torch.randn(2, 2, 15, 8)
    valid = torch.ones(2, 15).bool()
    valid[0, 4:8] = False
    current = torch.tensor([[.15, 0, 0, 0], [0, .15, 0, 0]])
    trace = torch.tensor([[[.8, .9]], [[-1., -1.]]])
    prior = current + .15 * trajectory_prior(trace, 4)
    native = guided_probabilities(q, k, valid, torch.zeros_like(prior))
    guided = guided_probabilities(q, k, valid, prior)
    torch.testing.assert_close(guided[..., 4:], native[..., 4:])
    torch.testing.assert_close(guided[..., :4].sum(-1), native[..., :4].sum(-1))
    assert guided[0, :, :, 4:8].count_nonzero() == 0
    assert not torch.equal(guided[..., :4], native[..., :4])


def test_one_native_pass_and_final_ten_slots_for_training_and_sampling(tmp_path):
    cfg = ESkillConfig.tiny(current_grounding='action_execution', vocab_size=72, train_planner=True)
    params = TrainConfig('', '', '', '', model=cfg, smoke=True, device='cpu', validation_fraction=0,
                         cameras=('image', 'wrist_image'), precision='float32')
    source = replace(params, training_stage='motion_planner', loss_weights=None,
                     model=replace(cfg, current_grounding='motion_planner'))
    planner = create_model(source)
    planner.planner_calibration = {}
    path = tmp_path/'plan.pt'
    save_checkpoint(path, model=planner, optimizer=torch.optim.AdamW(planner.parameters()), config=source,
                    step=1, stats={}, resources={'pretrained_path': None, 'tokenizer_path': None})
    # Use explicit nonexistent fixture paths, avoiding cwd as an asset identity.
    params.tokenizer_path = 'unused-tokenizer'
    model = create_model(params, init_checkpoint=path)
    torch.testing.assert_close(model.planner.native_queries, planner.planner.native_queries)
    inputs = dict(images=torch.randn(2, 2, 3, 32, 32), image_mask=torch.tensor([[True, False], [True, True]]),
                  tokens=torch.ones(2, 4, dtype=torch.long), token_mask=torch.ones(2, 4).bool())
    targets = dict(skill_labels=torch.tensor([0, 1]), skill_valid=torch.tensor([True, False]),
                   trace=torch.rand(2, 8, 2), trace_valid=torch.ones(2, 8).bool(),
                   trace_terminal_hold=torch.zeros(2, 8).bool(), boundary_xy=torch.rand(2, 2, 2),
                   boundary_valid=torch.ones(2, 2).bool(), center_px_scale=torch.full((2, 2), 32),
                   current_box=torch.rand(2, 4), current_mask=torch.rand(2, 32, 32),
                   actions=torch.rand(2, 10, 32), action_valid=torch.ones(2, 10, 32).bool(),
                   action_sample_valid=torch.tensor([True, False]))
    with patch.object(model, 'encode_observation', wraps=model.encode_observation) as encoder:
        losses = model(inputs, targets)
        assert encoder.call_count == 1
        losses['action'].backward()
        assert any(p.grad is not None for p in model.planner.parameters())
        encoder.reset_mock()
        actions, prediction = model.sample_actions(inputs, steps=3)
        assert encoder.call_count == 1 and actions.shape == (2, 10, 32)
        assert prediction['event_motion_latents'].shape == (2, 10, cfg.hidden_dim)
    condition, _ = model.predict_condition(inputs)
    assert condition.mask.shape == (2, 2 * 16 + 4 + 10)
    assert not condition.mask[0, 16:32].any()
    # Paired interventions reuse the host observation, preserving its native K/V.
    features = model.plan_features(inputs, cache_prefix=True)
    donor = model.plan_from_features(features, skill_id=torch.tensor([1, 0]))
    with patch.object(model, 'encode_observation', side_effect=AssertionError('re-encoded the host')):
        changed, prediction = model.condition_from_features(features, current_prior=features['current_map'].flip(-1),
                                                            motion_plan=donor)
    torch.testing.assert_close(prediction['event_motion_latents'], donor['event_motion_latents'])
    torch.testing.assert_close(changed.keys[0][:, :, :-10], features['prefix'].keys[0])
    assert prediction['skill_id'].tolist() == [1, 0]
    assert model.loss_counts['action'] == 320
    if research.ACTION_ABLATIONS:
        for kv, current, motion in [(False, 0., 0.), (True, .3, 0.), (True, 0., .3)]:
            ablation = create_model(replace(params, model=replace(cfg, action_motion_kv=kv, action_current_ratio=current,
                                                                  action_motion_ratio=motion)))
            cond, _ = ablation.predict_condition(inputs)
            assert cond.mask.shape[-1] == 36 + 10 * int(kv)
            assert (cond.attention is not None) == bool(current or motion)
    else:
        with pytest.raises(ValueError, match='unified'):
            replace(cfg, action_motion_kv=False)

def test_action_clock_training_resume_and_online_policy(tmp_path):
    import sentencepiece
    from grounding_fixtures import ToyTokenizer
    from test_motion_planner_data import planner_dataset
    from openpi.augmentation import GroundingAugmentation
    from openpi.policies.grounding_dataset import GroundingDataset
    from openpi.policies.policy_config import create_trained_policy
    from scripts.train import train

    upstream = planner_dataset(tmp_path).config
    meta_path = tmp_path / 'episode.json'
    meta = json.loads(meta_path.read_text())

    def ref(key):
        return dict(uri='arrays.npz', key=key)

    meta['cameras']['wrist_image'] = dict(meta['cameras']['image'])
    meta['actions'] = dict(frame_indices=ref('action_frames'), values=ref('controls'), valid=ref('control_valid'),
                           conditioning_image_indices={'image': ref('pairing'), 'wrist_image': ref('wrist_pairing')})
    meta_path.write_text(json.dumps(meta))
    with np.load(tmp_path / 'arrays.npz') as archive:
        arrays = dict(archive)
    np.savez(tmp_path / 'arrays.npz', **arrays, action_frames=[0, 1, 2], pairing=[0, 0, 1], wrist_pairing=[0, -1, 1],
             controls=np.arange(21, dtype=np.float32).reshape(3, 7) / 21, control_valid=np.ones((3, 7), bool))
    corpus = tmp_path / 'task.txt'
    corpus.write_text(meta['instructions'][0])
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / 'tokenizer'),
                                             vocab_size=40, hard_vocab_limit=False, minloglevel=2)
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'actions': dict(method='quantile', q01=[-1]*7, q99=[1]*7)}))
    cfg = replace(upstream, training_stage='action_execution', model=replace(upstream.model,
                  current_grounding='action_execution', train_planner=True), loss_weights=None,
                  output_dir=str(tmp_path / 'run'), norm_stats=str(stats), tokenizer_path=str(tmp_path / 'tokenizer.model'),
                  cameras=('image', 'wrist_image'), batch_size=2, num_steps=1, num_workers=0, smoke=True,
                  device='cpu', precision='bfloat16', validation_fraction=0)
    dataset = GroundingDataset(cfg.data_index, cfg, ToyTokenizer())
    assert dataset.action_rows == [0, 1, 2] and dataset.samples[0] == dataset.samples[1]
    assert dataset[0]['inputs']['image_mask'].tolist() == [True, True]
    assert dataset[1]['inputs']['image_mask'].tolist() == [True, False]
    assert not np.array_equal(dataset[0]['targets']['actions'], dataset[1]['targets']['actions'])
    swapped = replace(cfg, grounding_augmentation=GroundingAugmentation(target_swap_prob=1.))
    row = GroundingDataset(cfg.data_index, swapped, ToyTokenizer())[2]
    assert row['metadata']['counterfactual'] == 'source' and row['inputs']['image_mask'].all()
    assert row['targets']['boundary_valid'].tolist() == [True, False]
    assert not row['targets']['action_sample_valid'] and not row['targets']['trace_valid'].any()

    path = train(cfg)
    train(replace(cfg, num_steps=2), resume=path)
    assert torch.load(path, weights_only=False, mmap=True)['step'] == 2
    policy = create_trained_policy(path, device='cpu')
    result = policy.infer(dict(images=[arrays['rgb'][0], None], prompt=meta['instructions'][0]), steps=2)
    assert result['actions'].shape == (10, 7) and np.isfinite(result['actions']).all()

def test_no_interface_matches_native_pi05_with_identical_weights_and_noise():
    from openpi.models.pi0 import Pi0

    cfg = ESkillConfig.tiny(current_grounding='action_execution', action_motion_kv=False,
                            action_current_ratio=0., action_motion_ratio=0., vocab_size=128)
    params = TrainConfig('', '', '', '', model=cfg, smoke=True, device='cpu',
                         cameras=('image', 'wrist_image'), precision='float32')
    model = create_model(params).eval()
    native = Pi0(cfg).eval()
    weights = model.state_dict()
    native.load_state_dict({key: weights[key] for key in native.state_dict()})
    data = dict(images=torch.randn(2, 2, 3, 32, 32), image_mask=torch.tensor([[True, True], [True, False]]),
                tokens=torch.ones(2, 4, dtype=torch.long), token_mask=torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]]).bool())
    noise = torch.randn(2, cfg.action_horizon, cfg.action_dim)
    expected, _ = native.sample_actions(data, steps=3, noise=noise.clone())
    actual, _ = model.sample_actions(data, steps=3, noise=noise.clone())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
