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
from openpi.training.metrics import reduce_diagnostics


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


def test_nearest_prior_counts_repeated_stops_once_and_masks_invalid_points():
    points = torch.tensor([[[.125, .125], [.875, .875]]], requires_grad=True)
    repeated = torch.cat((points[:, :1], points[:, 1:].expand(-1, 9, -1)), dim=1)
    expected = trajectory_prior(points, 16)
    torch.testing.assert_close(trajectory_prior(repeated, 16), expected)
    torch.testing.assert_close(expected[0, 0], expected[0, 15])
    assert trajectory_prior(repeated, 16, legacy_mean=True)[0, 15] > expected[0, 15]
    invalid = torch.tensor([[[float('nan'), 0.], [2., 0.]]])
    torch.testing.assert_close(trajectory_prior(torch.cat((points, invalid), 1), 16), expected)
    actual = trajectory_prior(torch.cat((points, invalid), 0), 16)
    torch.testing.assert_close(actual.sum(1), torch.tensor([1., 0.]))
    actual[:, 1].sum().backward()
    assert torch.isfinite(points.grad).all()


def test_historical_action_checkpoint_restores_original_protocol():
    from openpi.policies.policy_config import load_train_config

    cfg = TrainConfig('', '', '', '', model=ESkillConfig.tiny(current_grounding='action_execution'))
    saved = cfg.to_dict()
    for key in ('action_text_kv', 'action_motion_prior'):
        del saved['model'][key]
    restored = load_train_config({'config': saved}).model
    assert restored.action_text_kv and restored.action_motion_prior == 'legacy_mean_8'
    assert not cfg.model.action_text_kv and cfg.model.action_motion_prior == 'nearest_10'


@pytest.mark.parametrize('learned_alpha', [True, False], ids=['learned', 'uniform'])
def test_one_native_pass_and_final_ten_slots_for_training_and_sampling(tmp_path, learned_alpha):
    cfg = ESkillConfig.tiny(current_grounding='action_execution', vocab_size=72, train_planner=True,
                            learned_alpha=learned_alpha)
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
    mismatched = replace(params, model=replace(cfg, learned_alpha=not learned_alpha))
    with pytest.raises(ValueError, match=r'model\.learned_alpha'):
        create_model(mismatched, init_checkpoint=path)
    model = create_model(params, init_checkpoint=path)
    assert hasattr(model.grounding, 'alpha_scorer') == learned_alpha
    torch.testing.assert_close(model.planner.native_queries, planner.planner.native_queries)
    inputs = dict(images=torch.randn(2, 2, 3, 32, 32), image_mask=torch.tensor([[True, False], [True, True]]),
                  tokens=torch.ones(2, 4, dtype=torch.long), token_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]).bool())
    targets = dict(skill_labels=torch.tensor([0, 1]), skill_valid=torch.tensor([True, False]),
                   trace=torch.rand(2, 8, 2), trace_valid=torch.ones(2, 8).bool(),
                   trace_terminal_hold=torch.zeros(2, 8).bool(), boundary_xy=torch.rand(2, 2, 2),
                   boundary_valid=torch.ones(2, 2).bool(), center_px_scale=torch.full((2, 2), 32),
                   current_box=torch.rand(2, 4), current_mask=torch.rand(2, 32, 32),
                   actions=torch.rand(2, 10, 32), action_valid=torch.ones(2, 10, 32).bool(),
                   action_sample_valid=torch.tensor([True, False]))
    with patch.object(model, 'encode_observation', wraps=model.encode_observation) as encoder:
        losses = model(inputs, targets, action_diagnostics=True)
        assert encoder.call_count == 1
        losses['action'].backward()
        assert any(p.grad is not None for p in model.planner.parameters())
        stats = reduce_diagnostics(model.diagnostic_statistics)
        assert stats['action_attention_count'] == 10
        for i in range(cfg.depth):
            masses = [stats[f'action_layer_{i:02d}_{role}_mass'] for role in ('main', 'wrist', 'text', 'motion', 'self')]
            assert sum(masses) == pytest.approx(1.)
            assert masses[1:3] == [0., 0.]  # Only the first sample has action supervision and its wrist is absent.
        encoder.reset_mock()
        actions, prediction = model.sample_actions(inputs, steps=3)
        assert encoder.call_count == 1 and actions.shape == (2, 10, 32)
        assert prediction['event_motion_latents'].shape == (2, 10, cfg.hidden_dim)
    condition, _ = model.predict_condition(inputs)
    assert condition.mask.shape == (2, 2 * 16 + 4 + 10)
    assert not condition.mask[0, 16:32].any()
    assert not condition.mask[:, 32:36].any() and condition.mask[:, 36:].all()
    torch.testing.assert_close(condition.position_mask[:, 32:36], inputs['token_mask'])
    # Paired interventions reuse the host observation, preserving its native K/V.
    features = model.plan_features(inputs, cache_prefix=True)
    if not learned_alpha:
        torch.testing.assert_close(features['alpha'], torch.tensor([[.5, .5, 0., 0.], [.25, .25, .25, .25]]))
        prefix = features['prefix']
        text = prefix.context[:, 32:36].float()
        expected_query = torch.stack([tokens[valid].mean(0) for tokens, valid in zip(text, inputs['token_mask'])])
        torch.testing.assert_close(features['current_query'], expected_query)
    donor = model.plan_from_features(features, skill_id=torch.tensor([1, 0]))
    with patch.object(model, 'encode_observation', side_effect=AssertionError('re-encoded the host')):
        changed, prediction = model.condition_from_features(features, current_prior=features['current_map'].flip(-1),
                                                            motion_plan=donor)
    torch.testing.assert_close(prediction['event_motion_latents'], donor['event_motion_latents'])
    torch.testing.assert_close(changed.keys[0][:, :, :-10], features['prefix'].keys[0])
    # Editing cached direct text alone cannot change the masked executor's velocity.
    changed_keys, changed_values = [[tensor.clone() for tensor in tensors] for tensors in (changed.keys, changed.values)]
    for key, value in zip(changed_keys, changed_values, strict=True):
        key[:, :, 32:36] += 20
        value[:, :, 32:36] += 50
    inputs_x, time = torch.randn(2, 10, cfg.action_dim), torch.ones(2)
    with torch.no_grad():
        expected = model.velocity(inputs_x, time, changed)
        torch.testing.assert_close(model.velocity(inputs_x, time, replace(changed, keys=changed_keys, values=changed_values)),
                                   expected)
    # Isolate boundary information: all future points invalid, two distinct valid endpoints.
    boundaries = dict(donor, start=torch.zeros(2, 2), end=torch.ones(2, 2), trace=torch.full((2, 8, 2), float('nan')))
    prior = model.execution(features['prefix'], boundaries, torch.zeros(2, 16)).attention.keywords['prior']
    torch.testing.assert_close(prior.sum(1), torch.full((2,), cfg.action_motion_ratio))
    torch.testing.assert_close(prior[:, 0], prior[:, -1])
    assert prediction['skill_id'].tolist() == [1, 0]
    assert model.loss_counts['action'] == 320
    if research.ACTION_ABLATIONS:
        from openpi.models.action_guidance import ActionGuidance

        text_enabled = ActionGuidance(replace(cfg, action_text_kv=True))
        text_enabled.load_state_dict(model.execution.state_dict())
        visible = text_enabled(features['prefix'], donor, features['current_map'])
        torch.testing.assert_close(visible.mask, visible.position_mask)
        torch.testing.assert_close(visible.position_mask, changed.position_mask)
        for left, right in zip(visible.keys, changed.keys, strict=True):
            torch.testing.assert_close(left, right)  # Includes RoPE on all ten final motion keys.
        for kv, current, motion in [(False, 0., 0.), (True, .3, 0.), (True, 0., .3)]:
            ablation = create_model(replace(params, model=replace(cfg, action_motion_kv=kv, action_current_ratio=current,
                                                                  action_motion_ratio=motion)))
            cond, _ = ablation.predict_condition(inputs)
            assert cond.mask.shape[-1] == 36 + 10 * int(kv)
            assert (cond.attention is not None) == bool(current or motion)
    else:
        with pytest.raises(ValueError, match='unified'):
            replace(cfg, action_motion_kv=False)

@pytest.mark.parametrize('learned_alpha', [True, False], ids=['learned', 'uniform'])
def test_action_clock_training_resume_and_online_policy(tmp_path, learned_alpha):
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
                  current_grounding='action_execution', train_planner=True, learned_alpha=learned_alpha), loss_weights=None,
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
    last = json.loads((path.parent / 'metrics.jsonl').read_text().splitlines()[-1])
    assert last['current_training'] == dict(vlm_trainable_parameters=0, current_trainable_parameters=0,
                                            current_align_weight=0.)
    assert last['aux']['action_text_attention_mass'] == 0. and last['aux']['action_attention_count'] > 0
    assert torch.load(path, weights_only=False, mmap=True)['step'] == 2
    policy = create_trained_policy(path, device='cpu')
    assert hasattr(policy.model.grounding, 'alpha_scorer') == learned_alpha
    result = policy.infer(dict(images=[arrays['rgb'][0], None], prompt=meta['instructions'][0]), steps=2)
    assert result['actions'].shape == (10, 7) and np.isfinite(result['actions']).all()
