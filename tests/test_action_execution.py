from dataclasses import replace
from unittest.mock import patch

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
