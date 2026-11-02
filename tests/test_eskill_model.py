"""Five-loss gradients, no target leakage, arm masks, cache and checkpoint contracts."""
from unittest import mock

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    cfg = ESkillConfig.tiny(vocab_size=128, max_token_len=8, gradient_checkpointing=True)
    vae = AutoencoderKL(
        down_block_types=("DownEncoderBlock2D",) * 2, up_block_types=("UpDecoderBlock2D",) * 2,
        block_out_channels=(16, 32), norm_num_groups=8, layers_per_block=1, latent_channels=4)
    result = Pi0ESkill(cfg, vae=vae)
    tokens = torch.arange(9)[:, None].expand(9, 5) + 1
    result.skill.text_prototypes.copy_(result.text_prototypes(tokens, torch.ones_like(tokens, dtype=torch.bool)))
    return result


def batch(cfg):
    b, h, m, s = 2, cfg.action_horizon, cfg.motion_queries, cfg.canvas_size
    inputs = dict(images=torch.randn(b, 1, 3, cfg.image_size, cfg.image_size),
                  image_mask=torch.ones(b, 1, dtype=torch.bool), tokens=torch.ones(b, 8, dtype=torch.long),
                  token_mask=torch.ones(b, 8, dtype=torch.bool), arm_presence=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
                  arm_state=torch.randn(b, 2, 15))
    targets = dict(
        actions=torch.randn(b, h, cfg.action_dim), action_valid=torch.ones(b, h, cfg.action_dim, dtype=torch.bool),
        skill_labels=torch.zeros(b, 2, dtype=torch.long), skill_valid=torch.ones(b, 2, dtype=torch.bool),
        future_skill_labels=torch.ones(b, 2, h, dtype=torch.long), future_skill_valid=torch.ones(b, 2, h, dtype=torch.bool),
        role_images=torch.randn(b, 2, 3, 3, s, s), role_weights=torch.ones(b, 2, 3, s, s),
        role_valid=torch.ones(b, 2, 3, dtype=torch.bool), trace=torch.rand(b, 2, m, 2),
        trace_valid=torch.ones(b, 2, m, dtype=torch.bool))
    return inputs, targets


def has_gradient(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_all_five_losses_backpropagate_and_frozen_modules_stay_frozen(model):
    inputs, targets = batch(model.config)
    losses = model(inputs, targets)
    assert set(losses) == {"action", "skill", "future", "grounding", "trace"}
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    for module in (model.skill.current_decoder, model.skill.future_decoder, model.skill.table,
                   model.grounding, model.visual_supervision.head, model.motion,
                   model.step_projection, model.action_out_proj):
        assert has_gradient(module)
    assert model.skill.text_prototypes.grad is None
    assert all(p.grad is None for p in model.visual_supervision.vae.parameters())
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())


def test_hard_selection_gradient_boundary_and_future_loss_reaches_event_features(model):
    inputs, targets = batch(model.config)
    losses = model(inputs, targets)
    losses["action"].backward()
    assert not has_gradient(model.skill.future_head)
    assert not has_gradient(model.skill.semantic_projection)
    assert has_gradient(model.skill.table)
    model.zero_grad(set_to_none=True)
    model(inputs, targets)["future"].backward()
    assert has_gradient(model.grounding)
    assert has_gradient(model.motion)
    assert has_gradient(model.skill.future_head)


def test_visual_loss_reaches_all_roles_without_action_or_vae_gradients(model):
    inputs, targets = batch(model.config)
    model(inputs, targets)['grounding'].backward()
    assert (model.grounding.queries.grad.abs().sum((1, 2)) > 0).all()
    assert has_gradient(model.visual_supervision.head)
    assert not has_gradient(model.action_out_proj)
    assert not has_gradient(model.visual_supervision.vae)


def test_future_skill_injection_starts_at_zero_and_learns_from_action_loss(model):
    inputs, targets = batch(model.config)
    noise = torch.randn_like(targets["actions"])
    time = torch.full((len(noise),), .5)
    optimizer = torch.optim.SGD(model.step_projection.parameters(), lr=.01)
    with mock.patch.object(model.step_projection, "forward", wraps=model.step_projection.forward) as project:
        condition, _ = model.predict_condition(inputs)
        vectors = project.call_args.args[0]
        vectors.retain_grad()
        assert torch.count_nonzero(condition.step_skill) == 0
        model.flow_loss(targets["actions"], targets["action_valid"], condition, noise=noise, time=time).backward()
        assert model.step_projection.weight.grad.abs().sum() > 0
        assert model.step_projection.bias.grad.abs().sum() > 0
        assert torch.count_nonzero(vectors.grad) == 0
        optimizer.step()
        model.zero_grad(set_to_none=True)

        updated, _ = model.predict_condition(inputs)
        vectors = project.call_args.args[0]
        vectors.retain_grad()
        assert torch.count_nonzero(updated.step_skill) > 0
        # Only the additive projection changed; shared event attention is preserved.
        torch.testing.assert_close(updated.mask, condition.mask, rtol=0, atol=0)
        for before, after in zip(condition.keys + condition.values, updated.keys + updated.values, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        model.flow_loss(targets["actions"], targets["action_valid"], updated, noise=noise, time=time).backward()
        assert vectors.grad.abs().sum() > 0


def test_online_conditions_independent_of_absent_arm_state_and_cached_once(model):
    model.eval()
    inputs, _ = batch(model.config)
    noise = torch.randn(2, model.config.action_horizon, model.config.action_dim)
    with mock.patch.object(model, "predict_condition", wraps=model.predict_condition) as predict:
        actions, _ = model.sample_actions(inputs, noise=noise, steps=3)
        assert predict.call_count == 1
    inputs["arm_state"][0, 1] += 1000
    other, _ = model.sample_actions(inputs, noise=noise, steps=3)
    torch.testing.assert_close(actions[0], other[0])


def test_invalid_supervision_is_masked_without_training_idle(model):
    inputs, targets = batch(model.config)
    for key in ("action_valid", "skill_valid", "future_skill_valid", "role_valid", "trace_valid"):
        targets[key].zero_()
    targets["skill_labels"].fill_(-1)
    targets["future_skill_labels"].fill_(-1)
    losses = model(inputs, targets)
    assert all(value == 0 for value in losses.values())
    sum(losses.values()).backward()


def test_base_checkpoint_load_requires_all_parameters(model, tmp_path):
    base = {key: value.detach().clone() for key, value in model.state_dict().items() if key in model._base_keys}
    path = tmp_path / "model.safetensors"
    save_file(base, str(path))
    other = Pi0ESkill(model.config)
    other.load_base_weights(path)
    for key in model._base_keys:
        torch.testing.assert_close(other.state_dict()[key], base[key])
    del base["action_in_proj.weight"]
    save_file(base, str(path))
    with pytest.raises(ValueError, match="action_in_proj"):
        other.load_base_weights(path)


def test_saved_inference_omits_only_visual_training_modules(model):
    model.eval()
    other = Pi0ESkill(model.config).eval()
    other.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("visual_supervision.")})
    assert other.visual_supervision is None
    inputs, _ = batch(model.config)
    noise = torch.randn(2, model.config.action_horizon, model.config.action_dim)
    expected = model.sample_actions(inputs, noise=noise, steps=2)[0]
    actual = other.sample_actions(inputs, noise=noise, steps=2)[0]
    torch.testing.assert_close(expected, actual)


def test_mixed_precision_forward_and_backward(model):
    inputs, targets = batch(model.config)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        losses = model(inputs, targets)
        loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert has_gradient(model.action_out_proj)


def test_eight_motion_queries_are_independent_of_sixteen_grounding_queries():
    config = ESkillConfig.tiny(vocab_size=128, role_queries=16)
    assert config.motion_offsets == (4, 8, 12, 16, 20, 24, 28, 32)
    model = Pi0ESkill(config).eval()
    inputs, _ = batch(config)
    with mock.patch.object(model.motion.trace_head, 'forward', side_effect=AssertionError('Inference ran trace head')):
        actions, prediction = model.sample_actions(inputs, steps=1)
    assert actions.shape == (2, config.action_horizon, config.action_dim)
    assert prediction['grounding'].shape == (2, 2, 3, 16, config.hidden_dim)
    assert prediction['motion'].shape == (2, 2, 8, config.hidden_dim)
    assert model.motion.trace_head(prediction['motion']).shape == (2, 2, 8, 2)


def test_trace_supervision_reaches_every_time_query_and_masks_nan_targets(model):
    inputs, targets = batch(model.config)
    targets['trace_valid'][0, 1] = False
    targets['trace'][0, 1] = float('nan')
    loss = model(inputs, targets)['trace']
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.motion.queries.grad).all()
    assert (model.motion.queries.grad.abs().sum(-1) > 0).all()
    assert has_gradient(model.motion.trace_head)
    assert has_gradient(model.grounding)
