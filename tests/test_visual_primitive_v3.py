"""V1/V2/V3 primitive information flow; model effects still need real training."""
from dataclasses import replace
from unittest import mock

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from openpi.models.grounding import Grounding
from openpi.models.motion import Motion
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill


def _inputs(config):
    return dict(
        images=torch.randn(2, 2, 3, config.image_size, config.image_size),
        image_mask=torch.ones(2, 2, dtype=torch.bool),
        tokens=torch.randint(1, config.vocab_size - 1, (2, 5)),
        token_mask=torch.ones(2, 5, dtype=torch.bool),
        arm_presence=torch.tensor([[True, False], [True, True]]),
    )


def _nonzero_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def test_version_validation_and_v0_backward_compatible_parameter_names():
    with pytest.raises(ValueError, match='visual_primitive_version'):
        ESkillConfig.tiny(visual_primitive_version='v4')
    torch.manual_seed(9)
    old = Pi0ESkill(ESkillConfig.tiny(vocab_size=128))
    assert old.config.visual_primitive_version == 'v0'
    assert not hasattr(old.grounding, 'text_attention')
    assert not hasattr(old.grounding, 'future_decoder')
    assert not hasattr(old.motion, 'goal_decoder')
    assert not hasattr(old.motion, 'spatial_projection')
    assert old.paligemma_with_expert is not None


@pytest.mark.parametrize('version', ('v0', 'v1', 'v2', 'v3'))
def test_versions_keep_the_same_action_and_auxiliary_tensor_contract(version):
    torch.set_num_threads(2)
    torch.manual_seed(13)
    config = ESkillConfig.tiny(vocab_size=128, visual_primitive_version=version)
    model = Pi0ESkill(config).eval()
    condition, prediction = model.predict_condition(_inputs(config))
    assert prediction['grounding'].shape == (2, 2, 2, config.role_queries, config.hidden_dim)
    assert prediction['motion'].shape == (2, 2, config.motion_queries, config.hidden_dim)
    assert condition.mask[:, -2 * config.motion_queries:].sum().item() == 3 * config.motion_queries
    assert torch.isfinite(prediction['grounding']).all()
    assert torch.isfinite(prediction['motion']).all()


def test_v1_text_can_change_current_grounding_with_fixed_scene_features():
    torch.manual_seed(17)
    config = ESkillConfig.tiny(visual_primitive_version='v1')
    model = Grounding(config).eval()
    context = torch.randn(2, 10, config.hidden_dim)
    text = torch.randn(2, 5, config.hidden_dim)
    mask = torch.ones(2, 10, dtype=torch.bool)
    text_mask = torch.ones(2, 5, dtype=torch.bool)
    arms = torch.randn(2, config.hidden_dim)
    a = model(context, mask, arms, text_context=text, text_mask=text_mask)
    b = model(context, mask, arms, text_context=text.flip(-1), text_mask=text_mask)
    assert not torch.allclose(a[:, :, 0], b[:, :, 0])
    with pytest.raises(ValueError, match='text tokens'):
        model(context, mask, arms)


def test_v2_future_uses_the_current_prediction():
    torch.manual_seed(19)
    config = ESkillConfig.tiny(visual_primitive_version='v2')
    model = Grounding(config).eval()
    context = torch.randn(1, 12, config.hidden_dim)
    text = torch.randn(1, 3, config.hidden_dim)
    mask = torch.ones(1, 12, dtype=torch.bool)
    tmask = torch.ones(1, 3, dtype=torch.bool)
    arms = torch.randn(2, config.hidden_dim)
    original = model(context, mask, arms, text_context=text, text_mask=tmask)
    original_forward = model.decoder.forward
    with mock.patch.object(
        model.decoder, 'forward',
        side_effect=lambda *a, **k: original_forward(*a, **k) + .1 * torch.randn_like(a[0]),
    ):
        changed = model(context, mask, arms, text_context=text, text_mask=tmask)
    assert not torch.allclose(original[:, :, 0], changed[:, :, 0])
    assert not torch.allclose(original[:, :, 1], changed[:, :, 1])


def test_v3_motion_is_goal_first_and_never_requires_fused_vlm_context():
    torch.manual_seed(23)
    config = ESkillConfig.tiny(visual_primitive_version='v3')
    motion = Motion(config).eval()
    b, d = 2, config.hidden_dim
    skill = torch.randn(b, 2, d)
    arms = torch.randn(2, d)
    presence = torch.tensor([[True, False], [True, True]])
    grounding = torch.randn(b, 2, 2, config.role_queries, d, requires_grad=True)
    spatial = torch.randn(b, 8, config.vlm_width)
    mask = torch.ones(b, 8, dtype=torch.bool)
    def run(objects):
        return motion(None, None, skill, arms, presence, grounding=objects,
                      spatial=spatial, spatial_mask=mask)
    baseline = run(grounding)
    new_current = grounding.detach().clone()
    new_current[0, 0, 0, 0, 0] += 3.
    new_future = grounding.detach().clone()
    new_future[0, 0, 1, 0, 1] += 3.
    assert not torch.allclose(baseline, run(new_current))
    assert not torch.allclose(baseline, run(new_future))
    baseline.square().sum().backward()
    assert grounding.grad[0, 0, 0].abs().sum() > 0
    assert grounding.grad[0, 0, 1].abs().sum() > 0
    assert _nonzero_grad(motion.goal_decoder) and _nonzero_grad(motion.spatial_projection)
    with pytest.raises(ValueError, match='Current/Future'):
        motion(None, None, skill, arms, presence)


def test_v3_action_loss_reaches_both_visual_roles_and_skill_without_state():
    torch.set_num_threads(2)
    torch.manual_seed(29)
    config = ESkillConfig.tiny(vocab_size=128, visual_primitive_version='v3')
    model = Pi0ESkill(config).eval()
    inputs = _inputs(config)
    condition, prediction = model.predict_condition(inputs)
    action = torch.randn(2, config.action_horizon, config.action_dim)
    loss = model.velocity(action, torch.full((2,), .5), condition).square().mean()
    loss.backward()
    for module in (model.grounding.decoder, model.grounding.future_decoder,
                   model.grounding.text_attention, model.motion.goal_decoder,
                   model.motion.spatial_projection, model.motion.layers,
                   model.skill.to_motion, model.auxiliary_kv):
        assert _nonzero_grad(module), type(module).__name__
    assert 'state' not in inputs and 'arm_state' not in inputs
    assert not hasattr(model, 'state_projection')
    assert prediction['motion'].shape[2] == config.motion_queries
