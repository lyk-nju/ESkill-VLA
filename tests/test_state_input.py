"""Disabled numeric state cannot influence ESkill predictions or losses."""
from dataclasses import replace

import pytest
torch = pytest.importorskip('torch')

from openpi.models.pi0_eskill import Pi0ESkill
from test_eskill_model import model, batch, has_gradient


def test_no_state_eskill_ignores_missing_or_changed_numeric_state(model):
    model = Pi0ESkill(replace(model.config, use_arm_state=False, discrete_state_input=False),
                       vae=model.visual_supervision.vae).eval()
    assert model.state_projection is None
    assert not any(key.startswith('state_projection.') for key in model.state_dict())
    inputs, targets = batch(model.config)
    del inputs['arm_state']
    noise = torch.randn_like(targets['actions'])
    time = torch.full((len(noise),), .5)
    with torch.no_grad():
        losses = model(inputs, targets, noise=noise, time=time)
        actions, prediction = model.sample_actions(inputs, noise=noise, steps=2)
        changed = dict(inputs, state=torch.full((2, 32), float('nan')),
                       arm_state=torch.full((2, 2, 15), float('nan')))
        changed_losses = model(changed, targets, noise=noise, time=time)
        changed_actions, changed_prediction = model.sample_actions(changed, noise=noise, steps=2)
    torch.testing.assert_close(actions, changed_actions, rtol=0, atol=0)
    for key in prediction:
        torch.testing.assert_close(prediction[key], changed_prediction[key], rtol=0, atol=0)
    for key in losses:
        torch.testing.assert_close(losses[key], changed_losses[key], rtol=0, atol=0)
    sum(model(inputs, targets, noise=noise, time=time).values()).backward()
    for module in (model.grounding, model.motion, model.skill.future_head, model.action_out_proj):
        assert has_gradient(module)


def test_no_state_recipe_disables_both_numeric_input_routes():
    from pathlib import Path
    from openpi.training.config import TrainConfig
    recipe = Path(__file__).resolve().parents[1] / 'examples/libero/train_eskill_no_state.json'
    config = TrainConfig.load(recipe)
    assert not config.model.use_arm_state and not config.model.discrete_state_input
    assert config.run_name.endswith('_no_state')
