import pytest
import torch
from torch.nn import functional as F

from openpi.models.lora import apply_grounding_lora
from openpi.models.motion_planner import CrossAttention, normalize_prior
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill


def build(joint=False):
    torch.manual_seed(12)
    conf = ESkillConfig.tiny(current_grounding='motion_planner', planner_joint=joint,
                             motion_terminal_hold=True, vocab_size=90)
    model = Pi0ESkill(conf)
    apply_grounding_lora(model, 2, 2)
    model.configure_stage('motion_planner')
    inputs = dict(images=torch.randn(3, 1, 3, 32, 32), image_mask=torch.ones(3, 1).bool(),
                  tokens=torch.tensor([[1, 6, 0], [1, 5, 2], [7, 0, 0]]),
                  token_mask=torch.tensor([[1, 1, 0], [1, 1, 1], [1, 0, 0]]).bool())
    targets = dict(skill_labels=torch.tensor([1, 0, -1]), skill_valid=torch.tensor([1, 1, 0]).bool(),
                   trace=torch.rand(3, 8, 2), trace_valid=torch.ones(3, 8).bool(),
                   trace_terminal_hold=torch.zeros(3, 8).bool(), current_mask=torch.zeros(3, 32, 32),
                   current_box=torch.tensor([[0., 0., .25, .25]]).repeat(3, 1),
                   center_px_scale=torch.full((3, 2), 32.))
    targets['current_mask'][:, :8, :8] = 1
    targets['trace_valid'][-1] = False
    return model, inputs, targets


def test_prefix_isolation_query_calibration_and_independent_readouts():
    model, inputs, targets = build()
    assert not hasattr(model.paligemma_with_expert, 'gemma_expert') and not hasattr(model, 'action_in_proj')
    assert not hasattr(model.grounding, 'subject_decoder') and not hasattr(model.grounding, 'gac_query')
    ref = model.encode_observation(inputs, attention_layers=model.config.attention_layers)
    n = ref.visual_tokens.size(1)
    expected = model.grounding.read_current(ref.context[:, n:], inputs['token_mask'], ref.text_image_attention)
    before = model.plan_features(inputs)
    for key in ('current_map', 'alpha', 'current_query', 'layer_maps'):
        torch.testing.assert_close(before[key], expected[key])
    torch.testing.assert_close(before['visual_hidden'], ref.context[:, :n])
    stats = model.calibrate_motion_queries([{'inputs': inputs, 'targets': targets}])
    assert stats['motion_input_rms'] == pytest.approx(stats['text_rms'], rel=1e-5)
    assert stats['query_to_time_rms'] == pytest.approx(1.)
    assert stats['xy_corner_l2'] > stats['xy_neighbor_l2'] > 0
    assert sum(stats[f'first_layer_{k}_mass'] for k in ('image','text','motion')) == pytest.approx(1.)
    padded = dict(inputs, tokens=F.pad(inputs['tokens'], (0, 4)), token_mask=F.pad(inputs['token_mask'], (0, 4)))
    pred = model.predict_plan(inputs)
    for key in ('current_map', 'motion_hidden', 'trace', 'skill_logits'):
        torch.testing.assert_close(model.predict_plan(padded)[key], pred[key], atol=1e-5, rtol=1e-5)
    anti = model.plan_from_features(pred, current_prior=pred['current_map'].roll(3, -1))
    difference = anti['first_readout'] - pred['first_readout']
    assert difference.norm() > 0
    torch.testing.assert_close(difference, difference[:, :1].expand_as(difference), atol=3e-6, rtol=1e-5)


def test_prior_mass_mask_fallback_and_exact_output_projection():
    torch.manual_seed(7)
    attn = CrossAttention(12, 12, 3)
    memory, query = torch.randn(3, 6, 12), torch.randn(3, 8, 12)
    valid = torch.ones(3, 6).bool()
    valid[:, -1] = False
    raw = torch.zeros(3, 6)
    raw[0, 1] = 1e-25
    raw[1, -1] = 10.
    raw[2, 3] = 2.
    p = normalize_prior(raw, valid)
    torch.testing.assert_close(p[:2], valid[:2].float() / 5)
    assert p[2, 3] == 1.
    got = attn(query, memory, valid, p, .3)
    q, k, v = map(attn.heads_of, (attn.q(query), attn.k(memory), attn.v(memory)))
    logits = (q @ k.transpose(-1, -2)) / 2
    a = logits.masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    mix = (.7 * a + .3 * p[:, None, None]) @ v
    expected = attn.out(mix.transpose(1, 2).flatten(2))
    torch.testing.assert_close(got, expected)
    with pytest.raises(ValueError):
        normalize_prior(raw, torch.zeros_like(valid))


def test_calibration_counts_value_pairs_across_all_microbatches():
    model, inputs, _ = build()
    boxes = torch.tensor([[[0., 0., .25, .25], [.75, .75, 1., 1.]]])
    def batches():
        for i in range(3):
            yield dict(inputs={k: v[i:i + 1] for k, v in inputs.items()},
                       targets=dict(positive_entity=torch.tensor([0 if i == 0 else -1]),
                                    entity_boxes=boxes, entity_valid=torch.ones(1, 2, dtype=torch.bool)))
    report = model.calibrate_motion_queries(batches)
    assert report['diagnostic_samples'] == report['calibration_samples'] == 3
    assert report['target_other_pairs'] == 1 and report['target_other_value_l2'] > 0


@pytest.mark.parametrize('joint', [False, True])
def test_gradients_sparse_experts_and_router_warmup(joint):
    model, inputs, targets = build(joint)
    losses = model(inputs, targets, router_only=True)
    assert set(losses) == {'skill'}
    losses['skill'].backward()
    assert model.planner.router[-1].weight.grad is not None
    assert model.planner.native_queries.grad is None
    model.zero_grad(set_to_none=True)
    route = model.predict_plan(inputs)['skill_id'][:2].tolist()
    losses = model(inputs, targets)
    sum(losses.values()).backward()
    assert model.planner.native_queries.grad.norm(dim=1).gt(0).all()
    assert (model.grounding.alpha_scorer[-1].weight.grad is not None) == joint
    proj = model.paligemma_with_expert.paligemma.language_model.layers[-1].self_attn.q_proj
    assert (proj.lora_B.grad is not None) == joint
    for layer in model.planner.layers:
        for k, expert in enumerate(layer.experts):
            assert (expert.down.weight.grad is not None) == (k in route)
    model.zero_grad(set_to_none=True)
    targets['trace_valid'].zero_()
    sum(model(inputs, targets).values()).backward()
    assert all(e.down.weight.grad is None for l in model.planner.layers for e in l.experts)


def test_planner_base_load_uses_only_paligemma(tmp_path):
    from safetensors.torch import save_file
    from openpi.models.pi0 import Pi0
    cfg = ESkillConfig.tiny(current_grounding='motion_planner', vocab_size=90)
    original = Pi0(cfg)
    state = {name: p.detach().clone() for name, p in original.state_dict().items()}
    path = tmp_path / 'base.safetensors'
    save_file(state, path)
    model = Pi0ESkill(cfg)
    model.load_base_weights(path)
    for key in model._base_keys:
        torch.testing.assert_close(model.state_dict()[key], state[key], atol=0, rtol=0)


def test_joint_trace_supervises_prior_and_native_queries_through_probability_fusion():
    model, inputs, targets = build(joint=True)
    model(inputs, targets)['trace'].backward()
    assert model.grounding.alpha_scorer[-1].weight.grad.norm() > 0
    assert model.planner.native_queries.grad.norm(dim=-1).gt(0).all()
    assert model.planner.router[-1].weight.grad is None
    layer = model.paligemma_with_expert.paligemma.language_model.layers[-1]
    assert layer.self_attn.q_proj.lora_B.grad.norm() > 0


def test_bfloat16_frozen_backbone_keeps_native_query_gradient_with_checkpointing():
    model, inputs, targets = build()
    model.paligemma_with_expert.to(torch.bfloat16)
    from dataclasses import replace
    model.config = replace(model.config, gradient_checkpointing=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        model.calibrate_motion_queries(inputs)
        losses = model(inputs, targets)
    losses['trace'].backward()
    assert model.planner.native_queries.grad.norm(dim=-1).gt(0).all()
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())
