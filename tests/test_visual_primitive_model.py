"""Native SDPA parity, per-layer KL and trainable paths for Visual Primitive."""
import pytest
import torch
import torch.nn.functional as F

from openpi.models.gemma import rotary, text_image_attention
from openpi.models.lora import apply_grounding_lora
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.models.visual_primitive import box_coverage, patch_coverage, spatial_kl


def build(*, checkpoint=False, learned=True):
    torch.manual_seed(37)
    config = ESkillConfig.tiny(current_grounding='visual_primitive', grounding_subject=True, vocab_size=128,
                               gradient_checkpointing=checkpoint, learned_alpha=learned)
    model = Pi0ESkill(config)
    if learned:
        apply_grounding_lora(model, 2, 2)
    model.configure_stage('visual_primitive')
    inputs = dict(images=torch.randn(2, 1, 3, 32, 32), image_mask=torch.ones(2, 1, dtype=torch.bool),
                  tokens=torch.tensor([[1, 6, 2], [12, 9, 0]]),
                  token_mask=torch.tensor([[True, True, True], [True, True, False]]))
    target = torch.zeros(2, 32, 32)
    target[0, :13, :12] = 1
    subject = torch.zeros_like(target)
    subject[:, 10:17, 20:30] = 1
    boxes = torch.tensor([[[0., 0., .375, .40625], [.6, .5, .9, .9]]]).expand(2, -1, -1)
    targets = dict(current_mask=target, subject_mask=subject, current_box=boxes[:, 0],
                   subject_box=boxes[:, 1], center_px_scale=torch.full((2, 2), 32.),
                   entity_boxes=boxes, entity_valid=torch.ones(2, 2, dtype=torch.bool),
                   positive_entity=torch.tensor([0, -1]))
    return model, inputs, targets


def test_native_attention_matches_sdpa_after_rotary_with_gqa_and_padding():
    torch.manual_seed(1)
    b, h, nh, n, v = 2, 4, 2, 10, 6
    mask = torch.tensor([[1]*10, [1]*8+[0]*2]).bool()
    pos = (mask.long().cumsum(-1) - 1).clamp_min(0)
    q = rotary(torch.randn(b, h, n, 8), pos)
    k = rotary(torch.randn(b, nh, n, 8), pos)
    got = text_image_attention(q, k, mask, v)
    # Identity values expose native SDPA's individual key probabilities.
    keys = k.repeat_interleave(h // nh, dim=1)
    values = torch.eye(n).expand(b, h, n, n)
    expected = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask[:, None, None])[:, :, v:, :v]
    torch.testing.assert_close(got, expected * mask[:, None, v:, None], atol=1e-6, rtol=1e-5)
    assert (got.sum(-1) < 1).all()
    assert got[1, :, -2:].count_nonzero() == 0


def test_zero_lora_keeps_backbone_hidden_and_uniform_alpha():
    model, inputs, _ = build()
    model.eval()
    with torch.no_grad():
        pred = model.predict_grounding(inputs)
        prefix = model.encode_observation(inputs)
        torch.testing.assert_close(pred['vlm_context'], prefix.context, rtol=0, atol=0)
        torch.testing.assert_close(pred['visual_tokens'], prefix.visual_tokens, rtol=0, atol=0)
        from openpi.models.pi0 import Pi0
        base = Pi0(model.config).eval()
        base.load_state_dict({name: value for name, value in model.state_dict().items() if name in base._base_keys})
        torch.testing.assert_close(base.encode_observation(inputs).context, pred['vlm_context'], rtol=0, atol=0)
        alpha = inputs['token_mask'].float() / inputs['token_mask'].sum(1, keepdim=True)
        torch.testing.assert_close(pred['alpha'], alpha)
        raw = pred['native_attention']
        mass = raw.sum(-1)
        torch.testing.assert_close(mass, pred['visual_attention_mass'])
        expected = ((raw / mass.unsqueeze(-1).clamp_min(1e-30)) * alpha[:, None, None, :, None]).sum(3).mean(2)
        torch.testing.assert_close(expected, pred['layer_maps'])
        torch.testing.assert_close(pred['current_map'], expected.mean(1))
        torch.testing.assert_close(pred['current_map'].sum(1), torch.ones(2))
        torch.testing.assert_close(pred['subject_map'].sum(1), torch.ones(2))
        swapped = model.predict_grounding(dict(inputs, tokens=inputs['tokens']+1))
        torch.testing.assert_close(swapped['subject_map'], pred['subject_map'], rtol=0, atol=0)
        assert not hasattr(model.grounding, 'current_decoder') and not hasattr(model, 'skill')


@pytest.mark.parametrize('checkpoint,bfloat', [(False, False), (True, True)])
def test_shared_alpha_and_native_lora_receive_both_gradients(checkpoint, bfloat):
    model, inputs, targets = build(checkpoint=checkpoint)
    layers = model.paligemma_with_expert.paligemma.language_model.layers
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bfloat):
        pred = model.predict_grounding(inputs)
        losses, counts, metrics = model.grounding.losses(pred, targets)
    assert {name: int(v) for name, v in counts.items()} == dict(gac=1, current_align=1, subject=2)
    assert all(value.dtype == torch.float32 and torch.isfinite(value) for value in losses.values())
    assert 'grounding_layer_1_gt_mass' in metrics
    for key in ('gac', 'current_align'):
        grads = torch.autograd.grad(losses[key], (pred['alpha'], model.grounding.alpha_scorer[-1].weight,
                                    *(layer.self_attn.q_proj.lora_B for layer in layers)), retain_graph=True)
        assert all(g.isfinite().all() and g.norm() > 0 for g in grads)
    isolated = torch.autograd.grad(losses['subject'], layers[-1].self_attn.q_proj.lora_B,
                                   allow_unused=True, retain_graph=True)[0]
    assert isolated is None
    sum(losses.values()).backward()
    assert model.grounding.subject_queries.grad.norm() > 0
    for name, p in model.named_parameters():
        if name.startswith('grounding.') or '.q_proj.lora_' in name:
            assert p.requires_grad and p.grad is not None and p.grad.isfinite().all(), name
        else:
            assert not p.requires_grad and p.grad is None, name


def test_target_overlap_and_layer_kl_on_head_average():
    mask = torch.zeros(1, 4, 4)
    mask[:, :3, 1:3] = 1.
    gt = patch_coverage(mask, 2)
    torch.testing.assert_close(gt, torch.tensor([[.5, .5, .25, .25]]))
    bbox = torch.tensor([[[.25, 0., .75, .75]]])
    torch.testing.assert_close(box_coverage(bbox, 2)[:, 0], gt)
    model, inputs, targets = build()
    pred = model.predict_grounding(inputs)
    side = 4
    raw = torch.rand_like(pred['native_attention'])
    raw = raw / raw.sum(-1, keepdim=True)
    maps = (raw * pred['alpha'][:, None, None, :, None]).sum(3)
    pred = dict(pred, layer_maps=maps.mean(2), current_map=maps.mean((1, 2)))
    losses, _, _ = model.grounding.losses(pred, targets)
    gt = patch_coverage(targets['current_mask'], side)
    per_layer, valid = spatial_kl(maps.mean(2), gt)
    after_mean, _ = spatial_kl(maps.mean((1, 2))[:, None], gt)
    each_head, _ = spatial_kl(maps.flatten(1, 2), gt)
    torch.testing.assert_close(losses['current_align'], per_layer[valid].mean())
    assert after_mean[valid].mean() < losses['current_align'] < each_head[valid].mean()


@pytest.mark.parametrize('empty_width', [False, True])
def test_no_valid_supervision_returns_connected_finite_zero(empty_width):
    model, inputs, targets = build()
    targets['entity_valid'].zero_()
    if empty_width:
        targets['entity_boxes'] = targets['entity_boxes'][:, :0]
        targets['entity_valid'] = targets['entity_valid'][:, :0]
    targets['subject_mask'].zero_()
    targets['current_mask'].zero_()
    losses = model(inputs, targets)
    assert all(v.item() == 0 for v in model.loss_counts.values())
    assert all(v.item() == 0 for v in losses.values())
    sum(losses.values()).backward()
    assert all(p.grad is None or (torch.isfinite(p.grad).all() and p.grad.norm() == 0) for p in model.parameters())


def test_frozen_baseline_starts_with_identical_subject_and_gac_projections():
    baseline, inputs, _ = build(learned=False)
    adapted, _, _ = build()
    assert not hasattr(baseline.grounding, 'alpha_scorer')
    for name, tensor in baseline.grounding.state_dict().items():
        torch.testing.assert_close(tensor, adapted.grounding.state_dict()[name], rtol=0, atol=0)
    with torch.no_grad():
        first = baseline.predict_grounding(inputs)
        second = adapted.predict_grounding(inputs)
    for key in ('current_map', 'subject_map', 'visual_attention_mass', 'alpha'):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
