"""The event boundary is independent of the fixed lookahead in data, gradients and evaluation."""

from dataclasses import replace
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from openpi.augmentation import affine_eskill_batch
from openpi.models.motion_planner import rms
from openpi.policies.eskill_dataset import EpisodeReader, event_boundary_targets, planning_event
from openpi.policies.grounding_dataset import collate_grounding
from openpi.policies.motion_planner import MotionPlannerPolicy, create_motion_planner_policy
from openpi.training.checkpoints import load_klal_weights, load_weights, save_checkpoint
from openpi.training.config import create_model
from openpi.transforms import horizontal_mirror
from scripts.analysis.motion_planner import episode_rows, native_report, plan_pair
from scripts.analysis.visual_primitive_eval import evaluate, motion_pair_metrics
from test_motion_planner_data import planner_dataset
from test_motion_planner_model import build
from test_motion_planner_training import _training_config


@pytest.mark.parametrize('version', ['boundary_first', 'joint_decode'])
def test_labels_before_short_lookahead_and_after_geometry(tmp_path, version):
    ds = planner_dataset(tmp_path)
    ds.config.model = replace(ds.config.model, planner_version=version)
    row = ds[0]
    assert row['metadata']['boundary_source_frame_indices'] == [0, 0]
    assert row['metadata']['trace_source_frame_indices'] == [1] * 8
    batch = collate_grounding([row])
    xy = batch['targets']['boundary_xy']
    torch.testing.assert_close(xy, torch.tensor([2.5, 4.5]).expand_as(xy) / 32)
    mirrored = horizontal_mirror(batch, torch.tensor([True]))['targets']
    torch.testing.assert_close(mirrored['boundary_xy'][..., 0], 1 - xy[..., 0])
    rot, _ = affine_eskill_batch(batch, torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([90.0]))
    torch.testing.assert_close(rot['targets']['boundary_xy'][0, 1], torch.tensor([1 - 4.5 / 32, 2.5 / 32]))
    crop, _ = affine_eskill_batch(batch, torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    assert not crop['targets']['boundary_valid'].any()
    edge = dict(batch, targets=dict(batch['targets'], boundary_xy=xy.clone()))
    edge['targets']['boundary_xy'][0, 0, 0] = 0.0
    assert horizontal_mirror(edge, torch.tensor([True]))['targets']['boundary_valid'].tolist() == [[False, True]]
    reader = EpisodeReader(tmp_path / 'episode.json')
    event = dict(planning_event(reader, 0), stop_frame=2, goal_frame=0)
    # The last saved observation is frame 1. Its invalid EEF must not fall back to frame 0.
    pix = reader.meta['effector_pixels']['single']['image']
    original = reader.array
    pix['valid'] = dict(key='eef_valid_test')
    reader.array = lambda ref: np.array([True, False]) if ref is pix['valid'] else original(ref)
    _, valid, sources = event_boundary_targets(reader, 0, event, image_size=32)
    assert valid.tolist() == [True, False] and sources.tolist() == [0, 1]


def test_boundary_graph_calibration_and_intervention():
    model, inputs, targets = build(version='boundary_first')
    planner = model.planner
    report = model.calibrate_motion_queries(inputs)
    assert report['query_to_time_rms'] == pytest.approx(1.0, rel=1e-6)
    assert report['end_input_rms'] == pytest.approx(report['fixed_input_rms'], rel=1e-6)
    assert report['fixed_input_rms'] == pytest.approx(report['text_rms'], rel=1e-6)
    assert planner.native_time[1].count_nonzero() == 0
    assert planner.native_time[0].count_nonzero() and len(planner.layers) == 1
    features = model.plan_features(inputs)
    prefix = model.plan_features(inputs, router_only=True)
    for key in ('visual_hidden', 'current_map', 'skill_logits'):
        torch.testing.assert_close(features[key], prefix[key], atol=1e-5, rtol=1e-5)
    plan = planner(features)
    x1 = plan['first_readout']
    assert x1.shape[1:] == (10, model.config.hidden_dim)
    assert plan['trace'].shape[1:] == (8, 2)
    memory = planner.memory_norm(planner.project(features['visual_hidden']) + planner.xy)
    residual = planner.motion_project(features['motion_hidden'])
    guided = planner.grounded_cross(
        planner.read_norm(residual), memory, features['visual_mask'], features['current_map'], 0.3
    )
    torch.testing.assert_close(x1, residual + guided)
    grad, coord = torch.autograd.grad(
        plan['trace'].sum(), (plan['boundary_latents'], plan['end']), allow_unused=True, retain_graph=True
    )
    assert grad.norm(dim=-1).gt(0).all() and coord is None
    donor = plan['boundary_latents'][:, 1:2].roll(1, 0)
    swapped = planner(features, end_latent=donor)
    torch.testing.assert_close(swapped['first_readout'][:, 1:2], donor)
    torch.testing.assert_close(swapped['first_readout'][:, :1], x1[:, :1])
    torch.testing.assert_close(swapped['first_readout'][:, 2:], x1[:, 2:])
    torch.testing.assert_close(planner(features, end_latent=x1[:, 1:2])['trace'], plan['trace'])
    assert rms(swapped['trace'] - plan['trace']) > 0
    targets.update(boundary_xy=torch.rand(3, 2, 2), boundary_valid=torch.ones(3, 2, dtype=torch.bool))
    targets['boundary_valid'][0, 1] = False
    targets['boundary_xy'][0, 1] = float('nan')
    losses = model(inputs, targets, planner_diagnostics=True)
    xy = torch.cat((targets['boundary_xy'], targets['trace']), dim=1)
    good = torch.cat((targets['boundary_valid'], targets['trace_valid']), dim=1)
    prediction = torch.cat((torch.stack((plan['start'], plan['end']), dim=1), plan['trace']), dim=1)
    torch.testing.assert_close(
        losses['trace'], F.smooth_l1_loss(prediction[good], xy[good], beta=model.config.trace_beta)
    )
    assert model.loss_counts['trace'] == good.sum()
    for role in ('start', 'end', 'future'):
        assert f'planner_offset_40_{role}_mass' in model.diagnostic_statistics
    targets['trace_valid'].zero_()
    model(inputs, targets)['trace'].backward()
    assert all(p.grad is None for p in planner.layers[0].experts.parameters())
    assert planner.native_queries.grad.norm() > 0
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())


def test_joint_slots_decode_once_after_one_expert_and_swap_before_temporal_attention():
    model, inputs, _ = build(version='joint_decode')
    planner = model.planner
    features = model.plan_features(inputs)
    route = torch.tensor([0, 1, 2])
    head_inputs = []
    with planner.trace_head.register_forward_pre_hook(lambda _, args: head_inputs.append(args[0])):
        plan = planner(features, skill_id=route)
    assert len(head_inputs) == 1 and head_inputs[0] is plan['event_motion_latents']
    first = plan['first_readout']
    visual_mask = features['visual_mask']
    memory = planner.memory_norm(planner.project(features['visual_hidden']) + planner.xy)
    layer = planner.layers[0]
    temporal = layer.temporal_step(first)
    visual = temporal + layer.cross(layer.cross_norm(temporal), memory, visual_mask)
    expected = layer.expert_step(visual, route)
    torch.testing.assert_close(plan['event_motion_latents'], expected)
    torch.testing.assert_close(plan['boundary_latents'], expected[:, :2])
    torch.testing.assert_close(plan['motion_tokens'], expected[:, 2:])
    coords = torch.cat((plan['start'][:, None], plan['end'][:, None], plan['trace']), dim=1)
    torch.testing.assert_close(coords, planner.trace_head(expected))
    bypass = planner(features, skill_id=route, remove_experts=True)
    torch.testing.assert_close(bypass['event_motion_latents'], visual)
    assert (bypass['event_motion_latents'] - expected).norm(dim=-1).gt(0).all()

    identity = planner(features, skill_id=route, end_latent=first[:, 1:2])
    torch.testing.assert_close(identity['event_motion_latents'], expected)
    donor = first[:, 1:2].roll(1, 0)
    swapped = planner(features, skill_id=route, end_latent=donor)
    for section in (slice(0, 1), slice(2, None)):
        torch.testing.assert_close(swapped['first_readout'][:, section], first[:, section])
    torch.testing.assert_close(swapped['first_readout'][:, 1:2], donor)
    for key in ('start', 'end', 'trace'):
        assert (swapped[key] - plan[key]).norm(dim=-1).gt(0).all()
    torch.testing.assert_close(swapped['skill_id'], route)


def test_joint_boundary_only_gradients_reach_only_routed_experts():
    model, inputs, targets = build(version='joint_decode')
    planner = model.planner
    features = model.plan_features(inputs)
    features['skill_logits'] = features['skill_logits'] + 100 * F.one_hot(torch.tensor([1, 3, 5]), 6)
    targets['trace_valid'].zero_()
    targets['trace'].fill_(float('nan'))
    valid = torch.tensor([[False, True], [True, False], [False, False]])
    xy = torch.rand(3, 2, 2).masked_fill(~valid[..., None], float('nan'))
    targets.update(boundary_xy=xy, boundary_valid=valid)
    outputs = []
    with planner.trace_head.register_forward_hook(lambda _, args, value: outputs.append(value)):
        losses, counts, stats = planner.supervision(features, targets, diagnostics=True)
    decoded, = outputs
    expected = F.smooth_l1_loss(decoded[:, :2][valid], xy[valid], beta=model.config.trace_beta)
    torch.testing.assert_close(losses['trace'], expected)
    assert counts['trace'] == 2
    for row in ('start', 'end', 'offset_40'):
        mass = [stats[f'planner_{row}_{role}_mass'] for role in ('start', 'end', 'future')]
        torch.testing.assert_close(sum(v[0] for v in mass), mass[0][1])
    decoded.retain_grad()
    losses['trace'].backward()
    torch.testing.assert_close(decoded.grad.norm(dim=-1) > 0, torch.cat((valid, targets['trace_valid']), dim=1))
    for i, expert in enumerate(planner.layers[0].experts):
        if i in (1, 3):
            assert all(p.grad is not None and p.grad.norm() > 0 for p in expert.parameters())
        else:
            assert all(p.grad is None for p in expert.parameters())
    assert all(p.grad is None for p in planner.router.parameters())
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())


@pytest.mark.parametrize('source_kind', ['full', 'visual_adapter', 'visual_full_vlm'])
def test_non_lora_transfer_and_standalone_restore(tmp_path, source_kind):
    cfg = _training_config(tmp_path, 'motion_planner')
    cfg = replace(
        cfg,
        model=replace(cfg.model, planner_version='boundary_first', planner_frozen_vlm_layers=(1,)),
        lora_rank=0,
        planner_init_step=10000,
        tokenizer_path=None,
    )
    model = create_model(cfg)
    source_cfg = _training_config(tmp_path, 'visual_primitive').to_dict()
    source_cfg.update(lora_rank=0, lora_alpha=999)
    source_cfg['model']['full_vlm_layers'] = [1]  # Source-side training setting.
    backbone_keys = {k for k in model._base_keys if '.language_model.layers.1.' in k}
    assert backbone_keys
    source_keys = model._base_keys if source_kind == 'full' else backbone_keys
    source_keys = source_keys | {k for k in model.state_dict() if k.startswith('grounding.alpha_scorer.')}
    before = model.state_dict()
    weights = {k: torch.full_like(before[k], 0.27) for k in source_keys}
    path = tmp_path / 'source.pt'
    saved = dict(config=source_cfg, model=weights, step=10000, resources={}, weights_kind=source_kind)
    torch.save(saved, path)
    load_klal_weights(model, path, cfg)
    for k in source_keys:
        torch.testing.assert_close(model.state_dict()[k], weights[k])
    model.planner_calibration = {'samples': []}
    path = tmp_path / 'trained.pt'
    save_checkpoint(
        path,
        model=model,
        optimizer=torch.optim.AdamW(model.planner.parameters()),
        config=replace(cfg, smoke=False),
        step=2,
        stats={},
        resources={},
    )
    snapshot = torch.load(path, weights_only=False)
    assert snapshot['weights_kind'] == 'full'
    restored = create_model(cfg, initialize=False)
    load_weights(restored, snapshot['model'])
    for k in model.state_dict():
        torch.testing.assert_close(restored.state_dict()[k], model.state_dict()[k])
    weights.pop(sorted(backbone_keys)[0])
    torch.save(saved, path)
    with pytest.raises(ValueError, match='Missing KLAL weights'):
        load_klal_weights(model, path, cfg)


@pytest.mark.parametrize('version', ['boundary_first', 'joint_decode'])
def test_boundary_eval_uses_normal_policy(tmp_path, version):
    planner_dataset(tmp_path)
    reader = EpisodeReader(tmp_path / 'episode.json')
    model, _, _ = build(version=version)

    def tokenizer(task, state=None):
        return np.array([1, 5 if 'milk' in task else 6]), np.ones(2, bool)

    policy = MotionPlannerPolicy(model, tokenizer, 'float32')
    row = next(episode_rows(policy, reader, [0], {}))
    assert row['end']['source_frame'] == 0 and row['start']['error_px'] >= 0
    pair = plan_pair(
        policy,
        reader.rgb('image', 0),
        ['pick milk', 'pick pudding'],
        regions=[None, None],
        observed=np.ones(8, bool),
    )
    result = pair['conditions']
    assert pair['planner_version'] == version
    assert result['end_swap']['skill_id'] == result['task_a']['skill_id']
    if version == 'boundary_first':
        assert result['end_swap']['start_displacement_px'] == 0
        np.testing.assert_allclose(result['end_swap']['end'], result['task_b']['end'], atol=1e-6)
    else:
        rgb = reader.rgb('image', 0)
        a, b = (policy.features(rgb, task) for task in pair['tasks'])
        donor = policy.read_plan(b)['first_readout'][:, 1:2]
        swapped = policy.read_plan(a, skill_id=a['skill_logits'].argmax(-1), end_latent=donor)
        for key in ('start', 'end', 'trace'):
            np.testing.assert_allclose(result['end_swap'][key], swapped[key][0].numpy(), atol=1e-6)
        assert result['end_swap']['start_displacement_px'] > 0
    eid = reader.meta['episode_id']
    manifest = tmp_path / 'frames.json'
    manifest.write_text(json.dumps({'rows': [{'episode_id': eid, 'frame': i} for i in (1, 0)]}))
    index = tmp_path / 'index.json'
    native = native_report(policy, index, manifest)
    assert native['planner_version'] == version
    assert native['episodes'] == 1 and native['events'] == native['frames'] == 2
    assert [r['frame'] for r in native['rows']] == [1, 0]
    assert native['metrics']['end_error_px']['n'] == 2

    Image.fromarray(reader.rgb('image', 0)).save(tmp_path / 'frame.png')
    cases = [
        dict(
            id=target,
            split='B',
            parent=eid,
            scene='one',
            image='frame.png',
            task='pick ' + target,
            target=target,
            pair_id='same',
            skill='Pick',
            boxes={'milk': [1, 1, 4, 4], 'pudding': [20, 2, 24, 8]},
        )
        for target in ('milk', 'pudding')
    ]
    manifest.write_text(json.dumps(dict(protocol='technical', metric_space='source', rows=cases)))
    scored = evaluate(policy, manifest, tmp_path / 'pair_results', panels=0, interventions=True)
    pair = scored['pairs'][0]
    assert scored['planner_version'] == version
    assert pair['interventions']['end_swap']['swap_source'] == 'pre_temporal_end_slot'
    assert len(pair['interventions']['prior_only']['trace_displacement_px']) == 8
    assert scored['rows'][0]['motion']['target_box_px'] == [1., 1., 4., 4.]


def test_paired_motion_uses_bbox_edges_for_both_planner_versions():
    a = dict(trace_px=[[115., 10.]] * 4 + [[95., 10.]] * 4,
             target_box_px=[0., 0., 100., 20.], target_center_px=[50., 10.])
    b = dict(trace_px=[[95., 10.]] * 4 + [[115., 10.]] * 4,
             target_box_px=[110., 0., 120., 20.], target_center_px=[115., 10.])
    for boundary in (False, True):
        if boundary:
            a.update(start_px=[1., 2.], end_px=[95., 10.])
            b.update(start_px=[1., 2.], end_px=[115., 10.])
        metrics = motion_pair_metrics(a, b)
        assert metrics['last4_both_closer_to_own_target'] == metrics['last4_direction_correct'] == 1.
        assert metrics['last4_toward_b_px'] == 20.
        if boundary:
            assert metrics['end_toward_b_px'] == 20.
        tied = dict(a, trace_px=[[105., 10.]] * 8)
        assert motion_pair_metrics(tied, b)['last4_both_closer_to_own_target'] == 0.
        assert 'last4_direction_correct' not in motion_pair_metrics(a, a)


def test_full_non_lora_v1_before_layer_metadata(tmp_path, monkeypatch):
    cfg = replace(_training_config(tmp_path, 'motion_planner'), lora_rank=0)
    model = create_model(cfg)
    old = cfg.to_dict()
    for field in ('planner_version', 'planner_frozen_vlm_layers'):
        del old['model'][field]
    checkpoint = tmp_path / 'legacy.pt'
    torch.save(dict(weights_kind='full', config=old, model=model.state_dict()), checkpoint)
    monkeypatch.setattr('openpi.policies.motion_planner.TokenizePrompt', lambda *args: None)
    restored = create_motion_planner_policy(checkpoint).model
    assert restored.config.planner_version == 'trajectory'
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value)
    with pytest.raises(ValueError, match='explicit planner_frozen_vlm_layers'):
        load_weights(restored, {}, adapter_only=True)
