"""Digital reflection and the two-stage training boundary, including fresh handoff/resume."""
from dataclasses import replace
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig, create_model
from openpi.training.data_loader import create_dataset
from openpi.training.metrics import motion_gradient_balance
from openpi.transforms import horizontal_mirror, mirror_eligible
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from scripts.train import train
from test_eskill_model import batch, gradient, model
from test_training_data import saved_episode


def test_spatial_reflection_is_involutive_and_masks_selected_body_state():
    inputs, targets = batch(ESkillConfig.tiny())
    targets['trace_valid'][0, 0, 2] = False
    targets['role_weights'] = torch.rand_like(targets['role_weights'])
    original = dict(inputs=inputs, targets=targets, metadata=[{}, {}])
    selected = torch.tensor([True, False])
    flipped = horizontal_mirror(original, selected)
    for group, name in (('inputs', 'images'), ('targets', 'role_images'), ('targets', 'role_weights')):
        torch.testing.assert_close(flipped[group][name][0], original[group][name][0].flip(-1))
        torch.testing.assert_close(flipped[group][name][1], original[group][name][1])
    for group, name, valid_name in (('targets', 'trace', 'trace_valid'),):
        good = original[group][valid_name][0]
        torch.testing.assert_close(flipped[group][name][0, ..., 0][good], 1-original[group][name][0, ..., 0][good])
        torch.testing.assert_close(flipped[group][name][0, ..., 0][~good], original[group][name][0, ..., 0][~good])
        torch.testing.assert_close(flipped[group][name][..., 1], original[group][name][..., 1])
    for name in ('arm_presence', 'tokens'):
        assert flipped['inputs'][name] is inputs[name]
    for name in ('actions', 'action_valid', 'skill_labels', 'trace_valid', 'gripper', 'gripper_valid'):
        assert flipped['targets'][name] is targets[name]
    restored = horizontal_mirror(flipped, selected)
    for group in ('inputs', 'targets'):
        for key, value in original[group].items():
            torch.testing.assert_close(restored[group][key], value)
    assert [x['mirrored'] for x in flipped['metadata']] == [True, False]


@pytest.mark.parametrize('text,expected', [('put mug on RIGHT plate', False), ('leftmost_cup', False),
    ('turn clockwise', False), ('pick cup in front of bowl', True), ('open top drawer', True)])
def test_language_eligibility(text, expected):
    assert mirror_eligible(text) == expected


def test_mirrored_motion_uses_no_proprioceptive_inputs(model):
    model.eval()
    inputs, targets = batch(model.config)
    mirrored = horizontal_mirror(dict(inputs=inputs, targets=targets, metadata=[{}, {}]),
                                 torch.tensor([True, False]))['inputs']
    assert 'arm_state_mask' not in mirrored and 'arm_state' not in mirrored
    with torch.no_grad():
        _, before = model.predict_condition(mirrored, with_action=False)
        changed = dict(mirrored, arm_state=torch.full((2, 2, 8), float('nan')))
        _, after = model.predict_condition(changed, with_action=False)
    torch.testing.assert_close(before['motion'], after['motion'], rtol=0, atol=0)
    assert torch.isfinite(after['motion']).all()


def test_all_mirrored_batch_keeps_visual_gradients(model):
    model.configure_stage('stage1')
    inputs, targets = batch(model.config)
    sample = horizontal_mirror(dict(inputs=inputs, targets=targets, metadata=[{}, {}]),
                               torch.tensor([True, True]))
    losses = model(sample['inputs'], sample['targets'])
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()
    assert not hasattr(model, "state_projection")
    for module in (model.skill.shared, model.grounding, model.motion):
        assert gradient(module) > 0


def test_stage_gradients_and_forward_boundaries(model):
    inputs, targets = batch(model.config)
    model.configure_stage('stage1')
    with mock.patch.object(model, 'flow_loss', side_effect=AssertionError('action loss in stage1')), \
         mock.patch.object(model, 'action_condition', side_effect=AssertionError('action K/V in stage1')):
        losses = model(inputs, targets)
    assert set(losses) == {'skill', 'current', 'future_object', 'trace', 'gripper'}
    sum(losses.values()).backward()
    for module in (model.skill, model.grounding, model.motion, model.visual_supervision.head):
        assert gradient(module) > 0
    assert gradient(model.auxiliary_kv) == gradient(model.paligemma_with_expert) == 0
    assert not any(p.requires_grad for p in model.visual_supervision.vae.parameters())
    model.zero_grad(set_to_none=True)
    model.configure_stage('stage2')
    losses = model(inputs, targets)
    assert set(losses) == {'action', 'skill', 'current', 'future_object', 'trace', 'gripper'}
    losses['action'].backward(retain_graph=True)
    assert gradient(model.auxiliary_kv) > 0
    assert gradient(model.paligemma_with_expert.gemma_expert) > 0
    for module in (model.skill.shared, model.grounding, model.motion):
        assert gradient(module) > 0  # Action alone must reach every continuous prior.
    assert gradient(model.skill.classifier) == 0  # Hard Top-1 is not differentiable.
    assert gradient(model.motion.trace_head) == gradient(model.motion.gripper_head) == 0
    model.zero_grad(set_to_none=True)
    sum(losses.values()).backward()
    for module in (model.skill.classifier, model.motion.trace_head, model.motion.gripper_head,
                   model.visual_supervision.head):
        assert gradient(module) > 0
    assert gradient(model.visual_supervision.vae) == 0
    assert gradient(model.paligemma_with_expert.paligemma) == 0


def test_stage_configs_reject_mirrored_actions():
    base = TrainConfig('index', 'tokenizer', 'out', 'stats', model=ESkillConfig.tiny())
    for stage in ('joint', 'stage2'):
        with pytest.raises(ValueError, match='mirroring'):
            replace(base, training_stage=stage, horizontal_flip_prob=.5, loss_weights=None)
    with pytest.raises(ValueError, match='frozen VLM'):
        replace(base, training_stage='stage1', model=replace(base.model, freeze_vlm=False))
    unfrozen = replace(base, training_stage='stage2', model=replace(base.model, freeze_vlm=False))
    assert not unfrozen.model.freeze_vlm
    with pytest.raises(ValueError, match='discrete_state_input'):
        replace(base, training_stage='stage1', horizontal_flip_prob=.5,
                model=replace(base.model, discrete_state_input=True), loss_weights=None)


@pytest.mark.parametrize('freeze_vlm,rank', [(True, 0), (False, 0), (True, 2)])
def test_stage_switch_restores_joint_scope_and_preserves_frozen_vae(freeze_vlm, rank):
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused',
        model=ESkillConfig.tiny(vocab_size=128, freeze_vlm=freeze_vlm), lora_rank=rank,
        device='cpu', precision='float32', smoke=True)
    net = create_model(cfg, initialize=False)
    original = {name for name, p in net.named_parameters() if p.requires_grad}
    weights = set(net.state_dict())
    for stage in ('stage1', 'joint', 'stage1', 'joint') if rank else ('stage1', 'stage2', 'joint', 'stage1', 'joint'):
        net.configure_stage(stage)
        assert not any(p.requires_grad for p in net.visual_supervision.vae.parameters())
        if stage == 'stage1':
            assert not any(p.requires_grad for p in net.paligemma_with_expert.gemma_expert.parameters())
        if stage == 'joint':
            assert {name for name, p in net.named_parameters() if p.requires_grad} == original
    assert set(net.state_dict()) == weights
    with pytest.raises(ValueError, match='Unknown training stage'):
        net.configure_stage('invalid')
    assert net.training_stage == 'joint'
    assert {name for name, p in net.named_parameters() if p.requires_grad} == original


def test_unfrozen_stage2_vlm_receives_fm_and_auxiliary_gradients():
    torch.set_num_threads(2)
    config = TrainConfig('unused', 'unused', 'unused', 'unused',
        model=ESkillConfig.tiny(vocab_size=128, freeze_vlm=False, gradient_checkpointing=True),
        training_stage='stage2', device='cpu', precision='float32', smoke=True)
    net = create_model(config, initialize=False)
    vlm = net.paligemma_with_expert.paligemma
    assert vlm.vision_tower.is_gradient_checkpointing
    net.configure_stage('stage1')
    assert not any(p.requires_grad for p in vlm.parameters())
    net.configure_stage('stage2')
    assert all(p.requires_grad for p in vlm.parameters())
    inputs, targets = batch(net.config)
    losses = net(inputs, targets)
    for loss in (losses['action'], sum(value for key, value in losses.items() if key != 'action')):
        net.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        for module in (vlm.vision_tower, vlm.multi_modal_projector, vlm.language_model):
            assert gradient(module) > 0
        assert gradient(net.visual_supervision.vae) == 0
    net.configure_stage('stage1')
    assert not any(p.requires_grad for p in vlm.parameters())


@pytest.mark.parametrize('freeze_vlm', [True, False])
def test_stage_handoff_and_exact_mirror_resume(tmp_path, freeze_vlm):
    import sentencepiece
    torch.set_num_threads(2)
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('pick cup then place it\nput cup on left plate\n')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / 'tokenizer'),
        vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    def edit(meta, arrays):
        if meta['episode_id'] == 'two':
            meta['instructions'] = ['put cup on left plate']
    index = tmp_path / 'index.json'
    write_index(index, [saved_episode(tmp_path, identity=x, edit=edit) for x in ('one', 'two')])
    config = TrainConfig(str(index), str(tmp_path / 'tokenizer.model'), str(tmp_path / 'stage1'),
        str(tmp_path / 'stats.json'), model=ESkillConfig.tiny(vocab_size=128, motion_stride=1),
        training_stage='stage1', horizontal_flip_prob=.5, skill_sampling=True, batch_size=4,
        num_steps=2, save_every=2, log_every=1, val_interval=0, validation_fraction=0,
        num_workers=0, device='cpu', precision='float32', smoke=True)
    Path(config.norm_stats).write_text(json.dumps(compute(config)))
    checkpoint = train(config)
    config.num_steps = 4  # Resume crosses an epoch; retain the mirror RNG and weighted draw.
    train(config, resume=checkpoint)
    resumed = torch.load(checkpoint, weights_only=False)
    direct = torch.load(train(replace(config, output_dir=str(tmp_path / 'direct'))), weights_only=False)
    for name in direct['model']:
        torch.testing.assert_close(resumed['model'][name], direct['model'][name], rtol=0, atol=0)
    logs = [json.loads(x) for x in (checkpoint.parent / 'metrics.jsonl').read_text().splitlines()]
    other_logs = [json.loads(x) for x in (tmp_path / 'direct/metrics.jsonl').read_text().splitlines()]
    assert [x['samples'] for x in logs] == [x['samples'] for x in other_logs]
    assert any(s['mirrored'] for x in logs for s in x['samples'])
    assert all(not s['mirrored'] for x in logs for s in x['samples'] if s['episode_id'] == 'two')
    with pytest.raises(ValueError, match='Stage 1'):
        create_trained_policy(checkpoint)
    with pytest.raises(ValueError, match='changing model'):
        train(replace(config, model=replace(config.model, motion_min_horizon_frames=0), num_steps=5),
              resume=checkpoint)
    weights = dict(action=1., skill=.02, current=.01, future_object=.01, trace=.25, gripper=.05)
    stage2 = replace(config, training_stage='stage2', horizontal_flip_prob=0., loss_weights=weights,
                     model=replace(config.model, freeze_vlm=freeze_vlm, gradient_checkpointing=True),
                     init_checkpoint=str(checkpoint), output_dir=str(tmp_path / 'stage2'), num_steps=2,
                     motion_grad_interval=1)
    dataset = create_dataset(stage2, 'train')
    reader = dataset.dataset.reader(0)
    with mock.patch.object(reader, 'goal', wraps=reader.goal) as goal:
        targets = dataset[0]['targets']
    assert goal.called
    assert {'role_images', 'trace', 'gripper', 'skill_labels'} <= set(targets)
    model = create_model(stage2, init_checkpoint=checkpoint)
    with pytest.raises(ValueError, match='motion_min_horizon_frames'):
        create_model(replace(stage2, model=replace(stage2.model, motion_min_horizon_frames=0)),
                     init_checkpoint=checkpoint)
    assert model.visual_supervision is not None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trained = torch.load(train(stage2), weights_only=False)
    assert trained['step'] == 2  # Fresh optimizer/step, not the Stage-1 step of 4.
    for name, value in frozen.items():
        torch.testing.assert_close(value, trained['model'][name], rtol=0, atol=0)
    assert not torch.equal(trained['model']['action_out_proj.weight'], resumed['model']['action_out_proj.weight'])
    assert not torch.equal(trained['model']['auxiliary_kv.0.weight'], resumed['model']['auxiliary_kv.0.weight'])
    for prefix in ('skill.', 'grounding.', 'motion.', 'visual_supervision.head.'):
        assert any(not torch.equal(value, resumed['model'][name]) for name, value in trained['model'].items()
                   if name.startswith(prefix)), prefix
    if not freeze_vlm:
        for module in ('vision_tower', 'multi_modal_projector', 'language_model'):
            prefix = f'paligemma_with_expert.paligemma.model.{module}.'
            assert any(not torch.equal(value, resumed['model'][name])
                       for name, value in trained['model'].items() if name.startswith(prefix)), prefix
    stage2.num_steps = 3
    stage2_checkpoint = Path(stage2.output_dir) / 'checkpoint.pt'
    with pytest.raises(ValueError, match='changing model'):
        train(replace(stage2, model=replace(stage2.model, freeze_vlm=not freeze_vlm)),
              resume=stage2_checkpoint)
    train(stage2, resume=stage2_checkpoint)
    continued = torch.load(stage2_checkpoint, weights_only=False)
    direct = torch.load(train(replace(stage2, output_dir=str(tmp_path / 'stage2_direct'), motion_grad_interval=0)), weights_only=False)
    for name in direct['model']:
        torch.testing.assert_close(continued['model'][name], direct['model'][name], rtol=0, atol=0)
    records = [json.loads(line) for line in (Path(stage2.output_dir) / 'metrics.jsonl').read_text().splitlines()]
    assert all(row['motion_gradients']['fm_norm'] > 0 for row in records)
    assert all(row['motion_gradients']['loss_weights'] == weights for row in records)
    assert create_trained_policy(Path(stage2.output_dir) / 'checkpoint.pt').model.visual_supervision is None


@pytest.mark.parametrize('stage', ['stage1', 'stage2'])
def test_validation_reports_only_active_stage_losses(model, stage):
    from openpi.training.validation import validation_losses
    model.configure_stage(stage)
    inputs, targets = batch(model.config)
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused', model=model.config,
                      training_stage=stage, device='cpu', precision='float32')
    rng = torch.get_rng_state().clone()
    results = validation_losses(model, [dict(inputs=inputs, targets=targets)], cfg)
    assert set(results) == set(cfg.loss_keys) | {'loss'}
    assert torch.isfinite(torch.tensor(list(results.values()))).all()
    assert torch.equal(rng, torch.get_rng_state())


def stage_worker(rank, rendezvous, stage, freeze_vlm):
    from copy import deepcopy
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    torch.manual_seed(19)
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused',
                      model=ESkillConfig.tiny(vocab_size=128, gradient_checkpointing=True, freeze_vlm=freeze_vlm),
                      training_stage=stage, device='cpu', precision='float32', smoke=True)
    net = create_model(cfg, initialize=False)
    inputs, targets = batch(net.config)
    for key in ('skill_valid', 'role_valid', 'trace_valid', 'gripper_valid'):
        targets[key][0] = False
    noise, time = torch.randn_like(targets['actions']), torch.tensor([.2, .8])
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        wrapper = DistributedDataParallel(net, broadcast_buffers=False)
        local = lambda data: {key: val[rank:rank+1] for key, val in data.items()}
        # Two iterations detect unused-parameter reduction errors on the next forward.
        for iteration in range(2):
            net.zero_grad(set_to_none=True)
            losses = wrapper(local(inputs), local(targets), noise=noise[rank:rank+1], time=time[rank:rank+1])
            balance = motion_gradient_balance(net, losses) if stage == 'stage2' else None
            sum(losses.values()).backward()
            reference = deepcopy(net)
            reference.zero_grad(set_to_none=True)
            with mock.patch('openpi.models.model.dist.is_initialized', return_value=False):
                expected = reference(inputs, targets, noise=noise, time=time)
                expected_balance = motion_gradient_balance(reference, expected) if stage == 'stage2' else None
            sum(expected.values()).backward()
            if balance is not None:
                for key in balance:
                    assert balance[key] == pytest.approx(expected_balance[key], rel=3e-3, abs=2e-5)
            for name, p in net.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, name
                    torch.testing.assert_close(p.grad, reference.get_parameter(name).grad, rtol=3e-3, atol=2e-5, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('stage,freeze_vlm', [('stage1', True), ('stage2', True), ('stage2', False)])
def test_stage_ddp_matches_global_batch(tmp_path, stage, freeze_vlm):
    import torch.multiprocessing as mp
    mp.spawn(stage_worker, args=((tmp_path / 'rendezvous').as_uri(), stage, freeze_vlm), nprocs=2, join=True)
