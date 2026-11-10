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
from openpi.transforms import horizontal_mirror, mirror_eligible
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from scripts.train import train
from test_eskill_model import batch, gradient, model
from test_training_data import saved_episode


def test_reflection_keeps_physics_and_masks_and_is_an_involution():
    inputs, targets = batch(ESkillConfig.tiny())
    inputs['effector_start_valid'][0, 1] = False
    targets['trace_valid'][0, 0, 2] = False
    targets['role_weights'] = torch.rand_like(targets['role_weights'])
    original = dict(inputs=inputs, targets=targets, metadata=[{}, {}])
    selected = torch.tensor([True, False])
    flipped = horizontal_mirror(original, selected)
    for group, name in (('inputs', 'images'), ('targets', 'role_images'), ('targets', 'role_weights')):
        torch.testing.assert_close(flipped[group][name][0], original[group][name][0].flip(-1))
        torch.testing.assert_close(flipped[group][name][1], original[group][name][1])
    for group, name, valid_name in (('inputs', 'effector_start', 'effector_start_valid'),
                                    ('targets', 'trace', 'trace_valid')):
        good = original[group][valid_name][0]
        torch.testing.assert_close(flipped[group][name][0, ..., 0][good], 1-original[group][name][0, ..., 0][good])
        torch.testing.assert_close(flipped[group][name][0, ..., 0][~good], original[group][name][0, ..., 0][~good])
        torch.testing.assert_close(flipped[group][name][..., 1], original[group][name][..., 1])
    for name in ('arm_state', 'arm_presence', 'tokens'):
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
    model.visual_supervision = None
    with mock.patch.object(model, 'read_motion', side_effect=AssertionError('aux targets in stage2')):
        losses = model(inputs, {k: targets[k] for k in ('actions', 'action_valid')})
    assert set(losses) == {'action'}
    losses['action'].backward()
    assert gradient(model.auxiliary_kv) > 0
    assert gradient(model.paligemma_with_expert.gemma_expert) > 0
    assert gradient(model.skill) == gradient(model.grounding) == gradient(model.motion) == 0


def test_stage_configs_reject_mirrored_actions():
    base = TrainConfig('index', 'tokenizer', 'out', 'stats', model=ESkillConfig.tiny())
    for stage in ('joint', 'stage2'):
        with pytest.raises(ValueError, match='mirroring'):
            replace(base, training_stage=stage, horizontal_flip_prob=.5, loss_weights=None)
    with pytest.raises(ValueError, match='frozen VLM'):
        replace(base, training_stage='stage1', model=replace(base.model, freeze_vlm=False))


def test_stage_handoff_and_exact_mirror_resume(tmp_path):
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
    stage2 = replace(config, training_stage='stage2', horizontal_flip_prob=0., loss_weights=None,
                     init_checkpoint=str(checkpoint), output_dir=str(tmp_path / 'stage2'), num_steps=2)
    dataset = create_dataset(stage2, 'train')
    with mock.patch('openpi.policies.eskill_dataset.EpisodeReader.goal', side_effect=AssertionError('goal read')):
        assert set(dataset[0]['targets']) == {'actions', 'action_valid'}
    model = create_model(stage2, init_checkpoint=checkpoint)
    assert model.visual_supervision is None
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed['model'][name], rtol=0, atol=0)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trained = torch.load(train(stage2), weights_only=False)
    assert trained['step'] == 2  # Fresh optimizer/step, not the Stage-1 step of 4.
    for name, value in frozen.items():
        torch.testing.assert_close(value, trained['model'][name], rtol=0, atol=0)
    assert not torch.equal(trained['model']['action_out_proj.weight'], resumed['model']['action_out_proj.weight'])
    assert not torch.equal(trained['model']['auxiliary_kv.0.weight'], resumed['model']['auxiliary_kv.0.weight'])
    assert not any(k.startswith('visual_supervision.') for k in trained['model'])
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


def stage_worker(rank, rendezvous, stage):
    from copy import deepcopy
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    torch.manual_seed(19)
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused', model=ESkillConfig.tiny(vocab_size=128),
                      training_stage=stage, device='cpu', precision='float32', smoke=True)
    net = create_model(cfg, initialize=False)
    inputs, targets = batch(net.config)
    if stage == 'stage1':
        for key in ('skill_valid', 'role_valid', 'trace_valid', 'gripper_valid'):
            targets[key][0] = False
    noise, time = torch.randn_like(targets['actions']), torch.tensor([.2, .8])
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        wrapper = DistributedDataParallel(net, broadcast_buffers=False)
        local = lambda data: {key: val[rank:rank+1] for key, val in data.items()}
        # Two iterations detect unused-parameter reduction errors on the next forward.
        for _ in range(2):
            net.zero_grad(set_to_none=True)
            losses = wrapper(local(inputs), local(targets), noise=noise[rank:rank+1], time=time[rank:rank+1])
            sum(losses.values()).backward()
            reference = deepcopy(net)
            reference.zero_grad(set_to_none=True)
            with mock.patch('openpi.models.model.dist.is_initialized', return_value=False):
                expected = reference(inputs, targets, noise=noise, time=time)
            sum(expected.values()).backward()
            for name, p in net.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None, name
                    torch.testing.assert_close(p.grad, reference.get_parameter(name).grad, rtol=3e-3, atol=2e-5, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('stage', ['stage1', 'stage2'])
def test_stage_ddp_matches_global_batch(tmp_path, stage):
    import torch.multiprocessing as mp
    mp.spawn(stage_worker, args=((tmp_path / 'rendezvous').as_uri(), stage), nprocs=2, join=True)
