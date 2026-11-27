"""Planner checkpoint inheritance and sparse AdamW semantics across ranks."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest
import sentencepiece
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from openpi.models.research import build_planner
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeReader
from openpi.policies.motion_planner import create_motion_planner_policy
from openpi.training.checkpoints import load_weights, save_checkpoint
from openpi.training.config import TrainConfig, create_model
from scripts.train import train


def _training_config(tmp_path, mode):
    sizes = ESkillConfig.tiny(current_grounding=mode, grounding_subject=False,
                              vocab_size=128, motion_terminal_hold=True)
    return TrainConfig('', '', str(tmp_path), '', model=sizes, device='cpu', precision='float32',
                       training_stage=mode, lora_rank=2, lora_alpha=2, smoke=True, validation_fraction=0,
                       loss_weights=dict(current_align=.1) if mode == 'visual_primitive' else None)


def test_planner_transfer_and_compact_restore(tmp_path):
    visual = _training_config(tmp_path, 'visual_primitive')
    source = create_model(visual)
    with torch.no_grad():
        for name, param in source.named_parameters():
            if '.q_proj.lora_' in name or name.startswith('grounding.alpha_scorer.'):
                param.fill_(.31)
    source_file = tmp_path/'klal.pt'
    # Use the same compact layout as an actual KLAL run.
    resources = dict(pretrained_path=None, tokenizer_path=None)
    visual.tokenizer_path = None
    save_checkpoint(source_file, model=source, optimizer=torch.optim.AdamW(source.parameters()),
                    config=replace(visual, smoke=False), step=20000, stats={}, resources=resources)
    config = _training_config(tmp_path, 'motion_planner')
    config.tokenizer_path = None
    config.planner_init_step = 20000
    technical = torch.load(source_file, weights_only=False)
    technical['step'] = 32
    technical_file = tmp_path/'technical_klal.pt'
    torch.save(technical, technical_file)
    with pytest.raises(ValueError, match='Expected KLAL source step 20000, got 32'):
        create_model(config, init_checkpoint=technical_file)
    model = create_model(config, init_checkpoint=source_file)
    assert model.planner_initialization['source_step'] == 20000
    original = source.state_dict()
    for key, param in model.named_parameters():
        if '.q_proj.lora_' in key or key.startswith('grounding.alpha_scorer.'):
            assert not param.requires_grad
            torch.testing.assert_close(param, original[key])
    model.planner_calibration = dict(samples=[dict(episode_id='a', frame=1)])
    model.planner.query_time_scale.fill_(.42)
    optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
    path = tmp_path/'planner.pt'
    save_checkpoint(path, model=model, optimizer=optimizer, config=replace(config, smoke=False),
                    step=11, stats={}, resources=resources)
    saved = torch.load(path, weights_only=False)
    assert saved['weights_kind'] == 'planner_adapter'
    assert saved['planner_calibration'] == model.planner_calibration
    restored = create_model(config)
    load_weights(restored, saved['model'], adapter_only=True)
    for key, value in saved['model'].items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    without_alpha = {k: v for k, v in saved['model'].items() if 'alpha_scorer' not in k}
    with pytest.raises(ValueError, match='Incomplete'):
        load_weights(restored, without_alpha, adapter_only=True)


@pytest.mark.parametrize('version', ['trajectory', 'boundary_first', 'joint_decode'])
def test_router_warmup_resume_keeps_calibrated_parameters(tmp_path, version):
    from test_motion_planner_data import planner_dataset

    fixture = planner_dataset(tmp_path)
    corpus = tmp_path / 'corpus'
    corpus.write_text('put the milk in the basket\nput the chocolate pudding in the basket\n')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path/'sp'),
                                            vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    stats = tmp_path/'stats.json'
    stats.write_text('{}')
    config = replace(_training_config(tmp_path, 'motion_planner'), data_index=fixture.config.data_index,
                     tokenizer_path=str(tmp_path/'sp.model'), output_dir=str(tmp_path/'run'), norm_stats=str(stats),
                     batch_size=2, num_workers=0, num_steps=1, planner_router_warmup_steps=1,
                     planner_calibration_samples=2, skill_sampling=True)
    if version != 'trajectory':
        config = replace(config, model=replace(config.model, planner_version=version), lora_rank=0)
    checkpoint = train(config)
    if version == 'joint_decode':
        split = replace(config, model=replace(config.model, planner_version='boundary_first'))
        with pytest.raises(ValueError, match='Cannot resume after changing model'):
            train(split, resume=checkpoint)
    warmup = torch.load(checkpoint, weights_only=False)
    calibration = warmup['planner_calibration']
    assert calibration['calibration_samples'] == 2
    assert {s['frame'] for s in calibration['samples']} == {0, 1}
    assert calibration['motion_input_rms'] == pytest.approx(calibration['text_rms'], rel=.01)
    assert calibration['target_other_value_l2'] > 0
    first_step = json.loads((tmp_path/'run/metrics.jsonl').read_text())
    assert first_step['router_only'] and all(v is None for v in first_step['expert_gradients'].values())

    resume = train(replace(config, num_steps=2), resume=checkpoint)
    direct = train(replace(config, num_steps=2, output_dir=str(tmp_path/'direct')))
    a, b = [torch.load(path, weights_only=False) for path in (resume, direct)]
    assert a['planner_calibration'] == b['planner_calibration'] == calibration
    for key, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][key], rtol=0, atol=0)
    if version == 'joint_decode':
        policy = create_motion_planner_policy(resume)
        assert policy.model.planner.joint_slots
        plan = policy.infer(EpisodeReader(tmp_path / 'episode.json').rgb('image', 0), 'pick up milk')
        assert plan['event_motion_latents'].shape == (1, 10, config.model.hidden_dim)


def _sparse_worker(rank, rendezvous, version):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(10)
        cfg = ESkillConfig.tiny(current_grounding='motion_planner', planner_version=version)
        net = build_planner(cfg)
        ddp = DistributedDataParallel(net, find_unused_parameters=True, broadcast_buffers=False,
                                      gradient_as_bucket_view=True)
        opt = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=.1)
        features = dict(visual_hidden=torch.randn(1,16,cfg.vlm_width),
                        motion_hidden=torch.randn(1,len(net.native_queries),cfg.vlm_width),
                        visual_mask=torch.ones(1,16,dtype=torch.bool), current_map=torch.rand(1,16))
        for step in (0, 1):
            # Step 1: rank 0 uses Pick, rank 1 uses Place. Step 2: rank 0 uses Place,
            # rank 1 has a Pick prediction but no valid Trace on its synthetic sample.
            route = torch.tensor([rank if step == 0 else 1 - rank])
            active = torch.tensor([step == 0 or rank == 0])
            opt.zero_grad(set_to_none=True)
            output = ddp(features, skill_id=route, active=active)
            supervised = output['end'] if version == 'joint_decode' else output['trace']
            (supervised.square().mean() * active.float().mean()).backward()
            for layer in net.layers:
                for k, expert in enumerate(layer.experts):
                    used = k == 1 or (step == 0 and k == 0)
                    assert all((p.grad is not None) == used for p in expert.parameters())
                    if used:
                        assert expert.up.weight.grad.norm() > 0
            param = net.layers[0].experts[0].up.weight
            before = param.detach().clone()
            opt.step()
            if step == 0:
                assert not torch.equal(param, before)
                assert opt.state[param]['exp_avg'].norm() > 0
                momentum = opt.state[param]['exp_avg'].clone()
            else:
                torch.testing.assert_close(param, before, atol=0, rtol=0)
                torch.testing.assert_close(opt.state[param]['exp_avg'], momentum, atol=0, rtol=0)
                assert opt.state[param]['step'] == 1
            for expert in net.layers[0].experts[:2]:
                gathered = [torch.empty_like(expert.up.weight) for _ in range(2)]
                dist.all_gather(gathered, expert.up.weight.detach())
                torch.testing.assert_close(*gathered, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason='requires CPU DDP')
@pytest.mark.parametrize('version', ['trajectory', 'boundary_first', 'joint_decode'])
def test_sparse_expert_grad_none_preserves_momentum(tmp_path, version):
    mp.spawn(_sparse_worker, args=((tmp_path / 'rendezvous').as_uri(), version), nprocs=2, join=True)
