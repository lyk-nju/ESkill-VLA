"""Skill/Motion information flow, semantic experts, fixed-time heads and action visibility."""
from dataclasses import replace
from unittest import mock

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')
AutoencoderKL = pytest.importorskip('diffusers').AutoencoderKL

from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill


def batch(config):
    b, m, s = 2, getattr(config, 'motion_queries', 8), getattr(config, 'canvas_size', 32)
    inputs = dict(images=torch.randn(b, 2, 3, config.image_size, config.image_size),
                  image_mask=torch.ones(b, 2, dtype=torch.bool), tokens=torch.ones(b, 5, dtype=torch.long),
                  token_mask=torch.ones(b, 5, dtype=torch.bool),
                  arm_presence=torch.tensor([[1, 0], [1, 1]], dtype=torch.bool),
                  arm_state=torch.randn(b, 2, 8), effector_start=torch.rand(b, 2, 2),
                  effector_start_valid=torch.ones(b, 2, dtype=torch.bool))
    targets = dict(actions=torch.randn(b, config.action_horizon, config.action_dim),
                   action_valid=torch.ones(b, config.action_horizon, config.action_dim, dtype=torch.bool),
                   skill_labels=torch.tensor([[0, -1], [1, 5]]), skill_valid=torch.ones(b, 2, dtype=torch.bool),
                   role_images=torch.randn(b, 2, 2, 3, s, s), role_weights=torch.ones(b, 2, 2, s, s),
                   role_valid=torch.ones(b, 2, 2, dtype=torch.bool),
                   trace=torch.rand(b, 2, m, 2), trace_valid=torch.ones(b, 2, m, dtype=torch.bool),
                   gripper=torch.randint(0, 2, (b, 2, m)).float(),
                   gripper_valid=torch.ones(b, 2, m, dtype=torch.bool),
                   future_padded=torch.zeros(b, 2, m, dtype=torch.bool))
    return inputs, targets


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(7)
    config = ESkillConfig.tiny(vocab_size=128, max_token_len=8)
    vae = AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 2,
                        up_block_types=('UpDecoderBlock2D',) * 2, block_out_channels=(16, 32),
                        layers_per_block=1, latent_channels=4, norm_num_groups=8)
    return Pi0ESkill(config, vae=vae)


def gradient(module):
    return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)


def has_gradient(module):
    return gradient(module) > 0


def test_small_experts_and_independent_time_queries():
    config = ESkillConfig()
    assert config.action_horizon == 10 and config.arm_state_dim == 8
    assert config.motion_offsets_s == (.25, .5, .75, 1., 1.25, 1.5, 1.75, 2.)
    assert (config.skill_dim, config.skill_mlp, config.skill_depth) == (512, 2048, 2)
    assert config.active_skill_ids == tuple(range(6))
    assert ESkillConfig.tiny(action_horizon=3).action_horizon == 3
    with pytest.raises(ValueError, match='offsets'):
        replace(config, motion_offsets_s=(.25, .1))
    with pytest.raises(ValueError, match='six'):
        replace(config, active_skill_ids=tuple(range(9)))


def test_forward_six_losses_and_local_denominators(model):
    inputs, targets = batch(model.config)
    loss = model(inputs, targets)
    assert set(loss) == {'action', 'skill', 'current', 'future_object', 'trace', 'gripper'}
    assert all(torch.isfinite(x) for x in loss.values())
    assert model.loss_counts['skill'] == 3
    assert model.loss_counts['current'] == 3 and model.loss_counts['future_object'] == 3
    assert model.loss_counts['trace'] == 3 * model.config.motion_queries
    assert model.loss_counts['gripper'] == 3 * model.config.motion_queries
    assert model.diagnostic_statistics['gripper_accuracy'][1] == model.loss_counts['gripper']
    sum(loss.values()).backward()
    for module in (model.skill.classifier, model.skill.decoder, model.skill.shared,
                   model.grounding, model.motion.trace_head, model.motion.gripper_head,
                   model.visual_supervision.head, model.action_out_proj):
        assert gradient(module) > 0
    assert all(p.grad is None for p in model.visual_supervision.vae.parameters())
    assert all(p.grad is None for p in model.paligemma_with_expert.paligemma.parameters())


def test_action_gradient_reaches_continuous_skill_and_grounding_not_router(model):
    inputs, targets = batch(model.config)
    model(inputs, targets)['action'].backward()
    for module in (model.skill.decoder, model.skill.shared, model.skill.to_motion,
                   model.grounding, model.motion, model.auxiliary_kv):
        assert gradient(module) > 0
    assert gradient(model.skill.classifier) == 0
    assert gradient(model.motion.trace_head) == 0 and gradient(model.motion.gripper_head) == 0
    assert not hasattr(model.skill, 'table') and not hasattr(model, 'step_projection')


def test_predicted_top1_only_selected_expert_has_nonzero_gradient(model):
    inputs, _ = batch(model.config)
    with torch.no_grad():
        model.skill.classifier.weight.zero_()
        model.skill.classifier.bias.fill_(-10)
        model.skill.classifier.bias[3] = 10
    condition, prediction = model.predict_condition(inputs)
    assert (prediction['skill_ids'] == 3).all()
    (condition.values[0].square().sum()).backward()
    for index, expert in enumerate(model.skill.experts):
        assert (gradient(expert) > 0) == (index == 3)
        # Zero-gradient edges permit ordinary DDP without rank-dependent unused parameters.
        assert all(p.grad is not None for p in expert.parameters())


def test_two_grounding_roles_and_only_motion_action_slots(model):
    inputs, _ = batch(model.config)
    condition, prediction = model.predict_condition(inputs)
    prefix_count = 2 * (model.config.image_size // model.config.patch_size) ** 2 + inputs['tokens'].shape[1]
    assert prediction['grounding'].shape == (2, 2, 2, model.config.role_queries, model.config.hidden_dim)
    assert prediction['motion'].shape == (2, 2, 8, model.config.hidden_dim)
    assert condition.mask.shape == (2, prefix_count + 16)
    assert condition.mask[0, prefix_count:].sum() == 8
    assert condition.mask[1, prefix_count:].sum() == 16


def test_readout_uses_known_start_and_same_eight_tokens(model):
    inputs, _ = batch(model.config)
    _, prediction = model.predict_condition(inputs)
    with torch.no_grad():
        model.motion.trace_head[-1].weight.zero_()
        model.motion.trace_head[-1].bias.zero_()
    trace, gripper = model.read_motion(prediction)
    torch.testing.assert_close(trace, prediction['effector_start'][:, :, None].expand(-1, -1, 8, -1))
    assert gripper.shape == (2, 2, 8)


def test_absent_arm_state_and_coordinates_cannot_change_present_action(model):
    model.eval()
    inputs, _ = batch(model.config)
    noise = torch.randn(2, model.config.action_horizon, model.config.action_dim)
    expected = model.sample_actions(inputs, noise=noise, steps=2)[0]
    inputs['arm_state'][0, 1] = float('nan')
    inputs['effector_start'][0, 1] = float('nan')
    actual = model.sample_actions(inputs, noise=noise, steps=2)[0]
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


def test_terminal_padding_and_unsupported_classes_never_supervise(model):
    inputs, targets = batch(model.config)
    targets['skill_labels'].fill_(7)
    targets['future_padded'].fill_(True)
    targets['trace'].fill_(float('nan'))
    targets['gripper'].fill_(float('nan'))
    losses = model(inputs, targets)
    assert losses['skill'] == losses['trace'] == losses['gripper'] == 0
    assert model.loss_counts['skill'] == model.loss_counts['trace'] == model.loss_counts['gripper'] == 0
    sum(losses.values()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_future_target_masks_do_not_hide_online_motion(model):
    model.eval()
    inputs, targets = batch(model.config)
    noise = torch.randn_like(targets['actions'])
    time = torch.full((2,), .5)
    expected = model(inputs, targets, noise=noise, time=time)['action']
    targets['skill_valid'].zero_()
    targets['role_valid'].zero_()
    targets['trace_valid'].zero_()
    targets['gripper_valid'].zero_()
    actual = model(inputs, targets, noise=noise, time=time)['action']
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_action_visibility_both_cameras_preserves_rope_and_upstream_features(model):
    inputs, _ = batch(model.config)
    model.config = replace(model.config, action_patch_mask_prob=(1., 1.))
    model.eval()
    full, full_prediction = model.predict_condition(inputs)
    model.train()
    masked, masked_prediction = model.predict_condition(inputs)
    patches = 2 * (model.config.image_size // model.config.patch_size) ** 2
    assert not masked.mask[:, :patches].any()
    assert masked.mask[:, patches:].any()
    torch.testing.assert_close(full.position_mask, masked.position_mask)
    for key in full_prediction:
        torch.testing.assert_close(full_prediction[key], masked_prediction[key], rtol=0, atol=0)
    for x, y in zip(full.keys + full.values, masked.keys + masked.values, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    from openpi.models import pi0
    with mock.patch.object(pi0, 'qkv', wraps=pi0.qkv) as qkv:
        model.velocity(torch.randn(2, model.config.action_horizon, model.config.action_dim),
                       torch.full((2,), .5), masked)
    expected = full.position_mask.sum(-1, keepdim=True) + torch.arange(model.config.action_horizon)
    for call in qkv.call_args_list:
        torch.testing.assert_close(call.args[-1], expected)


def test_unmasked_training_examples_and_inference_skip_auxiliary_readouts(model):
    model.config = replace(model.config, action_patch_mask_prob=(1., 1.), action_unmasked_prob=1.)
    inputs, _ = batch(model.config)
    condition, _ = model.predict_condition(inputs)
    torch.testing.assert_close(condition.mask, condition.position_mask)
    model.eval()
    with (mock.patch.object(model.motion.trace_head, 'forward', side_effect=AssertionError('trace head at inference')),
         mock.patch.object(model.motion.gripper_head, 'forward', side_effect=AssertionError('gripper head at inference')),
         mock.patch.object(model, 'action_visibility', wraps=model.action_visibility) as visibility):
        actions, _ = model.sample_actions(inputs, steps=3)
    assert torch.isfinite(actions).all() and visibility.call_count == 1


def test_inference_checkpoint_and_bfloat16(model):
    model.eval()
    other = Pi0ESkill(model.config).eval()
    other.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith('visual_supervision.')})
    inputs, targets = batch(model.config)
    noise = torch.randn_like(targets['actions'])
    torch.testing.assert_close(model.sample_actions(inputs, noise=noise, steps=1)[0],
                               other.sample_actions(inputs, noise=noise, steps=1)[0])
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        losses = model(inputs, targets)
    sum(losses.values()).backward()
    assert all(torch.isfinite(x) for x in losses.values())
    assert gradient(model.skill.shared) > 0


def distributed_worker(rank, rendezvous):
    from copy import deepcopy
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    torch.set_num_threads(1)
    torch.manual_seed(17)
    config = ESkillConfig.tiny(vocab_size=128, max_token_len=8)
    vae = AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 2,
                        up_block_types=('UpDecoderBlock2D',) * 2, block_out_channels=(16, 32),
                        layers_per_block=1, latent_channels=4, norm_num_groups=8)
    model = Pi0ESkill(config, vae=vae)
    inputs, targets = batch(config)
    captured = []
    handle = model.skill.classifier.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    with torch.no_grad():
        model.predict_condition(inputs)
    handle.remove()
    # Establish two distinct predicted specialists under ONE common classifier.
    first, second = captured[0][0], captured[0][2]
    direction = (first - second) / (first - second).norm()
    midpoint = (first + second).dot(direction) / 2
    with torch.no_grad():
        model.skill.classifier.weight.zero_()
        model.skill.classifier.bias.fill_(-100)
        model.skill.classifier.weight[0].copy_(direction)
        model.skill.classifier.weight[1].copy_(-direction)
        model.skill.classifier.bias[0] = -midpoint
        model.skill.classifier.bias[1] = midpoint
        _, prediction = model.predict_condition(inputs)
        assert prediction['skill_ids'][0, 0] != prediction['skill_ids'][1, 0]
    noise, time = torch.randn_like(targets['actions']), torch.tensor([.3, .7])
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        wrapped = DistributedDataParallel(model, broadcast_buffers=False)
        # One rank lacks all auxiliary labels; another iteration lacks them globally.
        for globally_empty in (False, True):
            for key in ('skill_valid', 'role_valid', 'trace_valid', 'gripper_valid'):
                targets[key].fill_(not globally_empty)
                targets[key][0] = False
            local_inputs = {key: value[rank:rank+1] for key, value in inputs.items()}
            local_targets = {key: value[rank:rank+1] for key, value in targets.items()}
            model.zero_grad(set_to_none=True)
            local_losses = wrapped(local_inputs, local_targets, noise=noise[rank:rank+1], time=time[rank:rank+1])
            sum(local_losses.values()).backward()
            logged = torch.stack([value.detach().clone() for value in local_losses.values()])
            dist.all_reduce(logged)
            logged /= 2
            reference = deepcopy(model)
            reference.zero_grad(set_to_none=True)
            with mock.patch('openpi.models.model.dist.is_initialized', return_value=False):
                expected = reference(inputs, targets, noise=noise, time=time)
            sum(expected.values()).backward()
            torch.testing.assert_close(logged, torch.stack(list(expected.values())), rtol=1e-4, atol=1e-5)
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                actual = parameter.grad
                desired = reference.get_parameter(name).grad
                assert actual is not None and desired is not None, name
                torch.testing.assert_close(actual, desired, rtol=3e-3, atol=2e-5, msg=name)
    finally:
        dist.destroy_process_group()


def test_ddp_matches_global_batch_with_sparse_routes_and_empty_labels(tmp_path):
    import torch.distributed as dist
    import torch.multiprocessing as mp
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Requires Gloo')
    mp.spawn(distributed_worker, args=((tmp_path / 'rendezvous').as_uri(),), nprocs=2, join=True)
