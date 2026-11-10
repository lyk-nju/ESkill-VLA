"""Full-parameter AMP, visual recomputation and the eight-GPU recipe/launcher."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest
import torch

from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig, create_model
from test_eskill_model import batch


def test_visual_recomputation_preserves_full_model_gradients_and_fp32_optimizer():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    config = TrainConfig('unused', 'unused', 'unused', 'unused', device='cpu', smoke=True,
                         model=ESkillConfig.tiny(vocab_size=128, freeze_vlm=False,
                                                gradient_checkpointing=True),
                         backbone_dtype='float32', precision='bfloat16')
    model = create_model(config, initialize=False)
    reference = deepcopy(model)
    reference.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing_disable()
    x, y = batch(config.model)
    noise, time = torch.randn_like(y['actions']), torch.full((2,), .5)
    calls = []
    layer = model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers[0]
    hook = layer.register_forward_pre_hook(lambda *args: calls.append(1))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model(x, y, noise=noise, time=time)
        expected = reference(x, y, noise=noise, time=time)
    sum(actual.values()).backward()
    sum(expected.values()).backward()
    hook.remove()
    assert len(calls) == 2  # The vision layer actually recomputed during backward.
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key])
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == torch.float32
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad)
        else:
            assert name.startswith('visual_supervision.vae.'), name
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    optimizer.step()
    assert all(state['exp_avg'].dtype == state['exp_avg_sq'].dtype == torch.float32
               for state in optimizer.state.values())


@pytest.mark.parametrize('resume_step', [0, 10000])
def test_gpu10_recipe_and_eight_card_pipeline(tmp_path, monkeypatch, resume_step):
    from experiments.eskill_libero40 import run

    config = TrainConfig.load(Path(__file__).resolve().parents[1] / 'examples/libero/train_eskill.json')
    config.batch_size, config.num_steps, config.save_every = 16, 40000, 5000
    config.model = __import__('dataclasses').replace(config.model, freeze_vlm=False)
    config.adam_fused = True
    assert config.batch_size * 8 == 128
    assert config.lora_rank == 0 and not config.model.freeze_vlm
    assert config.backbone_dtype == 'float32' and config.precision == 'bfloat16'
    assert config.adam_fused and config.model.gradient_checkpointing
    assert config.save_every == 5000 and config.num_steps == 40000
    assert config.validation_fraction == .1 and config.split_path.endswith('split.json')
    assert config.model.arm_state_dim == 8 and not config.model.discrete_state_input
    config.output_dir = str(tmp_path/'run/train')
    path = tmp_path/'recipe.json'
    config.save(path)
    simulator = tmp_path/'venv/bin/python'
    simulator.parent.mkdir(parents=True)
    simulator.symlink_to(sys.executable)
    assert simulator.resolve() != simulator
    monkeypatch.setattr(sys, 'argv', ['run.py', '--config', str(path), '--gpus', *map(str, range(8)),
                                     '--sim-python', str(simulator), '--resume-step', str(resume_step)])
    if resume_step:
        train = Path(config.output_dir)
        train.mkdir(parents=True)
        (train/f'checkpoint_{resume_step:06d}.pt').touch()
        (train/'training_check.json').write_text(json.dumps(dict(step=resume_step, all_ranks_equal=True)))
    # Restore launcher globals after the test; no training subprocesses are launched.
    for name in ('SETTINGS', 'DATA', 'SIM', 'OPENPI'):
        monkeypatch.setattr(run, name, getattr(run, name))
    jobs, evaluations = [], []

    def job(name, command, env, timeout):
        assert '--nproc_per_node=8' in command
        assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'
        settings = json.loads(Path(command[command.index('--config')+1]).read_text())
        step = settings['num_steps']
        assert ('--resume' in command) == (step > 10000)
        jobs.append(step)
        train = Path(settings['output_dir'])
        train.mkdir(parents=True, exist_ok=True)
        (train/'training_check.json').write_text(json.dumps(dict(step=step, all_ranks_equal=True)))

    def evaluate(gpu, suite, step, checkpoint):
        evaluations.append((gpu, suite, step))
        return dict(episodes=50, errors=0)

    monkeypatch.setattr(run, 'job', job)
    monkeypatch.setattr(run, 'evaluate', evaluate)
    monkeypatch.setattr(run.subprocess, 'check_output', lambda *args, **kwargs: 'test\n')
    preflight = []
    monkeypatch.setattr(run.subprocess, 'run', lambda command, **kwargs: preflight.append(command))
    run.main()
    assert jobs == ([10000] if not resume_step else []) + [20000, 30000, 40000]
    assert len(evaluations) == 16 and {row[0] for row in evaluations} == {0, 1, 2, 3}
    runtime = json.loads((tmp_path/'run/runtime.json').read_text())
    assert runtime['world_size'] == 8 and runtime['global_batch_size'] == 128
    assert runtime['sim_python'] == str(simulator)  # Preserve the venv, including its symlink.
    assert preflight[0][0] == str(simulator) and '--list-tasks' in preflight[0]
    assert json.loads((tmp_path/'run/status.json').read_text())['phase'] == 'complete'
