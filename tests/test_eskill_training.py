"""ESkill configuration, weighted validation, SFT transfer and resumed policy roundtrip."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('diffusers')
sentencepiece = pytest.importorskip('sentencepiece')

from openpi.models.model import masked_mean
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import ESkillConfig, Pi0Config, SKILLS
from openpi.policies.policy_config import create_trained_policy
from openpi.training.checkpoints import resource_identities
from openpi.training.config import TrainConfig, create_model
from openpi.training.metrics import phase_metrics
from openpi.training.validation import validation_losses
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from scripts.train import train
from test_training_data import saved_episode


def test_config_roundtrip_selects_one_eskill_design():
    base = dict(data_index='index', tokenizer_path='tokenizer', output_dir='output', norm_stats='stats')
    config = TrainConfig.from_dict(base)
    assert type(config.model) is ESkillConfig
    assert config.model.skill_dim == 512 and config.model.skill_mlp == 2048
    assert config.model.active_skill_ids == tuple(range(6))
    assert config.model.motion_offsets_s == tuple(np.arange(1, 9) * .25)
    assert TrainConfig.from_dict(config.to_dict()).to_dict() == config.to_dict()
    baseline = TrainConfig.from_dict(dict(base, model_type='pi05'))
    assert type(baseline.model) is Pi0Config and baseline.loss_weights == {'action': 1.}
    with pytest.raises(ValueError, match='loss_weights'):
        replace(config, loss_weights={'action': 1.})
    with pytest.raises(ValueError, match='first camera'):
        replace(config, cameras=('wrist_image', 'image'))


def test_eskill_dashboard_has_no_removed_supervision():
    metrics = phase_metrics('train', dict(loss=12., action=1., skill=2., current=3.,
                                         future_object=4., trace=5., gripper=6.),
                            dict(current_skill_accuracy=.8, current_skill_count=3,
                                 gripper_accuracy=.5, gripper_accuracy_count=2,
                                 gripper_transition_accuracy=1., gripper_transition_accuracy_count=0))
    assert metrics['train/skill_loss'] == 2.
    assert metrics['train/current_object_loss'] == 3.
    assert metrics['train/future_object_loss'] == 4.
    assert metrics['train/gripper_loss'] == 6.
    assert metrics['aux_train/gripper_accuracy'] == .5
    assert not any('future_skill' in key or 'grounding_ee' in key or 'transition' in key for key in metrics)


def test_eskill_validation_uses_effective_loss_counts_and_restores_rng():
    class Model(torch.nn.Module):
        def forward(self, inputs, targets):
            torch.rand(1)
            self.loss_counts = {key: valid.sum() for key, valid in targets['valid'].items()}
            return {key: masked_mean(targets['value'][key], valid)
                    for key, valid in targets['valid'].items()}
    config = TrainConfig('unused', 'unused', 'unused', 'unused', model_type='eskill',
                         model=ESkillConfig.tiny(), device='cpu', precision='float32')
    batches = []
    for value, n in ((9., 1), (1., 9)):
        valid = {key: torch.arange(9) < n for key in config.loss_weights}
        batches.append(dict(inputs={}, targets=dict(value={key: torch.full((9,), value) for key in valid},
                                                   valid=valid, actions=torch.ones(1),
                                                   action_valid=torch.ones(1, dtype=torch.bool))))
    model = Model().train()
    rng = torch.get_rng_state().clone()
    result = validation_losses(model, batches, config)
    for key in config.loss_weights:
        assert result[key] == pytest.approx(1.8)
    assert result['loss'] == pytest.approx(10.8)
    assert model.training and torch.equal(torch.get_rng_state(), rng)


def test_eskill_sft_transfer_train_resume_and_online_roundtrip(tmp_path):
    torch.set_num_threads(2)
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('\n'.join(text for _, text in SKILLS) + '\npick cup then place it')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / 'tokenizer'),
        vocab_size=64, hard_vocab_limit=False, minloglevel=2)

    def edit(meta, arrays):
        arrays['timestamps'] = np.array([0., .25, .5, .75], np.float64)
        meta['timestamps_s'] = dict(uri='arrays.npz', key='timestamps', shape=[4], dtype='float64')
        meta['actions']['gripper_convention'] = dict(channels={'single': 6}, close_sign=1,
            alignment='observation_then_action', evidence='fixture commands follow each observation')
        arrays['state'][:, -8:-5] = arrays['eef']
        arrays['actions'][:, 6] = [-1., -1., 1., 1., -1., -1.]
    index = tmp_path / 'index.json'
    write_index(index, [saved_episode(tmp_path, identity=identity, edit=edit) for identity in ('one', 'two')])
    base_model = Pi0Config.tiny(vocab_size=128, action_horizon=10, discrete_state_input=False)
    source = TrainConfig(str(index), str(tmp_path / 'tokenizer.model'), str(tmp_path / 'sft'),
                         str(tmp_path / 'stats.json'), model_type='pi05', model=base_model,
                         validation_fraction=0, num_workers=0, device='cpu', precision='float32', smoke=True)
    stats = compute(source)
    Path(source.norm_stats).write_text(json.dumps(stats))
    torch.manual_seed(12)
    sft_weights = Pi0(base_model).state_dict()
    sft = tmp_path / 'sft.pt'
    torch.save(dict(model=sft_weights, config=source.to_dict(), norm_stats=stats,
                    resources=resource_identities(source)), sft)
    cfg = ESkillConfig.tiny(vocab_size=128, action_patch_mask_prob=(.5,), action_unmasked_prob=.25)
    config = replace(source, model_type='eskill', model=cfg, init_checkpoint=str(sft),
                     loss_weights=None, output_dir=str(tmp_path / 'staged'), num_steps=2,
                     batch_size=3, save_every=2, log_every=2, val_interval=0,
                     learning_rate=1e-3, norm_stats=str(tmp_path / 'eskill_stats.json'))
    eskill_stats = compute(config)
    assert eskill_stats['actions'] == stats['actions'] and eskill_stats['state'] == stats['state']
    assert 'arm_state' not in eskill_stats
    Path(config.norm_stats).write_text(json.dumps(eskill_stats))
    initialized = create_model(config, init_checkpoint=sft)
    for name, value in sft_weights.items():
        torch.testing.assert_close(initialized.state_dict()[name], value, rtol=0, atol=0)
    assert not hasattr(initialized, 'step_projection')

    checkpoint = train(config)
    config.num_steps = 3
    train(config, resume=checkpoint)
    staged = torch.load(checkpoint, weights_only=False)
    direct = replace(config, output_dir=str(tmp_path / 'direct'), norm_stats=str(tmp_path / 'eskill_stats.json'))
    continuous = torch.load(train(direct), weights_only=False)
    assert staged['step'] == continuous['step'] == 3
    for name in staged['model']:
        torch.testing.assert_close(staged['model'][name], continuous['model'][name], rtol=0, atol=0)
    records = [json.loads(row) for row in (checkpoint.parent / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in records] == [1, 2, 3]
    assert all(np.isfinite(row[key]) for row in records for key in config.loss_weights)
    assert all(row['expert_gradients']['shared'] > 0 for row in records)
    sft.unlink()  # ESkill inference uses its own checkpoint, not the initialization run.
    policy = create_trained_policy(checkpoint)
    observation = dict(images=[np.zeros((32, 32, 3), np.uint8)], state=np.zeros(8, np.float32),
                       prompt='pick cup', effector_start=np.array([[.5, .5], [0., 0.]], np.float32),
                       effector_start_valid=np.array([True, False]))
    result = policy.infer(observation, steps=2)
    assert result['actions'].shape == (10, 7) and np.isfinite(result['actions']).all()
    assert 'future_skills' not in result
    assert result['current_skills'][1] == -1
    assert policy.model.visual_supervision is None
