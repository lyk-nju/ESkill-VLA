import pytest

from openpi.models.pi0_config import ESkillConfig, Pi0Config
from openpi.training.config import TrainConfig


def test_mask_padding_cannot_be_used_with_unmasked_loss():
    with pytest.raises(ValueError, match='mask_action_loss'):
        TrainConfig('data', 'tokenizer', 'output', 'stats', action_padding='mask',
                    model=ESkillConfig(mask_action_loss=False))


def test_sft_statistics_require_explicit_matching_method(tmp_path):
    import torch
    from scripts.compute_norm_stats import compute
    from scripts.preprocess.export import write_index
    from test_training_data import saved_episode
    index = tmp_path / 'index.json'
    write_index(index, [saved_episode(tmp_path)])
    source = tmp_path / 'old.pt'
    for stats in ({'method': 'quantile', 'state': {}, 'actions': {}},
                  {'state': {'method': 'standard'}, 'actions': {'method': 'standard'}}):
        torch.save({'config': {}, 'norm_stats': stats}, source)
        config = TrainConfig(str(index), 'unused', 'unused', 'unused', validation_fraction=0,
                             model_type="pi05", init_checkpoint=str(source))
        with pytest.raises(ValueError, match='statistics must explicitly use quantile'):
            compute(config)


def test_eskill_rejects_image_augmentation_it_cannot_apply_to_labels():
    with pytest.raises(ValueError, match='augment_images'):
        TrainConfig('data', 'tokenizer', 'output', 'stats', model_type='eskill',
                    model=ESkillConfig(augment_images=True))
    TrainConfig('data', 'tokenizer', 'output', 'stats', model_type='pi05', model=Pi0Config(augment_images=True))


@pytest.mark.parametrize('stats', [{'state': {'mean': [0.], 'std': [1.]}, 'method': 'standard'},
                                   {'state': {'method': 'quantlie', 'q01': [0.], 'q99': [1.]}}])
def test_normalization_method_must_be_explicit_per_field(stats):
    import numpy as np
    from openpi.transforms import Normalize
    with pytest.raises(ValueError, match="'state' must set method"):
        Normalize(stats).apply('state', np.zeros(1))


def test_resources_may_move_but_content_must_match(tmp_path):
    import json
    import shutil
    from openpi.training.checkpoints import check_resources, resource_identities
    from scripts.preprocess.export import write_index
    from test_training_data import saved_episode
    first = tmp_path / 'first'
    write_index(first / 'index.json', [saved_episode(first)])
    (first / 'tokenizer.model').write_bytes(b'tokens')
    config = TrainConfig(str(first / 'index.json'), str(first / 'tokenizer.model'), 'out', 'stats')
    saved = dict(resources=resource_identities(config))
    moved = tmp_path / 'moved'
    shutil.copytree(first, moved)
    config.data_index, config.tokenizer_path = str(moved / 'index.json'), str(moved / 'tokenizer.model')
    check_resources(saved, config)
    episode = next(moved.glob('*/episode.json'))
    meta = json.loads(episode.read_text())
    meta['events'][0]['skill'] = 'Place'
    episode.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='data_index content differs'):
        check_resources(saved, config)
    with pytest.raises(ValueError, match='no resource identities'):
        check_resources({}, config)


def test_file_and_checkpoint_share_nested_config_loading(tmp_path):
    import copy
    import json
    from openpi.policies.policy_config import load_train_config
    data = dict(data_index='data', tokenizer_path='tokens', output_dir='out', norm_stats='stats',
                model={'hidden_dim': 64, 'discrete_state_input': False},
                wandb_info={'mode': 'disabled'})
    original = copy.deepcopy(data)
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(data))
    loaded = TrainConfig.load(path)
    restored = load_train_config({'config': data}, device=loaded.device)
    assert loaded.to_dict() == restored.to_dict()
    assert TrainConfig.from_dict(data).to_dict() == loaded.to_dict()
    assert data == original


def test_checkpoint_config_accepts_omitted_model_and_keeps_explicit_legacy_horizon():
    from openpi.policies.policy_config import load_train_config
    minimal = dict(data_index='data', tokenizer_path='tokens', output_dir='out', norm_stats='stats', device='cpu')
    for settings in (minimal, dict(minimal, model_type='eskill')):
        assert load_train_config({'config': settings}).to_dict() == TrainConfig.from_dict(settings).to_dict()
    saved = dict(minimal, model_type='eskill', model={'skill_dim': 64})
    assert load_train_config({'config': saved}).model.motion_min_horizon_frames == 0
    saved['model']['motion_min_horizon_frames'] = 10
    assert load_train_config({'config': saved}).model.motion_min_horizon_frames == 10
