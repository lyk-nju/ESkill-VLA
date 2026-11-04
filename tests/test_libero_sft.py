"""One episode format for π0.5 and ESkill: views, action tail and normalization are explicit settings."""
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.policies.libero_policy import LiberoInputs
from openpi.transforms import ModelTransforms, Normalize


def dual_camera_episode(tmp_path):
    """The fixture episode with synchronized actions and a second, encoded wrist view."""
    from scripts.preprocess.export import write_index
    from test_training_data import saved_episode

    path = saved_episode(tmp_path)
    meta = json.loads(path.read_text())
    with np.load(path.parent / 'arrays.npz') as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['actions'] = arrays['actions'][:4]
    arrays['action_valid'] = arrays['action_valid'][:4]
    arrays['action_indices'] = arrays['pairing'] = np.arange(4)
    np.savez(path.parent / 'arrays.npz', **arrays)
    for ref in (*[meta['actions'][k] for k in ('values', 'valid', 'frame_indices')],
                meta['actions']['conditioning_image_indices']['image']):
        ref['shape'][0] = 4
    encoded, offsets = [], [0]
    for frame in range(4):
        data = BytesIO()
        Image.fromarray(np.full((32, 32, 3), frame + 100, np.uint8)).save(data, format='PNG')
        encoded.append(data.getvalue())
        offsets.append(offsets[-1] + len(encoded[-1]))
    np.savez(path.parent / 'images.npz', wrist_image_bytes=np.frombuffer(b''.join(encoded), np.uint8),
             wrist_image_offsets=np.asarray(offsets, np.int64))
    image = meta['cameras']['image']
    meta['cameras']['wrist_image'] = dict(
        frame_indices=image['frame_indices'], valid=image['valid'], height=32, width=32, calibration=None,
        rgb=dict(uri='images.npz', key='wrist_image_bytes', offsets_key='wrist_image_offsets',
                 encoding='jpeg_sequence', shape=[4, 32, 32, 3], dtype='uint8'))
    meta['actions']['conditioning_image_indices']['wrist_image'] = meta['actions']['conditioning_image_indices']['image']
    path.write_text(json.dumps(meta))
    write_index(tmp_path / 'index.json', [path])
    return tmp_path / 'index.json', arrays


@pytest.mark.parametrize('auxiliary', [False, True])
def test_dual_camera_views_and_repeated_tail_come_from_one_episode(tmp_path, auxiliary):
    index, arrays = dual_camera_episode(tmp_path)
    data = EpisodeDataset(index, action_horizon=3, cameras=['image', 'wrist_image'], auxiliary=auxiliary,
                          action_padding='repeat_last', skill_names=ESkillConfig().skill_names)
    sample = data[3]
    images = LiberoInputs()(sample['inputs'])['images']
    reader = data.reader(0)
    np.testing.assert_array_equal(images[0], reader.rgb('image', 3))
    assert len(images) == 2 and np.all(images[1] == 103)
    np.testing.assert_array_equal(sample['targets']['actions'], np.repeat(arrays['actions'][3:], 3, axis=0))
    assert sample['targets']['action_valid'].all()
    if auxiliary:
        # Future skills still stop at the last recorded action.
        assert sample['targets']['future_skill_valid'][0].tolist() == [True, False, False]
        assert not sample['targets']['future_skill_valid'][1].any()
        assert [len(roles) for roles in sample['targets']['role_sources']] == [3, 3]
    else:
        assert set(sample['targets']) == {'actions', 'action_valid'}


def test_masked_tail_and_missing_camera_are_explicit(tmp_path):
    index, arrays = dual_camera_episode(tmp_path)
    data = EpisodeDataset(index, action_horizon=3, cameras=['image'], auxiliary=False, action_padding='mask')
    targets = data[3]['targets']
    np.testing.assert_array_equal(targets['actions'][0], arrays['actions'][3])
    assert targets['action_valid'][0].all() and not targets['action_valid'][1:].any()
    assert not targets['actions'][1:].any()
    with pytest.raises(ValueError, match="camera 'missing'"):
        EpisodeDataset(index, action_horizon=3, cameras=['image', 'missing'])


def test_quantile_roundtrip_and_action_only_batch(tmp_path):
    index, arrays = dual_camera_episode(tmp_path)
    data = EpisodeDataset(index, action_horizon=3, cameras=['image', 'wrist_image'], auxiliary=False)
    stats = {name: dict(method='quantile', q01=[-2.] * size, q99=[3.] * size)
             for name, size in (('state', 8), ('actions', 7))}
    actions = arrays['actions']
    norm = Normalize(stats)
    np.testing.assert_allclose(norm.apply('actions', norm.apply('actions', actions), inverse=True), actions, atol=1e-6)

    def tokenizer(prompt, state):
        assert prompt == 'pick cup then place it' and state is None
        return np.zeros(8, np.int64), np.ones(8, bool)

    config = ESkillConfig.tiny(discrete_state_input=False, mask_action_loss=False)
    sample = data[1]
    sample['inputs'] = LiberoInputs(use_arm_state=False)(sample['inputs'])
    batch = ModelTransforms(config, stats, tokenizer)(sample)
    assert batch['inputs']['images'].shape == (2, 3, 32, 32)
    assert set(batch['inputs']) == {'images', 'image_mask', 'tokens', 'token_mask', 'state', 'arm_presence'}
    assert set(batch['targets']) == {'actions', 'action_valid'}
    assert not batch['targets']['actions'][:, 7:].any()

    stats['arm_state'] = dict(method='standard', mean=[[0.] * 15] * 2, std=[[2.] * 15] * 2)
    np.testing.assert_array_equal(Normalize(stats).apply('arm_state', np.ones((2, 15))), .5)


def test_standard_flow_loss_includes_zero_padded_dimensions():
    torch = pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from openpi.models.pi0 import Pi0
    from test_eskill_model import batch
    config = ESkillConfig.tiny(vocab_size=128, mask_action_loss=False)
    model = Pi0(config).eval()
    inputs, targets = batch(config)
    condition, _ = model.predict_condition(inputs)
    action = targets['actions']
    noise, time = torch.randn_like(action), torch.full((len(action),), .5)
    expected = (model.velocity(.5 * noise + .5 * action, time, condition) - (noise - action)).square().mean()
    actual = model.flow_loss(action, torch.zeros_like(action, dtype=torch.bool), condition, noise=noise, time=time)
    torch.testing.assert_close(actual, expected)
