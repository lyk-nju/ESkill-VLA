"""Ordinary SFT must preserve source pairing and keep events out of the inputs."""
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.libero_dataset import LiberoSFTDataset
from openpi.policies.libero_policy import LiberoSFTInputs
from openpi.transforms import ModelTransforms, Normalize


def test_event_join_preserves_sft_views_and_tail_but_masks_future_padding(tmp_path):
    from openpi.policies.eskill_dataset import EpisodeReader
    from openpi.policies.libero_dataset import LiberoESkillDataset
    from openpi.policies.libero_policy import LiberoInputs
    from scripts.preprocess.export import write_index
    from test_training_data import saved_episode

    path = saved_episode(tmp_path)
    meta = json.loads(path.read_text())
    # This join is for synchronized LIBERO trajectories, unlike the asynchronous fixture.
    with np.load(path.parent / 'arrays.npz') as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays['actions'] = arrays['actions'][:4]
    arrays['action_valid'] = arrays['action_valid'][:4]
    arrays['action_indices'] = arrays['pairing'] = np.arange(4)
    np.savez(path.parent / 'arrays.npz', **arrays)
    for ref in (*[meta['actions'][k] for k in ('values', 'valid', 'frame_indices')],
                meta['actions']['conditioning_image_indices']['image']):
        ref['shape'][0] = 4
    path.write_text(json.dumps(meta))
    write_index(tmp_path / 'events.json', [path])
    reader = EpisodeReader(path)
    sft = tmp_path / 'sft'
    sft.mkdir()
    (sft / 'index.json').write_text(json.dumps(dict(episodes=[dict(
        episode_id='one', task=meta['instructions'][0], start=0, frames=4)])))
    offsets = np.zeros((4, 2, 2), np.int64)
    with (sft / 'images.bin').open('wb') as file:
        for frame in range(4):
            for camera, image in enumerate((reader.rgb('image', frame), np.full((32, 32, 3), frame + 100, np.uint8))):
                encoded = BytesIO()
                Image.fromarray(image).save(encoded, format='PNG')
                offsets[frame, camera] = file.tell(), len(encoded.getvalue())
                file.write(encoded.getvalue())
    np.save(sft / 'image_offsets.npy', offsets)
    np.save(sft / 'state.npy', arrays['state'][:, 7:])
    np.save(sft / 'actions.npy', arrays['actions'])
    data = LiberoESkillDataset(tmp_path / 'events.json', sft_index=sft / 'index.json',
                              action_horizon=3, skill_names=ESkillConfig().skill_names)
    sample = data[3]
    images = LiberoInputs()(sample['inputs'])['images']
    np.testing.assert_array_equal(images[0], reader.rgb('image', 3))
    assert np.all(images[1] == 103)
    np.testing.assert_array_equal(sample['targets']['actions'], np.repeat(arrays['actions'][3:], 3, axis=0))
    assert sample['targets']['action_valid'].all()
    assert sample['targets']['future_skill_valid'][0].tolist() == [True, False, False]
    assert not sample['targets']['future_skill_valid'][1].any()
    assert sample['targets']['role_images'].shape[:2] == (2, 3)
    data.sft.arrays['actions'] = np.zeros_like(arrays['actions'])
    with pytest.raises(ValueError, match='SFT/event row mismatch'):
        data[3]


def test_packed_dual_camera_pairing_tail_and_quantile_roundtrip(tmp_path):
    entries = [dict(episode_id=str(i), task='pick cup', start=i * 2, frames=2) for i in range(2)]
    index = tmp_path / 'index.json'
    index.write_text(json.dumps(dict(episodes=entries)))
    offsets = np.zeros((4, 2, 2), np.int64)
    with (tmp_path / 'images.bin').open('wb') as file:
        for frame in range(4):
            for camera in range(2):
                data = BytesIO()
                Image.fromarray(np.full((16, 16, 3), 10 * frame + camera, np.uint8)).save(data, format='PNG')
                encoded = data.getvalue()
                offsets[frame, camera] = file.tell(), len(encoded)
                file.write(encoded)
    actions = np.arange(28, dtype=np.float32).reshape(4, 7) / 10
    np.save(tmp_path / 'actions.npy', actions)
    np.save(tmp_path / 'state.npy', np.ones((4, 8), np.float32))
    np.save(tmp_path / 'image_offsets.npy', offsets)
    dataset = LiberoSFTDataset(index, action_horizon=3)
    sample = dataset[1]
    assert np.all(sample['inputs']['image'] == 10)
    assert np.all(sample['inputs']['wrist_image'] == 11)
    np.testing.assert_array_equal(sample['targets']['actions'], np.repeat(actions[1:2], 3, axis=0))
    assert sample['targets']['action_valid'].all()
    stats = dict(method='quantile')
    for name, size in (('state', 8), ('actions', 7)):
        stats[name] = dict(q01=[-2.] * size, q99=[3.] * size)
    norm = Normalize(stats)
    np.testing.assert_allclose(norm.apply('actions', norm.apply('actions', actions), inverse=True), actions, atol=1e-6)

    def tokenizer(prompt, state):
        assert prompt == 'pick cup' and state is None
        return np.zeros(8, np.int64), np.ones(8, bool)

    config = ESkillConfig.tiny(discrete_state_input=False, mask_action_loss=False)
    transforms = ModelTransforms(config, stats, tokenizer)
    sample['inputs'] = LiberoSFTInputs()(sample['inputs'])
    batch = transforms(sample)
    assert batch['inputs']['images'].shape == (2, 3, 32, 32)
    assert set(batch['inputs']) == {'images', 'image_mask', 'tokens', 'token_mask', 'state'}
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
