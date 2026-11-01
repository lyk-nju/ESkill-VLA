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
