"""Saved-head readout, GT isolation, action parity and new two-role panel output."""
from dataclasses import replace
from pathlib import Path
import json

import av
import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL

from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig, create_model
from openpi.training.checkpoints import resource_identities
from scripts.analysis.predictions import collect, VisualReadout
from scripts.prediction_viewer import main
from scripts.analysis.prediction_viewer import Renderer
from scripts.analysis.rollout_predictions import RecordedPolicy, load_rollout
from test_attention_viewer import checkpoints


@pytest.fixture(scope='module')
def decoder_checkpoint(checkpoints):
    root, paths = checkpoints
    saved = torch.load(paths['eskill'], map_location='cpu', weights_only=False)
    config = TrainConfig.from_dict(saved['config'])
    vae = AutoencoderKL(down_block_types=('DownEncoderBlock2D',) * 2,
                        up_block_types=('UpDecoderBlock2D',) * 2, block_out_channels=(16, 32),
                        layers_per_block=1, norm_num_groups=8, latent_channels=4)
    vae.save_pretrained(root / 'vae')
    config = replace(config, vae_path=str(root / 'vae'), model=replace(config.model, motion_stride=1))
    model = create_model(config)
    path = root / 'with_decoder.pt'
    torch.save(dict(model=model.state_dict(), config=config.to_dict(), norm_stats=saved['norm_stats'],
                    resources=resource_identities(config), weights_kind='full', step=0), path)
    return path, config


def test_readout_matches_native_actions_and_future_targets_cannot_change_predictions(decoder_checkpoint, monkeypatch):
    from scripts.analysis import predictions
    path, config = decoder_checkpoint
    source = EpisodeDataset(config.data_index, action_horizon=10, auxiliary=False, include_state=False)
    policy = create_trained_policy(path)
    noise = torch.randn(10, 32, generator=torch.Generator().manual_seed(4)).numpy()
    expected = policy.infer(source[0]['inputs'], noise=noise, steps=2)['actions']
    seen = []
    original_infer = predictions.create_trained_policy
    original_item = EpisodeDataset.__getitem__
    def factory(*args, **kwargs):
        result = original_infer(*args, **kwargs)
        original = result.infer
        def infer(observation, **options):
            assert set(observation) == {'images', 'prompt'}
            value = original(observation, **options)
            seen.append('infer')
            return value
        result.infer = infer
        return result
    def item(self, index):
        if self.auxiliary:
            assert seen and seen[-1] == 'infer'
            seen.append('GT')
        return original_item(self, index)
    monkeypatch.setattr(predictions, 'create_trained_policy', factory)
    monkeypatch.setattr(EpisodeDataset, '__getitem__', item)
    frames, _, info = collect(path, 'one', device='cpu', steps=2, seed=4, max_frames=1,
                              object_word='cup', target_word='it')
    value = frames[0]
    np.testing.assert_array_equal(value.prediction['actions'], expected)
    assert value.prediction['role_rgb'].shape == (2, 2, 32, 32, 3)
    assert value.prediction['role_roundtrip'].shape == (2, 2, 32, 32, 3)
    assert value.prediction['skill_prob'].shape == (2, 6)
    assert value.prediction['trace'].shape == (2, 8, 2)
    assert value.targets['trace_valid'][0].any()
    assert info['technical'] and 'MATCHED' in info['visual_readout']
    language = value.attention.language_attention
    assert language['valid'].all() and language['maps'].shape == (3, 1, 4, 4)
    np.testing.assert_allclose(language['probabilities'].sum(-1), 1., atol=1e-6)
    np.testing.assert_allclose(language['maps'].sum((-1, -2)), 1., atol=1e-6)
    np.testing.assert_allclose(language['probabilities'][:, :16].sum(-1), language['view_mass'][:, 0])
    original_canvases = predictions.role_canvases
    def changed_targets(*args, **kwargs):
        target = original_canvases(*args, **kwargs)
        target['role_images'].fill(1.)
        return target
    monkeypatch.setattr(predictions, 'role_canvases', changed_targets)
    changed, _, _ = collect(path, 'one', device='cpu', steps=2, seed=4, max_frames=1)
    for key in ('actions', 'trace', 'gripper_prob', 'skill_prob', 'role_rgb'):
        np.testing.assert_array_equal(changed[0].prediction[key], value.prediction[key])
    np.testing.assert_array_equal(changed[0].attention.language_attention['maps'][2], language['maps'][2])
    assert not changed[0].attention.language_attention['valid'][:2].any()


def test_missing_head_is_explicit_and_new_panel_saves_all_readouts(checkpoints, decoder_checkpoint, tmp_path):
    _, paths = checkpoints
    config = TrainConfig.from_dict(torch.load(paths['eskill'], weights_only=False)['config'])
    missing = VisualReadout(paths['eskill'], config, 'cpu')
    assert missing.module is None and missing.status == 'VISUAL HEAD NOT SAVED'
    path, _ = decoder_checkpoint
    video = tmp_path / 'predictions.mp4'
    attention = tmp_path / 'attention.mp4'
    main(['--checkpoint', str(path), '--episode', 'one', '--video', str(video), '--device', 'cpu',
          '--max-frames', '2', '--steps', '2', '--attention-video', str(attention), '--check-action-parity',
          '--object-word', 'cup', '--target-word', 'it'])
    with av.open(str(video)) as stream:
        frames = list(stream.decode(video=0))
    assert len(frames) == 2 and (frames[0].width, frames[0].height) == (1920, 1340)
    assert attention.exists() and attention.with_suffix('.json').exists()
    with np.load(video.with_suffix('.npz')) as data:
        assert data['pred_role_rgb'].shape == (2, 2, 2, 32, 32, 3)
        assert data['pred_trace'].shape == (2, 2, 8, 2)
        assert data['pred_gripper_prob'].shape == (2, 2, 8)
        assert not data['motion_attention'][:, 1].any()
        assert data['image_attention'].shape == (2, 1, 4, 4)
        assert data['input_images'].shape == (2, 1, 32, 32, 3)
        np.testing.assert_allclose(data['key_mass'].sum(1), 1., atol=1e-6)
        assert data['language_maps'].shape == (2, 3, 1, 4, 4)
        assert data['language_valid'].all()
        np.testing.assert_allclose(data['language_probabilities'].sum(-1), 1., atol=1e-6)
    report = json.loads(video.with_suffix('.json').read_text())
    assert report['gt_usage'].startswith('Read only after')
    assert report['action_parity_max_abs_error'] == 0
    assert sum(report['action_attention']['0']['key_mass'].values()) == pytest.approx(1., abs=1e-6)
    assert report['language_attention']['layout']['target']['phrase'] == 'it'
    assert report['language_groups'] == ['target', 'object', 'instruction']


@pytest.mark.parametrize('replan_steps', [3, 5])
def test_actual_rollout_keeps_actions_rng_and_holds_maps_on_their_source_input(decoder_checkpoint, tmp_path, replan_steps):
    from examples.libero import evaluate as evaluation
    from openpi.policies.libero_policy import LiberoWirePolicy
    from test_libero_evaluation import FakeEnvironment, observation
    path, config = decoder_checkpoint
    policy = create_trained_policy(path)
    metadata = dict(cameras=['image'], requires_state=False, image_resize_on_server=True,
                    prediction_recording=dict(directory=str(tmp_path / 'captures'), schema_version=1))
    task = 'pick cup then place it'
    native = LiberoWirePolicy(policy, config.to_dict(), steps=2)
    torch.manual_seed(31)
    expected = native.infer(evaluation.policy_input(observation(), task, metadata=metadata))
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    recorder = RecordedPolicy(policy, path, config, tmp_path / 'captures', object_word='cup',
                              target_word='it', check_action_parity=True)
    assert torch.equal(before, torch.get_rng_state())
    wire = LiberoWirePolicy(recorder, config.to_dict(), steps=2)

    class Client:
        calls = 0
        def reset(self):
            pass
        def infer(self, payload):
            result = wire.infer(payload)
            if self.calls == 0:
                np.testing.assert_array_equal(result['actions'], expected['actions'])
                assert torch.equal(expected_rng, torch.get_rng_state())
            self.calls += 1
            return result

    class Environment(FakeEnvironment):
        def step(self, action):
            _, reward, done, info = super().step(action)
            obs = observation(self.steps)
            obs['agentview_image'] = (obs['agentview_image'].astype(int) + self.steps * 5).astype(np.uint8)
            return obs, reward, done, info
    client = Client()
    result = evaluation.rollout(Environment(), client, 'published state', task, max_steps=12,
        wait_steps=0, replan_steps=replan_steps, capture_video=False, policy_metadata=metadata)
    assert result['error'] is None and result['steps'] == 9
    assert client.calls == len(range(0, 9, replan_steps))
    record = tmp_path / 'rollout.json'
    record.write_text(json.dumps(dict(instruction=task, prediction_recording=metadata['prediction_recording'],
        control_fps=20, replan_steps=replan_steps, error=None, success=True,
        inference_requests=result['inference_requests'])))
    arrays = dict(actions=result['actions'], chunk_ids=result['chunk_ids'],
                  diagnostic_images=np.array(result['diagnostic_images']))
    arrays.update({f'prediction_{i:04d}': values for i, values in enumerate(result['predictions'])})
    np.savez_compressed(record.with_suffix('.npz'), **arrays)
    frames, settings, info, images = load_rollout(record)
    assert sorted(frames) == list(range(0, 9, replan_steps))
    assert len(images) == 10
    assert not frames[0].targets['trace_valid'].any()
    assert frames[0].prediction['role_roundtrip'] is None
    renderer = Renderer(frames, settings, info, images)
    first, held, updated = (renderer.frame(n) for n in (0, 1, replan_steps))
    np.testing.assert_array_equal(first[792:], held[792:])
    assert not np.array_equal(first[184:760, 36:612], held[184:760, 36:612])
    assert not np.array_equal(first[792:], updated[792:])
    with pytest.raises(ValueError, match='overwrite the input rollout'):
        main(['--rollout-record', str(record), '--video', str(record.with_suffix('.mp4'))])
    video = tmp_path / 'rendered.mp4'
    main(['--rollout-record', str(record), '--video', str(video)])
    report = json.loads(video.with_suffix('.json').read_text())
    assert report['timeline'] == 'actual_rollout_control_frames' and report['fps'] == 20
    assert report['rendered_frame_count'] == 10
    with np.load(video.with_suffix('.npz')) as data:
        np.testing.assert_array_equal(data['display_source_frames'], [renderer.source_frame(n) for n in range(10)])
    with av.open(str(video)) as stream:
        assert sum(1 for _ in stream.decode(video=0)) == 10
    with av.open(report['original_video']) as stream:
        originals = list(stream.decode(video=0))
        assert len(originals) == 10 and (originals[0].width, originals[0].height) == (6, 6)
    arrays['actions'][0, 0] += 1
    np.savez_compressed(record.with_suffix('.npz'), **arrays)
    with pytest.raises(AssertionError):
        load_rollout(record)
