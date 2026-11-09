"""Unified RoboTwin observations, next targets, pixel preservation and missing annotations."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from openpi.policies.eskill_dataset import EpisodeDataset, load_episode
from openpi.training.config import TrainConfig
from openpi.training.data_loader import input_adapter
from scripts.preprocess.export import write_index
from scripts.preprocess.robotwin.episodes import CAMERAS, export_episode, observation_rows


def source_episode(tmp_path):
    path = tmp_path / 'stack' / 'aloha_agilex' / 'raw' / 'episode_0000000.hdf5'
    path.parent.mkdir(parents=True)
    with h5py.File(path, 'w') as f:
        f.attrs.update(format='robotwin.raw.v2', complete=True, task='stack', frame_count=4,
                       seed=0, collection_fingerprint='test', native_entities='{}',
                       ee_reference='world_wxyz_get_ee_pose_is_endpose_false_tcp_offset_local_x_0.12m')
        f['timestamp'] = [0., .004, .004, .008]
        f['observations/physical_step'] = [0, 1, 1, 2]
        f['instructions'] = json.dumps(['stack bowls'])
        f['evidence'] = json.dumps({'subtasks': [{'label': 'grasp', 'arm': 'left', 'start_frame': 0, 'end_frame': 2}]})
        commands = np.repeat(np.array([0., 1., 1., 2.])[:, None], 14, axis=1)
        f['observations/joint_action/vector'] = commands
        f['control_trace/physical_step'] = [0, 1]
        f['control_trace/action'] = commands[[2, 3]]
        for arm in ('left', 'right'):
            f[f'observations/joint_state/{arm}_arm'] = np.ones((4, 6)) * .2
            f[f'observations/joint_state/{arm}_gripper_command'] = np.ones(4) * .3
            f[f'observations/endpose/{arm}_endpose'] = np.tile([0., 0., 1., 1., 0., 0., 0.], (4, 1))
            base = f'observations/native_regions/regions/{arm}_gripper'
            f[base + '/full_boxes_xyxy'] = np.tile([1, 2, 8, 9], (4, 1))
            f[base + '/full_boxes_valid'] = np.ones(4, bool)
        for camera in CAMERAS.values():
            base = f'observations/observation/{camera}'
            f[base + '/rgb'] = np.broadcast_to(np.arange(4, dtype=np.uint8)[:, None, None, None], (4, 16, 20, 3))
            f[base + '/intrinsic_cv'] = np.tile([[10., 0, 10], [0, 10, 8], [0, 0, 1]], (4, 1, 1))
            f[base + '/extrinsic_cv'] = np.tile(np.eye(4)[:3], (4, 1, 1))
    return path


def test_export_preserves_pixels_targets_and_unknown_labels(tmp_path):
    source = source_episode(tmp_path)
    output = tmp_path / 'saved'
    path = Path(export_episode(source, output))
    reader = load_episode(path)
    meta = reader.meta
    np.testing.assert_array_equal(reader.array(meta['alignment_metadata']['source_rows']), [0, 2, 3])
    np.testing.assert_array_equal(reader.array(meta['timestamps_s']), [0, .004, .008])
    np.testing.assert_array_equal(reader.array(meta['actions']['values'])[:, 0], [1, 2])
    assert meta['events'] == []
    for camera in CAMERAS:
        assert np.all(reader.rgb(camera, 1) == 2)  # Last duplicate row, preserved losslessly.
    index = output / 'index.json'
    write_index(index, [path])
    data = EpisodeDataset(index, action_horizon=3, action_padding='repeat_last',
                          cameras=('image', 'left_wrist_image', 'right_wrist_image'), auxiliary=True)
    sample = data[0]
    assert sample['metadata']['frame'] == 0
    np.testing.assert_array_equal(sample['targets']['actions'][:, 0], [1, 2, 2])
    assert not sample['targets']['skill_valid'].any()
    assert not sample['targets']['future_skill_valid'].any()
    assert sample['targets']['trace_valid'].all()
    assert sample['targets']['role_sources'][0][0][1] is not None
    assert sample['targets']['role_sources'][0][1][1] is None
    assert 'effector_pixels' in meta


def test_command_mapping_is_checked_against_executed_trace(tmp_path):
    path = source_episode(tmp_path)
    with h5py.File(path, 'r+') as f:
        f['control_trace/action'][0, 0] = 999.
    with pytest.raises(ValueError, match='differs from last applied control'):
        export_episode(path, tmp_path / 'bad')


def test_non_monotonic_source_time_is_not_silently_reordered():
    with pytest.raises(ValueError, match='monotone'):
        observation_rows([0, 3, 2])


@pytest.mark.parametrize('model_type,enabled,expected', [('pi05', True, False), ('eskill', False, False), ('eskill', True, True)])
def test_body_state_is_only_built_for_enabled_eskill(model_type, enabled, expected):
    from openpi.models.pi0_config import ESkillConfig
    cfg = TrainConfig('unused', 'unused', 'unused', 'unused', model_type=model_type,
                      platform='robotwin', model=ESkillConfig(use_arm_state=enabled))
    result = input_adapter(cfg)(dict(state=np.arange(14), state_valid=np.ones(14, bool), images=[], prompt='task'))
    assert ('arm_state' in result) == expected
    if expected:
        assert result['arm_state_valid'].sum() == 14
        assert not result['arm_state_valid'][:, 7:].any()


def test_readback_check_and_render_use_saved_episodes(tmp_path):
    from scripts.preprocess.robotwin.check import check_saved_episodes
    from scripts.preprocess.robotwin.render import render_saved_episode
    source = source_episode(tmp_path)
    output = tmp_path / 'saved'
    path = Path(export_episode(source, output))
    index = output / 'index.json'
    write_index(index, [path])
    result = check_saved_episodes(index, tmp_path / 'check.json')
    assert result['episodes'][0]['images_compared'] == 12
    # Rendering is self-contained even when source HDF5 is unavailable.
    source.rename(source.with_suffix('.unavailable'))
    result = render_saved_episode(path, tmp_path / 'review.mp4')
    assert result['frames'] == 3 and result['manual_review'] == 'pending'
