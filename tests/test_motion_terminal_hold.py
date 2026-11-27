"""Event-end 2D stop supervision, preserving real minimum lookahead and commands."""
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.augmentation import ImageAugmentation, augment_eskill_batch, affine_eskill_batch
from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig
from openpi.training.data_loader import episode_dataset
from scripts.preprocess.libero.audit_training import audit
from scripts.preprocess.libero.review_motion import save_targets
from test_eskill_data import dataset, fixture
from test_augmentation import sample


@pytest.mark.parametrize('stop, sources', [
    (21, [5, 10, 15, 20, 20, 20, 20, 20]),
    (19, [5, 10, 15, 18, 18, 18, 18, 18]),
])
def test_event_stop_and_between_query_endpoint(tmp_path, stop, sources):
    data = dataset(fixture(tmp_path, frames=61, pick_stop=stop), motion_stride=5,
                   motion_min_horizon_frames=10, motion_terminal_hold=True)
    row = data[0]
    t, meta = row['targets'], row['metadata']
    np.testing.assert_array_equal(meta['trace_source_frame_indices'][0], sources)
    np.testing.assert_allclose(t['trace'][0, 3:], np.tile(t['trace'][0, 3], (5, 1)))
    assert t['trace_valid'][0].all() and not t['trace_valid'][1].any()
    held = np.arange(5, 41, 5) > stop - 1
    np.testing.assert_array_equal(t['trace_terminal_hold'][0], held)
    np.testing.assert_array_equal(t['terminal_padded_mask'][0], held)
    np.testing.assert_array_equal(t['future_observed'][0], ~held)
    np.testing.assert_array_equal(t['gripper_valid'][0], ~held)
    assert not t['trace_terminal_hold'][1].any()
    assert meta['goal_frames'][0] == stop - 1
    # The optional target policy changes neither current input nor other supervision.
    old = dataset(data.paths[0].parent.parent / 'index.json', motion_stride=5,
                  motion_min_horizon_frames=10)[0]
    for key in ('actions', 'action_valid', 'skill_labels', 'gripper', 'gripper_valid'):
        np.testing.assert_array_equal(t[key], old['targets'][key])
    for key in ('state', 'state_valid'):
        np.testing.assert_array_equal(row['inputs'][key], old['inputs'][key])
    for key in ('future_camera_rows', 'future_source_frame_indices', 'goal_frames'):
        np.testing.assert_array_equal(meta[key], old['metadata'][key])


def test_short_event_keeps_minimum_real_points_then_holds_later_endpoint(tmp_path):
    data = dataset(fixture(tmp_path, frames=61), motion_stride=5,
                   motion_min_horizon_frames=10, motion_terminal_hold=True)
    row = data[3]
    t = row['targets']
    assert row['metadata']['trace_source_frame_indices'][0].tolist() == [8, 13, 13, 13, 13, 13, 13, 13]
    assert t['future_cross_event'][0].tolist() == [True, True] + [False] * 6
    assert t['trace_terminal_hold'][0].tolist() == [False, False] + [True] * 6
    assert t['gripper_valid'][0].tolist() == [True, True] + [False] * 6
    assert t['gripper'][0, :2].tolist() == [0., 1.]
    assert row['metadata']['goal_frames'][0] == 3 and t['skill_labels'][0] == 0


@pytest.mark.parametrize('current, sources', [(7, [8] * 8), (8, [8] * 8)])
def test_episode_tail_is_derived_hold_not_observation_or_command(tmp_path, current, sources):
    row = dataset(fixture(tmp_path), motion_min_horizon_frames=2, motion_terminal_hold=True)[current]
    t = row['targets']
    assert row['metadata']['trace_source_frame_indices'][0].tolist() == sources
    assert t['trace_valid'][0].all()
    np.testing.assert_array_equal(t['trace_terminal_hold'][0], t['future_episode_tail'][0])
    assert t['gripper_valid'][0].sum() == (current == 7)
    assert not (t['future_observed'] & t['trace_terminal_hold']).any()


@pytest.mark.parametrize('invalid', ['projection', 'missing'])
def test_invalid_endpoint_is_not_replaced_with_an_earlier_valid_point(tmp_path, invalid):
    def edit(meta, arrays):
        if invalid == 'projection':
            arrays['eef_valid'][8] = False
        else:
            stream = meta['effector_positions']['single']
            arrays['eef_frames'] = np.arange(8, dtype=np.int64)
            stream['frame_indices'] = dict(uri='arrays.npz', key='eef_frames', shape=[8], dtype='int64')
            for field in ('values', 'valid'):
                ref = stream[field]
                arrays[ref['key']] = arrays[ref['key']][:-1]
                ref['shape'][0] = 8
    index = fixture(tmp_path, edit=edit)
    row = dataset(index, motion_min_horizon_frames=2, motion_terminal_hold=True)[8]
    t = row['targets']
    assert t['trace_terminal_hold'][0].all() and not t['trace_valid'].any()
    reason = 'future_projection_invalid' if invalid == 'projection' else 'future_effector_missing'
    assert t[reason][0].all()
    report = audit(index, motion_stride=1, min_horizon_frames=2, terminal_hold=True)
    assert report['effective_supervision']['trace_zero_reasons'][reason] >= 1


@pytest.mark.parametrize('problem', ['gap', 'invalid_event'])
def test_internal_missing_points_and_unknown_events_do_not_become_holds(tmp_path, problem):
    def edit(meta, arrays):
        if problem == 'gap':
            arrays['eef_valid'][1] = False
        else:
            meta['events'][0]['boundary_valid'] = False
    t = dataset(fixture(tmp_path, edit=edit), motion_terminal_hold=True)[0]['targets']
    assert not t['trace_valid'][0, 0] and not t['trace_terminal_hold'][0, 0]
    if problem == 'invalid_event':
        assert not t['trace_valid'].any() and not t['trace_terminal_hold'].any()
    else:
        assert t['trace_valid'][0, 1:].all()


def test_hold_endpoint_and_minimum_window_are_per_arm(tmp_path):
    def edit(meta, arrays):
        meta['arm_ids'] = ['left', 'right']
        xyz = meta['effector_positions'].pop('single')
        meta['effector_positions'] = dict(left=xyz, right=deepcopy(xyz))
        for event in meta['events']:
            event['arm'] = 'left'
        other = deepcopy(meta['events'][1])
        other.update(event_id='right_event', arm='right', start_frame=0)
        meta['events'].append(other)
        meta['actions']['gripper_convention']['channels'] = dict(left=6, right=6)
    row = dataset(fixture(tmp_path, edit=edit), motion_min_horizon_frames=2, motion_terminal_hold=True)[3]
    assert row['metadata']['trace_source_frame_indices'].tolist() == [[4, 5, 5, 5, 5, 5, 5, 5],
                                                                   [4, 5, 6, 7, 8, 8, 8, 8]]


def test_loader_and_persisted_review_use_the_same_hold_policy(tmp_path):
    index = fixture(tmp_path)
    data = dataset(index, motion_stride=1, motion_min_horizon_frames=2, motion_terminal_hold=True)
    assert data.motion_terminal_hold
    output = tmp_path / 'labels.npz'
    save_targets(data.reader(0), output, np.arange(1, 9), min_horizon_frames=2, terminal_hold=True)
    with np.load(output) as saved:
        for i in (0, 3, 8):
            row = data[i]
            for key in ('trace', 'trace_valid', 'trace_terminal_hold', 'gripper_valid'):
                np.testing.assert_array_equal(saved[key][i], row['targets'][key])
            np.testing.assert_array_equal(saved['trace_source_frame_indices'][i],
                                          row['metadata']['trace_source_frame_indices'])
    original = audit(index, motion_stride=1, min_horizon_frames=2)['effective_supervision']
    hold = audit(index, motion_stride=1, min_horizon_frames=2, terminal_hold=True)['effective_supervision']
    assert hold['gripper'] == original['gripper']
    assert hold['trace']['samples_without_targets'] == 0
    observed, held = hold['trace']['observed_valid_by_offset'], hold['trace']['terminal_hold_valid_by_offset']
    np.testing.assert_array_equal(observed, original['trace']['valid_by_offset'])
    np.testing.assert_array_equal(np.array(observed) + held, hold['trace']['valid_by_offset'])
    assert not ESkillConfig().motion_terminal_hold  # Legacy/checkpoint omission preserves prior policy.






def test_augmentation_transforms_holds_and_counts_them_as_trace_supervision():
    b = sample()
    b['targets']['trace'][0, 0, 1] = b['targets']['trace'][0, 0, 0]
    b['targets']['trace_terminal_hold'] = torch.tensor([[[False, True], [False, False]]])
    b['targets']['terminal_padded_mask'] = b['targets']['trace_terminal_hold'].clone()
    transformed, _ = affine_eskill_batch(b, torch.tensor([[.1, .2, .8, .8]]), torch.tensor([5.]))
    torch.testing.assert_close(transformed['targets']['trace'][0, 0, 0], transformed['targets']['trace'][0, 0, 1])
    assert transformed['targets']['trace_valid'][0, 0].all()
    colored = augment_eskill_batch(b, ImageAugmentation(color_prob=1.))
    assert colored['metadata'][0]['augmentation']['counts']['trace_before'] == 2
    assert colored['metadata'][0]['augmentation']['counts']['trace_after'] == 2
