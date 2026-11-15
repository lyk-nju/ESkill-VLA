"""ESkill saved-frame queries, Event boundaries, command semantics and raw body geometry."""
import json
from unittest import mock

import numpy as np
import pytest

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeDataset, EpisodeReader, SCHEMA_VERSION
from openpi.policies.libero_policy import LiberoInputs
from openpi.transforms import ModelTransforms
from scripts.preprocess.export import EpisodeWriter, write_index
from scripts.preprocess.libero.audit_training import audit


def fixture(root, *, times=True, edit=None, frames=9, pick_stop=4):
    w, n = EpisodeWriter(), frames
    ids = w.add("indices", np.arange(n))
    def sequence(name, values):
        return dict(values=w.add(name, values), frame_indices=ids,
                    valid=w.add(name + "_valid", np.ones_like(values, dtype=bool)))
    region = dict(frame_indices=ids, boxes_xyxy=w.add("boxes", np.tile([8, 8, 20, 20], (n, 1))),
                  boxes_valid=w.add("boxes_valid", np.ones(n, bool)), masks=None, masks_valid=None,
                  region_extent="full_in_frame")
    eef = np.stack((np.arange(n) * .1, np.zeros(n), np.zeros(n)), -1).astype(np.float32)
    states = np.zeros((n, 15), np.float32)
    states[:, 7:10], states[:, -2:] = eef, .03
    actions = np.zeros((n, 7), np.float32)
    actions[:, 0], actions[:, -1] = .01, np.where(np.arange(n) % 2, 1, -1)
    extrinsic = np.eye(4)
    extrinsic[2, 3] = 10
    camera = dict(intrinsics=w.add("intrinsic", np.array([[10., 0, 15.5], [0, 10., 15.5], [0, 0, 1]])),
                  extrinsics=w.add("extrinsic", extrinsic), image_transform=w.add("pixel_transform", np.eye(3)),
                  extrinsics_convention="world_to_camera_cv", world_frame="world")
    meta = dict(schema_version=SCHEMA_VERSION, episode_id="eskill_fixture", frame_count=n,
                source=dict(dataset="fixture", revision="fixture", episode_id="0", parent_trajectory_id="parent"),
                instructions=["pick then place"], arm_ids=["single"], timeline="timestamps" if times else "source_steps",
                timestamps_s=w.add("timestamps", np.arange(n) * .25) if times else None,
                nominal_fps=None, state=sequence("state", states),
                actions={**sequence("actions", actions), "conditioning_image_indices": {"image": ids},
                         "gripper_convention": dict(channels={"single": 6}, close_sign=1,
                                                    alignment="observation_then_action", evidence="fixture execution")},
                cameras={"image": dict(frame_indices=ids, valid=w.add("rgb_valid", np.ones(n, bool)),
                                       rgb=w.add("rgb", np.zeros((n, 32, 32, 3), np.uint8)),
                                       width=32, height=32, calibration=camera)},
                effector_positions={"single": {**sequence("eef", eef), "unit": "m", "coordinate_frame": "world",
                                                "reference_point": "fixture_site"}},
                effector_regions={}, quality={"manual_review": "pending"}, events=[])
    for name, start, stop, skill in [("pick", 0, pick_stop, "Pick"), ("place", pick_stop, n, "Place")]:
        meta["events"].append(dict(event_id=name, arm="single", start_frame=start, stop_frame=stop,
                                   goal_frame=stop - 1, boundary_valid=True, skill=skill, skill_valid=True,
                                   object_regions={"image": region}))
    if edit:
        edit(meta, w.arrays)
    path = w.save(root / "episode", meta)
    index = root / "index.json"
    write_index(index, [path])
    return index


def dataset(index, **kwargs):
    cfg = ESkillConfig.tiny()
    return EpisodeDataset(index, action_horizon=cfg.action_horizon, skill_names=cfg.skill_names,
                          motion_queries=8, image_size=cfg.image_size,
                          **(dict(motion_stride=1, motion_min_horizon_frames=0) | kwargs))


def test_no_clock_is_required_but_command_semantics_are_required(tmp_path):
    index = fixture(tmp_path, times=False)
    assert dataset(index)[0]['targets']['trace_valid'][0, :3].all()
    index = fixture(tmp_path / "command", edit=lambda m, a: m["actions"].pop("gripper_convention"))
    with pytest.raises(ValueError, match="aperture is not a command"):
        dataset(index)


def test_shared_queries_are_event_local_and_terminal_points_have_no_loss(tmp_path):
    data = dataset(fixture(tmp_path))
    sample = data[0]
    target = sample["targets"]
    assert len(target["role_sources"][0]) == 2
    assert "future_skill_labels" not in target
    np.testing.assert_array_equal(target["trace_valid"][0], [True] * 3 + [False] * 5)
    np.testing.assert_array_equal(target["gripper_valid"][0], target["trace_valid"][0])
    np.testing.assert_array_equal(target["gripper"][0, :3], [1, 0, 1])
    assert target["future_event_tail"][0, 3:].all()
    assert not target["terminal_padded_mask"].any()
    assert not target["trace_valid"][1].any()
    assert not data[8]["targets"]["trace_valid"].any()
    assert data[8]["targets"]["future_episode_tail"][0].all()


def test_five_frame_queries_ignore_unknown_or_irregular_physical_time(tmp_path):
    def edit(meta, arrays):
        meta['events'][0]['stop_frame'] = 8
        meta['events'][0]['goal_frame'] = 7
        meta['events'][1]['start_frame'] = 8
        arrays['timestamps'] = np.array([0, .1, .3, .4, 1, 1.2, 1.3, 2, 3.])
    data = dataset(fixture(tmp_path, edit=edit), motion_stride=5)
    target = data[0]['targets']
    assert target['trace_valid'][0].tolist() == [True] + [False] * 7
    assert data[0]['metadata']['future_source_frame_indices'][0, 0] == 5
    assert target['future_event_tail'][0, 1]  # Frame 10 is beyond this Pick Event and Episode.
    assert not target['terminal_padded_mask'].any()
    assert 'future_times_s' not in data[0]['metadata']


def test_offsets_count_saved_camera_frames_independently_of_sample_stride(tmp_path):
    def edit(meta, arrays):
        meta['events'][0].update(stop_frame=8, goal_frame=7)
        meta['events'][1]['start_frame'] = 8
        indices = np.array([0, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        arrays['camera_indices'] = indices
        arrays['pairing'] = np.searchsorted(indices, np.arange(9), side='right') - 1
        stream = meta['cameras']['image']
        stream['frame_indices'] = dict(uri='arrays.npz', key='camera_indices', shape=[8], dtype='int64')
        meta['actions']['conditioning_image_indices']['image'] = dict(
            uri='arrays.npz', key='pairing', shape=[9], dtype='int64')
        for field in ('rgb', 'valid'):
            ref = stream[field]
            arrays[ref['key']] = arrays[ref['key']][indices]
            ref['shape'] = list(arrays[ref['key']].shape)
    data = dataset(fixture(tmp_path, times=False, edit=edit), motion_stride=5, frame_stride=2)
    sample = data[0]
    assert sample['metadata']['future_source_frame_indices'][0, 0] == 6  # Fifth saved future RGB row.
    assert sample['targets']['trace_valid'][0, 0]


def test_invalid_current_projection_keeps_valid_future_supervision(tmp_path):
    def edit(meta, arrays):
        arrays['eef_valid'][0] = False
    sample = dataset(fixture(tmp_path, edit=edit))[0]
    assert set(sample['inputs']) == {'images', 'state', 'state_valid', 'prompt'}
    assert sample['targets']['trace_valid'][0, :3].all()
    assert not sample['targets']['trace_valid'][0, 3:].any()
    assert sample['targets']['gripper_valid'][0, :3].all()


def test_path_and_gripper_validity_are_independent(tmp_path):
    def edit(meta, arrays):
        arrays["eef_valid"][1] = False
        arrays["actions"][2, -1] = 0
    target = dataset(fixture(tmp_path, edit=edit))[0]["targets"]
    assert not target["trace_valid"][0, 0] and target["gripper_valid"][0, 0]
    assert target["trace_valid"][0, 1] and not target["gripper_valid"][0, 1]


def test_minimum_horizon_crosses_event_using_real_positions_and_commands(tmp_path):
    data = dataset(fixture(tmp_path), motion_min_horizon_frames=2)
    sample = data[3]  # Last Pick frame; next two recorded observations belong to Place.
    target = sample['targets']
    assert target['trace_valid'][0].tolist() == [True, True] + [False] * 6
    assert target['gripper_valid'][0].tolist() == [True, True] + [False] * 6
    assert target['future_cross_event'][0].tolist() == [True, True] + [False] * 6
    assert target['future_event_tail'][0].all()  # Boundary facts no longer imply exclusion.
    assert target['gripper'][0, :2].tolist() == [0., 1.]  # Actual frame-4/5 commands.
    np.testing.assert_array_equal(sample['metadata']['future_source_frame_indices'][0, :2], [4, 5])
    assert sample['metadata']['goal_frames'][0] == 3
    assert target['skill_labels'][0] == 0  # Current Pick, not next Place.
    assert 'effector_start' not in sample['inputs']
    assert not target['future_padded'].any()


def test_minimum_horizon_is_a_floor_and_never_repeats_episode_tail(tmp_path):
    data = dataset(fixture(tmp_path), motion_min_horizon_frames=2)
    assert data[0]['targets']['trace_valid'][0].tolist() == [True] * 3 + [False] * 5
    target = data[7]['targets']
    assert target['trace_valid'][0].tolist() == [True] + [False] * 7
    assert not target['future_cross_event'][0].any()
    assert target['future_episode_tail'][0, 1:].all()
    assert not data[8]['targets']['trace_valid'].any()
    assert not data[8]['targets']['gripper_valid'].any()


def test_cross_event_queries_keep_projection_and_command_validity_independent(tmp_path):
    def edit(meta, arrays):
        arrays['eef_valid'][4] = False
        arrays['actions'][5, -1] = 0
    target = dataset(fixture(tmp_path, edit=edit), motion_min_horizon_frames=2)[3]['targets']
    assert target['future_cross_event'][0, :2].all()
    assert target['trace_valid'][0, :2].tolist() == [False, True]
    assert target['gripper_valid'][0, :2].tolist() == [True, False]
    assert target['future_projection_invalid'][0, 0]
    assert target['future_command_missing'][0, 1]


def test_minimum_horizon_is_per_arm(tmp_path):
    import copy
    def edit(meta, arrays):
        meta['arm_ids'] = ['left', 'right']
        position = meta['effector_positions'].pop('single')
        meta['effector_positions'] = {'left': position, 'right': copy.deepcopy(position)}
        for event in meta['events']:
            event['arm'] = 'left'
        other = copy.deepcopy(meta['events'][1])
        other.update(event_id='right_event', arm='right', start_frame=0)
        meta['events'].append(other)
        meta['actions']['gripper_convention']['channels'] = dict(left=6, right=6)
    sample = dataset(fixture(tmp_path, edit=edit), motion_min_horizon_frames=2)[3]
    assert sample['targets']['future_cross_event'][0, :2].all()
    assert not sample['targets']['future_cross_event'][1].any()
    assert sample['targets']['trace_valid'].sum(1).tolist() == [2, 5]


def test_five_frame_stride_extends_across_event_but_not_episode(tmp_path):
    data = dataset(fixture(tmp_path), motion_stride=5, motion_min_horizon_frames=10)
    assert data[0]['targets']['future_cross_event'][0, 0]
    assert data[0]['targets']['trace_valid'][0, 0]
    assert data[0]['targets']['future_episode_tail'][0, 1:].all()


def test_cross_event_floor_counts_saved_rows_with_source_index_gaps(tmp_path):
    def edit(meta, arrays):
        frames = np.array([0, 3, 4, 5, 6, 7, 8], np.int64)
        arrays['camera_frames'] = frames
        stream = meta['cameras']['image']
        stream['frame_indices'] = dict(uri='arrays.npz', key='camera_frames', shape=[7], dtype='int64')
        arrays['pairing'] = np.searchsorted(frames, np.arange(9), side='right') - 1
        meta['actions']['conditioning_image_indices']['image'] = dict(
            uri='arrays.npz', key='pairing', shape=[9], dtype='int64')
        for field in ('rgb', 'valid'):
            ref = stream[field]
            arrays[ref['key']] = arrays[ref['key']][frames]
            ref['shape'] = list(arrays[ref['key']].shape)
    sample = dataset(fixture(tmp_path, edit=edit), motion_min_horizon_frames=2)[0]
    assert sample['metadata']['future_source_frame_indices'][0, :2].tolist() == [3, 4]
    assert sample['targets']['future_cross_event'][0, :2].tolist() == [False, True]
    assert sample['targets']['trace_valid'][0, :2].all()


def test_missing_diagnostics_distinguish_projection_effector_and_command(tmp_path):
    def edit(meta, arrays):
        arrays['eef_valid'][1] = False
        for name, sequence, frames in (
                ('effector', meta['effector_positions']['single'], np.array([0, 1, 3, 4, 5, 6, 7, 8])),
                ('command', meta['actions'], np.array([0, 1, 2, 4, 5, 6, 7, 8]))):
            key = name + '_frames'
            arrays[key] = frames.astype(np.int64)
            sequence['frame_indices'] = dict(uri='arrays.npz', key=key, shape=[8], dtype='int64')
            for field in ('values', 'valid'):
                ref = sequence[field]
                arrays[ref['key']] = arrays[ref['key']][frames]
                ref['shape'] = list(arrays[ref['key']].shape)
        meta['actions']['conditioning_image_indices']['image'] = meta['actions']['frame_indices']
    target = dataset(fixture(tmp_path, edit=edit))[0]['targets']
    assert 'future_missing' not in target
    assert target['future_observed'][0, :3].all()  # All three queries are inside Pick.
    assert target['future_projection_invalid'][0, :3].tolist() == [True, False, False]
    assert target['future_effector_missing'][0, :3].tolist() == [False, True, False]
    assert target['future_command_missing'][0, :3].tolist() == [False, False, True]
    assert target['trace_valid'][0, :3].tolist() == [False, False, True]
    assert target['gripper_valid'][0, :3].tolist() == [True, True, False]
    assert not target['future_event_tail'][0, :3].any()
    assert not target['future_episode_tail'][0, :3].any()


@pytest.mark.parametrize('problem', ['missing_event', 'invalid_boundary'])
def test_unknown_event_is_not_a_future_observation_gap(tmp_path, problem):
    def edit(meta, arrays):
        if problem == 'missing_event':
            meta['events'] = []
        else:
            meta['events'][0]['boundary_valid'] = False
    sample = dataset(fixture(tmp_path, edit=edit))[0]
    target = sample['targets']
    assert 'future_missing' not in target
    assert sample['metadata']['event_ids'][0] is None
    assert not target['future_observed'][0].any()
    assert not target['trace_valid'][0].any() and not target['gripper_valid'][0].any()
    assert not target['future_event_tail'][0].any()


@pytest.mark.parametrize("skill, label", [("Hold", 6), ("unknown", -1)])
def test_unsupported_skills_keep_identity_and_do_not_disable_other_targets(tmp_path, skill, label):
    def edit(meta, arrays):
        meta["events"][0]["skill"] = skill
    target = dataset(fixture(tmp_path, edit=edit))[0]["targets"]
    assert target["skill_labels"][0] == label and not target["skill_valid"][0]
    assert target["trace_valid"][0].any() and target["gripper_valid"][0].any()


def test_rgb_only_adapter_and_transform_ignore_state_and_need_no_state_statistics(tmp_path):
    data = dataset(fixture(tmp_path), include_state=False)
    sample = data[1]
    assert set(sample['inputs']) == {'images', 'prompt'}
    adapter = LiberoInputs(include_state=False)
    def tokenize(text, state):
        assert state is None
        return np.array([1]), np.array([True])
    transform = ModelTransforms(ESkillConfig.tiny(), {}, tokenize)
    expected = transform.inputs(adapter(sample['inputs']))
    noisy = dict(sample['inputs'], state=np.full(15, np.nan), arm_state=np.full((2, 8), np.nan))
    actual = transform.inputs(adapter(noisy))
    assert set(actual) == {'images', 'image_mask', 'tokens', 'token_mask', 'arm_presence'}
    for key in actual:
        np.testing.assert_array_equal(actual[key], expected[key])


def test_audit_reports_query_coverage_and_crossing_counts(tmp_path):
    result = audit(fixture(tmp_path), execute_prefix=2, motion_stride=1)
    assert result["physical_time_ready_episodes"] == result["command_ready_episodes"] == 1
    assert result["crossing_action_chunks"] > result["crossing_execute_prefixes"] > 0
    assert result["samples_without_same_event_query"] == 2
    assert result["skill_event_counts"] == {"Pick": 1, "Place": 1}


def test_transform_applies_configured_future_expansion_only_to_targets(tmp_path):
    sample = dataset(fixture(tmp_path))[0]
    sample['inputs'] = LiberoInputs(include_state=False)(sample['inputs'])
    stats = {k: dict(method='standard', mean=[0.] * n, std=[1.] * n)
             for k, n in [('state', 8), ('actions', 7)]}
    tokenizer = lambda text, state: (np.array([1]), np.array([True]))
    previous = ModelTransforms(ESkillConfig.tiny(future_bbox_expand_px=0), stats, tokenizer)(sample)
    expanded = ModelTransforms(ESkillConfig.tiny(future_bbox_expand_px=8), stats, tokenizer)(sample)
    for key in previous['inputs']:
        np.testing.assert_array_equal(previous['inputs'][key], expanded['inputs'][key])
    np.testing.assert_array_equal(previous['targets']['role_weights'][:, 0],
                                  expanded['targets']['role_weights'][:, 0])
    assert expanded['targets']['role_weights'][0, 1].sum() > previous['targets']['role_weights'][0, 1].sum()
    assert expanded['metadata']['frame'] == previous['metadata']['frame']
