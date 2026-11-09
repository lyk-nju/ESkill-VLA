"""Action pairing, Event-owned goals and independent fixed-time 2D motion targets."""
import json

import numpy as np
import pytest
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeDataset, SCHEMA_VERSION
from openpi.training.config import TrainConfig, split_episodes
from openpi.transforms import role_canvas, role_canvases, spatial_trace
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import EpisodeWriter, write_index


def calibration(writer, size, depth=10.):
    pose = np.eye(4)
    pose[2, 3] = depth
    return dict(intrinsics=writer.add("camera_k", np.array([[10., 0, (size-1)/2], [0, 10., (size-1)/2], [0, 0, 1.]])),
                extrinsics=writer.add("camera_pose", pose), image_transform=writer.add("camera_pixels", np.eye(3)),
                extrinsics_convention="world_to_camera_cv", world_frame="world")


def saved_episode(root, identity="one", parent=None, *, events=True, edit=None):
    writer = EpisodeWriter()
    def array(key, value):
        return writer.add(key, np.asarray(value))
    indices = array("indices", np.arange(4))
    rgb = np.zeros((4, 32, 32, 3), np.uint8)
    for t in range(4):
        rgb[t, ..., 0] = t * 40
        rgb[t, :16, :, 1] = 200
        rgb[t, 16:, :, 2] = 200
    region = lambda key, box: dict(
        frame_indices=indices, boxes_xyxy=array(key, np.tile(box, (4, 1)).astype(np.float32)),
        boxes_valid=array(key + "_valid", np.ones(4, bool)), masks=None, masks_valid=None,
        visibility=None, region_extent="full_in_frame", evidence={})
    old_object = region("old_box", [1, 1, 5, 5])
    new_object = region("new_box", [22, 22, 29, 29])
    def event(name, start, stop, goal, skill, region):
        return dict(event_id=name, arm="single", start_frame=start, stop_frame=stop, goal_frame=goal,
                    boundary_valid=True, skill=skill, skill_valid=True, object_regions={"image": region})
    states = np.arange(60, dtype=np.float32).reshape(4, 15) / 10
    meta = dict(
        schema_version=SCHEMA_VERSION, episode_id=identity,
        source=dict(dataset="fixture", revision="test", episode_id=identity,
                    parent_trajectory_id=parent or identity),
        instructions=["pick cup then place it"], arm_ids=["single"], frame_count=4,
        timeline="source_steps", timestamps_s=None, nominal_fps=None,
        cameras={"image": dict(frame_indices=indices, valid=array("rgb_valid", np.ones(4, bool)),
                               rgb=array("rgb", rgb), height=32, width=32, calibration=calibration(writer, 32))},
        state=dict(frame_indices=indices, values=array("state", states),
                   valid=array("state_valid", np.ones_like(states, bool))),
        actions=dict(frame_indices=array("action_indices", [0, 0, 1, 2, 2, 3]),
                     values=array("actions", np.arange(42, dtype=np.float32).reshape(6, 7)),
                     valid=array("action_valid", np.ones((6, 7), bool)),
                     conditioning_image_indices={"image": array("pairing", [0, 0, 0, 2, 2, 3])}),
        effector_regions={"single": {"image": old_object}},
        effector_positions={"single": dict(
            frame_indices=indices, values=array("eef", np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 3, 1]], np.float32)),
            valid=array("eef_valid", np.ones((4, 3), bool)),
            unit="m", coordinate_frame="world", reference_point="test_grasp_site")},
        events=[event("pick", 0, 1, 1, "Pick", old_object), event("place", 1, 4, 3, "Place", new_object)] if events else [],
        quality={"manual_review": "pending", "warnings": ["source_units_pending"]})
    if edit is not None:
        edit(meta, writer.arrays)  # Change labels before derived fields are computed at save time.
    return writer.save(root / identity, meta)


def dataset(tmp_path, edit=None):
    path = saved_episode(tmp_path, edit=edit)
    index = tmp_path / "index.json"
    write_index(index, [path])
    config = ESkillConfig.tiny()
    return EpisodeDataset(index, action_horizon=4, skill_names=config.skill_names,
                          motion_queries=8, motion_stride=1, image_size=32), index


def test_action_windows_use_action_time_and_paired_observation(tmp_path):
    data, _ = dataset(tmp_path)
    item = data[2]  # action at 1 observes frame 0
    assert item["metadata"]["frame"] == 0
    assert item["targets"]["skill_labels"].tolist() == [0, -1]
    assert item["targets"]["future_skill_labels"][0].tolist() == [1, 1, 1, 1]
    assert item["targets"]["future_skill_valid"][1].sum() == 0
    # Goal at 1 belongs to old Pick even though Place already owns the interval at 1.
    reader = data.reader(0)
    goal = reader.goal("pick", "image")
    expected = role_canvas(goal["rgb"], goal["region"], 32)[0]
    canvases = role_canvases(item["targets"]["role_sources"], 32)["role_images"]
    np.testing.assert_array_equal(canvases[0, 2], expected.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)
    new_region = reader.region(reader.meta["events"][1]["object_regions"]["image"], 1)
    assert not np.array_equal(expected, role_canvas(reader.rgb("image", 1), new_region, 32)[0])
    assert set(item["inputs"]) == {"images", "state", "state_valid", "prompt"}
    assert len(item["inputs"]["images"]) == 1
    assert reader.meta["timestamps_s"] is None
    assert reader.meta["quality"]["manual_review"] == "pending"


def test_padding_and_invalid_labels_never_become_idle(tmp_path):
    data, _ = dataset(tmp_path)
    item = data[4]["targets"]
    assert item["future_skill_labels"][0].tolist() == [1, 1, -1, -1]
    assert not item["action_valid"][2:].any()
    reader = data.reader(0)
    reader.array(reader.meta["actions"]["valid"])[4] = False
    assert not data[4]["targets"]["future_skill_valid"][:, 0].any()
    reader.meta["events"][1]["skill_valid"] = False
    assert not data[3]["targets"]["future_skill_valid"].any()
    assert role_canvases(data[3]["targets"]["role_sources"], 32)["role_valid"][0].all()
    reader.meta["events"][0]["boundary_valid"] = False
    assert not data[0]["targets"]["skill_valid"].any()
    assert data[0]["targets"]["trace_valid"][0].all()  # Event validity does not gate EEF targets.


def test_spatial_trace_preserves_scale_turn_and_hold_motion():
    points = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 2, 0]])
    value, valid = spatial_trace(points, 5)
    np.testing.assert_allclose(value, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]])
    assert valid.all()
    assert spatial_trace(np.zeros((3, 3)), 4)[1].all()
    assert not spatial_trace([[np.nan, 0, 0]], 4)[1].any()


def test_missing_path_points_do_not_bridge_gap(tmp_path):
    def edit(meta, arrays):
        arrays["eef_valid"][3] = False
    data, _ = dataset(tmp_path, edit)
    assert not data[3]["targets"]["trace_valid"].any()


def test_motion_crosses_event_boundary_and_holds_episode_endpoint(tmp_path):
    data, _ = dataset(tmp_path)
    sample = data[2]  # Action at 1 is conditioned on image at 0.
    targets = sample["targets"]
    assert targets["trace"].shape == (2, 8, 2)
    np.testing.assert_array_equal(sample["metadata"]["future_frame_indices"], np.arange(1, 9))
    np.testing.assert_array_equal(sample["metadata"]["future_source_frame_indices"][0], [1, 2, 3, 3, 3, 3, 3, 3])
    np.testing.assert_array_equal(targets["future_observed"][0], [True]*3 + [False]*5)
    np.testing.assert_array_equal(targets["future_padded"][0], [False]*3 + [True]*5)
    assert targets["trace_valid"][0].all() and not targets["trace_valid"][1].any()
    assert targets["trace"][0, 0, 1] != targets["trace"][0, 1, 1]  # Motion continues after Pick.
    np.testing.assert_array_equal(targets["trace"][0, 2:], np.repeat(targets["trace"][0, 2:3], 6, axis=0))
    # Motion does not depend on Events: an episode saved without any gives the same trace.
    no_events = tmp_path / "no_events"
    write_index(no_events / "index.json", [saved_episode(no_events, events=False)])
    config = ESkillConfig.tiny()
    without_events = EpisodeDataset(no_events / "index.json", action_horizon=4, skill_names=config.skill_names,
                                    motion_queries=8, motion_stride=1, image_size=32)[2]
    np.testing.assert_array_equal(without_events["targets"]["trace"], targets["trace"])
    assert not without_events["targets"]["skill_valid"].any()


@pytest.mark.parametrize("change", ["no_calibration", "moving_camera"])
def test_missing_or_moving_calibration_masks_only_motion(tmp_path, change):
    def edit(meta, arrays):
        if change == "no_calibration":
            meta["cameras"]["image"]["calibration"] = None
        else:  # Per-row extrinsics: future points would need the current view; not stored.
            arrays["camera_pose"] = np.repeat(arrays["camera_pose"][None], 4, axis=0)
            meta["cameras"]["image"]["calibration"]["extrinsics"]["shape"] = [4, 4, 4]
    data, _ = dataset(tmp_path, edit)
    assert data.reader(0).meta["effector_pixels"] == {}
    assert not data[0]["targets"]["trace_valid"].any()
    assert data[0]["targets"]["skill_valid"][0]


def test_offscreen_endpoint_does_not_hold_an_earlier_valid_point(tmp_path):
    def edit(meta, arrays):
        arrays["eef"][-1, 0] = 1e4
    data, _ = dataset(tmp_path, edit)
    target = data[0]["targets"]
    np.testing.assert_array_equal(target["trace_valid"][0], [True, True] + [False]*6)
    assert not target["future_padded"].any()
    assert not target["trace"][~target["trace_valid"]].any()


def motion_dataset(root, rows, *, invalid_middle=False):
    """A five-timestamp public axis with an independently sampled EEF stream."""
    writer = EpisodeWriter()
    add = writer.add
    indices = add("indices", np.arange(5))
    points = np.array([[0, 0, 0], [.05, 0, 0], [.1, 0, 0], [.1, .05, 0], [.1, .1, 0]], np.float32)
    rows = np.asarray(rows, np.int64)
    valid = np.ones((len(rows), 3), bool)
    if invalid_middle:
        valid[rows == 2, 0] = False
    def numeric(key, values):
        return dict(frame_indices=indices, values=add(key, values),
                    valid=add(key + "_valid", np.ones_like(values, bool)))
    meta = dict(
        schema_version=SCHEMA_VERSION, episode_id="async",
        source=dict(dataset="fixture", revision="test", episode_id="async"),
        instructions=["pick cup"], arm_ids=["single"], frame_count=5,
        timeline="timestamps", timestamps_s=add("times", np.arange(5) * .05), nominal_fps=None,
        cameras={"image": dict(frame_indices=indices, valid=add("rgb_valid", np.ones(5, bool)),
                               rgb=add("rgb", np.zeros((5, 16, 16, 3), np.uint8)), height=16, width=16,
                               calibration=calibration(writer, 16, 1.))},
        state=numeric("state", np.zeros((5, 8), np.float32)),
        actions=dict(**numeric("actions", np.zeros((5, 7), np.float32)),
                     conditioning_image_indices={"image": indices}),
        effector_regions={}, effector_positions={"single": dict(
            frame_indices=add("eef_rows", rows), values=add("eef", points[rows]),
            valid=add("eef_valid", valid), unit="m", coordinate_frame="world",
            reference_point="test_grasp_site")},
        events=[dict(event_id="pick", arm="single", start_frame=0, stop_frame=5, goal_frame=4,
                     boundary_valid=True, skill="Pick", skill_valid=True, object_regions={})])
    path = writer.save(root / "episode", meta)
    write_index(root / "index.json", [path])
    return EpisodeDataset(root / "index.json", action_horizon=3, skill_names=["Pick"],
                          motion_queries=4, motion_stride=1, image_size=16)


@pytest.mark.parametrize("rows", [[0, 1, 2, 3, 4], [0, 2, 4]])
def test_motion_requires_exact_future_indices_without_interpolating_gaps(tmp_path, rows):
    targets = motion_dataset(tmp_path, rows)[0]["targets"]
    expected = np.array([[.53125, .5], [.5625, .5], [.5625, .53125], [.5625, .5625]])
    valid = np.isin([1, 2, 3, 4], rows)
    np.testing.assert_array_equal(targets["trace_valid"][0], valid)
    np.testing.assert_allclose(targets["trace"][0], np.where(valid[:, None], expected, 0), atol=1e-7)
    assert not targets["trace_valid"][1].any()


def test_ambiguous_duplicate_eef_indices_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="increasing"):
        motion_dataset(tmp_path, [0, 2, 2, 4])[0]


@pytest.mark.parametrize("rows,invalid_middle", [([2, 4], False), ([0, 2], False),
                                                ([], False), ([0, 2, 4], True)])
def test_missing_records_mask_only_corresponding_motion_points(tmp_path, rows, invalid_middle):
    data = motion_dataset(tmp_path, rows, invalid_middle=invalid_middle)
    targets = data[0]["targets"]
    expected = np.isin([1, 2, 3, 4], rows)
    if invalid_middle:
        expected[1] = False
    np.testing.assert_array_equal(targets["trace_valid"][0], expected)
    # Holding must reach the actual episode endpoint, not the last available EEF record.
    assert bool(data[4]["targets"]["trace_valid"][0].all()) == (4 in rows)
    assert targets["skill_valid"][0] and targets["action_valid"].all()


@pytest.mark.parametrize("field,value", [
    ("unit", None), ("unit", "mm"),
    ("coordinate_frame", None), ("coordinate_frame", "base"), ("coordinate_frame", "tool"),
    ("reference_point", None), ("reference_point", ""), ("reference_point", "unknown"),
])
def test_unconfirmed_geometry_masks_only_motion(tmp_path, field, value):
    before = dataset(tmp_path / "confirmed")[0][3]["targets"]
    assert before["trace_valid"][0].all()
    np.testing.assert_allclose(before["trace"][0, -1], [.5 + 20/352, .5 + 30/352])
    def edit(meta, arrays):
        positions = meta["effector_positions"]["single"]
        if value is None:
            del positions[field]
        else:
            positions[field] = value
    after = dataset(tmp_path / "unconfirmed", edit)[0][3]["targets"]
    assert not after["trace_valid"].any() and not after["trace"].any()
    for key in before.keys() - {"trace", "trace_valid", "future_observed", "future_padded", "role_sources"}:
        np.testing.assert_array_equal(after[key], before[key])


@pytest.mark.parametrize('available', [False, True])
def test_missing_eef_values_are_allowed_only_when_marked_invalid(tmp_path, available):
    def edit(meta, arrays):
        arrays['eef'][1] = np.nan
        arrays['eef_valid'][1] = available
    if available:
        with pytest.raises(ValueError, match='Invalid numeric sequence'):
            dataset(tmp_path, edit=edit)
    else:
        data, _ = dataset(tmp_path, edit=edit)
        targets = data[0]['targets']
        assert not targets['trace_valid'][0, 0]
        assert targets['trace_valid'][0, 1:].all()
        assert np.isfinite(targets['trace']).all()


def test_splits_and_statistics_use_training_episodes_once(tmp_path):
    paths = [saved_episode(tmp_path, f"ep{i}", parent=f"parent{i//2}") for i in range(20)]
    index = tmp_path / "index.json"
    # Two views can share a parent in pre-existing indices; write_index normally rejects duplicates.
    index.write_text(json.dumps({"episodes": [
        dict(episode_id=f"ep{i}", path=str(p.relative_to(tmp_path))) for i, p in enumerate(paths)]}))
    splits = split_episodes(index, .5, 42)
    assert splits["train"] and splits["val"]
    for i in range(0, 20, 2):
        assert (f"ep{i}" in splits["train"]) == (f"ep{i+1}" in splits["train"])
    config = TrainConfig(str(index), "unused", str(tmp_path / "run"), "unused", validation_fraction=.5)
    stats = compute(config)
    n = len(splits["train"])
    assert stats["state"]["count"] == [4 * n] * 8
    assert stats["actions"]["count"] == [6 * n] * 7
    assert stats["arm_state"]["count"][1] == [0] * 15
    assert set(stats["provenance"]["episode_ids"]) == set(splits["train"])
    np.testing.assert_allclose(stats["actions"]["mean"], np.arange(42).reshape(6, 7).mean(0))


def test_canvas_preserves_layout_and_clips_screen():
    image = np.full((16, 32, 3), 255, np.uint8)
    canvas, weight, valid = role_canvas(image, {"bbox": [24, 0, 40, 8]}, 32, margin=0)
    assert valid and canvas[:8].sum() == 0  # vertical letterbox
    assert canvas[:, :24].sum() == 0 and canvas[8:16, 24:].min() == 255
    assert weight[8:16, 24:].min() == 1
    assert not role_canvas(image, {"bbox": [40, 0, 45, 8]}, 32, margin=0)[2]


@pytest.mark.parametrize('camera', ['image', 'wrist_image'])
def test_missing_camera_pairing_cannot_read_last_frame(tmp_path, camera):
    import copy
    def edit(meta, arrays):
        meta['cameras']['wrist_image'] = copy.deepcopy(meta['cameras']['image'])
        wrist_ref = copy.deepcopy(meta['actions']['conditioning_image_indices']['image'])
        wrist_ref['key'] = 'wrist_pairing'
        meta['actions']['conditioning_image_indices']['wrist_image'] = wrist_ref
        arrays['wrist_pairing'] = arrays['pairing'].copy()
        key = meta['actions']['conditioning_image_indices'][camera]['key']
        arrays[key][0] = -1
    path = saved_episode(tmp_path, edit=edit)
    index = tmp_path / 'index.json'
    write_index(index, [path])  # -1 remains a legal storage representation of missing data.
    data = EpisodeDataset(index, action_horizon=4, cameras=('image', 'wrist_image'), auxiliary=False)
    with pytest.raises(ValueError, match=f'action row 0 has no paired image for {camera}'):
        data[0]
    assert data[1]['metadata']['frame'] == 0
