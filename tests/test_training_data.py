"""Action pairing, Event-owned goals and Event-local physical-time motion targets."""
import json

import numpy as np
import pytest
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeDataset, SCHEMA_VERSION
from openpi.training.config import TrainConfig, split_episodes
from openpi.transforms import role_canvas, role_canvases
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
    xyz = np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 3, 1]], np.float32)
    states[:, -8:-5] = xyz
    commands = np.arange(42, dtype=np.float32).reshape(6, 7)
    commands[:, 6] = [-1, -1, 1, 1, -1, -1]
    meta = dict(
        schema_version=SCHEMA_VERSION, episode_id=identity,
        source=dict(dataset="fixture", revision="test", episode_id=identity,
                    parent_trajectory_id=parent or identity),
        instructions=["pick cup then place it"], arm_ids=["single"], frame_count=4,
        timeline="timestamps", timestamps_s=array("times", np.arange(4) * .25), nominal_fps=None,
        cameras={"image": dict(frame_indices=indices, valid=array("rgb_valid", np.ones(4, bool)),
                               rgb=array("rgb", rgb), height=32, width=32, calibration=calibration(writer, 32))},
        state=dict(frame_indices=indices, values=array("state", states),
                   valid=array("state_valid", np.ones_like(states, bool))),
        actions=dict(frame_indices=array("action_indices", [0, 0, 1, 2, 2, 3]),
                     values=array("actions", commands),
                     gripper_convention=dict(channels={"single": 6}, close_sign=1,
                         alignment="observation_then_action", evidence="fixture execution order"),
                     valid=array("action_valid", np.ones((6, 7), bool)),
                     conditioning_image_indices={"image": array("pairing", [0, 0, 0, 2, 2, 3])}),
        effector_regions={"single": {"image": old_object}},
        effector_positions={"single": dict(
            frame_indices=indices, values=array("eef", xyz),
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
                          motion_stride=1, motion_min_horizon_frames=0, image_size=32), index


def test_action_windows_use_action_time_and_paired_observation(tmp_path):
    data, _ = dataset(tmp_path)
    item = data[2]  # Action at frame 1 observes frame 0.
    assert item["metadata"]["frame"] == 0
    assert item["metadata"]["action_frames"].tolist() == [1, 2, 2, 3]
    assert item["targets"]["skill_labels"].tolist() == [0, -1]
    assert not item["targets"]["trace_valid"].any()  # Pick ends before the first future query.
    reader = data.reader(0)
    goal = reader.goal("pick", "image")
    expected = role_canvas(goal["rgb"], goal["region"], 32, expand_px=8)[0]
    canvases = role_canvases(item["targets"]["role_sources"], 32)["role_images"]
    np.testing.assert_array_equal(canvases[0, 1], expected.transpose(2, 0, 1).astype(np.float32) / 127.5 - 1)
    next_region = reader.region(reader.meta["events"][1]["object_regions"]["image"], 1)
    assert not np.array_equal(expected, role_canvas(reader.rgb("image", 1), next_region, 32)[0])
    assert set(item["inputs"]) == {"images", "state", "state_valid", "prompt"}
    assert reader.meta["quality"]["manual_review"] == "pending"


def test_padding_and_invalid_labels_never_become_idle(tmp_path):
    data, _ = dataset(tmp_path)
    item = data[4]["targets"]
    assert not item["action_valid"][2:].any()
    assert item["skill_labels"].tolist() == [1, -1]
    reader = data.reader(0)
    reader.meta["events"][1]["skill_valid"] = False
    assert not data[3]["targets"]["skill_valid"].any()
    assert role_canvases(data[3]["targets"]["role_sources"], 32)["role_valid"][0].all()
    assert data[3]["targets"]["trace_valid"][0, 0]
    reader.meta["events"][1]["boundary_valid"] = False
    assert not data[3]["targets"]["trace_valid"].any()


def test_missing_path_values_do_not_disable_gripper_commands(tmp_path):
    def edit(meta, arrays):
        arrays["eef_valid"][3] = False
    data, _ = dataset(tmp_path, edit)
    target = data[3]["targets"]
    assert not target["trace_valid"].any()
    assert target["gripper_valid"][0, 0]
    assert target["future_projection_invalid"][0, 0]


def test_motion_remains_within_event_and_never_fills_terminal_loss(tmp_path):
    data, _ = dataset(tmp_path)
    target = data[3]["targets"]
    np.testing.assert_array_equal(target["trace_valid"][0], [True] + [False] * 7)
    assert not target["trace_valid"][1].any()
    assert not target["future_padded"].any()
    assert target["future_event_tail"][0, 1:].all()
    assert not data[5]["targets"]["trace_valid"].any()
    no_events = tmp_path / "no_events"
    write_index(no_events / "index.json", [saved_episode(no_events, events=False)])
    without = EpisodeDataset(no_events / "index.json", action_horizon=4, skill_names=ESkillConfig().skill_names)[0]
    assert not without["targets"]["trace_valid"].any()
    assert not without["targets"]["skill_valid"].any()
    assert without["targets"]["action_valid"].all()


@pytest.mark.parametrize("change", ["no_calibration", "moving_camera"])
def test_training_requires_calibrated_effector_pixels(tmp_path, change):
    def edit(meta, arrays):
        if change == "no_calibration":
            meta["cameras"]["image"]["calibration"] = None
        else:
            arrays["camera_pose"] = np.repeat(arrays["camera_pose"][None], 4, axis=0)
            meta["cameras"]["image"]["calibration"]["extrinsics"]["shape"] = [4, 4, 4]
    data, _ = dataset(tmp_path, edit)
    assert data.reader(0).meta["effector_pixels"] == {}
    with pytest.raises(ValueError, match="calibrated"):
        data[0]


def test_offscreen_endpoint_is_invalid_without_clamping_or_holding(tmp_path):
    def edit(meta, arrays):
        arrays["eef"][-1, 0] = 1e4
    data, _ = dataset(tmp_path, edit)
    target = data[3]["targets"]
    assert not target["trace_valid"].any() and not target["trace"].any()
    assert target["gripper_valid"][0, 0]
    assert target["future_projection_invalid"][0, 0]


def test_splits_and_statistics_use_training_episodes_once(tmp_path):
    paths = [saved_episode(tmp_path, f"ep{i}", parent=f"parent{i//2}") for i in range(20)]
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"episodes": [
        dict(episode_id=f"ep{i}", path=str(p.relative_to(tmp_path))) for i, p in enumerate(paths)]}))
    splits = split_episodes(index, .5, 42)
    for i in range(0, 20, 2):
        assert (f"ep{i}" in splits["train"]) == (f"ep{i+1}" in splits["train"])
    config = TrainConfig(str(index), "unused", str(tmp_path / "run"), "unused", validation_fraction=.5)
    stats = compute(config)
    n = len(splits["train"])
    assert stats["state"]["count"] == [4 * n] * 8
    assert stats["actions"]["count"] == [6 * n] * 7
    assert "arm_state" not in stats
    assert set(stats["provenance"]["episode_ids"]) == set(splits["train"])
    raw = np.arange(42).reshape(6, 7).astype(float)
    raw[:, 6] = [-1, -1, 1, 1, -1, -1]
    np.testing.assert_allclose(stats["actions"]["mean"], raw.mean(0))


def test_canvas_preserves_layout_and_clips_screen():
    image = np.full((16, 32, 3), 255, np.uint8)
    canvas, weight, valid = role_canvas(image, {"bbox": [24, 0, 40, 8]}, 32, margin=0)
    assert valid and canvas[:8].sum() == 0
    assert canvas[:, :24].sum() == 0 and canvas[8:16, 24:].min() == 255
    assert weight[8:16, 24:].min() == 1
    assert not role_canvas(image, {"bbox": [40, 0, 45, 8]}, 32, margin=0)[2]


@pytest.mark.parametrize('shape', [(256, 256), (512, 512), (128, 256)])
def test_only_future_expands_eight_canvas_pixels_without_changing_source(shape):
    h, w = shape
    image = np.full((h, w, 3), 200, np.uint8)
    box = np.array([w // 3, h // 3, 2 * w // 3, 2 * h // 3])
    sources = [[(image, box), (image, box)], [(None, None), (None, None)]]
    previous = role_canvases(sources, 256, future_bbox_expand_px=0)
    result = role_canvases(sources, 256)
    np.testing.assert_array_equal(result['role_images'][0, 0], previous['role_images'][0, 0])
    np.testing.assert_array_equal(result['role_weights'][0, 0], previous['role_weights'][0, 0])
    old_y, old_x = np.where(previous['role_weights'][0, 1] > 0)
    new_y, new_x = np.where(result['role_weights'][0, 1] > 0)
    assert (new_x.min(), new_y.min(), new_x.max(), new_y.max()) == (
        old_x.min() - 8, old_y.min() - 8, old_x.max() + 8, old_y.max() + 8)
    assert result['role_images'][0, 1, :, new_y.min() + 2, new_x.min() + 2].min() > -1
    assert result['role_valid'].tolist() == [[True, True], [False, False]]
    np.testing.assert_array_equal(box, [w // 3, h // 3, 2 * w // 3, 2 * h // 3])


def test_future_expansion_clips_to_rgb_and_keeps_missing_region_invalid():
    image = np.full((128, 256, 3), 255, np.uint8)
    sources = [[(image, [0, 0, 256, 128]), (image, [0, 0, 256, 128])],
               [(None, None), (image, None)]]
    result = role_canvases(sources, 256)
    weight = result['role_weights'][0, 1]
    assert not weight[:64].any() and not weight[192:].any()
    assert weight[64:192].min() == 1
    assert not result['role_valid'][1].any()
    assert not result['role_weights'][1].any()


@pytest.mark.parametrize("camera", ["image", "wrist_image"])
def test_missing_camera_pairing_cannot_read_last_frame(tmp_path, camera):
    import copy
    def edit(meta, arrays):
        meta["cameras"]["wrist_image"] = copy.deepcopy(meta["cameras"]["image"])
        wrist_ref = copy.deepcopy(meta["actions"]["conditioning_image_indices"]["image"])
        wrist_ref["key"] = "wrist_pairing"
        meta["actions"]["conditioning_image_indices"]["wrist_image"] = wrist_ref
        arrays["wrist_pairing"] = arrays["pairing"].copy()
        arrays[meta["actions"]["conditioning_image_indices"][camera]["key"]][0] = -1
    path = saved_episode(tmp_path, edit=edit)
    index = tmp_path / "index.json"
    write_index(index, [path])
    data = EpisodeDataset(index, action_horizon=4, cameras=("image", "wrist_image"), auxiliary=False)
    with pytest.raises(ValueError, match=f"action row 0 has no paired image for {camera}"):
        data[0]
    assert data[1]["metadata"]["frame"] == 0
