"""Action-time labels, Event-owned goals, padding and split isolation."""
import json

import numpy as np
from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import ESkillDataset, SCHEMA_VERSION
from openpi.training.config import TrainConfig, split_episodes
from openpi.transforms import role_canvas, spatial_trace
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import EpisodeWriter, write_index


def saved_episode(root, identity="one", parent=None):
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
                               rgb=array("rgb", rgb), height=32, width=32)},
        state=dict(frame_indices=indices, values=array("state", states),
                   valid=array("state_valid", np.ones_like(states, bool))),
        actions=dict(frame_indices=array("action_indices", [0, 0, 1, 2, 2, 3]),
                     values=array("actions", np.arange(42, dtype=np.float32).reshape(6, 7)),
                     valid=array("action_valid", np.ones((6, 7), bool)),
                     conditioning_image_indices={"image": array("pairing", [0, 0, 0, 2, 2, 3])}),
        effector_regions={"single": {"image": old_object}},
        effector_positions={"single": dict(
            frame_indices=indices, values=array("eef", np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [2, 3, 1]], np.float32)),
            valid=array("eef_valid", np.ones((4, 3), bool)))},
        events=[event("pick", 0, 1, 1, "Pick", old_object), event("place", 1, 4, 3, "Place", new_object)],
        quality={"manual_review": "pending", "warnings": ["source_units_pending"]})
    return writer.save(root / identity, meta)


def dataset(tmp_path):
    path = saved_episode(tmp_path)
    index = tmp_path / "index.json"
    write_index(index, [path])
    config = ESkillConfig.tiny()
    return ESkillDataset(index, action_horizon=4, skill_names=config.skill_names,
                         motion_queries=4, canvas_size=32), index


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
    np.testing.assert_array_equal(item["targets"]["role_images"][0, 2], expected)
    new_region = reader.region(reader.meta["events"][1]["object_regions"]["image"], 1)
    assert not np.array_equal(expected, role_canvas(reader.rgb("image", 1), new_region, 32)[0])
    assert set(item["inputs"]) == {"image", "state", "state_valid", "prompt"}
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
    assert data[3]["targets"]["role_valid"][0].all()
    reader.meta["events"][0]["boundary_valid"] = False
    assert not data[0]["targets"]["skill_valid"].any()
    assert not data[0]["targets"]["trace_valid"].any()


def test_spatial_trace_preserves_scale_turn_and_hold_motion():
    points = np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0], [2, 2, 0]])
    value, valid = spatial_trace(points, 5)
    np.testing.assert_allclose(value, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]])
    assert valid.all()
    assert spatial_trace(np.zeros((3, 3)), 4)[1].all()
    assert not spatial_trace([[np.nan, 0, 0]], 4)[1].any()


def test_missing_path_points_do_not_bridge_gap(tmp_path):
    data, _ = dataset(tmp_path)
    reader = data.reader(0)
    reader.array(reader.meta["effector_positions"]["single"]["valid"])[3] = False
    assert not data[3]["targets"]["trace_valid"].any()


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
