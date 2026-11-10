"""ESkill physical-time queries, Event boundaries, command semantics and raw body geometry."""
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


def fixture(root, *, times=True, edit=None):
    w, n = EpisodeWriter(), 9
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
    for name, start, stop, skill in [("pick", 0, 4, "Pick"), ("place", 4, n, "Place")]:
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
                          motion_queries=8, motion_offsets_s=cfg.motion_offsets_s, image_size=cfg.image_size, **kwargs)


def test_no_clock_or_semantics_fails_before_rgb_decode(tmp_path):
    index = fixture(tmp_path, times=False)
    with mock.patch.object(EpisodeReader, "rgb", side_effect=AssertionError("decoded RGB")):
        with pytest.raises(ValueError, match="physical timestamps"):
            dataset(index)
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


def test_removed_steps_are_missing_not_nearest_or_endpoint_targets(tmp_path):
    def edit(meta, arrays):
        arrays["timestamps"] = np.array([0, .25, .5, 1, 1.25, 1.5, 1.75, 2, 2.25])
    target = dataset(fixture(tmp_path, edit=edit))[0]["targets"]
    assert target["future_missing"][0, 2]
    assert not target["trace_valid"][0, 2] and not target["gripper_valid"][0, 2]
    assert target["trace_valid"][0, 3]  # +1s is a real record, still in Pick.


def test_explicit_control_clock_keeps_its_source_step_mapping(tmp_path):
    index = fixture(tmp_path, times=False)
    manifest = tmp_path / "timing.json"
    manifest.write_text(json.dumps(dict(episodes={"eskill_fixture": dict(
        source_step_indices=[0, 1, 2, 4, 5, 6, 7, 8, 9], control_frequency_hz=4,
        evidence="verified retained source index fixture")})))
    target = dataset(index, timing_path=manifest)[0]["targets"]
    assert target["future_missing"][0, 2]
    assert target["trace_valid"][0, 3]


def test_path_and_gripper_validity_are_independent(tmp_path):
    def edit(meta, arrays):
        arrays["eef_valid"][1] = False
        arrays["actions"][2, -1] = 0
    target = dataset(fixture(tmp_path, edit=edit))[0]["targets"]
    assert not target["trace_valid"][0, 0] and target["gripper_valid"][0, 0]
    assert target["trace_valid"][0, 1] and not target["gripper_valid"][0, 1]


@pytest.mark.parametrize("skill, label", [("Hold", 6), ("unknown", -1)])
def test_unsupported_skills_keep_identity_and_do_not_disable_other_targets(tmp_path, skill, label):
    def edit(meta, arrays):
        meta["events"][0]["skill"] = skill
    target = dataset(fixture(tmp_path, edit=edit))[0]["targets"]
    assert target["skill_labels"][0] == label and not target["skill_valid"][0]
    assert target["trace_valid"][0].any() and target["gripper_valid"][0].any()


def test_raw_current_body_and_online_projection_match_saved_pixels(tmp_path):
    data = dataset(fixture(tmp_path))
    sample = data[1]
    adapter = LiberoInputs(use_arm_state=True)
    stats = {k: dict(method="standard", mean=[100.] * n, std=[7.] * n) for k, n in [("state", 8), ("actions", 7)]}
    transform = ModelTransforms(ESkillConfig.tiny(), stats, lambda text, state: (np.array([1]), np.array([True])))
    offline = transform.inputs(adapter(sample["inputs"]))
    np.testing.assert_array_equal(offline["arm_state"][0], sample["inputs"]["state"][-8:])
    assert not np.array_equal(offline["state"][:8], offline["arm_state"][0])
    reader = data.reader(0)
    calibration = reader.meta["cameras"]["image"]["calibration"]
    live = dict(sample["inputs"])
    live.pop("effector_start")
    live.pop("effector_start_valid")
    live["main_camera"] = dict(intrinsic=reader.array(calibration["intrinsics"]),
                               world_to_camera=reader.array(calibration["extrinsics"]),
                               image_transform=reader.array(calibration["image_transform"]), image_shape=(32, 32))
    online = transform.inputs(adapter(live))
    np.testing.assert_allclose(online["effector_start"], offline["effector_start"])
    np.testing.assert_array_equal(online["effector_start_valid"], offline["effector_start_valid"])
    result = transform(dict(sample, inputs=adapter(sample["inputs"])))
    assert result["targets"]["role_images"].shape == (2, 2, 3, 32, 32)


def test_audit_reports_query_coverage_and_crossing_counts(tmp_path):
    result = audit(fixture(tmp_path), execute_prefix=2)
    assert result["physical_time_ready_episodes"] == result["command_ready_episodes"] == 1
    assert result["crossing_action_chunks"] > result["crossing_execute_prefixes"] > 0
    assert result["samples_without_same_event_query"] == 2
    assert result["skill_event_counts"] == {"Pick": 1, "Place": 1}
