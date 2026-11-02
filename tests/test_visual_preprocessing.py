"""Software contract tests with synthetic RGB and fake outputs, never model inference."""
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import hashlib
import subprocess
import sys

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.preprocess.libero.guidedvla.source import load_episode
from scripts.preprocess.text_annotation.annotate import prepare, annotate_episode, validate_text
from scripts.preprocess.visual_grounding.annotate import target_catalog, locate_episode, track_episode
from scripts.preprocess.visual_grounding.qwen import pixel_box, parse_json
from scripts.preprocess.visual_grounding.render import Regions, render_episode
from scripts.preprocess.visual_grounding.sam import mask_box, Sam


def test_offline_imports_and_default_output_do_not_load_models():
    code = """
import sys
from pathlib import Path
from scripts.preprocess.artifacts import ROOT, processed_root
import scripts.preprocess.visual_grounding.render
import scripts.preprocess.visual_grounding.annotate
import scripts.preprocess.text_annotation.annotate
assert ROOT == Path.cwd() / 'data'
assert processed_root('libero') == ROOT / 'libero/guidedvla/processed'
assert processed_root('robotwin') == ROOT / 'robotwin/processed'
assert not any(name.split('.')[0] in {'torch', 'transformers', 'sam3', 'sapien'} for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).resolve().parents[1])


@pytest.fixture
def episode():
    frames = np.zeros((4, 24, 32, 3), dtype=np.uint8)
    frames[..., 0], frames[..., 1], frames[..., 2] = 180, 30, 10
    intervals = [dict(source_id="a", arm="left", start_frame=0, stop_frame=3, goal_frame=2,
                      source_skill="grasp", source_kind="primitive"),
                 dict(source_id="b", arm="right", start_frame=2, stop_frame=4, goal_frame=3,
                      source_skill="place", source_kind="primitive")]
    return SimpleNamespace(entry={"key": "test/episode_0", "sha256": "synthetic"}, count=4, fps=10,
                           arms=("left", "right"), instructions=["hand over red cube"],
                           cameras={"head": frames, "wrist": frames[:-1]}, evidence={"intervals": intervals},
                           rgb=lambda camera, f: frames[f].copy() if 0 <= f < (3 if camera == "wrist" else 4) else None)


def labels(episode):
    result = prepare(episode)
    result["objects"] = [{"id": "cube", "description": "red cube"}]
    for e in result["events"]:
        e.update(skill="Pick", object_id="cube", description="handle cube", status="model_proposed")
    return result


def test_location_coordinate_contract():
    assert pixel_box([250, 250, 750, 750], 320, 240) == [80, 60, 240, 180]
    for box in ([0, 0, 1001, 1000], [500, 0, 400, 20], [float("nan"), 0, 2, 3]):
        with pytest.raises(ValueError):
            pixel_box(box, 320, 240)
    assert parse_json('```json\n{"visible": false}\n```') == {"visible": False}
    with pytest.raises(ValueError):
        parse_json('explanation {"visible": true}')


def test_cross_arm_handover_uses_one_object_track(episode):
    record = labels(episode)
    catalog = target_catalog(episode, validate_text(episode, record))
    assert [t["id"] for t in catalog] == ["gripper_left", "gripper_right", "cube"]
    assert catalog[-1]["events"] == ["a", "b"]
    assert prepare(episode)["events"][0]["skill"] is None
    record["events"][0]["object_id"] = "unlisted"
    with pytest.raises(ValueError):
        validate_text(episode, record)


def test_text_model_cannot_invent_or_move_events(episode):
    class Client:
        metadata = {"test_double": True}
        def ask(self, prompt, images):
            return {"objects": [], "events": [{"source_id": "invented", "skill": "Pick",
                                                "description": "", "object_id": None}]}, "fake"
    record = annotate_episode(episode, Client(), {"skills": ["Pick"], "max_keyframes": 3, "camera": "head"})
    assert "invented" in record["annotation_error"]
    assert record["events"] == prepare(episode)["events"]
    assert record["objects"] == [] and record["raw_response"] == "fake"


class Locator:
    metadata = {"test_double": True}
    def locate(self, description, images):
        return {"visible": True, "frame_index": images[0][0], "bbox": [2, 3, 8, 9]}, "fake"


class Tracker:
    metadata = {"test_double": True}
    def track(self, directory, **kwargs):
        assert len(list(directory.glob("*.jpg"))) == kwargs["count"]
        # Out-of-order output must still preserve source frame -> packed-mask mapping.
        for frame in reversed(range(kwargs["count"])):
            mask = np.zeros((kwargs["height"], kwargs["width"]), dtype=bool)
            if frame != 1:
                mask[3:9, 2+frame:8+frame] = True
            yield frame, mask


@pytest.mark.parametrize("render_cameras", [["head", "wrist"], ["head"]])
def test_tracking_empty_masks_missing_endpoint_and_render(episode, tmp_path, render_cameras):
    prompts = locate_episode(episode, labels(episode), Locator(),
                             {"cameras": ["head", "wrist"], "keyframes_per_target": 3})
    result = track_episode(episode, prompts, Tracker(), tmp_path)
    index = Regions(tmp_path, result)
    first = index.index[("head", "cube", 0)]
    assert first["bbox"] == [2, 3, 8, 9]
    np.testing.assert_array_equal(mask_box(index.mask(first)), first["bbox"])
    assert index.index[("head", "cube", 1)]["status"] == "empty"
    assert index.index[("head", "cube", 1)]["bbox"] is None
    assert index.index[("wrist", "cube", 3)]["status"] == "missing_source_image"
    assert result["manual_review"] == "pending"
    target = tmp_path / "video/test_processed.mp4"
    report = render_episode(episode, result, tmp_path, target, render_cameras)
    assert report["frames"] == 4 and report["manual_review"] == "pending"
    with av.open(str(target)) as video:
        frames = list(video.decode(video=0))
        assert len(frames) == 4 and float(video.streams.video[0].average_rate) == 10
        rgb = frames[0].to_ndarray(format="rgb24")
        # Both arms retain their goal panels even with a single source camera.
        assert rgb.shape == (960, 960, 3)
        # Original red channel remains red through the renderer and codec.
        sample = rgb[150:250, 180:280].mean(axis=(0, 1))
        assert sample[0] > sample[2] + 90


def test_unlocalized_stays_missing_without_sam_call(episode, tmp_path):
    class Missing:
        metadata = {"test_double": True}
        def locate(self, description, images):
            return {"visible": False, "reason": "uncertain"}, "fake"
    class Unused:
        metadata = {}
        def track(self, *args, **kwargs):
            raise AssertionError("Must not track a missing seed")
    prompt = locate_episode(episode, labels(episode), Missing(), {"cameras": ["head"], "keyframes_per_target": 3})
    result = track_episode(episode, prompt, Unused(), tmp_path)
    assert result["regions"] == []
    assert all(r["status"] == "unlocalized" for r in result["coverage"])


def test_libero_reader_keeps_terminal_and_skill_runs(tmp_path):
    image = np.full((8, 10, 3), [180, 30, 10], dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    row = {"image": {"bytes": buffer.getvalue(), "path": None},
           "wrist_image": {"bytes": buffer.getvalue(), "path": None},
           "state": [0.]*8, "joint_state": [0.]*7, "actions": [0.]*7,
           "episode_index": 8, "observation.skill_text": "approach target"}
    table = pa.Table.from_pylist([{**row, "frame_index": f, "timestamp": f/10, "index": 40+f,
                                  "observation.skill_id": 1 if f < 2 else 3} for f in range(3)])
    path = tmp_path / "source.parquet"
    pq.write_table(table, path)
    sample = dict(episode_index=8, expected_frames=3, dataset_from_index=40, dataset_to_index=43, task="move cube")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    episode = load_episode(path, sample, fps=10, revision="test", expected_sha256=sha)
    assert episode.count == 3 and episode.arms == ("single",)
    np.testing.assert_array_equal(episode.rgb("image", 2), image)
    assert [(e["start_frame"], e["stop_frame"], e["goal_frame"]) for e in episode.evidence["intervals"]] == [(0, 2, 1), (2, 3, 2)]
    with pytest.raises(ValueError, match="hash mismatch"):
        load_episode(path, sample, fps=10, revision="test", expected_sha256="wrong")


def test_sam_adapter_tracks_both_directions_and_returned_ids(tmp_path):
    import torch
    class Predictor:
        def init_state(self, **kwargs):
            assert kwargs["offload_state_to_cpu"] is False
            return {"num_frames": 3}
        def add_new_points_or_box(self, **kwargs):
            np.testing.assert_allclose(kwargs["box"], [[.1, .2, .5, .6]])
            assert kwargs["rel_coordinates"] is True
        def propagate_in_video(self, state, **kwargs):
            frames = [1, 0] if kwargs["reverse"] else [1, 2]
            for frame in frames:
                # Target 1 is returned second; selecting initial slot 0 is incorrect.
                yield frame, [7, 1], None, torch.stack([torch.zeros(1, 8, 10), torch.ones(1, 8, 10)]), None
    backend = Sam({"device": "cpu"})
    backend.predictor = Predictor()
    results = list(backend.track(tmp_path, seed_frame=1, bbox=[1, 1.6, 5, 4.8], width=10, height=8, count=3))
    assert sorted(frame for frame, _ in results) == [0, 1, 2]
    assert all(mask.all() for _, mask in results)
