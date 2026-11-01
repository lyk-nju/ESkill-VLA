"""Explicit readers for historical XPolicyLab v1 and complete RoboTwin raw v2."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

from scripts.preprocess import DATA_ROOT, PROJECT_ROOT

SOURCE_ROOT = DATA_ROOT / "robotwin"
ROBOTWIN_ROOT = PROJECT_ROOT.parent / "RoboTwin"
sys.path.append(str(ROBOTWIN_ROOT))
from XPolicyLab.utils.process_data import decode_image_bit

FORMATS = ("xpolicylab-v1", "robotwin.raw.v2")
CAMERAS = {"head_camera": "cam_head", "left_camera": "cam_left_wrist",
           "right_camera": "cam_right_wrist", "front_camera": "cam_third_view"}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config(path=SOURCE_ROOT / "config.yaml"):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    if config["source_format"] not in FORMATS:
        raise ValueError(f"Explicit source_format required: {FORMATS}")
    for key in ("source_root", "check_root"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return config


def inventory(config):
    root = Path(config["source_root"])
    folder = "data" if config["source_format"] == "xpolicylab-v1" else "raw"
    entries = []
    for path in sorted(root.glob(f"*/*/{folder}/episode_*.hdf5")):
        task_dir = path.parent.parent
        item = {"task": task_dir.parent.name, "episode": path.stem,
                "hdf5": str(path.resolve()), "format": config["source_format"]}
        if config["source_format"] == "xpolicylab-v1":
            item["events"] = str(task_dir / "events" / f"{path.stem}.json")
        entries.append(item)
    if not entries:
        raise ValueError(f"No {config['source_format']} episodes at {root}")
    return entries


def freeze_entry(entry):
    item = dict(entry)
    item["sha256"] = digest(item["hdf5"])
    if item.get("events"):
        item["events_sha256"] = digest(item["events"])
    return item


def verify_entry(entry):
    for key, hash_key in (("hdf5", "sha256"), ("events", "events_sha256")):
        if entry.get(hash_key) and digest(entry[key]) != entry[hash_key]:
            raise ValueError(f"Source changed since sampling: {entry[key]}")


def _json_dataset(dataset):
    value = dataset[()]
    return json.loads(value.decode() if isinstance(value, bytes) else str(value))


@dataclass
class Episode:
    entry: dict
    fps: float
    count: int
    instructions: list
    cameras: dict
    signals: dict
    poses: dict
    calibration: dict
    evidence: dict
    semantics: dict

    @property
    def arms(self):
        return ("left", "right")

    def rgb(self, camera, frame):
        values = self.cameras[camera]
        if not 0 <= frame < len(values):
            return None  # Missing terminal image stays missing.
        return decode_image_bit(values[frame])


def iter_episodes(manifest_path):
    """Adapt existing fixed samples to shared preprocessing; retain raw primitives."""
    for entry in read_json(manifest_path)["samples"]:
        verify_entry(entry)
        episode = load_episode(entry)
        episode.entry = {**episode.entry, "key": f"{entry['task']}/{entry['episode']}"}
        intervals = []
        for index, primitive in enumerate(episode.evidence.get("subtasks", [])):
            start, end = int(primitive["start_frame"]), int(primitive["end_frame"])
            if primitive["arm"] not in episode.arms or not 0 <= start <= end < episode.count:
                raise ValueError(f"Invalid source primitive bounds/arm: {primitive}")
            intervals.append({"source_id": f"primitive_{index:04d}", "arm": primitive["arm"],
                              "start_frame": start, "stop_frame": end+1, "goal_frame": end,
                              "source_skill": primitive.get("label", primitive.get("primitive", "")),
                              "source_object": primitive.get("actor", ""),
                              "source_kind": "script_primitive_not_verified_event"})
        episode.evidence = {**episode.evidence, "intervals": intervals}
        yield episode


def load_episode(entry):
    if entry["format"] not in FORMATS:
        raise ValueError("Unsupported source format")
    cameras, signals, poses, calibration = {}, {}, {}, {}
    with h5py.File(entry["hdf5"], "r") as f:
        if entry["format"] == "robotwin.raw.v2":
            if f.attrs.get("format") != "robotwin.raw.v2" or not f.attrs.get("complete"):
                raise ValueError("Not a completed raw v2 episode")
            fps, count = float(f.attrs["frequency"]), int(f.attrs["frame_count"])
            evidence = _json_dataset(f["evidence"])
            obs = f["observations"]
            for group in ("joint_action", "joint_state"):
                if group in obs:
                    signals.update({f"{group}/{k}": v[:] for k, v in obs[group].items()})
            for arm in ("left", "right"):
                key = f"endpose/{arm}_endpose"
                if key in obs:
                    poses[arm] = obs[key][:]
            for source_name, group in obs["observation"].items():
                name = CAMERAS.get(source_name, source_name)
                if "rgb" in group:
                    cameras[name] = group["rgb"][:]
                calibration[name] = {k: group[k][:] for k in
                                     ("intrinsic_cv", "cam2world_gl", "extrinsic_cv") if k in group}
            semantics = {k: str(v) for k, v in f.attrs.items() if "semantics" in k or k in ("ee_reference", "clock")}
            timestamp, frame_index = f["timestamp"][:], f["frame_index"][:]
        else:
            if "data_format_version" not in f or f["data_format_version"][()].decode() != "v1.0":
                raise ValueError("Not XPolicyLab v1.0")
            evidence = read_json(entry["events"]) if entry.get("events") else {}
            fps = float(f["additional_info/frequency"][()])
            for group in ("state", "action"):
                signals.update({f"{group}/{k}": v[:] for k, v in f[group].items()})
            count = int(evidence.get("frame_count", len(next(iter(f["state"].values())))))
            for arm in ("left", "right"):
                key = f"state/{arm}_ee_poses"
                if key in f:
                    poses[arm] = f[key][:]
                    action_key = f"action/{arm}_ee_poses"
                    if len(poses[arm]) + 1 == count and action_key in f:
                        poses[arm] = np.concatenate([poses[arm], f[action_key][-1:]], axis=0)
            for name, group in f["vision"].items():
                cameras[name] = group["colors"][:]
                calibration[name] = {k: group[k][:] for k in
                                     ("intrinsic_matrix", "extrinsics_matrix") if k in group}
            semantics = {"state": "historical_field_not_verified_as_measured_qpos",
                         "action": "stored_one_step_shift_do_not_shift_again",
                         "ee_reference": "requires_source_version_confirmation"}
            timestamp, frame_index = np.arange(count) / fps, np.arange(count)
        instructions = _json_dataset(f["instructions"])
        signals["timestamp"], signals["frame_index"] = timestamp, frame_index
    return Episode(entry, fps, count, instructions, cameras, signals, poses,
                   calibration, evidence, semantics)


def validate_episode(episode):
    errors, missing = [], []
    if not np.isfinite(episode.fps) or episode.fps <= 0 or episode.count < 2:
        errors.append("Invalid frequency or fewer than two frames")
    if not episode.instructions or not all(isinstance(s, str) and s for s in episode.instructions):
        errors.append("Missing task instruction")
    if not episode.cameras:
        errors.append("Missing RGB cameras")
    for arm in ("left", "right"):
        required = ([f"state/{arm}_arm_joint_states", f"state/{arm}_ee_joint_states",
                     f"action/{arm}_arm_joint_states", f"action/{arm}_ee_joint_states"]
                    if episode.entry["format"] == "xpolicylab-v1" else
                    [f"joint_action/{arm}_arm", f"joint_action/{arm}_gripper"])
        for key in required:
            if key not in episode.signals:
                errors.append(f"Missing required signal: {key}")
    for name, array in {**episode.signals, **episode.poses}.items():
        if not np.isfinite(array).all():
            errors.append(f"Non-finite {name}")
        expected = episode.count
        legacy_short = episode.entry["format"] == "xpolicylab-v1" and name.startswith(("state/", "action/"))
        if len(array) not in ({expected, expected - 1} if legacy_short else {expected}):
            errors.append(f"Timeline length mismatch: {name}={len(array)}, source={expected}")
    if not np.array_equal(episode.signals["frame_index"], np.arange(episode.count)):
        errors.append("Noncontiguous frame indices")
    if episode.semantics.get("clock") == "physics":
        if np.any(np.diff(episode.signals["timestamp"]) < 0):
            errors.append("Physical timestamps go backwards")
    elif episode.fps > 0 and not np.allclose(episode.signals["timestamp"], np.arange(episode.count) / episode.fps):
        errors.append("Timestamp/frequency mismatch")
    for name, frames in episode.cameras.items():
        if len(frames) != episode.count:
            if episode.entry["format"] == "xpolicylab-v1" and len(frames) == episode.count - 1:
                missing.append(f"{name}: terminal RGB missing")
            else:
                errors.append(f"RGB timeline mismatch: {name}")
        for frame in (0, len(frames) - 1):
            rgb = episode.rgb(name, frame)
            if rgb is None or rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
                errors.append(f"Invalid RGB: {name}/{frame}")
    for arm in ("left", "right"):
        if arm not in episode.poses:
            missing.append(f"{arm}: EE path unavailable")
        elif episode.poses[arm].ndim != 2 or episode.poses[arm].shape[1] < 3:
            errors.append(f"{arm}: invalid EE pose dimensions")
    return {"errors": errors, "missing": missing, "semantics": episode.semantics,
            "frames": episode.count, "fps": episode.fps,
            "manual_review": "pending"}
