"""Read fixed AtomicVLA/RLDS samples, preserving source labels and endpoint evidence."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re

import numpy as np
from PIL import Image

from scripts.preprocess.libero.rlds import load_records
from scripts.preprocess.artifacts import digest, read_json


def source_data(sample):
    """Read the exact cached source episode identified by the sample manifest."""
    if digest(sample['source_cache']) != sample['source_cache_sha256']:
        raise ValueError('Selected source excerpt changed')
    with np.load(sample['source_cache']) as data:
        return {key: data[key].copy() for key in data.files}


def rgb(data, frame):
    start, stop = data['image_offsets'][frame:frame+2]
    with Image.open(BytesIO(data['image_bytes'][start:stop].tobytes())) as image:
        return np.asarray(image.convert('RGB'))


@dataclass
class Episode:
    entry: dict
    fps: float
    count: int
    instructions: list
    cameras: dict
    signals: dict
    poses: dict
    evidence: dict
    arms: tuple = ("single",)

    def rgb(self, camera, frame):
        if not 0 <= frame < self.count:
            return None
        return np.asarray(Image.open(BytesIO(self.cameras[camera][frame])).convert("RGB"))


def source_intervals(annotation, n, *, resolve_shared_boundaries=True, apply_reviewed_skills=True):
    """Use [start, stop); a shared boundary frame belongs to the following Event.

    Source bounds stay in source_start_frame/source_end_frame. The raw conversion
    remains available for comparisons with earlier review batches. Explicit Push
    descriptions mislabeled Pick use the user's confirmed skill correction.
    """
    intervals = []
    for number, event in enumerate(annotation["segments"]):
        start, end = event["start_frame"], event["end_frame"]
        stop = n if end == n else end+1
        if not 0 <= start < stop <= n:
            raise ValueError("Invalid source event bounds; no automatic repair")
        description = re.search(r"This is step \d+/\d+ of task, (.*?), and the", event["chain_of_thought"])
        intervals.append({"source_id": f"event_{number:03d}", "arm": "single", "start_frame": start,
                          "stop_frame": stop, "goal_frame": stop-1, "source_start_frame": start,
                          "source_end_frame": end, "source_skill": event["primary_action_verb"],
                          "skill": event["primary_action_verb"].capitalize(),
                          "description": description[1] if description else event["chain_of_thought"],
                          "source_kind": "atomicvla_published_event"})
        row = intervals[-1]
        if (apply_reviewed_skills and row["skill"] == "Pick"
                and re.match(r"push\b", row["description"].strip(), flags=re.I)):
            row["skill"] = "Push"
            row["skill_correction"] = {
                "rule": "user_reviewed_explicit_push",
                "source_skill": "Pick", "skill": "Push",
                "skill_review": "user_confirmed",
                "reason": "压住操作物体，利用摩擦力沿支撑面平移",
                "review_record": "check/test/annotation_review/skill_review.json",
            }
    if resolve_shared_boundaries:
        for previous, following in zip(intervals, intervals[1:]):
            boundary = following["start_frame"]
            if previous["stop_frame"] <= boundary:
                continue
            if previous["stop_frame"] != boundary+1 or previous["start_frame"] >= boundary:
                raise ValueError("Overlap is not a single shared boundary frame")
            previous["boundary_correction"] = {
                "rule": "shared_frame_belongs_to_following_event",
                "source_stop_frame": previous["stop_frame"],
                "source_goal_frame": previous["goal_frame"],
            }
            previous.update(stop_frame=boundary, goal_frame=boundary-1)
    return intervals


def iter_episodes(manifest_path):
    manifest = read_json(manifest_path)
    root = Path(manifest["raw_root"])
    hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    for relative, expected in hashes.items():
        if digest(root/relative) != expected:
            raise ValueError("AtomicVLA raw shard hash mismatch")
    for sample, data in load_records(root, manifest["samples"]):
        n = sample["frames"]
        state = np.asarray(data["steps/observation/state"], np.float32).reshape(n, 8)
        actions = np.asarray(data["steps/action"], np.float32).reshape(n, 7)
        joints = np.asarray(data["steps/observation/joint_state"], np.float32).reshape(n, 7)
        annotation = sample["annotation"]
        if annotation is None or annotation["all_frames"] != n:
            raise ValueError("This test requires the verified source annotation linkage")
        intervals = source_intervals(annotation, n)
        entry = {"key": sample["key"], "number": sample["number"], "task": sample["task"],
                 "format": "atomicvla.libero.rlds", "revision": manifest["revision"],
                 "shard": sample["shard"], "record_index": sample["record_index"],
                 "sha256": hashes[sample["shard"]], "annotation_id": sample["annotation_candidate_id"],
                 "source_manifest_sha256": digest(manifest_path)}
        yield Episode(entry, 10.0, n, [sample["task"]],
                      {"image": data["steps/observation/image"], "wrist_image": data["steps/observation/wrist_image"]},
                      {"state": state, "actions": actions, "joint_state": joints}, {"single": state[:, :3]},
                      {"intervals": intervals, "fps_meaning": "display_only", "raw_sample": sample,
                       "endpoint_rule": "[start,stop); shared frame belongs to next Event; terminal stop=N; goal=stop-1"})
