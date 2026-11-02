"""Read the fixed GuidedVLA release without changing its images or timeline."""
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


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
        with Image.open(BytesIO(self.cameras[camera][frame]["bytes"])) as image:
            return np.asarray(image.convert("RGB")).copy()


def load_episode(path, sample, *, fps, revision, expected_sha256):
    path = Path(path)
    with path.open("rb") as stream:
        actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_hash != expected_sha256:
        raise ValueError(f"Source hash mismatch: {path}")
    table = pq.read_table(path, filters=[("episode_index", "=", sample["episode_index"])])
    data = table.to_pydict()
    count = table.num_rows
    frames = np.asarray(data["frame_index"])
    if count != sample["expected_frames"] or not np.array_equal(frames, np.arange(count)):
        raise ValueError("Episode length/frame mapping differs from fixed manifest")
    if not np.array_equal(data["index"], np.arange(sample["dataset_from_index"], sample["dataset_to_index"])):
        raise ValueError("Global source-row mapping differs from fixed manifest")
    if not np.allclose(data["timestamp"], frames / fps, atol=1e-5):
        raise ValueError("Timestamp/fps mismatch")
    skills = data["observation.skill_id"]
    stops = [0] + [i for i in range(1, count) if skills[i] != skills[i-1]] + [count]
    intervals = []
    for n, (start, stop) in enumerate(zip(stops, stops[1:])):
        intervals.append({"source_id": f"stage_{n:04d}", "arm": "single",
                          "start_frame": start, "stop_frame": stop, "goal_frame": stop-1,
                          "source_skill": data["observation.skill_text"][start],
                          "source_kind": "contiguous_skill_run_not_verified_event"})
    entry = {"key": f"episode_{sample['episode_index']:06d}", "episode_index": sample["episode_index"],
             "task": sample["task"], "format": "guidedvla.libero", "parquet": str(path.resolve()),
             "revision": revision, "sha256": actual_hash}
    signals = {k: np.asarray(data[k], dtype=np.float32) for k in ("state", "joint_state", "actions")}
    if any(not np.isfinite(v).all() for v in signals.values()):
        raise ValueError("Non-finite source state/action")
    return Episode(entry, fps, count, [sample["task"]],
                   {k: data[k] for k in ("image", "wrist_image")}, signals,
                   {"single": signals["state"][:, :3]}, {"intervals": intervals})


def iter_episodes(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    root = Path(manifest["raw_root"])
    info = json.loads((root / "meta/info.json").read_text())
    hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
    for sample in manifest["samples"]:
        yield load_episode(root / sample["source_parquet"], sample, fps=float(info["fps"]),
                           revision=manifest["revision"], expected_sha256=hashes[sample["source_parquet"]])
