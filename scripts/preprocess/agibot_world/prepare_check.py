"""Select and fetch a bounded, pinned set of original episodes for human review."""

import argparse
import concurrent.futures
import hashlib
import json
import pathlib
import random
import threading
import urllib.parse
import urllib.request

REVISION = "95b159a10305a0385f369aba5d4b16242526d17b"
from scripts.preprocess import DATA_ROOT

DATASET_DIR = DATA_ROOT / "agibot_world"
RAW_ROOT = pathlib.Path("/data1/yuankai/robot-research/data/external/agibot_world_2026/raw") / REVISION[:12]
CHECK_DIR = DATASET_DIR / "check"
SEED = 20261030
SELECTION = [
    ("opensourcev4-20261001", "task_100347"),
    ("opensourcev4-20261001", "task_69701"),
    ("opensourcev4-20261001", "task_75089"),
    ("opensourcev4-20261001", "task_87261"),
    ("opensourcev4-20261001", "task_87264"),
    ("opensourcev4-20261003", "task_100448"),
    ("opensourcev4-20261003", "task_111349"),
    ("opensourcev4-20261003", "task_75540"),
    ("opensourcev4-20261003", "task_86632"),
    ("opensourcev4-20261003", "task_87265"),
]
CAMERAS = ["top_head", "hand_left", "hand_right"]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select():
    manifest_path = CHECK_DIR / "manifest.json"
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if m["revision"] != REVISION or m["seed"] != SEED:
            raise ValueError("Existing selection uses a different protocol")
        return m
    source = RAW_ROOT / "files/PhysicalReasoning/.dataset_build/export_manifest.json"
    products = json.loads(source.read_text())["products"]
    rng = random.Random(SEED)
    samples = []
    for batch, task in SELECTION:
        candidates = sorted((p for p in products if p["path"].startswith(batch + "/lerobot/" + task + "/")), key=lambda p: p["path"])
        product = candidates[0]
        row = rng.choice(sorted(product["updatedEpisodes"], key=lambda e: e["localEpisode"]))
        info_path = "PhysicalReasoning/" + product["path"]
        base = info_path.rsplit("/meta/", 1)[0]
        shard = base.split("/")[-1]
        ep = row["localEpisode"]
        episode = f"episode_{ep:06d}"
        info = json.loads((RAW_ROOT / "files" / info_path).read_text())
        task_rows = [json.loads(l) for l in (RAW_ROOT / "files" / base / "meta/tasks.jsonl").read_text().splitlines()]
        episode_rows = [json.loads(l) for l in (RAW_ROOT / "files" / base / "meta/episodes.jsonl").read_text().splitlines()]
        length = next(r["length"] for r in episode_rows if r["episode_index"] == ep)
        filename = f"physical_reasoning_{batch.split('-')[-1]}_{task}_shard{shard}_ep{ep:06d}.mp4"
        samples.append({
            "sample_number": len(samples) + 1, "batch": batch, "task": task, "shard": shard,
            "episode_index": ep, "source_identity": {k: row[k] for k in ("source", "task_id", "episode_id")},
            "task_text": task_rows[0]["task"], "expected_frames": length, "fps": info["fps"],
            "info_path": info_path, "info_sha256": digest(RAW_ROOT / "files" / info_path),
            "parquet_path": base + "/data/chunk-000/" + episode + ".parquet",
            "video_paths": {cam: base + "/videos/chunk-000/observation.images." + cam + "/" + episode + ".mp4" for cam in CAMERAS},
            "output_video": "videos/" + filename,
        })
    identities = [tuple(x["source_identity"].values()) for x in samples]
    if len(set(identities)) != 10:
        raise ValueError("Selection contains duplicate source episodes")
    requirements = DATASET_DIR.parents[1] / "docs/data.md"
    result = {
        "dataset": "agibot-world/AgiBotWorld2026", "theme": "PhysicalReasoning", "revision": REVISION,
        "created_date": "2026-10-30", "seed": SEED, "sample_count": 10,
        "selection_method": "10 explicitly selected task/batch strata, 5 per batch; first shard per stratum; uniform random choice among mapped updatedEpisodes with Python random.Random; not a full-corpus quality estimate",
        "requirements_path": "docs/data.md", "requirements_sha256": digest(requirements),
        "source_manifest_sha256": digest(source), "raw_root": str(RAW_ROOT),
        "budget": {"new_download_bytes": 5_000_000_000, "video_file_limit_bytes": 150_000_000, "compute": "CPU, no inference"},
        "render": {"width": 1280, "height": 944, "fps": "native", "playback_speed": 1, "goal_preview": "source segment end-1, provisional; object identity unchecked", "spatial_labels": "original exact-frame labels only; no interpolation"},
        "samples": samples,
    }
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def fetch_all(manifest):
    lock = threading.Lock()
    previous_bytes = manifest.get("new_download_bytes", 0)
    used = [previous_bytes]
    paths = []
    for s in manifest["samples"]:
        paths.append((s["parquet_path"], 25_000_000))
        paths.extend((p, 150_000_000) for p in s["video_paths"].values())

    def fetch(job):
        path, cap = job
        dest = RAW_ROOT / "files" / path
        url = f"https://huggingface.co/datasets/agibot-world/AgiBotWorld2026/resolve/{REVISION}/{urllib.parse.quote(path)}"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".download")
            count = 0
            try:
                with urllib.request.urlopen(url, timeout=90) as response, tmp.open("wb") as output:
                    declared = int(response.headers.get("Content-Length", 0))
                    if declared > cap:
                        raise ValueError(f"File exceeds cap: {declared} > {cap}")
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > cap:
                            raise ValueError("File exceeded per-file budget")
                        with lock:
                            used[0] += len(block)
                            if used[0] > manifest["budget"]["new_download_bytes"]:
                                raise ValueError("Total download budget exceeded")
                        output.write(block)
                    if declared and count != declared:
                        raise ValueError("Incomplete HTTP response")
                tmp.rename(dest)
            except Exception:
                tmp.unlink(missing_ok=True)
                raise
        result = {"path": path, "url": url, "bytes": dest.stat().st_size, "sha256": digest(dest)}
        sidecar = dest.with_name(dest.name + ".source.json")
        if not sidecar.exists():
            sidecar.write_text(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    artifacts, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch, j): j[0] for j in paths}
        for future in concurrent.futures.as_completed(futures):
            try:
                item = future.result()
                artifacts.append(item)
                print(json.dumps({"fetched": item["path"], "bytes": item["bytes"]}), flush=True)
            except Exception as exc:
                error = {"path": futures[future], "error": str(exc)}
                errors.append(error)
                print(json.dumps(error), flush=True)
    manifest["artifacts"] = sorted(artifacts, key=lambda x: x["path"])
    manifest["fetch_errors"] = errors
    manifest["new_download_bytes"] = used[0]
    manifest.setdefault("fetch_attempts", []).append({"new_bytes": used[0] - previous_bytes, "errors": errors})
    (CHECK_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise RuntimeError(f"{len(errors)} source files could not be fetched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    m = select()
    print(json.dumps({"sample_count": len(m["samples"]), "selection": [(s["task"], s["episode_index"], s["expected_frames"]) for s in m["samples"]]}), flush=True)
    if args.fetch:
        fetch_all(m)
