"""Sample, download and render unchanged GuidedVLA LIBERO source annotations."""

import argparse
import concurrent.futures
import csv
import hashlib
import html
from io import BytesIO
import json
from pathlib import Path
import random
import time

import av
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import requests

from scripts.preprocess.libero.render import Episode as ReviewEpisode, Event as ReviewEvent, render_video, WIDTH, HEIGHT


from scripts.preprocess import DATA_ROOT

DATASET = DATA_ROOT / "libero/guidedvla"
PROJECT = DATASET.parents[2]
OUT = DATASET / "check/raw"
RAW = Path("/data1/yuankai/robot-research/data/external/guidedvla_libero/raw/477f79595e4b")
REVISION = "477f79595e4bc55829706fc655419523ae1da3b9"
FPS = 10


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def manifest():
    return json.loads((OUT / "manifest.json").read_text())


def prepare():
    if (OUT / "manifest.json").exists():
        raise FileExistsError("Manifest already exists; use it without resampling")
    episodes = pq.read_table(RAW / "meta/episodes/chunk-000/file-000.parquet", columns=[
        "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
        "dataset_from_index", "dataset_to_index",
    ]).to_pylist()
    episodes.sort(key=lambda e: e["episode_index"])
    tasks = sorted({task for e in episodes for task in e["tasks"]})
    rng = random.Random(42)
    selected = rng.sample(tasks, 10)
    siblings = {s["rfilename"]: s for s in json.loads((RAW / "repository.json").read_text())["siblings"]}
    samples, files = [], {}
    for n, task in enumerate(selected, 1):
        e = rng.choice([e for e in episodes if task in e["tasks"]])
        path = f"data/chunk-{e['data/chunk_index']:03d}/file-{e['data/file_index']:03d}.parquet"
        source = siblings[path]
        files[path] = {"path": path, "bytes": source["size"], "sha256": source["lfs"]["sha256"]}
        samples.append({
            "number": n, "episode_index": e["episode_index"], "task": task,
            "expected_frames": e["length"], "source_parquet": path,
            "dataset_from_index": e["dataset_from_index"], "dataset_to_index": e["dataset_to_index"],
            "output_video": f"video/{n:02d}_episode_{e['episode_index']:06d}_raw.mp4",
        })
    total = sum(f["bytes"] for f in files.values())
    if total > 1.2 * 1024 ** 3:
        raise ValueError(f"Download budget exceeded: {total}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "video").mkdir(exist_ok=True)
    save_json(OUT / "manifest.json", {
        "dataset": "ybwowen/libero", "revision": REVISION, "raw_root": str(RAW),
        "raw_meaning": "Unchanged GuidedVLA release, including author skill and object-mask annotations",
        "requirements_sha256": sha256(PROJECT / "docs/data.md"),
        "metadata_sha256": {p: sha256(RAW / p) for p in ["meta/info.json", "meta/episodes/chunk-000/file-000.parquet"]},
        "sampling": {"seed": 42, "method": "Ten sorted task texts sampled without replacement, then one episode per task using the same Python random.Random stream", "population_tasks": len(tasks), "population_episodes": len(episodes)},
        "render": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "speed": "one source frame per output frame", "new_annotations": False, "stage_endpoint": "last frame of current contiguous skill run, NOT confirmed semantic Event endpoint"},
        "download_bytes": total, "files": list(files.values()), "samples": samples,
        "manual_review": "pending",
    })
    print(json.dumps({"samples": len(samples), "files": len(files), "download_bytes": total,
                      "source_frames": sum(s["expected_frames"] for s in samples)}, ensure_ascii=False), flush=True)


def download_one(source):
    target = RAW / source["path"]
    if target.exists() and target.stat().st_size == source["bytes"] and sha256(target) == source["sha256"]:
        print("cached " + source["path"], flush=True)
        return
    url = f"https://huggingface.co/datasets/ybwowen/libero/resolve/{REVISION}/{source['path']}?download=true"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    for attempt in range(3):
        try:
            digest, size = hashlib.sha256(), 0
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as f:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
            if size != source["bytes"] or digest.hexdigest() != source["sha256"]:
                raise ValueError(f"Official LFS size/hash mismatch: {source['path']}")
            partial.replace(target)
            save_json(target.with_suffix(".parquet.source.json"), {"url": url, "bytes": size, "sha256": digest.hexdigest()})
            print("downloaded " + source["path"], flush=True)
            return
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download_one, manifest()["files"]))








def decode(value):
    with Image.open(BytesIO(value["bytes"])) as image:
        image.load()
        if image.size != (224, 224) or image.mode != "RGB":
            raise ValueError(f"Unexpected source image: {image.size} {image.mode}")
        return image.copy()






def read_episode(sample):
    source = RAW / sample["source_parquet"]
    table = pq.read_table(source, filters=[("episode_index", "=", sample["episode_index"])])
    table = table.sort_by([("frame_index", "ascending")])
    data = table.to_pydict()
    count = len(data["frame_index"])
    if count != sample["expected_frames"] or not np.array_equal(data["frame_index"], np.arange(count)):
        raise ValueError(f"Incomplete/duplicated source episode: {sample['episode_index']}")
    if not np.allclose(data["timestamp"], np.arange(count) / FPS, atol=1e-5):
        raise ValueError("Source timestamp grid does not match 10 fps")
    if not np.array_equal(data["index"], np.arange(sample["dataset_from_index"], sample["dataset_to_index"])):
        raise ValueError("Episode global index mapping does not match metadata")
    task_table = pq.read_table(RAW / "meta/tasks.parquet").to_pylist()
    task_mapping = {row["task_index"]: row["__index_level_0__"] for row in task_table}
    if {task_mapping[t] for t in data["task_index"]} != {sample["task"]}:
        raise ValueError("Task text mapping does not match fixed sample")
    for key, dim in [("state", 8), ("joint_state", 7), ("actions", 7)]:
        data[key] = np.asarray(data[key], dtype=np.float32)
        if data[key].shape != (count, dim) or not np.isfinite(data[key]).all():
            raise ValueError(f"Invalid {key} array")
    for key in ["agentview_attention_object_mask", "wrist_attention_object_mask"]:
        data[key] = np.asarray(data[key], dtype=np.float32)
        if data[key].shape != (count, 16, 16) or not np.isin(data[key], [0, 1]).all():
            raise ValueError(f"Invalid patch mask {key}")
    skills = np.asarray(data["observation.skill_id"]).reshape(-1)
    starts = np.r_[0, np.flatnonzero(skills[1:] != skills[:-1]) + 1]
    ends = np.r_[starts[1:], count]
    runs = [{"start_inclusive": int(a), "end_exclusive": int(b), "skill_id": int(skills[a]),
             "skill_text": data["observation.skill_text"][int(a)]} for a, b in zip(starts, ends)]
    return data, runs


def render_one(sample):
    data, runs = read_episode(sample)
    count = len(data["frame_index"])
    head = [decode(v) for v in data["image"]]
    wrist = [decode(v) for v in data["wrist_image"]]
    head_masks, wrist_masks = data["agentview_attention_object_mask"], data["wrist_attention_object_mask"]
    target = OUT / sample["output_video"]
    if target.exists() and sample.get("resume_existing", False):
        with av.open(str(target)) as existing:
            stream = existing.streams.video[0]
            if stream.frames != count or stream.width != manifest()['render']['width'] or stream.height != manifest()['render']['height'] or stream.average_rate != FPS:
                raise ValueError(f"Existing video metadata differs from protocol: {target}")
        print(f"resumed episode {sample['episode_index']}; full video decode still required", flush=True)
        return {
            **sample, "rendered_frames": count, "fps": FPS, "video_bytes": target.stat().st_size,
            "video_sha256": sha256(target), "source_rgb_frames_decoded": {"head": len(head), "wrist": len(wrist)},
            "source_numeric_finite": True, "source_frame_index_contiguous": True, "source_timestamp_matches_fps": True,
            "empty_mask_frames": {"head": int((head_masks.sum(axis=(1, 2)) == 0).sum()), "wrist": int((wrist_masks.sum(axis=(1, 2)) == 0).sum())},
            "skill_runs_derived_from_source_ids": runs, "new_model_annotations": False, "manual_review": "pending",
        }
    view = ReviewEpisode(
        key=f"episode_{sample['episode_index']:06d}", source='GuidedVLA / SOURCE PHASES',
        frame_count=count, rgb=lambda frame: np.asarray(head[frame]), task=sample['task'],
        events=[ReviewEvent(f"stage_{i:04d}", r['start_inclusive'], r['end_exclusive'],
                            r['end_exclusive']-1, r['skill_text']) for i, r in enumerate(runs)],
        gripper_state=data['state'][:, 6:], actions=data['actions'], eef_xyz=data['state'][:, :3],
        xyz_note='state[:3] / coordinate and unit pending', source_fps=FPS, playback_fps=FPS,
        annotation_label='SOURCE PATCH MASK ROLE UNVERIFIED', note='SOURCE PHASES / EVENT REVIEW PENDING')
    render_video(view, target)
    result = {
        **sample, "rendered_frames": count, "fps": FPS, "video_bytes": target.stat().st_size,
        "video_sha256": sha256(target), "source_rgb_frames_decoded": {"head": len(head), "wrist": len(wrist)},
        "source_numeric_finite": True, "source_frame_index_contiguous": True, "source_timestamp_matches_fps": True,
        "empty_mask_frames": {"head": int((head_masks.sum(axis=(1, 2)) == 0).sum()), "wrist": int((wrist_masks.sum(axis=(1, 2)) == 0).sum())},
        "skill_runs_derived_from_source_ids": runs, "new_model_annotations": False, "manual_review": "pending",
    }
    print(json.dumps({"episode": sample["episode_index"], "frames": count, "video": str(target)}, ensure_ascii=False), flush=True)
    return result


def render(resume=False):
    m = manifest()
    for f in m["files"]:
        p = RAW / f["path"]
        if not p.exists() or sha256(p) != f["sha256"]:
            raise ValueError(f"Source fingerprint mismatch: {p}")
    samples = [dict(s, resume_existing=resume) for s in m["samples"]]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(render_one, samples))
    save_json(OUT / "render_summary.json", {"revision": REVISION, "samples": results,
                                            "renderer_sha256": sha256(Path(__file__)),
                                            "font": "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                                            "pyav_version": av.__version__, "numpy_version": np.__version__,
                                            "total_frames": sum(r["rendered_frames"] for r in results), "manual_review": "pending"})
    with (OUT / "manual_review.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample", "episode_index", "task", "video", "images", "time_alignment", "skill_boundaries", "object_region", "endpoint_same_object", "state_motion_action", "issues_source_frame", "reviewer", "conclusion"])
        writer.writeheader()
        for s in m["samples"]:
            writer.writerow({"sample": s["number"], "episode_index": s["episode_index"], "task": s["task"], "video": s["output_video"], "conclusion": "待人工核验"})
    cards = []
    for s in m["samples"]:
        video, task = html.escape(s["output_video"]), html.escape(s["task"])
        cards.append(f'<article><h2>{s["number"]:02d} · episode {s["episode_index"]}</h2><p>{task}</p><video controls preload="metadata" src="{video}"></video><p>待人工核验 · {s["expected_frames"]} 帧 · 10 fps</p></article>')
    (OUT / "index.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LIBERO 原始数据核验</title><style>body{max-width:1440px;margin:24px auto;background:#14181e;color:#e9eef4;font-family:sans-serif;padding:0 20px}article{margin:36px 0}video{width:100%}p{line-height:1.6}</style><h1>GuidedVLA LIBERO：10 条原始数据</h1><p>绿色块为作者 16×16 对象区域；技能段末帧不是已核验的 Event 目标。未补标，人工结论待填写。</p>' + ''.join(cards) + '</html>')


def verify():
    m = manifest()
    rendered = json.loads((OUT / "render_summary.json").read_text())
    hashes = {r["episode_index"]: r["video_sha256"] for r in rendered["samples"]}
    results = []
    for s in m["samples"]:
        path = OUT / s["output_video"]
        count, timestamps, shapes = 0, [], set()
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            codec, rate = stream.codec_context.name, str(stream.average_rate)
            for frame in container.decode(stream):
                frame.to_ndarray(format="rgb24")
                timestamps.append(float(frame.pts * frame.time_base))
                shapes.add((frame.width, frame.height))
                count += 1
        expected = s["expected_frames"]
        digest = sha256(path)
        passed = count == expected and shapes == {(m['render']['width'], m['render']['height'])} and codec == "h264" and rate == "10" and np.allclose(timestamps, np.arange(expected) / FPS, atol=1e-5) and digest == hashes[s["episode_index"]]
        results.append({"episode_index": s["episode_index"], "video": s["output_video"], "frames": count, "expected_frames": expected,
                        "codec": codec, "fps": rate, "shapes": [list(v) for v in shapes], "full_decode_and_timestamps_pass": bool(passed),
                        "duration_seconds": count / FPS, "video_sha256": digest, "manual_review": "pending"})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    total_bytes = sum((OUT / s["output_video"]).stat().st_size for s in m["samples"])
    save_json(OUT / "verification.json", {"all_passed": all(r["full_decode_and_timestamps_pass"] for r in results),
                                          "samples": results, "total_video_bytes": total_bytes, "manual_review": "pending"})
    if not all(r["full_decode_and_timestamps_pass"] for r in results):
        raise RuntimeError("One or more source replay videos failed verification")
    if total_bytes > 300 * 1024 ** 2:
        raise RuntimeError("Video storage budget exceeded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "download", "render", "verify"])
    parser.add_argument("--resume", action="store_true", help="Reuse complete local videos after source and stream checks; full verify remains required")
    args = parser.parse_args()
    {"prepare": prepare, "download": download, "render": lambda: render(resume=args.resume), "verify": verify}[args.command]()
