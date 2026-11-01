"""Render source-only review videos, preserving frame/time and missing labels."""

import argparse
import bisect
import concurrent.futures
import csv
import hashlib
import json
import math
import pathlib
import platform
from fractions import Fraction

import av
import numpy as np
import pyarrow
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

from scripts.preprocess import DATA_ROOT

DATASET_DIR = DATA_ROOT / "agibot_world"
CHECK_DIR = DATASET_DIR / "check"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT = ImageFont.truetype(FONT_PATH, 17)
SMALL = ImageFont.truetype(FONT_PATH, 14)
TITLE = ImageFont.truetype(FONT_PATH, 20)
BG = (19, 23, 29)
WHITE = (228, 235, 242)
MUTED = (168, 180, 193)
YELLOW = (255, 210, 93)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def wrap(draw, text, width, font=FONT):
    lines, current = [], ""
    for word in str(text).split():
        candidate = (current + " " + word).strip()
        if draw.textlength(candidate, font=font) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    return lines + ([current] if current else [])


def txt(draw, xy, text, width, color=WHITE, font=FONT, max_lines=3):
    lines = wrap(draw, text, width, font)
    for line_no, line in enumerate(lines[:max_lines]):
        if line_no == max_lines - 1 and len(lines) > max_lines:
            line = line.rstrip() + " ..."
        draw.text((xy[0], xy[1] + line_no * 22), line, font=font, fill=color)


def hints(instruction):
    t = instruction.lower()
    left = "left arm" in t or "left hand" in t
    right = "right arm" in t or "right hand" in t
    if "both arms" in t or "both hands" in t or (left and right):
        return "BOTH (text hint; per-arm bounds unknown)"
    if left:
        return "LEFT (text hint)"
    if right:
        return "RIGHT (text hint)"
    if "chassis" in t or "waist" in t or "robot base" in t:
        return "BODY (text hint)"
    return "UNKNOWN (not inferred as idle/hold)"


def open_video(path):
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.codec_context.thread_count = 2
    return container, stream


def marked(image, annotation, heading):
    im = image.copy()
    draw = ImageDraw.Draw(im)
    width, height = im.size
    modalities = annotation.get("modalities", {}) if annotation else {}
    bbox = modalities.get("bbox", {})
    point = modalities.get("target_point", {})
    trace = modalities.get("trace_2d_gt", {})
    if trace.get("found") is True:
        points = trace.get("trace_xy_norm", [])
        points = [(round(p[0] * width), round(p[1] * height)) for p in points if len(p) == 2 and all(math.isfinite(v) for v in p)]
        if len(points) > 1:
            draw.line(points, fill=(51, 220, 244), width=3)
        for x, y in points:
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(51, 220, 244))
    if bbox.get("found") is True:
        b = bbox.get("bbox_xyxy_norm", [])
        if len(b) == 4 and all(math.isfinite(x) for x in b) and b[0] < b[2] and b[1] < b[3]:
            draw.rectangle((b[0] * width, b[1] * height, b[2] * width, b[3] * height), outline=(93, 255, 133), width=3)
    if point.get("found") is True:
        p = point.get("point_xy_norm", [])
        if len(p) == 2 and all(math.isfinite(x) for x in p):
            x, y = p[0] * width, p[1] * height
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(255, 210, 93), width=3)
    return im


def annotation_status(annotation):
    modalities = annotation.get("modalities", {}) if annotation else {}
    bbox = modalities.get("bbox", {})
    point = modalities.get("target_point", {})
    status = "bbox: " + ("source" if bbox.get("found") is True else "NO LABEL / not found")
    return status, str(bbox.get("target") or point.get("target") or "target unspecified")


class EndpointReader:
    def __init__(self, path, fps, total_frames):
        self.container, self.stream = open_video(path)
        self.fps = fps
        self.total_frames = total_frames
        self.cache = {}

    def read(self, index):
        if not 0 <= index < self.total_frames:
            return None
        if index in self.cache:
            return self.cache[index]
        origin = self.stream.start_time or 0
        pts = origin + round(Fraction(index, 1) / self.fps / self.stream.time_base)
        self.container.seek(pts, stream=self.stream, backward=True, any_frame=False)
        for frame in self.container.decode(self.stream):
            actual = round((frame.pts - origin) * self.stream.time_base * self.fps)
            if actual == index:
                im = frame.to_image().resize((640, 400), Image.Resampling.BILINEAR)
                self.cache[index] = im
                return im
            if actual > index:
                break
        return None

    def close(self):
        self.container.close()


def vector(state, schema, key):
    field = schema.get(key)
    return state[:, field["indices"]] if field else None


def values(array, index, precision=3):
    if array is None:
        return "MISSING"
    return " ".join(f"{float(x):.{precision}f}" for x in array[index])


def render_one(sample, manifest):
    raw = pathlib.Path(manifest["raw_root"]) / "files"
    info_file = raw / sample["info_path"]
    if sha256(info_file) != sample["info_sha256"]:
        raise ValueError("Metadata changed after sample selection")
    info = json.loads(info_file.read_text())
    table = pq.read_table(raw / sample["parquet_path"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    timestamps = np.asarray(table["timestamp"].to_pylist())
    frame_indices = np.asarray(table["frame_index"].to_pylist())
    count = len(states)
    schema = info["features"]["observation.state"]["field_descriptions"]
    effector = vector(states, schema, "state/end/arm_position")
    lgrip = vector(states, schema, "state/left_effector/position")
    rgrip = vector(states, schema, "state/right_effector/position")
    joints = vector(states, schema, "state/joint/position")
    eid = str(sample["episode_index"])
    segments = info.get("instruction_segments", {}).get(eid, [])
    starts = [s["start_frame_index"] for s in segments]
    if starts != sorted(starts):
        raise ValueError("Source segments are not ordered; explicit resolution is needed")
    affordance = info.get("affordance_annotations", {}).get("episodes", {}).get(eid, {})
    annotations = {s["seg_idx"]: {f["frame_index"]: f for f in s.get("frames", [])} for s in affordance.get("segments", [])}
    fps = Fraction(str(sample["fps"]))
    cameras = {name: open_video(raw / p) for name, p in sample["video_paths"].items()}
    streams = {name: container.decode(stream) for name, (container, stream) in cameras.items()}
    endpoint = EndpointReader(raw / sample["video_paths"]["top_head"], fps, count)
    output_path = CHECK_DIR / sample["output_video"]
    output_path.parent.mkdir(exist_ok=True)
    tmp = output_path.with_suffix(".tmp")
    frames_written = 0
    wrist_missing = {"hand_left": 0, "hand_right": 0}
    max_grid_error = {name: 0.0 for name in cameras}
    source_counts = {name: 0 for name in cameras}
    goal_missing = set()
    annotated_frames = 0
    native_bbox_frames = 0
    result = {"sample_number": sample["sample_number"], "output_video": sample["output_video"], "expected_frames": sample["expected_frames"], "parquet_rows": count, "state_shape": list(states.shape), "action_shape": list(actions.shape), "state_action_finite": bool(np.isfinite(states).all() and np.isfinite(actions).all()), "frame_index_contiguous": bool(np.array_equal(frame_indices, np.arange(count))), "episode_index_matches": set(table["episode_index"].to_pylist()) == {sample["episode_index"]}, "parquet_timestamp_grid_max_error_s": float(np.max(np.abs(timestamps - np.arange(count) / float(fps)))), "manual_review": "pending"}
    timeline = Image.new("RGB", (1280, 88), BG)
    td = ImageDraw.Draw(timeline)
    td.text((12, 2), "SOURCE SHARED TIMELINE (not separate arm events)", font=SMALL, fill=MUTED)
    for j, s in enumerate(segments):
        x1 = 12 + int(s["start_frame_index"] / max(count, 1) * 1255)
        x2 = 12 + int(s["end_frame_index"] / max(count, 1) * 1255)
        td.rectangle((x1, 28, max(x1 + 1, x2), 47), fill=(56 + j * 29 % 130, 85 + j * 41 % 120, 110 + j * 17 % 100))
    current_goal_key = None
    goal_image = None
    try:
        with av.open(str(tmp), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            out_stream = out.add_stream("libx264", rate=fps)
            out_stream.width, out_stream.height = 1280, 944
            out_stream.pix_fmt = "yuv420p"
            out_stream.codec_context.thread_count = 3
            out_stream.options = {"preset": "ultrafast", "crf": "25"}
            for index, head in enumerate(streams["top_head"]):
                source_counts["top_head"] += 1
                if index >= count:
                    continue
                canvas = Image.new("RGB", (1280, 944), BG)
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 8), f"PhysicalReasoning | {sample['task']} | {sample['batch']} / {sample['shard']} | episode {sample['episode_index']}", font=TITLE, fill=WHITE)
                txt(draw, (12, 38), sample["task_text"], 1240, font=FONT, max_lines=1)
                sindex = bisect.bisect_right(starts, index) - 1
                segment = segments[sindex] if sindex >= 0 and index < segments[sindex]["end_frame_index"] else None
                ann = annotations.get(sindex, {}).get(index) if segment else None
                annotated_frames += bool(ann)
                native_bbox_frames += bool(ann and ann.get("modalities", {}).get("bbox", {}).get("found"))
                head_image = head.to_image().resize((640, 400), Image.Resampling.BILINEAR)
                draw.text((8, 62), f"CURRENT HEAD | frame {index} | t={float(timestamps[index]):.3f}s", font=SMALL, fill=WHITE)
                canvas.paste(marked(head_image, ann, f"CURRENT HEAD | frame {index} | t={float(timestamps[index]):.3f}s"), (0, 80))
                status, target_name = annotation_status(ann)
                draw.text((8, 483), status, font=SMALL, fill=YELLOW)
                draw.text((8, 505), target_name[:70], font=SMALL, fill=MUTED)
                if head.pts is not None:
                    origin = cameras["top_head"][1].start_time or 0
                    max_grid_error["top_head"] = max(max_grid_error["top_head"], abs(float((head.pts - origin) * head.time_base) - index / float(fps)))
                if segment:
                    target = int(segment["end_frame_index"]) - 1
                    if current_goal_key != (sindex, target):
                        current_goal_key = (sindex, target)
                        snapshot = endpoint.read(target)
                        if snapshot is None:
                            goal_missing.add(target)
                            goal_image = Image.new("RGB", (640, 400), BG)
                            ImageDraw.Draw(goal_image).text((20, 140), "END PREVIEW MISSING / INVALID", font=TITLE, fill=YELLOW)
                        else:
                            goal_image = marked(snapshot, annotations.get(sindex, {}).get(target), f"STEP END-1 PREVIEW | frame {target} | object identity UNCHECKED")
                    canvas.paste(goal_image, (640, 80))
                    draw.text((648, 62), f"STEP END-1 | frame {target} | object identity UNCHECKED", font=SMALL, fill=WHITE)
                    status, target_name = annotation_status(annotations.get(sindex, {}).get(target))
                    draw.text((648, 483), status, font=SMALL, fill=YELLOW)
                    draw.text((648, 505), target_name[:70], font=SMALL, fill=MUTED)
                else:
                    draw.text((660, 220), "No source step at this frame", font=TITLE, fill=MUTED)
                for cam, x in [("hand_left", 0), ("hand_right", 320)]:
                    try:
                        frame = next(streams[cam])
                        source_counts[cam] += 1
                        canvas.paste(frame.to_image().resize((320, 264), Image.Resampling.BILINEAR), (x, 570))
                        if frame.pts is not None:
                            origin = cameras[cam][1].start_time or 0
                            max_grid_error[cam] = max(max_grid_error[cam], abs(float((frame.pts - origin) * frame.time_base) - index / float(fps)))
                    except StopIteration:
                        wrist_missing[cam] += 1
                        draw.text((x + 8, 600), "MISSING CAMERA FRAME", font=SMALL, fill=YELLOW)
                    draw.text((x + 8, 550), cam.upper() + " | source current frame", font=SMALL, fill=WHITE)
                px = 655
                if segment:
                    draw.text((px, 550), f"STEP {sindex:03d} [{segment['start_frame_index']}, {segment['end_frame_index']}) | {segment.get('skill','')}", font=FONT, fill=WHITE)
                    txt(draw, (px, 579), segment.get("instruction", ""), 610, max_lines=3)
                    txt(draw, (px, 648), "Arm: " + hints(segment.get("instruction", "")), 610, color=YELLOW, max_lines=1)
                draw.text((px, 678), "Grippers L / R (native values; closed != hold): " + values(lgrip, index) + " / " + values(rgrip, index), font=SMALL, fill=WHITE)
                draw.text((px, 703), "EEF native coordinates; units/frame NOT VERIFIED", font=SMALL, fill=YELLOW)
                if effector is not None:
                    draw.text((px, 727), "group 0: " + "  ".join(f"{x:.4f}" for x in effector[index, :3]), font=FONT, fill=WHITE)
                    draw.text((px, 752), "group 1: " + "  ".join(f"{x:.4f}" for x in effector[index, 3:]), font=FONT, fill=WHITE)
                txt(draw, (px, 783), "Joints: " + values(joints, index, 2), 610, font=SMALL, max_lines=1)
                draw.text((12, 527), "green: source bbox | yellow: interaction point | cyan: source future trace | no propagated boxes", font=SMALL, fill=MUTED)
                canvas.paste(timeline, (0, 850))
                draw = ImageDraw.Draw(canvas)
                xpos = 12 + int(index / max(count, 1) * 1255)
                draw.line((xpos, 871, xpos, 906), fill=(255, 255, 255), width=3)
                draw.text((12, 914), f"frame {index}/{count-1} | {index/float(fps):.2f}s / {count/float(fps):.2f}s | 1x native timing | HUMAN REVIEW PENDING", font=SMALL, fill=WHITE)
                frame = av.VideoFrame.from_image(canvas)
                frame.pts = index
                frame.time_base = 1 / fps
                for packet in out_stream.encode(frame):
                    out.mux(packet)
                frames_written += 1
                if index and index % 1500 == 0:
                    print(json.dumps({"sample": sample["sample_number"], "rendered": index, "total": count}), flush=True)
            for packet in out_stream.encode():
                out.mux(packet)
        for cam in ("hand_left", "hand_right"):
            for _ in streams[cam]:
                source_counts[cam] += 1
        tmp.replace(output_path)
    finally:
        endpoint.close()
        for container, stream in cameras.values():
            container.close()
    result.update(rendered_frames=frames_written, source_video_decoded_frames=source_counts, missing_wrist_frames=wrist_missing, source_pts_grid_max_error_s=max_grid_error, endpoint_frames_unavailable=sorted(goal_missing), source_annotated_frames_shown=annotated_frames, source_bbox_frames_shown=native_bbox_frames, output_bytes=output_path.stat().st_size, output_sha256=sha256(output_path))
    result["technical_pass"] = bool(frames_written == count == sample["expected_frames"] and all(v == count for v in source_counts.values()) and result["state_action_finite"] and result["frame_index_contiguous"] and result["episode_index_matches"] and not goal_missing and all(v < 1 / float(fps) for v in max_grid_error.values()))
    return result


def update_review(manifest):
    path = CHECK_DIR / "review.csv"
    if path.exists():
        return
    with path.open("w", newline="") as f:
        columns = ["sample_number", "task", "batch", "episode_index", "source_episode_id", "video", "video_normal", "time_alignment", "event_skill", "arm_and_concurrency", "hold", "current_object_bbox", "endpoint_object", "motion_action_semantics", "suitability", "reviewer", "notes"]
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for s in manifest["samples"]:
            row = {k: "待人工核验" for k in columns}
            row.update(sample_number=s["sample_number"], task=s["task"], batch=s["batch"], episode_index=s["episode_index"], source_episode_id=s["source_identity"]["episode_id"], video=s["output_video"], reviewer="", notes="")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", nargs="*", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((CHECK_DIR / "manifest.json").read_text())
    samples = [s for s in manifest["samples"] if not args.samples or s["sample_number"] in args.samples]
    update_review(manifest)
    report_path = CHECK_DIR / "technical_check.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"scope": "source file structure/time-grid/readability only; not semantic validation", "environment": {"python": platform.python_version(), "av": av.__version__, "numpy": np.__version__, "pyarrow": pyarrow.__version__}, "render_script_sha256": sha256(pathlib.Path(__file__)), "results": [], "errors": []}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_one, s, manifest): s for s in samples}
        for future in concurrent.futures.as_completed(futures):
            sample = futures[future]
            try:
                result = future.result()
                report["results"] = [r for r in report["results"] if r["sample_number"] != sample["sample_number"]] + [result]
                report["errors"] = [e for e in report["errors"] if e["sample_number"] != sample["sample_number"]]
                print(json.dumps({"completed": sample["sample_number"], "technical_pass": result["technical_pass"], "bytes": result["output_bytes"]}), flush=True)
            except Exception as exc:
                report["errors"].append({"sample_number": sample["sample_number"], "error": str(exc)})
                print(json.dumps(report["errors"][-1]), flush=True)
            report["results"].sort(key=lambda r: r["sample_number"])
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if report["errors"] or not all(r["technical_pass"] for r in report["results"]):
        raise SystemExit("One or more samples need technical investigation")


if __name__ == "__main__":
    main()
