"""Render original pixels and proposed regions without modifying annotation labels."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import textwrap

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.preprocess.artifacts import artifact, check_source, digest, episodes, processed_root, read_json, write_json

BLUE, GREEN, YELLOW = (80, 160, 255), (70, 220, 120), (255, 200, 60)
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


class Regions:
    def __init__(self, root, record):
        self.root = Path(root).resolve()
        self.index = {(r["camera"], r["target_id"], r["frame_index"]): r for r in record["regions"]}
        if len(self.index) != len(record["regions"]):
            raise ValueError("Duplicate camera/target/frame regions")
        self.archives = {}

    def mask(self, region):
        if not region or not region.get("mask"):
            return None
        path = (self.root / region["mask"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Mask path escapes processed directory")
        if path not in self.archives:
            with np.load(path, allow_pickle=False) as data:
                self.archives[path] = {k: data[k] for k in ("bits", "shape", "frames")}
        data = self.archives[path]
        slot = region["slot"]
        if int(data["frames"][slot]) != region["frame_index"]:
            raise ValueError("Mask slot/source-frame mapping mismatch")
        h, w = map(int, data["shape"])
        return np.unpackbits(data["bits"][slot], count=h*w).reshape(h, w).astype(bool)


def overlay(rgb, frame, camera, targets, regions, *, goal=False):
    if rgb is None:
        return None
    pixels = rgb.copy()
    boxes = []
    for target in targets:
        region = regions.index.get((camera, target["id"], frame))
        mask = regions.mask(region)
        if mask is None:
            continue
        if mask.shape != pixels.shape[:2]:
            raise ValueError("Mask and source RGB sizes differ")
        color = YELLOW if goal else BLUE if target["role"] == "gripper" else GREEN
        pixels[mask] = (.65*pixels[mask] + .35*np.asarray(color)).astype(np.uint8)
        if region["bbox"] is not None:
            boxes.append((region["bbox"], color))
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for (x1, y1, x2, y2), color in boxes:
        draw.rectangle((x1, y1, x2-1, y2-1), outline=color, width=2)
    return image


def make_frame(episode, frame, cameras, record, regions):
    font = ImageFont.truetype(FONT_PATH, 17)
    width, height, cell = 480*max(len(cameras), len(episode.arms)), 960, 480
    canvas = Image.new("RGB", (width, height), (24, 27, 32))
    draw = ImageDraw.Draw(canvas)

    def panel(image, x, y, label):
        draw.text((x+8, y), label[:65], font=font, fill="white")
        if image is None:
            draw.text((x+10, y+110), "MISSING / UNAVAILABLE", font=font, fill=YELLOW)
        else:
            image = image.copy()
            scale = min((cell-16)/image.width, 230/image.height)
            image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.BILINEAR)
            canvas.paste(image, (x+(cell-image.width)//2, y+30))

    draw.text((10, 8), f"{episode.entry['key']}  frame {frame}/{episode.count-1}  {frame/episode.fps:.2f}s", font=font, fill="white")
    legend = ("Blue=gripper | Green=object | Yellow=candidate end | MANUAL REVIEW PENDING"
              if width > 480 else "Blue=gripper | Green=object | Yellow=end")
    draw.text((10, 34), legend, font=font, fill="white")
    if width == 480:
        draw.text((10, 315), "Candidate regions / MANUAL REVIEW PENDING", font=font, fill="white")
    if record.get("annotation_error"):
        draw.text((10, 625), "TEXT ANNOTATION FAILED: labels unavailable", font=font, fill=YELLOW)
    active = [e for e in record["events"] if e["start_frame"] <= frame < e["stop_frame"]]
    objects = {e["object_id"] for e in active if e.get("object_id")}
    targets = [t for t in record["targets"] if t["role"] == "gripper" or t["id"] in objects]
    for column, camera in enumerate(cameras):
        rgb = episode.rgb(camera, frame)
        panel(Image.fromarray(rgb) if rgb is not None else None, column*cell, 65, f"{camera}: ORIGINAL")
        missing = [t["id"] for t in targets if not regions.index.get((camera, t["id"], frame), {}).get("bbox")]
        panel(overlay(rgb, frame, camera, targets, regions), column*cell, 340, f"{camera}: PROPOSED REGIONS")
        draw.text((column*cell+8, 605), "Missing: " + (", ".join(missing) or "none"), font=font, fill=YELLOW)
    head = cameras[0]
    for column, arm in enumerate(episode.arms):
        choices = [e for e in active if e["arm"] == arm]
        target_image, label = None, f"{arm}: Event unknown"
        if len(choices) > 1:
            label = f"{arm}: overlapping source intervals; review"
        elif len(choices) == 1:
            event = choices[0]
            identity = event.get("object_id")
            obj = [t for t in record["targets"] if t["role"] == "object" and t["id"] == identity]
            region = regions.index.get((head, identity, event["goal_frame"]))
            label = f"{arm}: {event.get('skill') or 'unknown'}; end frame {event['goal_frame']}"
            if obj and region and region["bbox"] is not None:
                target_image = overlay(episode.rgb(head, event["goal_frame"]), event["goal_frame"], head,
                                       obj, regions, goal=True)
        panel(target_image, column*cell, 650, label)
    lines = textwrap.wrap("Task: " + episode.instructions[0], max(40, width//10))[:2]
    for index, line in enumerate(lines):
        draw.text((10, 907+index*23), line, font=font, fill="white")
    return np.asarray(canvas)


def render_episode(episode, record, root, destination, cameras):
    check_source(episode, record)
    if not cameras or any(c not in episode.cameras for c in cameras):
        raise ValueError("Need valid source cameras")
    if "libero" in episode.entry.get("format", ""):
        from scripts.preprocess.libero.render import render_video
        from scripts.preprocess.libero.view import from_episode
        source_fps = None if episode.evidence.get("fps_meaning") == "display_only" else episode.fps
        view = from_episode(episode, record, Regions(root, record), source_fps=source_fps)
        report = render_video(view, destination)
        report.update(video=str(Path(destination).relative_to(root)), sha256=digest(destination), fps=episode.fps)
        return report
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".partial.mp4")
    regions = Regions(root, record)
    try:
        with av.open(str(temporary), "w") as container:
            stream = container.add_stream("libx264", rate=Fraction(str(episode.fps)))
            stream.width, stream.height, stream.pix_fmt = 480*max(len(cameras), len(episode.arms)), 960, "yuv420p"
            stream.options = {"crf": "20", "preset": "fast"}
            for frame in range(episode.count):
                image = make_frame(episode, frame, cameras, record, regions)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(temporary)) as container:
            decoded = sum(1 for _ in container.decode(video=0))
        if decoded != episode.count:
            raise ValueError("Rendered video frame count differs from source")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"key": episode.entry["key"], "video": str(path.relative_to(root)),
            "frames": episode.count, "fps": episode.fps, "sha256": digest(path), "manual_review": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = (args.output or processed_root(args.dataset)).resolve()
    manifest = read_json(root / "manifest.json")
    if manifest["source_manifest_sha256"] != digest(args.manifest) or manifest["dataset"] != args.dataset:
        raise ValueError("Processed manifest belongs to different input samples")
    rows = []
    for episode in episodes(args.dataset, args.manifest):
        record = read_json(artifact(root, "regions", episode.entry["key"]))
        if record["prompts_sha256"] != digest(artifact(root, "prompts", episode.entry["key"])):
            raise ValueError("Prompts changed after SAM tracking")
        name = episode.entry["key"].replace("/", "__") + "_processed.mp4"
        rows.append(render_episode(episode, record, root, root / "video" / name,
                                   manifest["config"]["cameras"][args.dataset]))
    write_json(root / "report.json", {"samples": rows, "manual_review": "pending"})
    with (root / "manual_review.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["key", "video", "gripper", "object", "identity", "event_end", "decision", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"key": row["key"], "video": row["video"], "decision": "pending"})


if __name__ == "__main__":
    main()
