"""Inventory Event targets, extract a review cohort, then render saved scene grounding.

python -m scripts.preprocess.libero.interaction_review --config <json> --stage inventory|extract|render
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import textwrap

import av
import numpy as np
from PIL import Image, ImageDraw

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.render import font
from .atomic_vla.interaction_targets import bind_target, entity_id, scene_entities


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def inventory(config):
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    index = Path(config["index"])
    samples = {s["key"]: s for s in json.loads(Path(config["sources"]).read_text())["samples"]}
    scenes, episodes = {}, []
    for entry in json.loads(index.read_text())["episodes"]:
        path = index.parent / entry["path"]
        meta = json.loads(path.read_text())
        sample = samples[meta["episode_id"]]
        bddl = sample["bddl"]
        if bddl not in scenes:
            scenes[bddl] = robosuite_parse_problem(bddl)
        scene = scenes[bddl]
        events = [{**{k: e[k] for k in ("event_id", "skill", "description", "start_frame", "stop_frame")},
                   "manipulated_instance": e["source_annotation"]["instance"],
                   "interaction_target": bind_target(e, scene)} for e in meta["events"]]
        reader = EpisodeReader(path, metadata=meta)
        available = {}
        for event in meta["events"]:
            source = event["source_annotation"]
            key = entity_id(source["instance"], source["scope"]["part"])
            region = event["object_regions"].get("image")
            if region and region.get("masks_valid"):
                frames = reader.array(region["frame_indices"])
                valid = reader.array(region["masks_valid"])
                available.setdefault(key, set()).update(frames[valid].tolist())
        for event in events:
            target = event["interaction_target"]
            needed = set(range(event["start_frame"], event["stop_frame"]))
            count = len(needed & available.get(target["entity_id"], set()))
            target["source_grounding_frames"] = count
            target["grounding_status"] = ("source_complete" if count == len(needed) else
                                          "source_partial" if count else "needs_render")
        episodes.append(dict(episode_id=meta["episode_id"], task=meta["instructions"][0], frames=meta["frame_count"],
            source_episode=str(path), source_episode_sha256=digest(path), bddl=bddl, bddl_sha256=digest(bddl),
            source_regions=str(Path(config["source_regions"]) / (meta["episode_id"]+".json")), events=events))
    # Explicit semantic coverage, not cherry-picking trajectories by visual quality.
    selectors = config["review_tasks"]
    selected = []
    for query in selectors:
        match = next(e for e in episodes if query in e["task"] and e["episode_id"] not in selected)
        selected.append(match["episode_id"])
    skills = Counter(e["skill"] for row in episodes for e in row["events"])
    statuses = Counter(e["interaction_target"]["status"] for row in episodes for e in row["events"])
    kinds = Counter(scene_entities(scenes[row["bddl"]])[e["interaction_target"]["entity_id"]]["kind"]
                    for row in episodes for e in row["events"] if e["interaction_target"]["status"] == "resolved")
    grounding = Counter(e["interaction_target"]["grounding_status"] for row in episodes for e in row["events"])
    result = dict(episodes=episodes, scenes=scenes, selected=selected, summary=dict(episodes=len(episodes),
        tasks=len(scenes), events=sum(skills.values()), skills=dict(skills), status=dict(statuses), target_kinds=dict(kinds),
        source_grounding=dict(grounding)),
        provenance=dict(index=str(index), index_sha256=digest(index), config=config,
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                        git_status=subprocess.check_output(["git", "status", "--short"], text=True)))
    out = Path(config["output"])
    write_json(out / "inventory.json", result)
    with (out / "events.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["episode_id", "event_id", "skill", "description", "manipulated", "target", "status", "grounding", "reason"])
        for row in episodes:
            for e in row["events"]:
                t = e["interaction_target"]
                writer.writerow([row["episode_id"], e["event_id"], e["skill"], e["description"],
                                 e["manipulated_instance"], t["entity_id"], t["status"], t["grounding_status"], t.get("reason", "")])
    print(json.dumps(result["summary"]), flush=True)


RED, GREEN, WHITE, BG = "#ff4d4d", "#52d98e", "#e9f0f5", "#10181f"
WIDTH, HEIGHT = 1920, 1280


class SceneRenderer:
    """Saved-label reader only: left source RGB, right target red / other instances green."""
    def __init__(self, path, number):
        self.reader, self.number = EpisodeReader(path), number
        self.meta = self.reader.meta
        self.entities = self.meta["scene_entities"]
        self.visible = [k for k, v in self.entities.items() if v["kind"] in {"object", "fixture"}]
        self.numbers = {key: i+1 for i, key in enumerate(self.entities)}

    def frame(self, frame):
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        event = self.reader.event_at("single", frame)
        target = event["interaction_target"]["entity_id"] if event else None
        title = self.meta["instructions"][0]
        draw.text((32, 16), f'{self.number:02d} | {self.meta["episode_id"]} | FRAME {frame}/{self.meta["frame_count"]-1}', font=font(22), fill=WHITE)
        for i, line in enumerate(textwrap.wrap(title, 100)):
            draw.text((32, 48+i*32), line, font=font(28, "bold"), fill=WHITE)
        draw.text((32, 128), "SOURCE MAIN CAMERA", font=font(24), fill=WHITE)
        draw.text((1024, 128), "TARGET RED | OTHERS GREEN | GRIPPER BLUE", font=font(22), fill=WHITE)
        rgb = Image.fromarray(self.reader.rgb("image", frame))
        size, y = 832, 172
        scaled = rgb.resize((size, size), Image.Resampling.BILINEAR)
        canvas.paste(scaled, (32, y))
        canvas.paste(scaled, (1024, y))
        selected = self.entities.get(target, {})
        keys = [k for k in self.visible if k != target and not
                (selected.get("kind") == "part" and self.entities[k]["instance"] == selected.get("instance"))]
        if target:
            keys.append(target)  # Draw the target last so it remains readable on overlaps.
        for key in keys:
            entity = self.entities[key]
            region = self.reader.region(entity["regions"].get("image"), frame)
            if region is None or region["bbox"] is None:
                continue
            x0, y0, x1, y1 = region["bbox"] * np.tile([size/rgb.width, size/rgb.height], 2)
            color = RED if key == target else GREEN
            # Saved upper edges are exclusive; display inside the image boundary.
            box = (1024+x0, y+y0, 1024+x1-1, y+y1-1)
            draw.rectangle(box, outline=color, width=5 if key == target else 3)
            label = str(self.numbers[key])
            tx, ty = box[0]+3, max(y, box[1]-27)
            draw.rectangle((tx, ty, tx+12+len(label)*14, ty+26), fill=BG)
            draw.text((tx+4, ty-2), label, font=font(22, "bold"), fill=color)
        gripper = self.reader.region(self.meta.get('effector_regions', {}).get('single', {}).get('image'), frame)
        if gripper and gripper['bbox'] is not None:
            x0,y0,x1,y1 = gripper['bbox']*np.tile([size/rgb.width,size/rgb.height],2)
            draw.rectangle((1024+x0,y+y0,1024+x1-1,y+y1-1),outline='#48b9ff',width=4)
        if event:
            draw.text((32, 1022), f'{event["skill"]} [{event["start_frame"]}, {event["stop_frame"]}) | {event["description"]}',
                      font=font(24), fill=WHITE)
        target_name = target or "UNRESOLVED"
        draw.text((32, 1056), "TARGET: " + target_name, font=font(24, "bold"), fill=RED)
        legend = list(dict.fromkeys([*self.visible, *([target] if target else [])]))
        for i, key in enumerate(legend):
            draw.text((32+(i%3)*628, 1100+(i//3)*30), f'{self.numbers[key]} {key}',
                      font=font(19), fill=RED if key == target else GREEN)
        draw.text((32, 1240), "Full in-frame geometry; occlusion ignored | Background surfaces excluded | 10 FPS playback | REVIEW PENDING",
                  font=font(19), fill="#99aab8")
        return np.asarray(canvas)


def render_saved(path, output, number):
    renderer = SceneRenderer(path, number)
    output.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(10))
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "20", "preset": "fast", "threads": "2"}
        for frame in range(renderer.meta["frame_count"]):
            for packet in stream.encode(av.VideoFrame.from_ndarray(renderer.frame(frame), format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    previews = output.parent.parent / "previews"
    previews.mkdir(exist_ok=True)
    for i, event in enumerate(renderer.meta["events"]):
        Image.fromarray(renderer.frame((event["start_frame"]+event["stop_frame"]-1)//2)).save(previews / f"{number:02d}_event_{i:02d}.jpg")
    return dict(episode_id=renderer.meta["episode_id"], frames=renderer.meta["frame_count"], video=str(output), sha256=digest(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["inventory", "extract", "render", "augment"], required=True)
    parser.add_argument("--number", type=int, help="One-based review selection; omit for all ten")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    sys.path.insert(0, config["libero_root"])
    if args.stage == 'augment':
        from .atomic_vla.scene_regions import prepare_online_episode
        from scripts.preprocess.export import write_index
        index = Path(config['source_index'])
        paths = []
        for row in json.loads(index.read_text())['episodes']:
            result = prepare_online_episode(index.parent/row['path'],
                Path(config['output'])/'episodes'/row['episode_id'], gpu=config['gpu'])
            paths.append(result)
            print(f'PREPARED {row["episode_id"]}', flush=True)
        write_index(Path(config['output'])/'index.json', paths)
        return
    if args.stage == "inventory":
        inventory(config)
        return
    out = Path(config["output"])
    manifest = json.loads((out / "inventory.json").read_text())
    by_key = {row["episode_id"]: row for row in manifest["episodes"]}
    for number, key in enumerate(manifest["selected"], 1):
        if args.number is not None and number != args.number:
            continue
        destination = out / "episodes" / key
        if args.stage == "extract":
            from .atomic_vla.scene_regions import extract_scene
            row = by_key[key]
            path = extract_scene(row, manifest["scenes"][row["bddl"]], destination, gpu=config["gpu"],
                                 include_robot=config.get('protect_robot', False))
            print(f"EXTRACTED {number:02d} {path}", flush=True)
        else:
            result = render_saved(destination / "episode.json", out / "video" / f"{number:02d}_{key}.mp4", number)
            write_json(out / "renders" / f"{number:02d}.json", result)
            print(f"RENDERED {number:02d} {key}", flush=True)
    if args.stage == 'extract':
        paths = [out / 'episodes' / key / 'episode.json' for key in manifest['selected']]
        if all(path.is_file() for path in paths):
            from scripts.preprocess.export import write_index
            write_index(out / 'index.json', paths)


if __name__ == "__main__":
    main()
