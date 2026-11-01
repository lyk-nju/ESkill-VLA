"""Qwen localization and SAM3 tracking are separate, replayable CLI stages."""
import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import yaml

from .qwen import Qwen
from .sam import Sam, mask_box
from scripts.preprocess.artifacts import artifact, check_source, digest, episodes, processed_root, read_json, write_json


def target_catalog(episode, text):
    """Physical objects have one identity across events and arms, including handover."""
    catalog = [{"id": f"gripper_{arm}", "role": "gripper", "arm": arm,
                "description": f"The robot's {arm} physical gripper (palm and fingers only)",
                "events": []} for arm in episode.arms]
    for item in text["objects"]:
        refs = [e["source_id"] for e in text["events"] if e.get("object_id") == item["id"]]
        if refs:
            catalog.append({"id": item["id"], "role": "object", "arm": None,
                            "description": item["description"], "events": refs})
    if len({t["id"] for t in catalog}) != len(catalog):
        raise ValueError("Object IDs must not collide with gripper identities")
    return catalog


def keyframes(episode, target, events, limit):
    related = [e for e in events if e["source_id"] in target["events"]]
    candidates = sorted({i for e in related for i in
                         (e["start_frame"], (e["start_frame"]+e["stop_frame"]-1)//2, e["goal_frame"])})
    if not candidates:
        candidates = [0, (episode.count-1)//2, episode.count-1]
    if len(candidates) > limit:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, limit, dtype=int)]
    return sorted(set(candidates))


def locate_episode(episode, text, client, config):
    check_source(episode, text)
    from scripts.preprocess.text_annotation.annotate import validate_text
    validate_text(episode, text)
    targets = target_catalog(episode, text)
    records = []
    for camera in config["cameras"]:
        if camera not in episode.cameras:
            raise ValueError(f"Unknown source camera: {camera}")
        for target in targets:
            frames = keyframes(episode, target, text["events"], config["keyframes_per_target"])
            images = [(f, episode.rgb(camera, f)) for f in frames]
            images = [(f, rgb) for f, rgb in images if rgb is not None]
            answer, raw = (client.locate(target["description"], images) if images else
                           ({"visible": False, "reason": "missing_source_images"}, None))
            records.append({"camera": camera, "target_id": target["id"], "supplied_frames": [f for f, _ in images],
                            "answer": answer, "raw_response": raw})
    return {"format": "eskill.grounding.prompts.v1", "source": episode.entry,
            "frame_count": episode.count, "targets": targets, "prompts": records,
            "events": text["events"], "annotation_error": text.get("annotation_error"),
            "manual_review": "pending", "model": client.metadata}


def track_episode(episode, prompts, tracker, output):
    check_source(episode, prompts)
    key = episode.entry["key"]
    records, coverage = [], []
    for camera in sorted({p["camera"] for p in prompts["prompts"]}):
        # No re-encoding to MP4 and no fps conversion: numeric JPEG names map to exact source frames.
        with tempfile.TemporaryDirectory(prefix="eskill-sam-") as temporary:
            directory = Path(temporary)
            source_frames = []
            size = None
            for frame in range(episode.count):
                rgb = episode.rgb(camera, frame)
                if rgb is None:
                    continue
                if size is not None and rgb.shape[:2] != size:
                    raise ValueError("Source image resolution changes within a camera")
                size = rgb.shape[:2]
                Image.fromarray(rgb).save(directory / f"{len(source_frames):06d}.jpg", quality=100, subsampling=0)
                source_frames.append(frame)
            # Missing interior frames cannot be silently compressed into a shorter timeline.
            if source_frames != list(range(len(source_frames))):
                raise ValueError("Interior source frame missing; requires an explicit tracking policy")
            for number, prompt in enumerate(p for p in prompts["prompts"] if p["camera"] == camera):
                identity = {"camera": camera, "target_id": prompt["target_id"]}
                answer = prompt["answer"]
                if not source_frames or not answer["visible"]:
                    coverage.append({**identity, "status": "unlocalized", "reason": answer.get("reason")})
                    continue
                seed = answer["frame_index"]
                if seed not in source_frames:
                    raise ValueError("Seed frame is not present in this camera")
                height, width = size
                box = np.asarray(answer["bbox"], dtype=float)
                if (box.shape != (4,) or not np.isfinite(box).all() or
                        not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)):
                    raise ValueError("Seed bbox is not original-image pixel xyxy")
                track_frames, masks = [], []
                for frame, mask in tracker.track(directory, seed_frame=seed, bbox=box,
                                                  width=width, height=height, count=len(source_frames)):
                    if frame in track_frames or not 0 <= frame < len(source_frames):
                        raise ValueError("Duplicate or out-of-range propagated frame")
                    track_frames.append(frame)
                    masks.append(mask)
                if set(track_frames) != set(range(len(source_frames))):
                    raise ValueError("SAM did not return every available source frame")
                order = np.argsort(track_frames)
                values = np.stack(masks)[order]
                relative = Path("masks") / key / f"{camera}_track_{number:03d}.npz"
                destination = Path(output) / relative
                if destination.exists():
                    raise FileExistsError(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Packed bits preserve full-resolution masks with much less storage.
                with destination.open("xb") as stream:
                    np.savez_compressed(stream, bits=np.packbits(values.reshape(len(values), -1), axis=1),
                                        shape=np.array([height, width]), frames=np.array(source_frames))
                for slot, mask in enumerate(values):
                    bbox = mask_box(mask)
                    records.append({**identity, "frame_index": source_frames[slot], "bbox": bbox,
                                    "status": "predicted" if bbox else "empty", "mask": str(relative), "slot": slot})
                for frame in range(len(source_frames), episode.count):
                    records.append({**identity, "frame_index": frame, "bbox": None,
                                    "status": "missing_source_image", "mask": None, "slot": None})
                coverage.append({**identity, "status": "tracked", "frames": len(source_frames),
                                 "nonempty": sum(mask_box(m) is not None for m in values)})
    return {"format": "eskill.grounding.regions.v1", "source": episode.entry,
            "frame_count": episode.count, "fps": episode.fps, "targets": prompts["targets"],
            "events": prompts["events"], "annotation_error": prompts.get("annotation_error"),
            "regions": records, "coverage": coverage,
            "manual_review": "pending", "bbox_convention": "pixel xyxy, upper bounds exclusive",
            "model": tracker.metadata}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["locate", "track"])
    parser.add_argument("--dataset", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path, help="Default: source-specific processed directory")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = args.output or processed_root(args.dataset)
    camera_config = {**config, "cameras": config["cameras"][args.dataset]}
    identity = {"dataset": args.dataset, "source_manifest": str(args.manifest.resolve()),
                "source_manifest_sha256": digest(args.manifest), "config": config,
                "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    manifest_path = output / "manifest.json"
    if args.stage == "locate":
        write_json(manifest_path, {**identity, "manual_review": "pending"})
        backend = Qwen(config["qwen"])
    else:
        previous = read_json(manifest_path)
        if any(previous[k] != v for k, v in identity.items()):
            raise ValueError("Configuration/source/code changed between localization and tracking")
        backend = Sam(config["sam"])
    for episode in episodes(args.dataset, args.manifest):
        key = episode.entry["key"]
        if args.stage == "locate":
            path = artifact(output, "text", key)
            record = locate_episode(episode, read_json(path), backend, camera_config)
            record["text_sha256"] = digest(path)
            write_json(artifact(output, "prompts", key), record)
        else:
            path = artifact(output, "prompts", key)
            prompts = read_json(path)
            if digest(artifact(output, "text", key)) != prompts["text_sha256"]:
                raise ValueError("Text labels changed after localization")
            record = track_episode(episode, prompts, backend, output)
            record["prompts_sha256"] = digest(path)
            write_json(artifact(output, "regions", key), record)
        print(f"{args.stage}: {key}", flush=True)


if __name__ == "__main__":
    main()
