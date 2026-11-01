"""Keep source intervals; assign draft skills and persistent object identities."""
import argparse
from pathlib import Path
import json

import numpy as np
import yaml

from scripts.preprocess.artifacts import artifact, check_source, digest, episodes, processed_root, write_json
from scripts.preprocess.visual_grounding.qwen import Qwen


def prepare(episode):
    events = [{**item, "skill": None, "description": None, "object_id": None,
               "status": "unannotated"} for item in episode.evidence["intervals"]]
    return {"format": "eskill.subtasks.draft.v1", "source": episode.entry, "frame_count": episode.count,
            "interval_convention": "[start_frame,stop_frame); goal_frame explicitly identifies source image",
            "events": events, "objects": [], "manual_review": "pending"}


def validate_text(episode, record):
    check_source(episode, record)
    objects = {o["id"]: o["description"] for o in record["objects"]}
    if len(objects) != len(record["objects"]) or any(not k or not v for k, v in objects.items()):
        raise ValueError("Objects need unique IDs and visual descriptions")
    ids = set()
    for event in record["events"]:
        start, stop, goal = event["start_frame"], event["stop_frame"], event["goal_frame"]
        if any(type(v) is not int for v in (start, stop, goal)) or not 0 <= start < stop <= episode.count:
            raise ValueError("Invalid interval bounds")
        if not start <= goal < stop or event["arm"] not in episode.arms:
            raise ValueError("Invalid arm or explicit goal frame")
        if event["source_id"] in ids:
            raise ValueError("Duplicate event identity")
        ids.add(event["source_id"])
        if event["object_id"] is not None and event["object_id"] not in objects:
            raise ValueError("Event references an unknown physical object")
    return record


def annotate_episode(episode, client, config):
    record = prepare(episode)
    if not record["events"]:
        raise ValueError("No source intervals; supply a reviewed interval table before semantic annotation")
    candidates = sorted({f for e in record["events"] for f in
                         (e["start_frame"], (e["start_frame"]+e["stop_frame"]-1)//2, e["goal_frame"])})
    maximum = config["max_keyframes"]
    if len(candidates) > maximum:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates)-1, maximum, dtype=int)]
    camera = config["camera"]
    images = [(f"camera={camera}, frame_index={f}", episode.rgb(camera, f)) for f in candidates]
    images = [(label, rgb) for label, rgb in images if rgb is not None]
    if not images:
        raise ValueError("No source images available for semantic annotation")
    prompt = (
        "Annotate robot subtasks using the supplied task, source intervals and sparse video frames. "
        "Keep every source_id; do not alter interval boundaries or arm identity. "
        "Source primitives/skill runs are candidates, NOT verified semantic Events. "
        "Return a SINGLE JSON object with objects:[{id,description}] and "
        "events:[{source_id,skill,description,object_id}]. Use one object ID per physical object "
        "throughout the episode, even across arm handovers. Distinguish manipulated object from "
        "destination/container. Never invent an object for idle or empty return. Static motion is "
        "not proof of Hold; gripper closure is not proof of grasp. Use null for uncertain skills "
        "or object identity. Descriptions must be short. No markdown.\n"
        + "Allowed skills: " + json.dumps(config["skills"]) + "\nTask: " + episode.instructions[0]
        + "\nSource intervals: " + json.dumps(episode.evidence["intervals"], ensure_ascii=False)
    )
    answer, raw = client.ask(prompt, images)
    try:
        record["objects"] = answer["objects"]
        proposed = {event["source_id"]: event for event in answer["events"]}
        if len(proposed) != len(answer["events"]) or set(proposed) != {e["source_id"] for e in record["events"]}:
            raise ValueError("Qwen omitted, duplicated or invented a source interval")
        for event in record["events"]:
            label = proposed[event["source_id"]]
            if label["skill"] is not None and label["skill"] not in config["skills"]:
                raise ValueError("Skill outside configured vocabulary")
            event.update({k: label[k] for k in ("skill", "description", "object_id")})
            event["status"] = "model_proposed"
        validate_text(episode, record)
    except (KeyError, TypeError, ValueError) as error:
        # Keep the failed response, but never publish a partially aligned event table.
        record = prepare(episode)
        record["annotation_error"] = str(error)
    record["raw_response"] = raw
    record["model"] = client.metadata
    record["prompt"] = prompt
    record["supplied_images"] = [label for label, _ in images]
    validate_text(episode, record)
    # Primitive overlap is evidence to review, never silently merge or force idle.
    overlaps = []
    for arm in episode.arms:
        ordered = sorted((e for e in record["events"] if e["arm"] == arm), key=lambda e: e["start_frame"])
        for i, event in enumerate(ordered):
            for other in ordered[i+1:]:
                if other["start_frame"] >= event["stop_frame"]:
                    break
                overlaps.append([event["source_id"], other["source_id"]])
    record["same_arm_overlaps_to_review"] = overlaps
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--dataset", choices=["libero", "robotwin"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    output = args.output or processed_root(args.dataset)
    backend = Qwen(config["qwen"]) if args.stage == "run" else None
    for episode in episodes(args.dataset, args.manifest):
        params = {**config, "camera": config["cameras"][args.dataset]}
        record = annotate_episode(episode, backend, params) if backend else prepare(episode)
        record["provenance"] = {"stage": args.stage, "config": config,
                                "source_manifest_sha256": digest(args.manifest),
                                "code_sha256": digest(__file__), "qwen_code_sha256": digest(Path(__file__).parents[1]/"visual_grounding/qwen.py")}
        write_json(artifact(output, "text", episode.entry["key"]), record)
        status = "annotation_failed" if record.get("annotation_error") else args.stage
        print(f"{status}: {episode.entry['key']}", flush=True)


if __name__ == "__main__":
    main()
