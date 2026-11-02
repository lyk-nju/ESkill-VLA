"""Keep source evidence separate from explicitly supplied project annotations."""
import argparse
import json
from pathlib import Path

import numpy as np

from .source import digest, load_episode, read_json, write_json


def validate_annotation(episode, annotation):
    if annotation["source_sha256"] != digest(episode.entry["hdf5"]):
        raise ValueError("Annotation refers to a different source episode")
    if episode.entry.get("events") and annotation.get("source_events_sha256") != digest(episode.entry["events"]):
        raise ValueError("Source event evidence changed since annotation")
    if annotation["frame_count"] != episode.count:
        raise ValueError("Annotation timeline differs from source")
    intervals = {"left": [], "right": []}
    for event in annotation["events"]:
        arm, start, end = event["arm"], event["start_frame"], event["end_frame"]
        if arm not in intervals or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Explicit arm and integer Event bounds required")
        # Current frames use [start, end); end identifies the target RGB frame.
        if not 0 <= start < end < episode.count or not event.get("skill"):
            raise ValueError(f"Invalid Event: {event}")
        intervals[arm].append((start, end))
    for arm, values in intervals.items():
        values.sort()
        if any(a[1] > b[0] for a, b in zip(values, values[1:])):
            raise ValueError(f"Overlapping Events on {arm}; opposite arms may overlap")
    seen = set()
    for region in annotation["regions"]:
        arm, frame, camera = region["arm"], region["frame_index"], region["camera"]
        role = region["role"]
        if arm not in intervals or role not in ("gripper", "object"):
            raise ValueError("Region requires arm and gripper/object role")
        if camera not in episode.cameras or not isinstance(frame, int) or not 0 <= frame < episode.count:
            raise ValueError("Region has invalid camera/frame")
        if role == "object" and not region.get("object"):
            raise ValueError("Object region requires a track identity for consistent endpoints")
        key = (arm, frame, camera, role, region.get("object"))
        if key in seen:
            raise ValueError(f"Duplicate region: {key}")
        seen.add(key)
        box = np.asarray(region["bbox"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("bbox must be finite pixel xyxy")
        rgb = episode.rgb(camera, frame)
        if rgb is None:
            raise ValueError("Cannot attach a region to a missing source image")
        h, w = rgb.shape[:2]
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            raise ValueError("bbox outside source image")


def build_annotation(episode, labels=None, *, provenance=None):
    labels = labels or {}
    output = {
        "format": "eskill.robotwin.annotations.v1", "source": episode.entry,
        "source_sha256": digest(episode.entry["hdf5"]), "frame_count": episode.count,
        "source_events_sha256": digest(episode.entry["events"]) if episode.entry.get("events") else None,
        "interval_convention": "[start_frame,end_frame); target image at end_frame",
        "bbox_convention": "pixel xyxy, upper bounds exclusive",
        "provenance": provenance or {"method": "source_evidence_only"},
        "source_intervals": episode.evidence.get("subtasks", []),
        "events": labels.get("events", []), "regions": labels.get("regions", []),
        "manual_review": "pending",
    }
    validate_annotation(episode, output)
    output["missing"] = [name for name in ("events", "regions") if not output[name]]
    return output


def event_at(annotation, arm, frame):
    return next((event for event in annotation["events"] if event["arm"] == arm
                 and event["start_frame"] <= frame < event["end_frame"]), None)


def region_at(annotation, arm, camera, frame, role, object_name=None):
    return next((region for region in annotation["regions"]
                 if region["arm"] == arm and region["camera"] == camera
                 and region["frame_index"] == frame and region["role"] == role
                 and (role == "gripper" or region.get("object") == object_name)), None)


def remaining_path(episode, event, frame, points=16):
    """Arc-length sampled metric relative XYZ; no invented path for missing EE."""
    if points < 2 or not event["start_frame"] <= frame < event["end_frame"]:
        raise ValueError("Invalid path query")
    values = episode.poses.get(event["arm"])
    if values is None or event["end_frame"] >= len(values):
        return None
    path = values[frame:event["end_frame"] + 1, :3]
    if not np.isfinite(path).all():
        raise ValueError("Non-finite EE path")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if distance[-1] == 0:
        return np.zeros((points, 3), dtype=np.float32)
    distance, indices = np.unique(distance, return_index=True)
    samples = np.linspace(0, distance[-1], points)
    return (np.stack([np.interp(samples, distance, path[indices, i]) for i in range(3)], axis=-1)
            - path[0]).astype(np.float32)


def native_event_at(record, arm, frame):
    return next((event for event in record["events"] if event["arm"] == arm
                 and event["start_frame"] <= frame < event["stop_frame"]), None)


def build_native_event_record(path):
    """Task rules over source primitives; preview labels keep their own endpoints."""
    import h5py
    path = Path(path)
    with h5py.File(path) as hdf:
        task = str(hdf.attrs["task"])
        rows = json.loads(hdf["frames"][()])
        evidence = json.loads(hdf["source_evidence"][()])
        entities = json.loads(hdf.attrs["entities"])
        roles = json.loads(hdf.attrs["region_roles"])
        poses = {arm: hdf[f"signals/endpose/{arm}_endpose"][:] for arm in ("left", "right")}
        commands = {arm: hdf[f"signals/joint_action/{arm}_gripper"][:].reshape(-1) for arm in poses}
        complete = bool(hdf.attrs["complete"])
        seed = int(hdf.attrs["seed"])
    if not rows:
        raise ValueError("No source frames for event preview")
    source = [{**p, "source_id": f"primitive_{i:04d}"} for i, p in enumerate(evidence["subtasks"])]
    observations = {(int(o["frame_index"]), o["arm"]): o for o in evidence.get("observations", [])}
    events = []

    def add(parts, skill, object_key=None, reason="direct mapping", arm=None, start=None, stop=None, **extra):
        if parts:
            arm = parts[0]["arm"]
            start = min(p["start_frame"] for p in parts)
            stop = max(p["end_frame"] for p in parts)+1
        if start is None or stop is None or start >= stop:
            return
        if object_key is not None and object_key not in entities:
            raise ValueError(f"Unresolved native entity: {object_key}")
        events.append(dict(arm=arm, skill=skill, object_key=object_key,
            object_label=roles[object_key]["label"] if object_key else None,
            start_frame=int(start), stop_frame=int(stop), goal_frame=int(stop)-1,
            source_ids=[p["source_id"] for p in parts], reason=reason, status="rule_proposed", **extra))

    for arm in ("left", "right"):
        parts = sorted([p for p in source if p["arm"] == arm], key=lambda p: (p["start_frame"], p["primitive_id"]))
        if not parts:
            continue
        if task == "beat_block_hammer":
            add([p for p in parts if p["label"] in ("grasp", "retreat")], "Pick", "object_hammer", "grasp plus immediate clearance")
            add([p for p in parts if p["label"] == "place"], "ToolContact", "object_hammer", "closed-gripper tool contact; not a release")
        elif task == "click_bell":
            add(parts, "Press", "object_bell", "click preparation, downward contact and withdrawal")
        elif task == "move_playingcard_away":
            for p in parts:
                skill = {"grasp": "Pick", "move_by_displacement": "Reposition", "open_gripper": "Release"}.get(p["label"], "Unknown")
                add([p], skill, "object_playingcards", "source primitive mapping; release does not assert stable placement")
        elif task == "lift_pot":
            add(parts, "Pick", "object_pot", "preparation, grasp and cooperative lift", cooperative_group="pot_pick")
        elif task == "place_burger_fries":
            key = "object_hamburg" if arm == "left" else "object_frenchfries"
            place = next((p for p in parts if p["label"] == "place"), None)
            if place:
                add([p for p in parts if p["end_frame"] < place["start_frame"]], "Pick", key, "grasp plus immediate lift")
                add([place], "Place", key)
                add([p for p in parts if p["start_frame"] > place["end_frame"]], "Return", reason="post-release clearance and return")
            else:
                add(parts, "Pick", key, "unfinished grasp/lift prefix")
        elif task == "place_can_basket":
            can_names = {e["name"] for e in entities["object_can"]}
            is_can_arm = any(p["label"] == "grasp" and p.get("actor") in can_names for p in parts)
            if is_can_arm:
                first = next(i for i, p in enumerate(parts) if p["label"] == "grasp")
                add(parts[:first+1], "Pick", "object_can")
                opened = next((i for i, p in enumerate(parts) if i > first and p["label"] == "open_gripper"), None)
                if opened is not None:
                    block = parts[first+1:opened+1]
                    add(block, "Place", "object_can", "placement, optional native fallback and explicit release",
                        zero_span_source_attempts=[p["source_id"] for p in block if p["label"] == "place" and p["start_frame"] == p["end_frame"]])
                    add(parts[opened+1:], "Return", reason="post-release clearance and return")
                else:
                    add(parts[first+1:], "Place", "object_can", "unfinished placement prefix")
            else:
                add(parts, "Pick", "object_basket", "basket grasp, extra closure and lift")
        else:
            raise ValueError(f"No reviewed preview rule for {task}")

    # One physical pot, one shared terminal state, separate measured arm paths.
    pot_events = [e for e in events if e.get("cooperative_group") == "pot_pick"]
    if pot_events:
        end = max(e["goal_frame"] for e in pot_events)
        for event in pot_events:
            event.update(stop_frame=end+1, goal_frame=end)

    for arm in ("left", "right"):
        ordered = sorted([e for e in events if e["arm"] == arm], key=lambda e: e["start_frame"])
        if any(a["stop_frame"] > b["start_frame"] for a,b in zip(ordered, ordered[1:])):
            raise ValueError(f"Same-arm event overlap: {arm}")
        cursor, previous = 0, None
        for following in [*ordered, None]:
            boundary = following["start_frame"] if following else len(rows)-1
            if cursor < boundary:
                key = previous.get("object_key") if previous else None
                held = key is not None and previous["skill"] in ("Pick", "Reposition", "Hold")
                proof = {"frames": boundary-cursor}
                skill, object_key = "Unknown", None
                if held:
                    seen = [observations.get((i, arm)) for i in range(cursor, boundary)]
                    positions = []
                    closed_contact = all(o and o["actor"] in {e["name"] for e in entities[key]}
                        and o["contact_points"] and not o["gripper_command_open"] for o in seen)
                    if closed_contact:
                        for i, o in zip(range(cursor,boundary), seen):
                            w,x,y,z = poses[arm][i,3:]
                            rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                            positions.append(rotation.T @ (np.asarray(o["actor_position"])-poses[arm][i,:3]))
                        drift=float(np.linalg.norm(np.asarray(positions)-positions[0],axis=1).max())
                        proof.update(relative_position_max_drift_m=drift, drift_limit_m=0.02)
                        if drift <= 0.02:
                            skill, object_key = "Hold", key
                    proof["all_frames_closed_contact"] = bool(closed_contact)
                else:
                    opened=bool(np.all(commands[arm][cursor:boundary] > 0.8))
                    contact_free=all(not observations.get((i,arm),{}).get("contact_points") for i in range(cursor,boundary))
                    proof.update(all_commands_open=opened, no_recorded_contact=contact_free)
                    if opened and contact_free:
                        skill="Idle"
                add([],skill,object_key,"evidence-checked gap between source primitives",arm=arm,
                    start=cursor,stop=boundary,evidence=proof)
            if following:
                cursor, previous = following["stop_frame"], following
    events.sort(key=lambda e:(e["arm"],e["start_frame"]))
    for i, event in enumerate(events):
        event["event_id"] = f"candidate_{i:04d}"
    record=dict(format="eskill.robotwin.primitive_events.preview.v1", source=str(path.resolve()),
        source_sha256=digest(path), task=task, seed=seed, frame_count=len(rows), source_complete=complete,
        interval_convention="source closed intervals; preview [start,stop), goal_frame=last source state; explicit training conversion required",
        terminal_state_frame=len(rows)-1, source_primitives=source, events=events, manual_review="pending")
    for event in events:
        if not 0 <= event["start_frame"] < event["stop_frame"] <= len(rows) or not event["start_frame"] <= event["goal_frame"] < event["stop_frame"]:
            raise ValueError("Candidate interval outside source")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", type=Path)
    inputs.add_argument("--native-record", type=Path)
    parser.add_argument("--labels", type=Path, help="Directory of explicitly supplied task/episode.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.native_record:
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, build_native_event_record(args.native_record))
        return
    for entry in read_json(args.manifest)["samples"]:
        from .source import verify_entry
        verify_entry(entry)
        episode = load_episode(entry)
        labels_path = args.labels / entry["task"] / f"{entry['episode']}.json" if args.labels else None
        labels = read_json(labels_path) if labels_path else None
        provenance = {"method": "explicit_labels", "path": str(labels_path),
                      "sha256": digest(labels_path)} if labels_path else None
        destination = args.output / entry["task"] / f"{entry['episode']}.json"
        if destination.exists():
            raise FileExistsError(f"Choose a new annotation output/version: {destination}")
        write_json(destination, build_annotation(episode, labels, provenance=provenance))


if __name__ == "__main__":
    main()
