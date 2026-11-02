"""Offline RGB replay. Rendering never creates or repairs semantic labels."""
import argparse
from pathlib import Path
import textwrap

import av
import cv2
import numpy as np

from .annotate import event_at, native_event_at, region_at, validate_annotation
from .source import load_episode, read_json

BLUE, GREEN, YELLOW = (80, 160, 255), (70, 220, 120), (255, 200, 60)


def text(canvas, value, x, y, color=(225, 225, 225), size=.55):
    cv2.putText(canvas, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, size,
                color, 1, cv2.LINE_AA)


def tile(canvas, rgb, x, y, width, height, label):
    text(canvas, label, x + 8, y + 20)
    if rgb is None:
        text(canvas, "EMPTY" if width < 250 else "MISSING / UNAVAILABLE", x + 15, y + height // 2, YELLOW)
        return
    scale = min(width / rgb.shape[1], (height - 32) / rgb.shape[0])
    w, h = round(rgb.shape[1] * scale), round(rgb.shape[0] * scale)
    image = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    left, top = x + (width - w) // 2, y + 30 + (height - 32 - h) // 2
    canvas[top:top + h, left:left + w] = image


def overlay(episode, annotation, camera, frame, *, arm=None, target=None):
    rgb = episode.rgb(camera, frame)
    if rgb is None:
        return None
    rgb = rgb.copy()
    if annotation:
        for region in annotation["regions"]:
            if region["camera"] != camera or region["frame_index"] != frame:
                continue
            if arm and (region["arm"] != arm or region["role"] != "object" or region.get("object") != target):
                continue
            x1, y1, x2, y2 = map(round, region["bbox"])
            color = YELLOW if arm else BLUE if region["role"] == "gripper" else GREEN
            cv2.rectangle(rgb, (x1, y1), (x2 - 1, y2 - 1), color, 2)
    return rgb


def make_frame(episode, frame, cameras, annotation=None):
    canvas = np.full((960, 1280, 3), 24, dtype=np.uint8)
    clock = (f"{episode.signals['timestamp'][frame]:.3f}s | nominal {episode.fps:g} fps"
             if episode.semantics.get("clock") == "physics" else f"{frame/episode.fps:.2f}s  {episode.fps:g} Hz")
    text(canvas, f"{episode.entry['task']} / {episode.entry['episode']}  frame {frame}/{episode.count-1}  "
         f"{clock}", 12, 24)
    text(canvas, "Blue: gripper | Green: current object | Yellow: same object at Event end | manual: pending", 12, 48)
    for index, camera in enumerate(cameras[:3]):
        rgb = overlay(episode, annotation, camera, frame) if camera in episode.cameras else None
        tile(canvas, rgb, index * 426, 60, 426, 300, camera)
    head = cameras[0]
    for index, arm in enumerate(("left", "right")):
        event = event_at(annotation, arm, frame) if annotation else None
        label = f"{arm}: Event unknown"
        rgb = None
        if event:
            label = f"{arm}: {event['skill']} [{event['start_frame']},{event['end_frame']})"
            if head in episode.cameras:
                rgb = overlay(episode, annotation, head, event["end_frame"], arm=arm, target=event.get("object"))
        tile(canvas, rgb, index * 426, 375, 426, 290, label)
        target_region = region_at(annotation, arm, head, event["end_frame"], "object", event.get("object")) if event else None
        text(canvas, "Event-end object region" if target_region else "Event-end object region: MISSING", index * 426 + 10, 690, YELLOW)
        if not event:
            primitives = [p.get("label", "?") for p in episode.evidence.get("subtasks", [])
                          if p.get("arm") == arm and p["start_frame"] <= frame <= p["end_frame"]]
            text(canvas, f"Source primitive: {', '.join(primitives) or 'unknown'}", index * 426 + 10, 715)
    text(canvas, "Recorded EE XYZ (isometric)", 865, 400)
    for arm, color in (("left", BLUE), ("right", YELLOW)):
        if arm not in episode.poses:
            continue
        xyz = episode.poses[arm][:, :3]
        xy = np.column_stack([xyz[:, 0] - xyz[:, 1] * .5, -xyz[:, 2] + xyz[:, 1] * .3])
        span = max(np.ptp(xy, axis=0).max(), .01)
        pixels = ((xy - xy.min(axis=0)) * (240 / span) + [880, 420]).astype(np.int32)
        cv2.polylines(canvas, [pixels], False, tuple(int(v * .5) for v in color), 1)
        if frame < len(pixels):
            cv2.circle(canvas, tuple(pixels[frame]), 4, color, -1)
            text(canvas, f"{arm}: {xyz[frame,0]:.3f}, {xyz[frame,1]:.3f}, {xyz[frame,2]:.3f}", 865, 700 if arm == "left" else 724, color, .45)
    for index, line in enumerate(textwrap.wrap("Task: " + " / ".join(episode.instructions[:1]), 140)[:2]):
        text(canvas, line, 12, 760 + index * 24)
    for index, arm in enumerate(("left", "right")):
        if episode.entry["format"] == "xpolicylab-v1":
            joint_key, gripper_key = f"state/{arm}_arm_joint_states", f"state/{arm}_ee_joint_states"
            kind = "stored state (semantics unverified)"
        else:
            joint_key, gripper_key = f"joint_state/{arm}_arm", f"joint_state/{arm}_gripper_command"
            kind = "measured arm / commanded gripper"
        joints, gripper = episode.signals.get(joint_key), episode.signals.get(gripper_key)
        if joints is not None and gripper is not None and frame < min(len(joints), len(gripper)):
            values = np.array2string(joints[frame], precision=2, suppress_small=True)
            line = f"{arm} {kind}: {values}  gripper={float(np.asarray(gripper[frame]).reshape(-1)[0]):.3f}"
        else:
            line = f"{arm}: state unavailable at this source frame"
        text(canvas, line, 12, 815 + index * 25, YELLOW, .46)
    for i, arm in enumerate(("left", "right")):
        y = 875 + i * 35
        text(canvas, arm, 12, y + 12)
        cv2.rectangle(canvas, (75, y), (1250, y + 15), (55, 55, 55), -1)
        if annotation:
            for event in annotation["events"]:
                if event["arm"] == arm:
                    x1 = 75 + round(event["start_frame"] / episode.count * 1175)
                    x2 = 75 + round(event["end_frame"] / episode.count * 1175)
                    cv2.rectangle(canvas, (x1, y), (x2, y + 15), BLUE if arm == "left" else YELLOW, -1)
        x = 75 + round(frame / episode.count * 1175)
        cv2.line(canvas, (x, y - 3), (x, y + 18), (255, 255, 255), 2)
    return canvas


def render_episode(episode, destination, *, cameras, annotation=None):
    if annotation:
        validate_annotation(episode, annotation)
    if not cameras or len(cameras) > 3:
        raise ValueError("Choose one to three cameras")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    from fractions import Fraction
    try:
        with av.open(str(temporary), "w") as output:
            stream = output.add_stream("libx264", rate=Fraction(str(episode.fps)))
            stream.width, stream.height, stream.pix_fmt = 1280, 960, "yuv420p"
            stream.options = {"crf": "23", "preset": "fast"}
            for index in range(episode.count):
                frame = av.VideoFrame.from_ndarray(make_frame(episode, index, cameras, annotation), format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(temporary)) as video:
            count = sum(1 for _ in video.decode(video=0))
        if count != episode.count:
            raise ValueError(f"Encoded frame count differs: {count}/{episode.count}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"video": str(destination), "frames": count, "fps": episode.fps}


def dashed_box(rgb, bbox, color):
    x1, y1, x2, y2 = bbox
    x2, y2 = x2-1, y2-1
    for x in range(x1, x2+1, 10):
        for y in (y1, y2):
            cv2.line(rgb, (x, y), (min(x+5, x2), y), color, 2)
    for y in range(y1, y2+1, 10):
        for x in (x1, x2):
            cv2.line(rgb, (x, y), (x, min(y+5, y2)), color, 2)


def draw_native_events(canvas, record, rows, frame):
    times = np.asarray([r["sim_time"] for r in rows], dtype=float)
    span = max(times[-1]-times[0], 1e-9)
    def px(index):
        return 85 + round((times[min(index,len(times)-1)]-times[0])/span*1175)
    for j, arm in enumerate(("left", "right")):
        event = native_event_at(record,arm,frame)
        active = f"{event['skill']} / {event.get('object_label') or '-'}" if event else "END STATE" if frame==record['terminal_state_frame'] else "Unknown"
        origins=','.join(event['source_ids']) if event else ''
        text(canvas, f"{arm.upper()}: {active}   {origins}",12,742+j*25,BLUE if arm=='left' else YELLOW,.49)
        y=808+j*55
        text(canvas,arm,12,y+16)
        cv2.rectangle(canvas,(85,y),(1260,y+28),(55,55,55),-1)
        for e in record["events"]:
            if e["arm"] != arm:
                continue
            x1,x2=px(e['start_frame']),px(e['stop_frame'])
            color=(90,90,90) if e['skill']=='Unknown' else BLUE if arm=='left' else YELLOW
            shade=tuple(int(c*(.65+.12*(int(e['event_id'].split('_')[-1])%3))) for c in color)
            cv2.rectangle(canvas,(x1,y),(max(x1+1,x2),y+28),shade,-1)
            width=cv2.getTextSize(e['skill'],cv2.FONT_HERSHEY_SIMPLEX,.43,1)[0][0]
            if x2-x1>width+6:
                text(canvas,e['skill'],x1+3,y+20,(245,245,245),.43)
        cursor=px(frame)
        cv2.line(canvas,(cursor,y-4),(cursor,y+32),(255,255,255),2)
    text(canvas,"Candidate events from source primitives | timeline uses simulation time | manual review: pending",12,914,(225,225,225),.47)


def render_full_gripper_record(path, destination, examples=None, *, event_record=None, preview_frames=None):
    """Only annotated camera views and real-RGB crops; no raw comparison panel."""
    import json
    import h5py
    from PIL import Image
    from collection.regions import extract_region
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    stats = {}
    with h5py.File(path) as hdf:
        rows = json.loads(hdf["frames"][()])
        entities = json.loads(hdf.attrs["entities"])
        roles = json.loads(hdf.attrs["region_roles"])
        if not rows:
            raise ValueError("No captured frames")
        if event_record:
            from .source import digest
            if event_record['source_sha256'] != digest(path) or event_record['frame_count'] != len(rows):
                raise ValueError("Events refer to a different native source")
        indices = list(range(len(rows))) if preview_frames is None else np.linspace(0,len(rows)-1,min(preview_frames,len(rows)),dtype=int).tolist()
        selected = set(np.linspace(0, len(rows)-1, min(8, len(rows)), dtype=int).tolist())
        fps = int(hdf.attrs["playback_fps"])
        with av.open(str(temporary), "w") as output:
            stream = output.add_stream("libx264", rate=fps)
            height=920 if event_record else 720
            stream.width, stream.height, stream.pix_fmt = 1280, height, "yuv420p"
            stream.thread_count = 2
            stream.options = {"crf": "18", "preset": "fast"}
            for i in indices:
                row=rows[i]
                canvas = np.full((height, 1280, 3), 24, dtype=np.uint8)
                text(canvas, f"{hdf.attrs['task']} | seed {hdf.attrs['seed']} | frame {i}/{len(rows)-1} | native expert", 12, 25)
                text(canvas, "LEFT = BLUE | RIGHT = YELLOW | thin solid: visible bbox | thick dashed: full bbox inside image", 12, 50)
                text(canvas, f"Simulation {row['sim_time']-rows[0]['sim_time']:.3f}s | physical step {row['physical_step']} | nominal {fps} fps review (irregular source sampling)", 12, 75)
                if not hdf.attrs["complete"]:
                    text(canvas, "PARTIAL", 1150, 25, YELLOW)
                elif preview_frames is not None:
                    text(canvas,"PREVIEW",1150,25,YELLOW)
                for j, camera in enumerate(("head_camera", "left_camera", "right_camera")):
                    rgb, ids = hdf[f"{camera}/rgb"][i], hdf[f"{camera}/actor_ids"][i]
                    painted = rgb.copy()
                    regions = {}
                    folder = Path(examples) / f"frame_{i:05d}" / camera if examples is not None and i in selected else None
                    if folder:
                        folder.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rgb).save(folder / "rgb.png")
                    for role, group in entities.items():
                        region = extract_region(rgb, ids, [v["id"] for v in group])
                        color = BLUE if role == "left_gripper" else YELLOW if role == "right_gripper" else (230, 120, 230) if roles[role]["kind"] == "reference" else GREEN
                        if region["bbox"]:
                            x1, y1, x2, y2 = region["bbox"]
                            cv2.rectangle(painted, (x1, y1), (x2-1, y2-1), color, 1)
                            if event_record and role in roles:
                                label=roles[role]['label']
                                width=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.3,1)[0][0]
                                text(painted,label,max(0,min(x1,rgb.shape[1]-width-1)),max(10,y1-3),color,.3)
                        if role not in ("left_gripper", "right_gripper"):
                            continue
                        full = extract_region(rgb, hdf[f"{camera}/{role}_full_mask"][i], [1])
                        regions[role] = full
                        saved = row["views"][camera][role]
                        stat = stats.setdefault(f"{camera}/{role}", dict(partial_occlusion=0, fully_occluded=0,
                            out_of_view=0, visible_outside_full=0, bbox_mismatch=0, full_bbox_not_containing_visible=0))
                        stat["bbox_mismatch"] += int(saved["full_bbox"] != full["bbox"] or saved["full_pixels"] != full["pixels"] or saved["bbox"] != region["bbox"] or saved["pixels"] != region["pixels"])
                        stat["visible_outside_full"] += int(np.count_nonzero(region["mask"] & ~full["mask"]))
                        stat["fully_occluded"] += int(full["pixels"] > 0 and region["pixels"] == 0)
                        stat["partial_occlusion"] += int(0 < region["pixels"] < full["pixels"])
                        stat["out_of_view"] += int(full["pixels"] == 0)
                        if full["bbox"]:
                            dashed_box(painted, full["bbox"], color)
                        if region["bbox"]:
                            b, f = region["bbox"], full["bbox"]
                            stat["full_bbox_not_containing_visible"] += int(f is None or not(f[0]<=b[0] and f[1]<=b[1] and f[2]>=b[2] and f[3]>=b[3]))
                        if folder:
                            for kind, value in (("visible", region), ("full", full)):
                                Image.fromarray(value["mask"].astype(np.uint8)*255).save(folder / f"{role}_{kind}_mask.png")
                                if value["crop"] is not None:
                                    Image.fromarray(value["crop"]).save(folder / f"{role}_{kind}_crop.png")
                    tile(canvas, painted, j*426, 85, 426, 310, camera + " / annotated")
                    for k, arm in enumerate(("left", "right")):
                        full = regions[f"{arm}_gripper"]
                        saved = row["views"][camera][f"{arm}_gripper"]
                        fraction = saved["visible_fraction"]
                        label = f"{arm}: {fraction:.0%} visible" if fraction is not None else f"{arm}: OUT OF VIEW"
                        tile(canvas, full["crop"], j*426+k*213, 400, 213, 255, label)
                text(canvas, "Crops use original RGB at the full bbox: occluders remain. Hidden gripper pixels are not synthesized.", 12, 680)
                text(canvas, "Task objects: green; references: purple. Full boxes cover only the same finger links, clipped to the camera view.", 12, 706)
                if event_record:
                    draw_native_events(canvas,event_record,rows,i)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(temporary)) as video:
            decoded = sum(1 for _ in video.decode(video=0))
        if decoded != len(indices):
            raise ValueError("Rendered video length differs from source")
        temporary.replace(destination)
        return dict(source=str(path), video=str(destination), frames=len(indices), source_frames=len(rows), fps=fps,
                    source_complete=bool(hdf.attrs["complete"]), views=stats,
                    simulation_seconds=rows[-1]["sim_time"]-rows[0]["sim_time"], manual_review="pending",
                    candidate_event_count=len(event_record['events']) if event_record else 0,
                    preview=preview_frames is not None)


def render_native_record(path, destination, examples=None, *, event_record=None, preview_frames=None):
    """Render native-ID evidence and verify saved bbox rows against lossless pixels."""
    import json
    import h5py
    from PIL import Image
    from collection.regions import extract_region

    with h5py.File(path) as header:
        full_gripper = bool(header.attrs.get("full_gripper", False))
    if full_gripper:
        return render_full_gripper_record(path, destination, examples,event_record=event_record,preview_frames=preview_frames)
    if event_record or preview_frames is not None:
        raise ValueError("Event overlay requires a full-gripper native record")

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    with h5py.File(path, "r") as hdf:
        rows = json.loads(hdf["frames"][()])
        entities = json.loads(hdf.attrs["entities"])
        roles = json.loads(hdf.attrs.get("region_roles", "{}"))
        colors = {key: BLUE if key == "left_gripper" else YELLOW if key == "right_gripper"
                  else (230, 120, 230) if roles.get(key, {}).get("kind") == "reference" else GREEN
                  for key in entities}
        crop_roles = ["left_gripper", "right_gripper"] + [
            key for key in entities if key not in ("left_gripper", "right_gripper")
            and roles.get(key, {}).get("show_crop", key == "object")][:2]
        crop_width = 480 // len(crop_roles)
        if not rows:
            raise ValueError("No captured frames")
        selected = set(np.linspace(0, len(rows)-1, min(8, len(rows)), dtype=int).tolist())
        fps = int(hdf.attrs["playback_fps"])
        temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
        with av.open(str(temporary), "w") as output:
            stream = output.add_stream("libx264", rate=fps)
            stream.width, stream.height, stream.pix_fmt = 1440, 1080, "yuv420p"
            stream.thread_count = 2
            stream.options = {"crf": "18", "preset": "fast"}
            for index, row in enumerate(rows):
                canvas = np.full((1080, 1440, 3), 24, dtype=np.uint8)
                text(canvas, f"{hdf.attrs['task']} | seed {hdf.attrs['seed']} | native expert | frame {index}/{len(rows)-1}", 12, 26)
                if not hdf.attrs["complete"]:
                    text(canvas, "PARTIAL / INCOMPLETE", 1100, 26, YELLOW)
                text(canvas, f"Simulation t={row['sim_time']-rows[0]['sim_time']:.3f}s | step={row['physical_step']} | Review playback: nominal {fps} fps (irregular source sampling)", 12, 52)
                text(canvas, "LEFT = BLUE | RIGHT = YELLOW | TASK OBJECTS = GREEN | REFERENCES = PURPLE | EMPTY = not visible", 12, 78)
                for camera_index, camera in enumerate(("head_camera", "left_camera", "right_camera")):
                    y = 90 + camera_index * 310
                    rgb, ids = hdf[f"{camera}/rgb"][index], hdf[f"{camera}/actor_ids"][index]
                    overlay_rgb = rgb.copy()
                    regions = {}
                    for role, group in entities.items():
                        region = extract_region(rgb, ids, [item["id"] for item in group])
                        regions[role] = region
                        key = f"{camera}/{role}"
                        stat = stats.setdefault(key, {"visible": 0, "empty": 0, "bbox_mismatch": 0,
                                                       "crop_mismatch": 0, "max_pixels": 0})
                        stat["visible" if region["pixels"] else "empty"] += 1
                        stat["max_pixels"] = max(stat["max_pixels"], region["pixels"])
                        saved = row["views"][camera][role]
                        stat["bbox_mismatch"] += int(saved["bbox"] != region["bbox"] or saved["pixels"] != region["pixels"])
                        color = colors[role]
                        if region["bbox"]:
                            x1, y1, x2, y2 = region["bbox"]
                            stat["crop_mismatch"] += int(not np.array_equal(region["crop"], rgb[y1:y2, x1:x2]))
                            mask = region["mask"]
                            overlay_rgb[mask] = (overlay_rgb[mask] * .60 + np.asarray(color) * .40).astype(np.uint8)
                            cv2.rectangle(overlay_rgb, (x1, y1), (x2-1, y2-1), color, 2)
                            if role in roles:
                                text(overlay_rgb, roles[role]["label"], x1, max(10, y1-3), color, .32)
                        if examples is not None and index in selected:
                            folder = Path(examples) / f"frame_{index:05d}" / camera
                            folder.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(region["mask"].astype(np.uint8) * 255).save(folder / f"{role}_mask.png")
                            if region["crop"] is not None:
                                Image.fromarray(region["crop"]).save(folder / f"{role}_crop.png")
                            Image.fromarray(rgb).save(folder / "rgb.png")
                    tile(canvas, rgb, 0, y, 480, 300, f"{camera}: original RGB")
                    tile(canvas, overlay_rgb, 480, y, 480, 300, "Native masks + tight visible bbox")
                    for j, role in enumerate(crop_roles):
                        color = colors[role]
                        region = regions[role]
                        crop = region["crop"]
                        label = "LEFT" if role == "left_gripper" else "RIGHT" if role == "right_gripper" else roles.get(role, {}).get("label", "OBJECT")
                        x = 960 + crop_width*j
                        tile(canvas, crop, x, y, crop_width, 235, label)
                        text(canvas, f"{region['pixels']} px", x+5, y+255, color, .40)
                        ids_label = ",".join(str(item["id"]) for item in entities[role])
                        text(canvas, "IDs " + (ids_label if len(ids_label)<18 else ids_label[:15]+"..."), x+5, y+278, color, .32)
                        cv2.rectangle(canvas, (x, y+28), (x+crop_width-2, y+235), color, 1)
                text(canvas, "Pixel bbox: xyxy, upper bounds exclusive. Crops use this camera and this exact source frame.", 12, 1040)
                text(canvas, "Native instance evidence only. Per-arm object assignment / Event / Pick / Place / Hold still require annotation.", 12, 1066)
                frame = av.VideoFrame.from_ndarray(canvas, format="rgb24")
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(temporary)) as video:
            decoded = sum(1 for _ in video.decode(video=0))
        if decoded != len(rows):
            raise ValueError("Rendered video length differs from source")
        temporary.replace(destination)
        return {"source": str(path), "video": str(destination), "frames": len(rows),
                "fps": fps, "source_complete": bool(hdf.attrs["complete"]),
                "source_simulation_seconds": rows[-1]["sim_time"] - rows[0]["sim_time"],
                "views": stats, "entities": entities, "manual_review": "pending"}


def render_collected_episode(path, destination, *, source_place_goals=False):
    """Review a production raw episode: native regions and recorded source primitives."""
    import json
    import h5py
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.mp4")
    try:
        with h5py.File(path, "r") as source, av.open(str(temporary), "w") as output:
            times = source["timestamp"][:]
            obs = source["observations"]
            evidence = json.loads(source["evidence"][()])
            primitives = evidence["subtasks"]
            identities = json.loads(source.attrs["native_entities"])
            regions = {key: {name: value[:] for name, value in group.items() if name != "full_mask"}
                       for key, group in obs["native_regions/regions"].items()}
            task, seed = str(source.attrs["task"]), int(source.attrs["seed"])
            goal_views, goal_records = [], []
            if source_place_goals:
                for primitive in [p for p in primitives if p["label"] == "place"][:3]:
                    ids = primitive["metadata"].get("actor_entity_ids", [])
                    key = next((key for key, entries in identities.items()
                                if ids and set(ids) == {entry["id"] for entry in entries}), None)
                    row = primitive["end_frame"]
                    image = obs["observation/head_camera/rgb"][row].copy()
                    valid = key is not None and bool(regions[key]["boxes_valid"][row])
                    box = regions[key]["boxes_xyxy"][row].tolist() if valid else None
                    if valid:
                        mask = np.isin(obs["native_regions/actor_ids"][row], ids)
                        image[mask] = (image[mask]*.55+np.asarray(GREEN)*.45).astype(np.uint8)
                        x1, y1, x2, y2 = box
                        cv2.rectangle(image, (x1,y1), (x2-1,y2-1), GREEN, 2)
                    label = f"Place end: {key or 'UNKNOWN'} id={','.join(map(str,ids))} f={row}"
                    if not valid:
                        label += " MISSING REGION"
                    goal_views.append((primitive["primitive_id"], image, label))
                    goal_records.append(dict(source_primitive_id=primitive["primitive_id"],
                        object_key=key, actor_entity_ids=ids, source_end_row=row,
                        time_s=float(times[row]), bbox=box, bbox_valid=valid))
            stream = output.add_stream("libx264", rate=15)
            stream.width, stream.height, stream.pix_fmt = 1280, 1240 if source_place_goals else 920, "yuv420p"
            stream.options = {"crf": "23", "preset": "fast"}
            span = max(times[-1]-times[0], 1e-9)
            def px(time):
                return 85+round((time-times[0])/span*1175)
            def target_name(p):
                if source_place_goals:
                    ids = p["metadata"].get("actor_entity_ids", [])
                    key = next((key for key, entries in identities.items()
                                if ids and set(ids) == {entry["id"] for entry in entries}), None)
                    if key:
                        return f"{key} id={','.join(map(str,ids))}"
                return p['actor'] or '-'
            for i, time in enumerate(times):
                canvas = np.full((stream.height, 1280, 3), 24, dtype=np.uint8)
                text(canvas, f"{task} | CLEAN | seed {seed} | source frame {i}/{len(times)-1} | t={time:.3f}s", 12, 25)
                text(canvas, "Blue / yellow: left / right gripper | dashed: full in frame | green: source target", 12, 49, size=.49)
                active = [p for p in primitives if p["start_sec"] <= time <= p["end_sec"]]
                active_ids = {identity for p in active for identity in p["metadata"].get("actor_entity_ids", [])}
                rgb = obs["observation/head_camera/rgb"][i].copy()
                for key, group in regions.items():
                    if key in ("left_gripper", "right_gripper"):
                        color = BLUE if key == "left_gripper" else YELLOW
                        if group["boxes_valid"][i]:
                            x1, y1, x2, y2 = map(int, group["boxes_xyxy"][i])
                            cv2.rectangle(rgb, (x1, y1), (x2-1, y2-1), color, 1)
                        if group["full_boxes_valid"][i]:
                            dashed_box(rgb, group["full_boxes_xyxy"][i], color)
                    elif group["boxes_valid"][i] and active_ids.intersection(entry["id"] for entry in identities[key]):
                        if source_place_goals:
                            mask = np.isin(obs["native_regions/actor_ids"][i], [entry["id"] for entry in identities[key]])
                            rgb[mask] = (rgb[mask]*.7+np.asarray(GREEN)*.3).astype(np.uint8)
                        x1, y1, x2, y2 = map(int, group["boxes_xyxy"][i])
                        cv2.rectangle(rgb, (x1, y1), (x2-1, y2-1), GREEN, 2)
                        text(rgb, key, x1, max(y1-4, 12), GREEN, .4)
                tile(canvas, rgb, 0, 60, 850, 650, "head camera: native regions")
                tile(canvas, obs["observation/left_camera/rgb"][i], 855, 60, 425, 315, "left wrist RGB")
                tile(canvas, obs["observation/right_camera/rgb"][i], 855, 390, 425, 315, "right wrist RGB")
                for j, arm in enumerate(("left", "right")):
                    color = BLUE if arm == "left" else YELLOW
                    current = [p for p in active if p["arm"] == arm]
                    label = " / ".join(f"{p['label']} ({target_name(p)})" for p in current) or "no active source primitive"
                    text(canvas, f"{arm.upper()}: {label}", 12, 742+j*25, color, .48)
                    y = 808+j*55
                    text(canvas, arm, 12, y+16, color)
                    cv2.rectangle(canvas, (85, y), (1260, y+28), (55, 55, 55), -1)
                    for p in primitives:
                        if p["arm"] != arm:
                            continue
                        x1, x2 = px(p["start_sec"]), px(p["end_sec"])
                        shade = tuple(int(c*(.55+.13*(p['primitive_id']%3))) for c in color)
                        cv2.rectangle(canvas, (x1, y), (max(x1+1, x2), y+28), shade, -1)
                        width = cv2.getTextSize(p["label"], cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                        if x2-x1 > width+6:
                            text(canvas, p["label"], x1+3, y+20, size=.4)
                    cv2.line(canvas, (px(time), y-4), (px(time), y+32), (255,255,255), 2)
                text(canvas, "SOURCE PRIMITIVES | physics clock | saved rows at nominal 15 fps, irregular sampling | manual: pending", 12, 914, size=.44)
                for index, (primitive_id, image, label) in enumerate(goal_views):
                    x = index*426
                    tile(canvas, image, x, 930, 426, 300, label)
                    if any(p["primitive_id"] == primitive_id for p in active):
                        cv2.rectangle(canvas, (x+2,932), (x+423,1230), GREEN, 2)
                for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        with av.open(str(temporary)) as video:
            decoded = sum(1 for _ in video.decode(video=0))
        if decoded != len(times):
            raise ValueError("Source/rendered frame count mismatch")
        temporary.replace(destination)
        return dict(task=task, seed=seed, frames=decoded, primitives=len(primitives),
                    source_place_endpoints=goal_records,
                    source=str(Path(path).resolve()), video=str(destination.resolve()), manual_review="pending")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--manifest", type=Path)
    source_group.add_argument("--native-record", type=Path)
    source_group.add_argument("--collected-record", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--examples", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--preview-frames", type=int)
    parser.add_argument("--source-place-goals", action="store_true",
                        help="Show up to three same-instance source Place endpoint views")
    args = parser.parse_args()
    if args.collected_record:
        from .source import write_json
        write_json(args.output.with_suffix(".json"), render_collected_episode(args.collected_record, args.output,
                   source_place_goals=args.source_place_goals))
        return
    if args.native_record:
        from .source import write_json
        report = render_native_record(args.native_record, args.output, args.examples,
            event_record=read_json(args.events) if args.events else None,preview_frames=args.preview_frames)
        write_json(args.output.with_suffix(".json"), report)
        return
    manifest = read_json(args.manifest)
    from .source import verify_entry
    entry = manifest["samples"][args.index]
    verify_entry(entry)
    render_episode(load_episode(entry), args.output, cameras=manifest["config"]["cameras"],
                   annotation=read_json(args.annotation) if args.annotation else None)


if __name__ == "__main__":
    main()
