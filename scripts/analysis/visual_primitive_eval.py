"""Evaluate saved Visual Primitive policies on a fixed same-image/perturbed-image manifest.

JSON fields: protocol, metric_space ("source" or square "model" pixels), rows.
Each row has id, split, parent (trajectory), scene, image (path relative to manifest),
task, boxes (entity -> raw-image xyxy pixel edges), target (entity ID or null for absent).
Optional: entity_masks (entity ID -> mask path), current_mask and subject_mask (binary image paths), subject_box (raw xyxy),
case, skill, pair_id (two different queries on identical RGB), move_id (two actual
layouts in reference/modified order), center_xy (original E0 target point, raw pixels).
For dual-view checkpoints use --episode-index with each row's episode_id (or parent) and frame.
Paired rows must match the source main RGB. Edited-main/wrist conflicts and wrist removal are separate runs.
Fit/Check and locked split names and membership come from the caller's manifest.
Use --panels 0 to omit PNGs. All maps are computed before accessing ground truth.
"""

import argparse
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw
import torch

from openpi.models.diagnostics import HIT_RADIUS_PX, spatial_centers
from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.policies.eskill_dataset import EpisodeReader
from openpi.transforms import normalized_image_boxes, resize_image, source_pixel_geometry


def map_array(tensor):
    return tensor[0].detach().float().cpu().numpy()


def peak3(weights):
    """Center of mass in the highest-mass 3x3 patch window, normalized model xy."""
    return spatial_centers(torch.as_tensor(weights))[0].numpy()


def box_overlap(box, side):
    """Fraction of each patch covered by the box; sum(map * fraction) gives ROI mass."""
    edges = np.arange(side + 1) / side
    overlap = (
        np.minimum(np.asarray(box[2:])[:, None], edges[1:])
        - np.maximum(np.asarray(box[:2])[:, None], edges[:-1])
    ).clip(0, 1) * side
    return (overlap[1, :, None] * overlap[0, None, :]).ravel()


def mask_overlap(path, shape, size, side):
    mask = np.asarray(Image.open(path).convert("L")) > 0
    if mask.shape != tuple(shape[:2]):
        raise ValueError(f"Mask/RGB shape mismatch: {path}")
    value = resize_image(mask.astype(np.float32), size, nearest=False)
    return value.reshape(side, size // side, side, size // side).mean((1, 3)).ravel()


def metric_point(point, shape, size, space):
    if space == "model":
        return point * size
    scale, offset = source_pixel_geometry(shape, size)
    return (point - offset) * scale


def center_box(box):
    return (np.asarray(box[:2]) + box[2:]) / 2


def map_score(weights, box, roi, distractors, shape, size, space, gt_center=None):
    has_target = box is not None and np.all(box[2:] > box[:2])
    result = dict(peak=float(weights.max()))
    if not weights.any():
        if (has_target or gt_center is not None) and roi.any():
            result.update(hit10=0.0, peak3_hit10=0.0, argmax_hit10=0.0)
        return result
    local, single = (p.numpy() for p in spatial_centers(torch.as_tensor(weights)))
    result.update(peak3_xy=metric_point(local, shape, size, space).tolist(),
                  argmax_xy=metric_point(single, shape, size, space).tolist())
    others = [float(weights @ other) for other in distractors]
    if others:
        result["max_distractor_mass"] = max(others)
    if not roi.any():
        return result
    probability = roi / roi.sum()
    positive = probability > 0
    kl = (probability[positive] * np.log(probability[positive] / weights[positive].clip(1e-30))).sum()
    mass = float(weights @ roi)
    result.update(roi_mass=mass, kl=float(kl))
    if others:
        result["mass_margin"] = mass - max(others)
        result["roi_mass_top1"] = float(mass > max(others))
    if has_target or gt_center is not None:
        center = np.asarray(gt_center if gt_center is not None else center_box(box), dtype=np.float32)
        result["true_center_xy"] = metric_point(center, shape, size, space).tolist()
        scale = size if space == "model" else source_pixel_geometry(shape, size)[0].astype(np.float32)
        for name, point in (("peak3", local), ("argmax", single)):
            error = float(np.linalg.norm((point - center) * scale))
            result[f"{name}_error_px"] = error
            result[f"{name}_hit10"] = float(error <= HIT_RADIUS_PX)
        # Keep the primary-readout fields for existing reports and Probe inputs.
        result.update(center_error_px=result["peak3_error_px"], hit10=result["peak3_hit10"])
    return result


def numeric_means(items):
    values = defaultdict(list)
    for row in items:
        for key, value in row.items():
            if type(value) in (int, float):
                values[key].append(value)
    return {k: dict(mean=float(np.mean(v)), n=len(v)) for k, v in sorted(values.items())}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        for key in (
            "all",
            "split/" + row["split"],
            "scene/" + row["scene"],
            "split_scene/" + row["split"] + "/" + row["scene"],
            "case/" + row["case"],
        ):
            groups[key].append(row)
        if row.get("skill"):
            groups["skill/" + row["skill"]].append(row)
    return {
        name: dict(
            queries=len(items),
            parents=len({r["parent"] for r in items}),
            current=numeric_means(row["current"] for row in items),
            **({"subject": numeric_means(row["subject"] for row in items)} if "subject" in items[0] else {}),
            alpha=numeric_means(row["alpha_stats"] for row in items),
            l12_h3_uniform=numeric_means(row.get("l12_h3_uniform", {}) for row in items),
            layers={layer: numeric_means(r["layers"][layer] for r in items) for layer in items[0]["layers"]},
        )
        for name, items in groups.items()
    }


def paired_rows(rows):
    result = []
    for key in ("pair_id", "move_id"):
        pairs = defaultdict(list)
        for row in rows:
            if row.get(key):
                pairs[(row["split"], row[key])].append(row)
        for (split, group), members in pairs.items():
            if len(members) != 2:
                raise ValueError(f"{key}={group} requires exactly two rows")
            a, b = members
            if a["parent"] != b["parent"] or a["scene"] != b["scene"]:
                raise ValueError(f"Pair spans parents or scenes: {group}")
            if key == "pair_id" and (a["rgb_hash"] != b["rgb_hash"] or a.get("wrist_hash") != b.get("wrist_hash")
                                     or a["task"] == b["task"]):
                raise ValueError(f"Query pair must share exact pixels and have distinct tasks: {group}")
            if key == "move_id" and (a["task"] != b["task"] or a["target"] != b["target"]):
                raise ValueError(f"Displacement pair changes the task or target: {group}")
            item = dict(
                id=group,
                kind=key,
                parent=a["parent"],
                scene=a["scene"],
                split=split,
                members=[a["id"], b["id"]],
            )
            if "motion" in a and "motion" in b:
                item.update(motion_pair_metrics(a["motion"], b["motion"]))
            if "hit10" in a["current"] and "hit10" in b["current"]:
                item["both_hit10"] = a["current"]["hit10"] * b["current"]["hit10"]
                for name in ("peak3_hit10", "argmax_hit10"):
                    if name in a["current"] and name in b["current"]:
                        item[f"both_{name}"] = a["current"][name] * b["current"][name]
                if "roi_mass_top1" in a["current"] and "roi_mass_top1" in b["current"]:
                    item["both_roi_mass_top1"] = a["current"]["roi_mass_top1"] * b["current"]["roi_mass_top1"]
                if key == "move_id" and "true_center_xy" in a["current"] and "true_center_xy" in b["current"]:
                    truth = np.subtract(b["current"]["true_center_xy"], a["current"]["true_center_xy"])
                    pred = np.subtract(
                        b["current"].get("predicted_center_xy", b["current"]["peak3_xy"]),
                        a["current"].get("predicted_center_xy", a["current"]["peak3_xy"]),
                    )
                    item["displacement_error_px"] = float(np.linalg.norm(pred - truth))
            result.append(item)
    return result


def motion_pair_metrics(a, b):
    """Trajectories, endpoints and target centers all use letterboxed model pixels."""
    ta, tb = np.asarray(a["trace_px"]), np.asarray(b["trace_px"])
    shift = tb - ta
    result = dict(last4_displacement_px=float(np.linalg.norm(shift[-4:], axis=1).mean()))
    if "start_px" in a:
        result["start_drift_px"] = float(np.linalg.norm(np.subtract(b["start_px"], a["start_px"])))
        end_shift = np.subtract(b["end_px"], a["end_px"])
        result["end_displacement_px"] = float(np.linalg.norm(end_shift))
    boxes = [row.get("target_box_px") for row in (a, b)]
    if all(box is not None for box in boxes) and not np.array_equal(*boxes):
        def distance(trace, box):
            points = trace[-4:]
            delta = np.maximum(np.maximum(box[:2] - points, points - box[2:]), 0.)
            return np.linalg.norm(delta, axis=-1).mean()

        box_a, box_b = boxes
        result["last4_both_closer_to_own_target"] = float(distance(ta, box_a) < distance(ta, box_b)
                                                          and distance(tb, box_b) < distance(tb, box_a))
    if a["target_center_px"] is None or b["target_center_px"] is None:
        return result
    ca, cb = np.asarray(a["target_center_px"]), np.asarray(b["target_center_px"])
    length = np.linalg.norm(cb - ca)
    if length < 1e-6:
        return result  # A stationary target still provides a displacement control.
    direction = (cb - ca) / length
    toward = float((shift[-4:] @ direction).mean())
    result.update(last4_toward_b_px=toward, last4_direction_correct=float(toward > 0))
    if "end_px" in a:
        result["end_toward_b_px"] = float(end_shift @ direction)
    return result


def heat_overlay(rgb, weights):
    raw = np.asarray(weights, dtype=np.float32)
    side = math.isqrt(raw.size)
    grid = Image.fromarray(raw.reshape(side, side)).resize(rgb.size, Image.Resampling.BILINEAR)
    level = np.asarray(grid)
    strength = (0.8 * (level - level.min()) / max(float(np.ptp(level)), 1e-12))[..., None]
    return Image.fromarray(
        (np.asarray(rgb) * (1 - strength) + np.array([255, 65, 0]) * strength).astype(np.uint8)
    )


def draw_panel(path, rgb, row, current, subject, layers, labels, pieces, alpha, mass):
    size = 224
    rgb_title = "RGB: red=Current" if subject is None else "RGB: red=Current, blue=Subject"
    tiles = [(rgb_title, None), ("Current", current)]
    if subject is not None:
        tiles.append(("Subject", subject))
    if len(alpha):
        tiles.append(("Text token weights", None))
    tiles += [
        (f"L{label}   image mass {m:.3f}", values)
        for label, values, m in zip(labels, layers, mass, strict=True)
    ]
    width = size * 4

    wrapped = textwrap.wrap(row["task"], width=105)
    # All effective tokenizer pieces, monotonically ordered. Display strongest per-input weights in title.
    best = np.argsort(-alpha)[:8]
    caption = "alpha: " + "   ".join(f"{pieces[i]} {alpha[i]:.2f}" for i in best)
    header = (len(wrapped) + 2) * 18 + 8
    panel = Image.new("RGB", (width, header + math.ceil(len(tiles) / 4) * (size + 22)), "#17202a")
    draw = ImageDraw.Draw(panel)

    def safe(t):
        return t.encode("ascii", "backslashreplace").decode("ascii")

    draw.multiline_text((6, 4), safe("\n".join([row["id"], *wrapped, caption])), fill="white", spacing=4)
    for i, (name, values) in enumerate(tiles):
        x, y = i % 4 * size, header + i // 4 * (size + 22)
        tile = rgb.copy() if values is None else heat_overlay(rgb, values)
        if i == 0:
            overlays = ImageDraw.Draw(tile)
            for field, color in [("normalized_box", "red"), ("normalized_subject_box", "cyan")]:
                if row.get(field) is not None:
                    xy = (np.asarray(row[field]) * tile.width).tolist()
                    overlays.rectangle(xy, outline=color, width=2)
        elif name == "Text token weights":
            tile = Image.new("RGB", (size, size), "#17202a")
            chart = ImageDraw.Draw(tile)
            for j, k in enumerate(best):
                y0 = 8 + j * 25
                chart.text((3, y0), safe(pieces[k])[:12], fill="white")
                chart.rectangle((91, y0, 91 + float(alpha[k] / alpha.max()) * 91, y0 + 9), fill="#6ad7df")
                chart.text((185, y0), f"{alpha[k]:.2f}", fill="white")
        panel.paste(tile.resize((size, size)), (x, y + 20))
        draw.text((x + 6, y + 3), name, fill="white")
    panel.save(path)


def historical_maps(prediction, side, decoder, precision):
    """Deterministic display maps. A regressed box and VAE luminance are distinct from native attention."""
    if "bbox_xyxy" in prediction and "subject_bbox_xyxy" in prediction:
        boxes = [map_array(prediction[k]) for k in ("bbox_xyxy", "subject_bbox_xyxy")]
        maps = np.stack([box_overlap(box, side) for box in boxes])
        kind = "regressed_bbox"
    else:
        if decoder is None or decoder.module is None:
            raise ValueError("Historical VAE needs its saved readout and matching FLUX VAE")
        anchors = torch.stack([prediction["grounding"], prediction["subject_grounding"]], dim=2)
        rgb, _ = decoder.decode(dict(grounding=anchors), None, np.array([True, False]), precision)
        intensity = torch.from_numpy(rgb[0].astype(np.float32)).mean(-1)[:, None]
        maps = torch.nn.functional.adaptive_avg_pool2d(intensity, (side, side)).flatten(1).numpy()
        boxes = None
        kind = "vae_canvas_mean_rgb"
    # A collapsed box or black canvas has no location.
    mass = maps.sum(1, keepdims=True)
    return maps / mass.clip(1e-12), boxes, kind


def rgb_hash(image):
    return hashlib.sha256(np.array(image.shape, dtype="<i8").tobytes() + image.tobytes()).hexdigest()


def wrist_for_case(case, image, paths, reader, mode):
    episode = case.get("episode_id", case["parent"])
    if episode not in paths or type(case.get("frame")) is not int:
        raise ValueError(f"{case['id']}: expected an indexed episode_id/parent and integer source frame")
    source = reader(paths[episode])
    frame = case["frame"]
    original = source.rgb("image", frame)
    if original is None or "wrist_image" not in source.meta["cameras"]:
        raise ValueError(f"{case['id']}: missing source camera at frame {frame}")
    same = np.array_equal(original, image)
    if same != (mode == "paired"):
        raise ValueError(f"{case['id']}: main RGB {'matches' if same else 'differs from'} source; "
                         f"use a manifest with only {mode} observations or evaluate with wrist removed")
    wrist = source.rgb("wrist_image", frame)
    if wrist is None:
        raise ValueError(f"{case['id']}: missing wrist observation at source frame {frame}")
    return wrist


def pick_place_timelines(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("timeline_id") is not None:
            groups[row["split"], row["parent"], row["timeline_id"]].append(row)
    result = []
    for (split, parent, name), seq in groups.items():
        seq.sort(key=lambda row: row["frame"])
        boundaries = {r["place_start_frame"] for r in seq}
        by_skill = {skill: {r["target"] for r in seq if r["skill"] == skill} for skill in ("Pick", "Place")}
        if (len(boundaries) != 1 or len({r["task"] for r in seq}) != 1 or
                len({r["frame"] for r in seq}) != len(seq) or any(len(ids) != 1 for ids in by_skill.values())):
            raise ValueError(f"Timeline {name} requires one task, one annotated Pick→Place transition and unique frames")
        boundary = boundaries.pop()
        source, receiver = by_skill["Pick"].pop(), by_skill["Place"].pop()
        if source == receiver or any(r["skill"] != ("Pick" if r["frame"] < boundary else "Place") for r in seq):
            raise ValueError(f"Timeline {name} has inconsistent Event targets or boundary")
        source_frame = receiver_frame = None
        for row in seq:
            distances = row["entity_error_px"]
            hits = [key for key in (source, receiver) if distances.get(key, math.inf) <= HIT_RADIUS_PX]
            if hits == [source] and source_frame is None:
                source_frame = row["frame"]
            if hits == [receiver] and source_frame is not None and receiver_frame is None:
                receiver_frame = row["frame"]
        result.append(dict(timeline_id=name, split=split, parent=parent, frames=len(seq), source=source,
                           receiver=receiver, place_start_frame=boundary, first_source_hit_frame=source_frame,
                           first_receiver_hit_after_source=receiver_frame,
                           switch_delay_frames=receiver_frame - boundary if receiver_frame is not None else None))
    return result


def evaluate(policy, manifest_path, output, *, panels=8, decoder=None, checkpoint=None,
             motion=False, interventions=False, wrist_mode=None, episode_index=None):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    space = manifest["metric_space"]
    if space not in ("source", "model"):
        raise ValueError("Declare metric_space as source or model for the 10px criterion")
    cases = manifest["rows"]
    if not cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Expected nonempty manifest with unique row IDs")
    dual = tuple(getattr(policy, "cameras", ("image",))) == ("image", "wrist_image")
    if wrist_mode is None:
        wrist_mode = "paired" if dual else "main_only"
    if ((dual and wrist_mode not in ("paired", "conflict", "removed"))
            or (not dual and wrist_mode != "main_only")):
        raise ValueError("Wrist evaluation requires a dual-view checkpoint")
    paths = {}
    source_reader = lru_cache(maxsize=4)(EpisodeReader)
    if wrist_mode in ("paired", "conflict"):
        if episode_index is None:
            raise ValueError("Supply --episode-index to verify the source of paired or conflicting wrist images")
        index = Path(episode_index)
        paths = {row["episode_id"]: index.parent / row["path"] for row in json.loads(index.read_text())["episodes"]}
    output.mkdir(parents=True, exist_ok=False)
    panel_path = output / "panels"
    if panels:
        panel_path.mkdir()
    digest = hashlib.sha256(manifest_path.read_bytes())
    size = policy.model.config.image_size
    planner = getattr(policy.model, "training_stage", None) == "motion_planner"
    motion = motion or interventions
    if motion and not planner:
        raise ValueError("Motion trajectory metrics require a Motion Planner checkpoint")
    labels = list(policy.model.config.attention_layers) if decoder is None else []
    rows, maps, data_hashes = [], [], {}
    wrist_digest = hashlib.sha256()
    for i, case in enumerate(cases):
        image_path = manifest_path.parent / case["image"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        main_hash = rgb_hash(image)
        wrist = wrist_for_case(case, image, paths, source_reader, wrist_mode) if wrist_mode in ("paired", "conflict") else None
        views = [image, wrist] if dual else [image]
        pred = (policy.features(image, case["task"], attention_diagnostics=True) if planner
                else policy.infer(views, case["task"]))
        plan = policy.read_plan(pred) if motion else None
        legacy_boxes = None
        if "current_map" in pred:
            current, layers = map_array(pred["current_map"]), map_array(pred["layer_maps"])
            subject = map_array(pred["subject_map"]) if "subject_map" in pred else None
            side = math.isqrt(current.size)
            mask = map_array(pred["text_mask"]).astype(bool)
            alpha = map_array(pred["alpha"])[mask]
            raw_mass = map_array(pred["visual_attention_mass"])
            mass = (raw_mass[..., mask] * alpha).sum(-1).mean(1)
            readout = "native_multilayer_attention"
        else:
            side = size // policy.model.config.patch_size
            (current, subject), legacy_boxes, readout = historical_maps(pred, side, decoder, policy.precision)
            layers, labels = [], []
            mass = alpha = mask = np.array([], dtype=np.float32)
        boxes = {key: normalized_image_boxes(box, image.shape, size)[0] for key, box in case["boxes"].items()}
        target = case.get("target")
        box = None if target is None else boxes[target]
        coverage = {k: box_overlap(v, side) for k, v in boxes.items()}
        for k, filename in case.get("entity_masks", {}).items():
            path = manifest_path.parent / filename
            coverage[k] = mask_overlap(path, image.shape, size, side)
            data_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        roi = np.zeros_like(current) if box is None else coverage[target]
        others = [v for k, v in coverage.items() if k != target and v.any()]
        gripper = None
        subject_roi = None
        if subject is not None:
            if case.get("subject_box") is not None:
                gripper = normalized_image_boxes(case["subject_box"], image.shape, size)[0]
            subject_roi = np.zeros_like(subject) if gripper is None else box_overlap(gripper, side)
        for k, region in [("current_mask", roi), ("subject_mask", subject_roi)]:
            if case.get(k):
                mp = manifest_path.parent / case[k]
                if region is not None:
                    region[:] = mask_overlap(mp, image.shape, size, side)
                # Hash the full locked manifest's inputs even when this policy only predicts Current.
                data_hashes[case[k]] = hashlib.sha256(mp.read_bytes()).hexdigest()
        explicit_center = case.get("center_xy")
        if explicit_center is not None:
            explicit_center = normalized_image_boxes(explicit_center * 2, image.shape, size)[0, :2]
        args = (box, roi, others, image.shape, size, space, explicit_center)
        result = {
            k: case.get(k)
            for k in ("id", "split", "parent", "scene", "task", "target", "skill", "pair_id", "move_id", "frame")
        }
        result.update(
            case=case.get("case", "native"),
            rgb_hash=main_hash,
            source_shape=list(image.shape[:2]),
            normalized_box=box.tolist() if box is not None else None,
            current=map_score(current, *args),
            layers={
                str(label): dict(map_score(layer, *args), visual_mass=float(m))
                for label, layer, m in zip(labels, layers, mass, strict=True)
            },
        )
        if dual:
            result["wrist_hash"] = rgb_hash(wrist) if wrist is not None else None
            wrist_digest.update(json.dumps([case["id"], result["wrist_hash"]]).encode())
            wrist_mass = (map_array(pred["wrist_attention_mass"])[..., mask] * alpha).sum(-1).mean(1)
            result["current"].update(main_attention_mass=float(mass.mean()), wrist_attention_mass=float(wrist_mass.mean()))
            for layer, value in zip(labels, wrist_mass, strict=True):
                result["layers"][str(layer)]["wrist_mass"] = float(value)
        if case.get("timeline_id") is not None:
            if type(case.get("place_start_frame")) is not int or type(case.get("frame")) is not int:
                raise ValueError("Timeline rows require frame and annotated place_start_frame")
            point = result["current"].get("peak3_xy")
            result.update(timeline_id=case["timeline_id"], place_start_frame=case["place_start_frame"],
                          entity_error_px={k: float(np.linalg.norm(point - metric_point(center_box(b), image.shape, size, space)))
                                           for k, b in boxes.items() if point is not None and coverage[k].any()})
        if plan is not None:
            center = explicit_center if explicit_center is not None else center_box(box) if box is not None else None
            result["motion"] = dict(trace_px=(map_array(plan["trace"]) * size).tolist(),
                                    skill_id=plan["skill_id"].item(),
                                    target_box_px=(box * size).tolist() if box is not None else None,
                                    target_center_px=(center * size).tolist() if center is not None else None)
            for role in ("start", "end"):
                if role in plan:
                    result["motion"][role + "_px"] = (map_array(plan[role]) * size).tolist()
        if subject is not None:
            result.update(normalized_subject_box=gripper.tolist() if gripper is not None else None,
                          subject=map_score(subject, gripper, subject_roi, [], image.shape, size, space))
        if legacy_boxes is not None:
            for name, pred_box in zip(("current", "subject"), legacy_boxes, strict=True):
                stats = result[name]
                stats["predicted_box"] = pred_box.tolist()
                if "true_center_xy" in stats:
                    # Native BBox readout is its predicted center, not a peak on a flat box.
                    pred_point = metric_point(center_box(pred_box), image.shape, size, space)
                    distance = float(np.linalg.norm(pred_point - stats["true_center_xy"]))
                    stats.update(
                        predicted_center_xy=pred_point.tolist(),
                        center_error_px=distance,
                        hit10=float(distance <= HIT_RADIUS_PX),
                    )
        if 12 in labels and "native_attention" in pred and pred["native_attention"].shape[2] > 3:
            # Historic single-head diagnostic shares this checkpoint and this exact prompt.
            head = pred["native_attention"][0, labels.index(12), 3].detach().float().cpu().numpy()[mask]
            denom = head.sum(-1, keepdims=True)
            single = (head / denom.clip(1e-12)).mean(0)
            result["l12_h3_uniform"] = dict(map_score(single, *args), visual_mass=float(denom.mean()))
        pieces = []
        result["alpha_stats"] = {}
        if target is None:
            result["current"]["absent_peak"] = float(current.max())
        if len(alpha):
            tokenizer = policy.transform.tokenizer if planner else policy.tokenizer
            ids, _ = tokenizer(case["task"])
            sp = getattr(tokenizer, "tokenizer", None)
            pieces = [
                sp.id_to_piece(int(j)).replace("▁", "_") if sp is not None else str(j)
                for j in ids[: len(mask)][mask]
            ]
            entropy = float(-(alpha * np.log(alpha.clip(1e-12))).sum())
            result["alpha_stats"] = dict(
                entropy=entropy, normalized_entropy=entropy / math.log(len(alpha)) if len(alpha) > 1 else 0.0
            )
            result["alpha"] = dict(pieces=pieces, weights=alpha.tolist())
            result["current"]["visual_mass"] = float(mass.mean())
            if target is None:
                result["current"]["absent_visual_mass"] = float(mass.mean())
        if i < panels or (
            panels
            and not result["current"].get("hit10", 1)
            and sum(r.get("failure_panel", False) for r in rows) < panels
        ):
            model_rgb = Image.fromarray(resize_image(image, size))
            filename = f"{i:05d}.jpg"
            draw_panel(
                panel_path / filename,
                model_rgb,
                result,
                current,
                subject,
                layers,
                labels,
                pieces,
                alpha,
                mass,
            )
            result["panel"] = "panels/" + filename
            result["failure_panel"] = result["current"].get("hit10") == 0
        rows.append(result)
        maps.append(current.astype(np.float32))
        data_hashes[case["image"]] = main_hash
    for path, sha in sorted(data_hashes.items()):
        digest.update(json.dumps([path, sha]).encode())
    pairs = paired_rows(rows)
    if interventions:
        from scripts.analysis.motion_planner import plan_pair
        by_id = {row["id"]: (case, row) for case, row in zip(cases, rows, strict=True)}
        for pair in pairs:
            if pair["kind"] != "pair_id":
                continue
            (case, a), (_, b) = (by_id[k] for k in pair["members"])
            centers = [r["motion"]["target_center_px"] for r in (a, b)]
            if (a["target"] == b["target"] or any(c is None for c in centers)
                    or (a["skill"] and b["skill"] and a["skill"] != b["skill"])):
                continue
            with Image.open(manifest_path.parent / case["image"]) as src:
                rgb = np.asarray(src.convert("RGB"))
            result = plan_pair(policy, rgb, [a["task"], b["task"]], regions=(None, None),
                               observed=[], target_centers=np.asarray(centers) / size)
            pair["same_predicted_skill"] = result["same_predicted_skill"]
            conditions = result["conditions"]
            pair["interventions"] = {k: conditions[k] for k in ("prior_only", "end_swap") if k in conditions}
    pair_summary = defaultdict(list)
    for pair in pairs:
        pair_summary[pair["kind"] + "/" + pair["split"]].append(pair)
        pair_summary[pair["kind"] + "/" + pair["split"] + "/" + pair["scene"]].append(pair)
    report = dict(
        protocol=manifest["protocol"],
        readout=readout,
        primary_center_readout="bbox_center" if readout == "regressed_bbox" else "peak3",
        center_reference=dict(current="manifest center_xy when supplied, otherwise bbox center"),
        checkpoint=str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        input_sha256=digest.hexdigest(),
        wrist_mode=wrist_mode,
        metric_space=space,
        model_size=size,
        layer_ids=labels,
        groups=summarize(rows),
        pair_summary={k: numeric_means(v) for k, v in pair_summary.items()},
        pairs=pairs,
        rows=rows,
    )
    if planner:
        report["planner_version"] = getattr(policy.model.config, "planner_version", "v1")
    if dual:
        report["wrist_sha256"] = wrist_digest.hexdigest()
    timelines = pick_place_timelines(rows)
    if timelines:
        report["timelines"] = timelines
    if "subject" in rows[0]:
        report["center_reference"]["subject"] = "bbox center"
    np.save(output / "current_maps.npy", np.stack(maps), allow_pickle=False)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-path")
    p.add_argument("--pretrained-path")
    p.add_argument("--wrist-mode", choices=("paired", "removed", "conflict"), help="Dual-view only; default paired")
    p.add_argument("--episode-index", type=Path, help="Grounding index for exact source-frame verification")
    p.add_argument("--vae-path", help="Matched VAE for historical decoded-canvas readouts")
    p.add_argument("--planner", action="store_true", help="Load a Motion Planner checkpoint and score its native Current maps")
    p.add_argument("--panels", type=int, default=8)
    p.add_argument("--motion", action="store_true", help="also score the predicted plan on the locked cases")
    p.add_argument("--interventions", action="store_true", help="run fixed-route Prior and End swaps on same-RGB pairs")
    args = p.parse_args()
    if args.planner:
        from openpi.policies.motion_planner import create_motion_planner_policy

        load_policy = create_motion_planner_policy
    else:
        load_policy = create_current_grounding_policy
    policy = load_policy(
        args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        pretrained_path=args.pretrained_path,
    )
    decoder = None
    if policy.model.training_stage == "grounding" and not policy.model.config.subject_bbox:
        from scripts.analysis.attention_viewer import load_config
        from scripts.analysis.predictions import VisualReadout

        decoder = VisualReadout(args.checkpoint, load_config(args.checkpoint), args.device, args.vae_path)
    result = evaluate(
        policy, args.manifest, args.output, panels=args.panels, decoder=decoder, checkpoint=args.checkpoint,
        motion=args.motion, interventions=args.interventions, wrist_mode=args.wrist_mode, episode_index=args.episode_index
    )
    print(json.dumps(result["groups"]["all"]["current"]))


if __name__ == "__main__":
    main()
