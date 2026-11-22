"""Evaluate saved Stage V policies on a fixed same-image/perturbed-image manifest.

JSON fields: protocol, metric_space ("source" or square "model" pixels), rows.
Each row has id, split, parent (trajectory), scene, image (path relative to manifest),
task, boxes (entity -> raw-image xyxy pixel edges), target (entity ID or null for absent).
Optional: entity_masks (entity ID -> mask path), current_mask and subject_mask (binary image paths), subject_box (raw xyxy),
case, skill, pair_id (two different queries on identical RGB), move_id (two actual
layouts in reference/modified order), center_xy (original E0 target point, raw pixels).
Fit/Check/A/B/C/D names and sample membership come from the caller's manifest.
Use --panels 0 to omit PNGs. All maps are computed before accessing ground truth.
"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw
import torch

from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.transforms import normalized_image_boxes, resize_geometry, resize_image


def map_array(tensor):
    return tensor[0].detach().float().cpu().numpy()


def grid_centers(side):
    y, x = np.indices((side, side))
    return np.stack((x + 0.5, y + 0.5), axis=-1) / side


def peak3(weights):
    """Center of mass in the highest-mass 3x3 patch window, normalized model xy."""
    side = math.isqrt(weights.size)
    value = weights.reshape(side, side)
    pad = np.pad(value, 1)
    windows = np.lib.stride_tricks.sliding_window_view(pad, (3, 3)).sum((-1, -2))
    y, x = np.unravel_index(windows.argmax(), windows.shape)
    region = np.s_[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
    return np.average(grid_centers(side)[region].reshape(-1, 2), weights=value[region].ravel(), axis=0)


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
    height, width = shape[:2]
    nh, nw, top, left = resize_geometry(shape, size)
    return (point * size - [left, top]) * [width / nw, height / nh]


def center_box(box):
    return (np.asarray(box[:2]) + box[2:]) / 2


def map_score(weights, box, roi, distractors, shape, size, space, gt_center=None):
    if not weights.any():
        return dict(peak=0.0, hit10=0.0) if box is not None and roi.any() else dict(peak=0.0)
    peak = peak3(weights)
    point = metric_point(peak, shape, size, space)
    plain_peak = metric_point(
        grid_centers(math.isqrt(len(weights))).reshape(-1, 2)[weights.argmax()], shape, size, space
    )
    result = dict(peak3_xy=point.tolist(), argmax_xy=plain_peak.tolist(), peak=float(weights.max()))
    others = [float(weights @ other) for other in distractors]
    if others:
        result["max_distractor_mass"] = max(others)
    if box is None or roi.sum() == 0:
        return result
    target = metric_point(gt_center if gt_center is not None else center_box(box), shape, size, space)
    delta = float(np.linalg.norm(point - target))
    probability = roi / roi.sum()
    positive = probability > 0
    kl = (probability[positive] * np.log(probability[positive] / weights[positive].clip(1e-12))).sum()
    mass = float(weights @ roi)
    result.update(
        center_error_px=delta,
        hit10=float(delta <= 10.0),
        roi_mass=mass,
        kl=float(kl),
        true_center_xy=target.tolist(),
        argmax_error_px=float(np.linalg.norm(plain_peak - target)),
    )
    if others:
        result["mass_margin"] = mass - max(others)
        result["roi_mass_top1"] = float(mass > max(others))
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
            subject=numeric_means(row["subject"] for row in items),
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
            if key == "pair_id" and (a["rgb_hash"] != b["rgb_hash"] or a["task"] == b["task"]):
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
            if "hit10" in a["current"] and "hit10" in b["current"]:
                item["both_hit10"] = a["current"]["hit10"] * b["current"]["hit10"]
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
    tiles = [
        ("RGB: red=Current, blue=Subject", None),
        ("Current", current),
        ("Subject", subject),
    ]
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
        elif i == 3 and len(alpha):
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


def evaluate(policy, manifest_path, output, *, panels=8, decoder=None, checkpoint=None):
    manifest_path, output = Path(manifest_path), Path(output)
    manifest = json.loads(manifest_path.read_text())
    space = manifest["metric_space"]
    if space not in ("source", "model"):
        raise ValueError("Declare metric_space as source or model for the 10px criterion")
    cases = manifest["rows"]
    if not cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("Expected nonempty manifest with unique row IDs")
    output.mkdir(parents=True, exist_ok=False)
    panel_path = output / "panels"
    if panels:
        panel_path.mkdir()
    digest = hashlib.sha256(manifest_path.read_bytes())
    size = policy.model.config.image_size
    labels = list(policy.model.config.attention_layers) if decoder is None else []
    rows, maps, data_hashes = [], [], {}
    for i, case in enumerate(cases):
        image_path = manifest_path.parent / case["image"]
        image = np.asarray(Image.open(image_path).convert("RGB"))
        rgb_hash = hashlib.sha256(np.array(image.shape, dtype="<i8").tobytes() + image.tobytes()).hexdigest()
        pred = policy.infer([image], case["task"])
        legacy_boxes = None
        if "current_map" in pred:
            current, subject, layers = [
                map_array(pred[k]) for k in ("current_map", "subject_map", "layer_maps")
            ]
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
        if case.get("subject_box") is not None:
            gripper = normalized_image_boxes(case["subject_box"], image.shape, size)[0]
        subject_roi = np.zeros_like(subject) if gripper is None else box_overlap(gripper, side)
        for k, region in [("current_mask", roi), ("subject_mask", subject_roi)]:
            if case.get(k):
                mp = manifest_path.parent / case[k]
                region[:] = mask_overlap(mp, image.shape, size, side)
                data_hashes[case[k]] = hashlib.sha256(mp.read_bytes()).hexdigest()
        explicit_center = case.get("center_xy")
        if explicit_center is not None:
            explicit_center = normalized_image_boxes(explicit_center * 2, image.shape, size)[0, :2]
        args = (box, roi, others, image.shape, size, space, explicit_center)
        result = {
            k: case.get(k)
            for k in ("id", "split", "parent", "scene", "task", "target", "skill", "pair_id", "move_id")
        }
        result.update(
            case=case.get("case", "native"),
            rgb_hash=rgb_hash,
            source_shape=list(image.shape[:2]),
            normalized_box=box.tolist() if box is not None else None,
            normalized_subject_box=gripper.tolist() if gripper is not None else None,
            current=map_score(current, *args),
            subject=map_score(subject, gripper, subject_roi, [], image.shape, size, space),
            layers={
                str(label): dict(map_score(layer, *args), visual_mass=float(m))
                for label, layer, m in zip(labels, layers, mass, strict=True)
            },
        )
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
                        hit10=float(distance <= 10),
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
            ids, _ = policy.tokenizer(case["task"])
            sp = getattr(policy.tokenizer, "tokenizer", None)
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
        data_hashes[case["image"]] = rgb_hash
    for path, sha in sorted(data_hashes.items()):
        digest.update(json.dumps([path, sha]).encode())
    pairs = paired_rows(rows)
    pair_summary = defaultdict(list)
    for pair in pairs:
        pair_summary[pair["kind"] + "/" + pair["split"]].append(pair)
        pair_summary[pair["kind"] + "/" + pair["split"] + "/" + pair["scene"]].append(pair)
    report = dict(
        protocol=manifest["protocol"],
        readout=readout,
        checkpoint=str(Path(checkpoint).resolve()) if checkpoint is not None else None,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        input_sha256=digest.hexdigest(),
        metric_space=space,
        model_size=size,
        layer_ids=labels,
        groups=summarize(rows),
        pair_summary={k: numeric_means(v) for k, v in pair_summary.items()},
        pairs=pairs,
        rows=rows,
    )
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
    p.add_argument("--vae-path", help="Matched VAE for historical decoded-canvas readouts")
    p.add_argument("--panels", type=int, default=8)
    args = p.parse_args()
    policy = create_current_grounding_policy(
        args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        pretrained_path=args.pretrained_path,
    )
    decoder = None
    if policy.model.training_stage != "visual_v4" and not policy.model.config.subject_bbox:
        from scripts.analysis.attention_viewer import load_config
        from scripts.analysis.predictions import VisualReadout

        decoder = VisualReadout(args.checkpoint, load_config(args.checkpoint), args.device, args.vae_path)
    result = evaluate(
        policy, args.manifest, args.output, panels=args.panels, decoder=decoder, checkpoint=args.checkpoint
    )
    print(json.dumps(result["groups"]["all"]["current"]))


if __name__ == "__main__":
    main()
