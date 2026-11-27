"""Recompute the locked counterfactual table using saved track coordinates."""

from collections import Counter
import csv
import json
from statistics import median

import numpy as np

from .assets import MOTION, read

RUNS = ("01/step_020000", "02", "03")
GROUPS = (
    "move_pick",
    "switch_pick",
    "random_layout_switch",
    "added_pick",
    "move_distractor",
)
FIELDS = ("direction_positive", "both_region_prefer", "current_both_hit10")


def score_pair(row, case, size):
    for key in ("id", "kind", "task", "suite", "stage"):
        if row[key] != case[key]:
            raise ValueError(f"{case['id']}: mismatched {key}")
    boxes = np.array(row["boxes"])
    np.testing.assert_allclose(boxes, np.array([case[s]["bbox"] for s in ("a", "b")]) / 256, atol=1e-7)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    traces = np.array([row[s]["trace"] for s in ("a", "b")])
    if traces.shape != (2, 8, 2) or not np.isfinite(traces).all():
        raise ValueError(f"{case['id']}: invalid trace")
    diff = centers[1] - centers[0]
    length = np.linalg.norm(diff)
    direction = diff / length if length > 1e-6 else None
    movement = (traces[1] - traces[0]) * size
    errors = [
        np.linalg.norm(np.array(row[s]["current_center"]) - c) * 256
        for s, c in zip(("a", "b"), centers, strict=True)
    ]
    np.testing.assert_allclose(errors, [row[s]["current_error_px"] for s in ("a", "b")], atol=1e-5)

    def distance(trace, box):
        gap = np.maximum(np.maximum(box[:2] - trace[-4:], trace[-4:] - box[2:]), 0)
        return np.linalg.norm(gap, axis=-1).min() * size

    prefer = all(distance(t, boxes[i]) + 1e-6 < distance(t, boxes[1 - i]) for i, t in enumerate(traces))
    prior = row["interventions"].get("prior_only")
    return dict(
        direction_positive=bool(direction is not None and (movement[-4:] @ direction).mean() > 0),
        both_region_prefer=bool(direction is not None and prefer),
        current_both_hit10=max(errors) <= 10,
        drift=float(np.linalg.norm(movement, axis=-1).mean()),
        prior=float(((np.asarray(prior["trace"]) - traces[0])[-4:] * size @ direction).mean())
        if prior and direction is not None
        else None,
    )


def reproduce(bundle):
    manifest = read(bundle.file(MOTION))
    cases, spec = manifest["pairs"], manifest["spec"]
    ids = [c["id"] for c in cases]
    if (
        len(set(ids)) != 202
        or Counter(c["kind"] for c in cases) != dict(zip(GROUPS, [48, 48, 48, 48, 10]))
        or spec["offsets"] != list(range(5, 41, 5))
        or spec["size"] != 224
    ):
        raise ValueError("Unexpected locked Motion population or time slots")
    versions = []
    for name in RUNS:
        prefix = "reference/motion/" + name + "/"
        run = read(bundle.file(prefix + "run.json"))
        if run["manifest_sha256"] != bundle.files[MOTION]["sha256"]:
            raise ValueError(f"{name}: wrong manifest")
        with bundle.file(prefix + "pairs.jsonl").open() as f:
            versions.append(
                [score_pair(json.loads(line), c, spec["size"]) for line, c in zip(f, cases, strict=True)]
            )
    with bundle.file("reference/motion/03/boundary_pair_cases.csv").open() as f:
        boundary = list(csv.DictReader(f))
    if [r["id"] for r in boundary] != ids:
        raise ValueError("Boundary table does not match pairs")
    groups = []
    for kind in GROUPS:
        idx = [i for i, c in enumerate(cases) if c["kind"] == kind]
        group = dict(kind=kind, pairs=len(idx))
        for field in FIELDS:
            group[field] = [int(sum(v[i][field] for i in idx)) for v in versions]
        group["trace_drift_median_px"] = [median(v[i]["drift"] for i in idx) for v in versions]
        group["prior_only_median_px"] = [
            median(p) if (p := [v[i]["prior"] for i in idx if v[i]["prior"] is not None]) else None
            for v in versions
        ]
        start, end, projected, swap = [], [], [], []
        for i in idx:
            r = boundary[i]

            def delta(role):
                return np.subtract(json.loads(r[role + "_b"]), json.loads(r[role + "_a"]))

            start.append(float(np.linalg.norm(delta("start").astype(np.float32))) * 224)
            end.append(float(np.linalg.norm(delta("end"))) * 224)
            boxes = np.array([cases[i][s]["bbox"] for s in ("a", "b")])
            axis = (boxes[1, :2] + boxes[1, 2:] - boxes[0, :2] - boxes[0, 2:]) / 2
            if np.linalg.norm(axis) > 1e-6:
                projected.append(float(delta("end") @ (axis / np.linalg.norm(axis))) * 224)
            if r["end_swap_trace_toward_target_px"]:
                swap.append(float(r["end_swap_trace_toward_target_px"]))
        group.update(start_drift_median_px=median(start), end_drift_median_px=median(end))
        for prefix, values in [("end", projected), ("end_swap", swap)]:
            if values:
                group[prefix + "_scored"] = len(values)
                group[prefix + "_positive"] = sum(v > 0 for v in values)
                group[prefix + "_projection_median_px"] = median(values)
        groups.append(group)
    actual = dict(
        model_order=["V1 frozen 20k", "V1 joint 20k", "Boundary V2 frozen 20k"],
        groups=groups,
        totals={k: [sum(g[k][i] for g in groups[:4]) for i in range(3)] for k in FIELDS},
    )
    expected = read(bundle.file("reference/motion/03/comparison.json"))
    for key in ("model_order", "totals"):
        if actual[key] != expected[key]:
            raise ValueError(f"Archived {key} differs")
    for got, gold in zip(groups, expected["groups"], strict=True):
        if got.keys() != gold.keys():
            raise ValueError(f"Fields differ in {got['kind']}")
        for k, x in gold.items():
            if isinstance(x, str) or (isinstance(x, list) and None in x):
                if got[k] != x:
                    raise ValueError(k)
            else:
                np.testing.assert_allclose(got[k], x, atol=1e-8, rtol=1e-10, err_msg=k)
    return actual
