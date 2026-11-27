"""Fit a detached CurrentMap -> center/offset/size probe from visual_primitive_eval outputs."""

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmark import unified_benchmark

unified_benchmark()
from eskill_bench import ROOT, revision
from eskill_bench.visual import center_box, metric_point, numeric_means


class MapBoxProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 5, 1),
        )

    def forward(self, spatial):
        side = math.isqrt(spatial.shape[-1])
        # Uniform attention averages to 1 in this parametrization.
        values = self.network(spatial.detach().reshape(-1, 1, side, side) * side**2)
        return values.flatten(2)

    def loss(self, spatial, boxes):
        pred = self(spatial)
        side = math.isqrt(spatial.shape[-1])
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5 * side
        cells = center.long().clamp(0, side - 1)
        index = cells[:, 1] * side + cells[:, 0]
        local = pred[torch.arange(len(index)), 1:, index].sigmoid()
        target = torch.cat((center - cells, boxes[:, 2:] - boxes[:, :2]), -1)
        return F.cross_entropy(pred[:, 0], index) + F.smooth_l1_loss(local, target, beta=0.05)

    @torch.no_grad()
    def boxes(self, spatial):
        pred = self(spatial)
        side = math.isqrt(spatial.shape[-1])
        cell = pred[:, 0].argmax(-1)
        offset, size = pred[torch.arange(len(cell)), 1:, cell].sigmoid().split(2, -1)
        center = (torch.stack((cell % side, cell // side), -1) + offset) / side
        return torch.cat((center - size / 2, center + size / 2), -1).clamp(0, 1)


def check_partitions(rows):
    fit_parents = {r["parent"] for r in rows if r["split"] == "Fit"}
    check_parents = {r["parent"] for r in rows if r["split"] == "Check"}
    locked_parents = {r["parent"] for r in rows if r["split"] not in ("Fit", "Check")}
    if not fit_parents or not check_parents:
        raise ValueError("Probe training requires the original Fit and Check partitions")
    if fit_parents & check_parents or (fit_parents | check_parents) & locked_parents:
        raise ValueError("Probe Fit/Check/locked evaluation share parent trajectories")


def iou(a, b):
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(overlap / max(1e-12, np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - overlap))


def fit_probe(run, output, steps, batch, seed):
    report = json.loads((run / "metrics.json").read_text())
    rows = report["rows"]
    check_partitions(rows)
    x = torch.from_numpy(np.load(run / "current_maps.npy", allow_pickle=False))
    usable = [i for i, r in enumerate(rows) if r["normalized_box"] is not None and "hit10" in r["current"]]
    boxes = torch.tensor([r["normalized_box"] or [0.0] * 4 for r in rows], dtype=torch.float)
    fit = torch.tensor([i for i in usable if rows[i]["split"] == "Fit"], dtype=torch.long)
    check = torch.tensor([i for i in usable if rows[i]["split"] == "Check"], dtype=torch.long)
    if min(len(fit), len(check), steps, batch) < 1:
        raise ValueError("Need positive training budget and valid targets in both Fit and Check")
    torch.manual_seed(seed)
    probe = MapBoxProbe()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3)
    best = float("inf")
    for step in range(1, steps + 1):
        ind = fit[torch.randint(len(fit), (min(batch, len(fit)),))]
        optimizer.zero_grad()
        probe.loss(x[ind], boxes[ind]).backward()
        optimizer.step()
        if step % max(1, steps // 10) == 0 or step == steps:
            with torch.no_grad():
                score = sum(float(probe.loss(x[g], boxes[g])) * len(g) for g in check.split(batch)) / len(
                    check
                )
            if score < best:
                best = score
                weights = {k: v.clone() for k, v in probe.state_dict().items()}
                best_step = step
    probe.load_state_dict(weights)
    output.mkdir(parents=True, exist_ok=False)
    predictions = torch.cat([probe.boxes(group) for group in x.split(batch)]).numpy()
    by_split = {}
    records = []
    for i in usable:
        row = rows[i]
        box = predictions[i]
        pred = metric_point(
            center_box(box), row["source_shape"], report["model_size"], report["metric_space"]
        )
        error = float(np.linalg.norm(pred - row["current"]["true_center_xy"]))
        record = dict(
            id=row["id"],
            split=row["split"],
            parent=row["parent"],
            normalized_box=box.tolist(),
            center_error_px=error,
            iou=iou(box, boxes[i].numpy()),
            hit10=float(error <= 10),
            native_peak3_xy=row["current"]["peak3_xy"],
            native_hit10=row["current"]["hit10"],
        )
        by_split.setdefault(row["split"], []).append(record)
        records.append(record)
    torch.save(
        dict(
            model=weights,
            selected_step=best_step,
            seed=seed,
            fit_loss_type="center CE + offset-size smooth L1",
            source_sha256=report["input_sha256"],
        ),
        output / "probe.pt",
    )
    summary = dict(
        experiment=revision(Path(__file__).resolve().parents[2]), benchmark=revision(ROOT), source_revisions=report.get("revisions"),
        fit_samples=len(fit),
        check_samples=len(check),
        selected_step=best_step,
        check_loss=best,
        seed=seed,
        batch=batch,
        steps=steps,
        by_split={k: numeric_means(v) for k, v in by_split.items()},
        rows=records,
    )
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    fit_probe(args.run, args.output, args.steps, args.batch, args.seed)


if __name__ == "__main__":
    main()
