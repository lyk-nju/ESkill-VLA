"""Compare runs on identical locked observations, resampling matched parent trajectories."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def parent_bootstrap(rows, draws, seed):
    """Micro-average paired differences, resampling entire parents with replacement."""
    groups = defaultdict(list)
    for parent, value in rows:
        groups[parent].append(value)
    keys = sorted(groups)
    sums = np.asarray([sum(groups[k]) for k in keys])
    counts = np.asarray([len(groups[k]) for k in keys])
    result = dict(delta=float(sums.sum() / counts.sum()), n=len(rows), parents=len(keys))
    if len(keys) > 1:
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(draws):
            sample = rng.integers(len(keys), size=len(keys))
            values.append(sums[sample].sum() / counts[sample].sum())
        result["ci95"] = np.quantile(values, [0.025, 0.975]).tolist()
    return result


def compare(reports, reference, *, draws=2000, seed=42):
    base = reports[reference]
    result = {}
    for name, r in reports.items():
        if r["metric_space"] == "model" and r["model_size"] != base["model_size"]:
            raise ValueError("Model-pixel distances require a shared model canvas")
        if r["input_sha256"] != base["input_sha256"]:
            raise ValueError("Comparisons require byte-identical manifests and RGB/labels")
        modes = {r.get("wrist_mode", "main_only"), base.get("wrist_mode", "main_only")}
        if modes == {"paired", "conflict"}:
            raise ValueError("Evaluate matched and conflicting camera inputs separately")
        if modes in ({"paired"}, {"conflict"}) and r["wrist_sha256"] != base["wrist_sha256"]:
            raise ValueError("Wrist images differ between runs with the same wrist protocol")
        if name == reference:
            continue
        groups = defaultdict(list)
        for kind, keys in [
            (
                "rows",
                (
                    "current.hit10",
                    "current.center_error_px",
                    "current.peak3_hit10",
                    "current.peak3_error_px",
                    "current.argmax_hit10",
                    "current.argmax_error_px",
                    "current.mass_margin",
                    "current.roi_mass",
                    "current.absent_peak",
                    "subject.hit10",
                    "subject.peak3_hit10",
                    "subject.peak3_error_px",
                    "subject.argmax_hit10",
                    "subject.argmax_error_px",
                    "subject.roi_mass",
                ),
            ),
            ("pairs", ("both_hit10", "both_peak3_hit10", "both_argmax_hit10",
                       "both_roi_mass_top1", "displacement_error_px")),
        ]:
            others = {(x["split"], x["id"], x.get("kind")): x for x in r[kind]}
            for a in base[kind]:
                b = others[a["split"], a["id"], a.get("kind")]
                for key in keys:
                    av, bv = a, b
                    for part in key.split("."):
                        av = av.get(part) if av is not None else None
                        bv = bv.get(part) if bv is not None else None
                    if av is None or bv is None:
                        continue
                    metric = a["kind"] + "." + key if kind == "pairs" else key
                    for subset in (
                        "all",
                        "split/" + a["split"],
                        "split_scene/" + a["split"] + "/" + a["scene"],
                    ):
                        groups[subset, metric].append((a["parent"], bv - av))
        result[name] = {}
        for (subset, metric), rows in groups.items():
            result[name].setdefault(subset, {})[metric] = parent_bootstrap(rows, draws, seed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, nargs="+", metavar="NAME=DIR")
    parser.add_argument("--reference", default="E0")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("draws must be positive")
    reports = {}
    for entry in args.run:
        name, folder = entry.split("=", 1)
        reports[name] = json.loads((Path(folder) / "metrics.json").read_text())
    base = reports[args.reference]
    result = dict(
        protocol=base["protocol"],
        input_sha256=base["input_sha256"],
        reference=args.reference,
        bootstrap_draws=args.draws,
        seed=args.seed,
        paired=compare(reports, args.reference, draws=args.draws, seed=args.seed),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "paired_deltas.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# {base['protocol']}",
        "",
        f"Pixel space: {base['metric_space']}. The primary readout is 3×3 peak for spatial maps "
        "and the regressed center for BBox models. ROI mass measures overlap with the annotated region.",
        "",
        "| Run | Split | Parents | Queries | Current primary hit ≤10 px | Current argmax hit ≤10 px | Current error, px | Current ROI mass | Subject primary hit ≤10 px | Subject argmax hit ≤10 px | Both queries hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in reports.items():
        for group, g in r["groups"].items():
            if not group.startswith("split/"):
                continue
            split = group.split("/", 1)[1]

            def mean(section, key, fmt):
                return format(section[key]["mean"], fmt) if key in section else "—"

            pair = r["pair_summary"].get("pair_id/" + split, {})
            lines.append(
                f"| {name} | {split} | {g['parents']} | {g['queries']} | "
                f"{mean(g['current'], 'hit10', '.1%')} | {mean(g['current'], 'argmax_hit10', '.1%')} | "
                f"{mean(g['current'], 'center_error_px', '.1f')} | {mean(g['current'], 'roi_mass', '.1%')} | "
                f"{mean(g.get('subject', {}), 'hit10', '.1%')} | {mean(g.get('subject', {}), 'argmax_hit10', '.1%')} | "
                f"{mean(pair, 'both_hit10', '.1%')} |"
            )
    (args.output / "results.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
