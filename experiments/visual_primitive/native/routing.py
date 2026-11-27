"""Conditional E1: fit a phase router over frozen source/destination attention."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np

if __package__:
    from . import evaluate as ev
else:
    import evaluate as ev


STAGES = ("Pick", "Place")
ROLES = ("source", "destination")
METHODS = (*ROLES, "oracle")
THRESHOLD = 10.0
REGULARIZATION = 0.05
FEATURES_PER_ROLE = ("log_visual_mass", "norm_entropy", "peak_probability",
                     "centroid_x", "centroid_y", "local_x", "local_y", "spread")
FEATURES = tuple(f"{role}_{feature}" for role in ROLES for feature in FEATURES_PER_ROLE) + (
    "distribution_overlap", "delta_local_x", "delta_local_y",
)


def eligible(row):
    """Require the original full instruction and an unambiguous ordinary Event."""
    if row.get("condition") != "native_event" or row.get("stage") not in STAGES:
        return False
    spans = row.get("phrases", {})
    if any(spans.get(role) is None for role in METHODS):
        return False
    if spans["source"] == spans["destination"]:
        return False
    role = ROLES[STAGES.index(row["stage"])]
    return spans["oracle"] == spans[role]


def feature_vector(source, destination):
    """Observable map features only. No row, target, bbox, or Event enters this interface."""
    p, q = source["map"], destination["map"]

    def role_features(values):
        x = values["map"]
        center = x @ ev.CENTERS
        spread = np.sqrt(np.sum(x * np.square(ev.CENTERS - center).sum(-1))) / ev.SIZE
        return [np.log(max(values["mass"], 1e-12)), ev.entropy(x), float(x.max()),
                *(center / ev.SIZE), *(values["position"] / ev.SIZE), spread]

    overlap = np.sqrt(p * q).sum()
    delta = (destination["position"] - source["position"]) / ev.SIZE
    return np.asarray([*role_features(source), *role_features(destination), overlap, *delta], dtype=np.float64)


def fit_role_heads(run, dev):
    """Rank fixed local3 heads using role GT on the fit parents exclusively."""
    stats = {}
    signature = None
    for rows, names, maps, mass, valid in ev.batches(run, "dev", dev):
        signature = maps.shape[2:4]
        n_heads = int(np.prod(signature))
        for role in ROLES:
            subset = np.where([row["split"] == "fit" and eligible(row) and row.get(role + "_bbox") is not None
                               for row in rows])[0]
            if not len(subset):
                continue
            q = names.index(role)
            spatial = maps[subset, q].reshape(len(subset), n_heads, ev.PATCHES)
            visual_mass = mass[subset, q].reshape(len(subset), n_heads)
            spatial = ev.normalize_maps(spatial, visual_mass)
            xy = ev.decode_maps(spatial.reshape(-1, ev.PATCHES), "local3").reshape(len(subset), n_heads, 2)
            good = valid[subset, q, None] & (visual_mass > 0) & (spatial.sum(-1) > 0)
            boxes = np.asarray([rows[i][role + "_bbox"] for i in subset])
            err = np.linalg.norm(xy - ev.bbox_center(boxes)[:, None, :], axis=-1)
            aggregate = stats.setdefault(role, np.zeros((4, n_heads)))
            aggregate[0] += len(subset)
            aggregate[1] += good.sum(0)
            aggregate[2] += (good & (err <= THRESHOLD)).sum(0)
            aggregate[3] += (good * err).sum(0)
    chosen, evidence = {}, {}
    for role in ROLES:
        planned, available, hits, err = stats[role]
        choices = [int(i) for i in np.flatnonzero(available)]
        if not choices:
            raise ValueError(f"No valid fit heads for {role}")
        head = max(choices, key=lambda i: (hits[i], available[i], -err[i] / available[i], -i))
        chosen[role] = [head]
        evidence[role] = dict(layer=head // signature[1], head=head % signature[1],
                              planned=int(planned[head]), available=int(available[head]),
                              hits=int(hits[head]))
    return chosen, evidence


def examples(run, dev, variants):
    """Use fixed local-peak decoding and scores in original-frame pixels."""
    for rows, names, maps, mass, valid in ev.batches(run, "dev", dev):
        subset = np.flatnonzero([eligible(row) for row in rows])
        if not len(subset):
            continue
        query_arrays = {}
        for method in METHODS:
            q = names.index(method)
            spatial = maps[subset, q].reshape(len(subset), -1, ev.PATCHES)
            raw_mass = mass[subset, q].reshape(len(subset), -1)
            query_arrays[method] = ev.normalize_maps(spatial, raw_mass), raw_mass, valid[subset, q]
        for variant, heads in variants.items():
            readouts = {method: [] for method in METHODS}
            for method, (normalized, raw_mass, usable) in query_arrays.items():
                role_names = [method] if method != "oracle" else list(ROLES)
                decoded = {}
                for role in role_names:
                    head_indices = list(range(normalized.shape[1])) if heads is None else heads[role]
                    pooled, visual_mass = ev.pool(normalized, raw_mass, head_indices, "equal")
                    xy = ev.decode_maps(pooled, "local3")
                    use = usable & (visual_mass > 0) & (pooled.sum(-1) > 0)
                    decoded[role] = [
                        dict(map=prob, mass=float(v), position=pos, valid=bool(good))
                        for prob, v, pos, good in zip(pooled, visual_mass, xy, use, strict=True)
                    ]
                for i, row_index in enumerate(subset):
                    # The upper-bound oracle knows the label-matched phrase and corresponding
                    # role ensemble. Deployable source/destination readouts ignore this stage.
                    role = ROLES[STAGES.index(rows[row_index]["stage"])] if method == "oracle" else method
                    readouts[method].append(decoded[role][i])
            for i, row_index in enumerate(subset):
                row = rows[row_index]
                queries = {method: readouts[method][i] for method in METHODS}
                use = all(queries[role]["valid"] for role in ROLES)
                x = feature_vector(*(queries[role] for role in ROLES)) if use else None
                yield variant, row, queries, x


def sigmoid(logits):
    return np.exp(-np.logaddexp(0, -logits))


def fit_phase(features, place):
    """Fit standardized ridge logistic regression using only the fit parent group."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(place, dtype=np.float64)
    if x.shape != (len(y), len(FEATURES)) or not np.isfinite(x).all() or set(y) != {0., 1.}:
        raise ValueError("Ridge phase fitting requires finite feature rows and both stages")
    m = x.mean(0)
    scale = np.where(x.std(0) > 1e-9, x.std(0), 1.)
    z = np.column_stack([np.ones(len(x)), (x - m) / scale])
    reg = REGULARIZATION * np.r_[0., np.ones(z.shape[1] - 1)]
    w = np.zeros(z.shape[1], np.float64)

    def objective(beta):
        logits = z @ beta
        loss = np.logaddexp(0, logits) - y * logits
        return np.mean(loss) + .5 * np.sum(reg * beta**2)

    for _ in range(60):
        p = sigmoid(z @ w)
        gradient = z.T @ (p - y) / len(z) + reg * w
        if np.linalg.norm(gradient, ord=np.inf) < 1e-9:
            break
        curvature = p * (1 - p)
        hessian = (z.T * curvature) @ z / len(z) + np.diag(reg)
        update = np.linalg.solve(hessian, gradient)
        previous = objective(w)
        for t in 2.0 ** -np.arange(15):
            new = w - t * update
            if objective(new) <= previous:
                w = new
                break
        else:
            raise ValueError("Phase router ridge fit failed to decrease its objective")
    return dict(feature_names=list(FEATURES), fit_mean=m.tolist(), fit_scale=scale.tolist(),
                intercept=float(w[0]), weight=w[1:].tolist(),
                l2=REGULARIZATION, decision_threshold=.5)


def predict_phase(model, observable_features):
    """Infer a role solely from the precomputed text/image attention features."""
    features = np.asarray(observable_features, dtype=np.float64)
    standardized = (features - model["fit_mean"]) / model["fit_scale"]
    place = float(sigmoid(standardized @ model["weight"] + model["intercept"]))
    return "Place" if place >= model["decision_threshold"] else "Pick", place


def error_to(readout, box):
    if not readout["valid"] or box is None:
        return None
    return float(np.linalg.norm(readout["position"] - ev.bbox_center(np.asarray(box))))


def score_check(row, readouts, observable, model):
    target_role = ROLES[STAGES.index(row["stage"])]
    answer = None if observable is None else predict_phase(model, observable)
    predicted_stage, probability = answer if answer is not None else (None, None)
    predicted_role = ROLES[STAGES.index(predicted_stage)] if predicted_stage is not None else None
    selections = {"oracle_phrase": "oracle", "oracle_stage": target_role, "predicted_stage": predicted_role}

    result = {k: row[k] for k in ("id", "episode_id", "frame", "parent_id", "suite", "task", "stage")}
    result.update(prompt=row["prompt"], predicted_stage=predicted_stage, expected_role=target_role,
                  predicted_role=predicted_role, p_place=probability,
                  stage_correct=predicted_stage == row["stage"])
    for mode, method in selections.items():
        readout = readouts.get(method) if method is not None else None
        err = None if readout is None else error_to(readout, row["current_bbox"])
        result[mode + "_error_px"] = err
        result[mode + "_hit"] = err is not None and err <= THRESHOLD
    for role in ROLES:
        readout = readouts[role]
        err = error_to(readout, row.get(role + "_bbox"))
        result.update({role + "_region_labeled": row.get(role + "_bbox") is not None,
                       role + "_error_px": err, role + "_hit": err is not None and err <= THRESHOLD,
                       role + "_visual_mass": readout["mass"],
                       role + "_entropy": float(ev.entropy(readout["map"])) if readout["valid"] else None,
                       role + "_pred_x": float(readout["position"][0]) if readout["valid"] else None,
                       role + "_pred_y": float(readout["position"][1]) if readout["valid"] else None})
    hit, stage_correct = result["predicted_stage_hit"], result["stage_correct"]
    if predicted_stage is None:
        attribution = "unavailable_router_features"
    elif stage_correct:
        attribution = "correct_stage_and_hit" if hit else "correct_stage_location_miss"
    else:
        attribution = "wrong_stage_but_hit" if hit else "wrong_stage_and_location_miss"
    result["outcome"] = attribution
    return result


def statistics(rows):
    result = dict(eligible=len(rows), parents=len({r["parent_id"] for r in rows}))
    for key in ("oracle_phrase", "oracle_stage", "predicted_stage", *ROLES):
        comparable = [r for r in rows if r.get(key + "_region_labeled", True)]
        error = [r[key + "_error_px"] for r in comparable if r[key + "_error_px"] is not None]
        hit = [int(r[key + "_hit"]) for r in comparable]
        result[key] = dict(planned=len(comparable), available=len(error), hits=sum(hit),
                           median_error_px=float(np.median(error)) if error else None,
                           rate_ci95=ev.clustered(hit, [r["parent_id"] for r in comparable]))
    assigned = [r for r in rows if r["predicted_stage"] is not None]
    result["phase"] = dict(available=len(assigned), correct=sum(r["stage_correct"] for r in assigned),
                           planned=len(rows), confusion={f"{a}→{b}": sum(
                               r["stage"] == a and r["predicted_stage"] == b for r in assigned
                           ) for a in STAGES for b in STAGES})
    result["attribution"] = dict(Counter(r["outcome"] for r in rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--allow-after-e0", action="store_true",
                        help="Run only after E0 supports conditional phase routing")
    args = parser.parse_args()
    if not args.allow_after_e0:
        parser.error("E1 requires a separate evidence-based decision after E0 and --allow-after-e0")
    run = args.run.resolve()
    data = run / "dev_rows.json"
    output = run / "e1_routing"
    if output.exists():
        raise FileExistsError(output)
    dev = ev.rows(data)
    parents = {split: {r["parent_id"] for r in dev if r["split"] == split} for split in ("fit", "check")}
    if {r["split"] for r in dev} != {"fit", "check"} or parents["fit"] & parents["check"]:
        raise ValueError("Expected independent frozen fit/check training parents")
    heads, head_evidence = fit_role_heads(run, dev)
    variants = {"all_equal": None, "fit_top1": heads}
    fit_x = {name: [] for name in variants}
    fit_y = {name: [] for name in variants}
    checking = {name: [] for name in variants}
    counts = Counter()
    for variant, row, query, x in examples(run, dev, variants):
        counts[(variant, row["split"], "eligible", row["stage"])] += 1
        if row["split"] == "check":
            checking[variant].append((row, query, x))
        elif x is not None:
            fit_x[variant].append(x)
            fit_y[variant].append(row["stage"] == "Place")
            counts[(variant, "fit", "available", row["stage"])] += 1
    models = {name: fit_phase(fit_x[name], fit_y[name]) for name in variants}
    variant_summaries, checks = {}, []
    for name in variants:
        predictions = [dict(variant=name, **score_check(row, query, x, models[name]))
                       for row, query, x in checking[name]]
        checks.extend(predictions)
        groups = [
            dict(suite=suite, stage=stage, **statistics([r for r in predictions if
                 (suite is None or r["suite"] == suite) and (stage is None or r["stage"] == stage)]))
            for suite in (None, "object", "goal") for stage in (None, *STAGES)
        ]
        variant_summaries[name] = dict(
            row_coverage=[dict(split=split, status=kind, stage=stage, count=total)
                          for (variant, split, kind, stage), total in sorted(counts.items()) if variant == name],
            heads="uniform over all native heads" if name == "all_equal" else head_evidence, groups=groups,
        )
    summary = dict(
        scope="Training-derived development check only. Place is a resolved ordinary Event diagnostic.",
        held_out_ABCD_accessed=False, fit_parents=len(parents["fit"]), check_parents=len(parents["check"]),
        dev_manifest_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),
        decode="local3", threshold_px=THRESHOLD, groups=variant_summaries["all_equal"]["groups"],
        variants=variant_summaries,
        inference_inputs="Frozen source/destination maps and all-token visual-mass summaries",
        oracle_phrase_inputs="Labeled target term and same stage-matched ensemble as oracle-stage",
        oracle_stage_inputs="Ground-truth Pick or Place stage for selecting source or destination readout",
    )
    ev.write(output / "summary.json", summary)
    ev.write(output / "routing_parameters.json", models)
    ev.write_csv(output / "per_sample.csv", checks)
    print(f"E1 dev-check report for {len(checking['all_equal'])} eligible full Event queries per variant: {output}")


if __name__ == "__main__":
    main()
