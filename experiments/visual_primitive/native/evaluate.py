"""Freeze head/readout choices on train dev, then score frozen A/B/C/D captures."""

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


SIZE = 256
GRID = 16
PATCHES = GRID**2
KS = (1, 3, 5, 8)
SEED = 17
DECODERS = ("peak", "local3", "centroid")
COMBINERS = ("mass", "equal")
DEPLOYABLE = ("source", "tail", "sentence")
Y, X = np.mgrid[:GRID, :GRID]
CENTERS = np.stack([X.ravel() + .5, Y.ravel() + .5], axis=-1) * (SIZE / GRID)
OFFSETS = np.array([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
NEIGHBORS_XY = np.stack([X.ravel(), Y.ravel()], axis=-1)[:, None, :] + OFFSETS
NEIGHBOR_VALID = ((NEIGHBORS_XY >= 0) & (NEIGHBORS_XY < GRID)).all(-1)
NEIGHBOR_INDEX = np.clip(NEIGHBORS_XY[..., 1], 0, GRID - 1) * GRID + np.clip(NEIGHBORS_XY[..., 0], 0, GRID - 1)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def write_csv(path, values):
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if not values:
        return
    fields = list(dict.fromkeys(k for row in values for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def finite(x):
    x = float(x)
    return x if np.isfinite(x) else None


def bbox_center(boxes):
    return (boxes[..., :2] + boxes[..., 2:]) / 2


def patch_overlap(boxes):
    """Fraction of each patch covered, including fractional boundary patches."""
    center = CENTERS[None, :]
    lo = np.maximum(boxes[:, None, :2], center - SIZE / GRID / 2)
    hi = np.minimum(boxes[:, None, 2:], center + SIZE / GRID / 2)
    return np.maximum(hi - lo, 0).prod(-1).astype(np.float32) / (SIZE / GRID)**2


def row_boxes(records, query, *, destination_labels):
    boxes = np.array([row["current_bbox"] for row in records], dtype=np.float32)
    mask = np.ones(len(boxes), dtype=bool)
    if query == "destination" and destination_labels:
        mask = np.array([row.get("destination_bbox") is not None for row in records])
        if mask.any():
            boxes[mask] = np.array([r["destination_bbox"] for r in records if r.get("destination_bbox") is not None])
    if not (np.isfinite(boxes).all() and np.all(boxes[:, 2:] > boxes[:, :2])):
        raise ValueError("Invalid GT box in manifest")
    return boxes, mask


def rows(path):
    value = read(path)
    if isinstance(value, dict):
        value = value["records"]
    ids = [row["id"] for row in value]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"Missing or duplicate sample IDs: {path}")
    return value


def batches(run, phase, manifest):
    by_id = {r["id"]: r for r in manifest}
    seen = set()
    shape = None
    shards = sorted((run / "captures" / phase).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No completed captures for {phase}")
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            ids = data["sample_ids"].astype(str).tolist()
            methods = data["methods"].astype(str).tolist()
            maps, mass, valid = data["maps"], data["mass"], data["valid"].astype(bool)
        signature = methods, maps.shape[2:4]
        bad_ids = set(ids) - set(by_id) | (set(ids) & seen)
        if bad_ids or len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate/unexpected capture IDs: {sorted(bad_ids)[:10]} in {shard}")
        if maps.ndim != 5 or maps.shape != (*mass.shape, PATCHES) or maps.shape[:2] != (len(ids), len(methods)) or valid.shape != maps.shape[:2]:
            raise ValueError(f"Unexpected capture layout: {shard}")
        if shape is not None and signature != shape:
            raise ValueError(f"Capture schema changed: {shard}")
        shape = signature
        seen.update(ids)
        records = [by_id[k] for k in ids]
        for start in range(0, len(ids), 32):
            take = slice(start, start + 32)
            yield records[take], methods, maps[take], mass[take], valid[take]
    if seen != set(by_id):
        raise ValueError(f"{phase}: captured {len(seen)}/{len(by_id)} samples")


def normalize_maps(maps, mass):
    if not np.isfinite(maps).all() or not np.isfinite(mass).all() or np.any(maps < -.0001) or np.any((mass < -.0001) | (mass > 1.0001)):
        raise ValueError("Nonfinite or out-of-range attention capture")
    x = np.maximum(maps.astype(np.float32), 0)
    denom = x.sum(-1, keepdims=True)
    return np.divide(x, denom, out=x, where=denom > 0)


def decode_maps(x, method):
    if method == "centroid":
        return x @ CENTERS
    peak = x.argmax(-1)
    if method == "peak":
        return CENTERS[peak]
    if method != "local3":
        raise ValueError(f"Unknown decoder: {method}")
    index, valid = NEIGHBOR_INDEX[peak], NEIGHBOR_VALID[peak]
    p = np.take_along_axis(x, index, axis=-1) * valid
    return (p[..., None] * CENTERS[index]).sum(-2) / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def pool(x, mass, heads, mode):
    if heads is None:  # Fixed uniform visual prior, useful for testing spatial bias.
        return np.full((len(x), PATCHES), 1 / PATCHES, dtype=np.float32), np.zeros(len(x), dtype=np.float32)
    selected, vis = x[:, heads], mass[:, heads]
    if mode == "equal":
        p = selected.mean(1)
    elif mode == "mass":
        p = np.einsum("nhp,nh->np", selected, vis) / np.maximum(vis.sum(-1, keepdims=True), 1e-12)
    else:
        raise ValueError(mode)
    return p, vis.mean(-1)


def entropy(x):
    """Conditional patch entropy divided by log(256)."""
    return -np.sum(x * np.log(np.maximum(x, 1e-12)), axis=-1) / np.log(PATCHES)


def spatial(x, vis, available, gt, decoder, count_as_valid=True):
    z = x.sum(-1)
    valid = available & (z > 0)
    if count_as_valid:
        valid &= vis > 0
    pred = decode_maps(x, decoder)
    error = np.linalg.norm(pred - bbox_center(gt), axis=-1)
    return pred, error, valid


def policy_specs(ranks, methods, nh):
    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(nh).tolist()
    head_sets = []
    for q in methods:
        h = ranks[q]
        head_sets.append(dict(query=q, name="all", heads=list(range(nh))))
        for k in KS:
            if k > nh:
                continue
            for family, lookup in (("random", shuffled), ("quality", h["quality"]), ("supervised", h["supervised"])):
                head_sets.append(dict(query=q, name=f"{family}{k}", heads=lookup[:k]))
    return head_sets


def accumulate_head_stats(head_stats, records, methods, maps, mass, available, threshold):
    by_part = np.array([str(r["split"]).removeprefix("dev_") for r in records])
    # Ranking and stability use the same fixed readout on both splits.
    for q, method in enumerate(methods):
        x = normalize_maps(maps[:, q].reshape(len(records), -1, PATCHES), mass[:, q]).astype(np.float32, copy=False)
        vis = mass[:, q].reshape(len(records), -1)
        gt, label_valid = row_boxes(records, method, destination_labels=True)
        if method == "destination" and not label_valid.any():
            # Without verified role boxes, do not rank against source labels.
            continue
        y = decode_maps(x, "local3")
        area = patch_overlap(gt)
        r = np.einsum("nhp,np->nh", x, area)
        e = entropy(x)
        err = np.linalg.norm(y - bbox_center(gt)[:, None, :], axis=-1)
        valid = available[:, q, None] & label_valid[:, None] & (vis > 0) & (x.sum(-1) > 0)
        fields = {"valid": np.ones_like(r), "hits": err <= threshold, "error": err,
                  "mass": vis, "entropy": e, "quality": vis * (1 - e),
                  "region": r, "raw_region": r * vis,
                  "region_enrichment": r / np.maximum(area.mean(-1, keepdims=True), 1e-12)}
        for part in ("fit", "check"):
            mask = valid & (by_part == part)[:, None]
            accum = head_stats.setdefault((method, part), {k: np.zeros(x.shape[1], dtype=np.float64) for k in fields})
            for k, z in fields.items():
                accum[k] += (z * mask).sum(0)


def head_orders(head_stats, methods, nh):
    ranks = {}
    for method in methods:
        h = head_stats.get((method, "fit"))
        if h is None:
            # Missing role supervision prevents a learned ranking. Keep a deterministic index only for visualization.
            ranks[method] = dict(supervised=list(range(nh)), quality=list(range(nh)), supervised_valid=False)
            continue
        denom = np.maximum(h["valid"], 1)
        indices = range(nh)
        rank = {
            "supervised": sorted(indices, key=lambda i: (-h["hits"][i]/denom[i], h["error"][i]/denom[i],
                                                         -h["raw_region"][i]/denom[i], i)),
            "quality": sorted(indices, key=lambda i: (-h["quality"][i]/denom[i], i)),
        }
        ranks[method] = dict(**rank, supervised_valid=bool(h["valid"].max()))
    return ranks


def head_table(stats, ranks, methods, layers, heads):
    table = []
    for q in methods:
        for i in range(layers * heads):
            row = dict(query=q, layer=i // heads, head=i % heads,
                       fit_supervised_rank=ranks[q]["supervised"].index(i) + 1 if ranks[q]["supervised_valid"] else None,
                       fit_quality_rank=ranks[q]["quality"].index(i) + 1 if ranks[q]["supervised_valid"] else None)
            for part in ("fit", "check"):
                v = stats.get((q, part))
                n = int(v["valid"][i]) if v else 0
                row[part + "_valid"] = n
                if n:
                    for name in ("hits", "error", "mass", "entropy", "region", "raw_region", "quality", "region_enrichment"):
                        row[f"{part}_{name}"] = int(v[name][i]) if name == "hits" else float(v[name][i]/n)
            table.append(row)
    return table


def compare_readouts(run, manifest, specs, methods, ranks, threshold):
    accum, check_rows = {}, defaultdict(list)
    indexed = defaultdict(list)
    for p in specs:
        indexed[p["query"]].append(p)
    for rr, names, mm, mass, valid in batches(run, "dev", manifest):
        keep = np.array([str(r["split"]).removeprefix("dev_") == "check" for r in rr])
        if not keep.any():
            continue
        rr = [r for r, use in zip(rr, keep) if use]
        for q, name in enumerate(names):
            x = normalize_maps(mm[keep, q].reshape(len(rr), -1, PATCHES), mass[keep, q])
            v = mass[keep, q].reshape(len(rr), -1)
            gt, eligible = row_boxes(rr, name, destination_labels=True)
            if name == "destination" and not ranks[name]["supervised_valid"]:
                continue
            for p in indexed[name]:
                for combiner in COMBINERS:
                    prob, vis = pool(x, v, p["heads"], combiner)
                    for decoder in DECODERS:
                        _, err, effective = spatial(prob, vis, valid[keep, q] & eligible, gt, decoder)
                        hit = effective & (err <= threshold)
                        key = name, p["name"], combiner, decoder
                        row = accum.setdefault(key, np.zeros(4, dtype=np.float64))
                        row += np.array([eligible.sum(), effective.sum(), hit.sum(), err[effective].sum()])
                        if name in DEPLOYABLE and p["name"].startswith("supervised"):
                            center_hit = np.linalg.norm(bbox_center(gt) - SIZE/2, axis=-1) <= threshold
                            ids = [(s.get("group_id", s.get("parent_id", s["id"]))) for s in rr]
                            check_rows[key].extend(zip(ids, hit.astype(float) - center_hit.astype(float)))
    selected = []
    for p in specs:
        keys = [k for k in accum if k[:2] == (p["query"], p["name"])]
        if keys:
            def quality(k):
                n, effective, hits, error = accum[k]
                return hits / max(n, 1), effective / max(n, 1), -error / max(effective, 1), -COMBINERS.index(k[2]), -DECODERS.index(k[3])
            chosen = max(keys, key=quality)
            n, effective, hits, error = accum[chosen]
            diag = dict(check_planned=int(n), check_valid=int(effective), check_hits=int(hits),
                        check_mean_error=finite(error / effective) if effective else None)
        else:
            chosen = p["query"], p["name"], "mass", "local3"
            diag = dict(check_planned=0, check_valid=0, check_hits=0, check_mean_error=None)
        selected.append(dict(**p, combiner=chosen[2], decoder=chosen[3], **diag))
    return selected, check_rows


def clustered(values, groups, seed=SEED, n_boot=2000):
    """Paired/micro mean 95% bootstrap by parent or frozen case group."""
    a = defaultdict(list)
    for x, g in zip(values, groups):
        if x is not None and np.isfinite(x):
            a[str(g)].append(float(x))
    if not a:
        return None
    sums = np.array([sum(v) for v in a.values()])
    counts = np.array([len(v) for v in a.values()])
    if len(sums) < 2:
        low = high = None
    else:
        indices = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
        q = np.quantile(sums[indices].sum(1) / counts[indices].sum(1), [.025, .975])
        low, high = map(float, q)
    return dict(mean=float(sums.sum()/counts.sum()), low=low, high=high, groups=len(a), samples=int(counts.sum()))


def check_roles(run, dev, primary, threshold):
    """Read the primary locked heads with different words on train-check frames only."""
    grouped = defaultdict(lambda: np.zeros(4, dtype=np.float64))
    for rs, methods, maps, mass, valid in batches(run, "dev", dev):
        check = np.array([str(r["split"]).removeprefix("dev_") == "check" for r in rs])
        for q, name in enumerate(methods):
            if name not in ("oracle", "source", "destination", "sentence", "locate"):
                continue
            x = normalize_maps(maps[:, q].reshape(len(rs), -1, PATCHES), mass[:, q])
            prob, vis = pool(x, mass[:, q].reshape(len(rs), -1), primary["heads"], primary["combiner"])
            xy = decode_maps(prob, primary["decoder"])
            usable = valid[:, q] & (vis > 0) & (prob.sum(-1) > 0)
            for role in ("current", "source", "destination"):
                for i, row in enumerate(rs):
                    box = row.get(role + "_bbox")
                    if check[i] and box is not None:
                        err = np.linalg.norm(xy[i] - bbox_center(np.array(box)))
                        available = usable[i]
                        grouped[(row["stage"], name, role)] += [
                            1, available, available and err <= threshold, err if available else 0
                        ]
    return [dict(stage=stage, query=q, role=role, planned=int(n), valid=int(valid), hits=int(hits),
                 mean_error=finite(error / valid) if valid else None)
            for (stage, q, role), (n, valid, hits, error) in sorted(grouped.items())]


def select(run):
    path = run / "dev_rows.json"
    dev = rows(path)
    split = defaultdict(set)
    for r in dev:
        part = str(r["split"]).removeprefix("dev_")
        if part not in {"fit", "check"} or not r.get("parent_id"):
            raise ValueError("Dev must be independently split by verified train parent")
        split[part].add(r["parent_id"])
    if not split["fit"] or not split["check"] or split["fit"] & split["check"]:
        raise ValueError("Nonempty, disjoint train parents required for fit/check")
    threshold = 10
    head_stats = {}
    for rs, methods, maps, mass, valid in batches(run, "dev", dev):
        layers, heads = maps.shape[2:4]
        accumulate_head_stats(head_stats, rs, methods, maps, mass, valid, threshold)
    ranks = head_orders(head_stats, methods, layers * heads)
    specs = policy_specs(ranks, methods, layers * heads)
    # Do not present an unlabeled index-order fallback as supervised ranking.
    specs = [p for p in specs if not p["name"].startswith("supervised") or ranks[p["query"]]["supervised_valid"]]
    picks, check_rows = compare_readouts(run, dev, specs, methods, ranks, threshold)
    eligible = [p for p in picks if p["query"] in DEPLOYABLE and p["name"].startswith("supervised")
                and p["check_valid"] > 0]
    if not eligible:
        raise ValueError("No deployable supervised candidate with valid dev-check predictions")
    primary = max(eligible, key=lambda p: (p["check_hits"] / p["check_planned"],
                                          p["check_valid"] / p["check_planned"],
                                          -p["check_mean_error"], -len(p["heads"]),
                                          -DEPLOYABLE.index(p["query"])))
    q = primary["query"]
    dev_g = check_rows[(q, primary["name"], primary["combiner"], primary["decoder"])]
    g, d = zip(*dev_g)
    candidates = head_table(head_stats, ranks, methods, layers, heads)
    write_csv(run / "dev_head_rankings.csv", candidates)
    fit_check_overlap = {}
    for method in methods:
        fit, check = head_stats.get((method, "fit")), head_stats.get((method, "check"))
        if fit is None or check is None or not check["valid"].max():
            continue
        check_rank = sorted(range(layers * heads),
                            key=lambda i: (-check["hits"][i]/max(check["valid"][i], 1),
                                           check["error"][i]/max(check["valid"][i], 1), i))
        fit_check_overlap[method] = {str(k): len(set(ranks[method]["supervised"][:k]) & set(check_rank[:k])) for k in KS}
    lock = dict(schema=1, dev_manifest_sha256=digest(path), test_used=False, hit_threshold_px=threshold,
                sizes=dict(dev_queries=len(dev), fit_parents=len(split["fit"]), check_parents=len(split["check"])),
                layers=layers, heads=heads, seed=SEED, methods=methods, policies=picks, primary=primary,
                dev_primary_delta_vs_fixed_center=clustered(d, g), dev_head_fit_check_top_k_overlap=fit_check_overlap,
                dev_check_role_diagnostics=check_roles(run, dev, primary, threshold),
                selection_definition="Train fit ranks each head using local3 center hit then error; train check selects aggregation/readout and primary source/tail/sentence policy; no test labels are read here.")
    if (run / "selection.json").exists():
        raise FileExistsError("Selection lock already exists")
    write(run / "selection.json", lock)
    print(f"LOCKED {q}/{primary['name']}/{primary['combiner']}/{primary['decoder']}, {primary['check_hits']}/{primary['check_planned']} dev-check")


def group(row):
    if row.get("unit") == "D" and row.get("category") and row["condition"] in ("added_query","native_with_added"):
        return f"D:{row['suite']}:{row['task']}:{row['category']}"
    return row.get("group_id") or row.get("parent_id") or row.get("task") or row["id"]


def score_batch(rs, methods, maps, mass, available, specs, primary, threshold):
    """Yield predictions with maps; retain raw head masses independently from normalizing maps."""
    for q, query in enumerate(methods):
        x = normalize_maps(maps[:, q].reshape(len(rs), -1, PATCHES), mass[:, q])
        vis = mass[:, q].reshape(len(rs), -1)
        gt, _ = row_boxes(rs, query, destination_labels=False)
        area = patch_overlap(gt)
        dboxes, deligible = row_boxes(rs, query, destination_labels=True)
        for p in specs[query]:
            prob, mean_vis = pool(x, vis, p["heads"], p["combiner"])
            pred, error, valid = spatial(prob, mean_vis, available[:, q], gt, p["decoder"])
            fg = np.einsum("np,np->n", prob, area)
            true_fg = (np.einsum("nhp,np->nh", x[:, p["heads"]], area) * vis[:, p["heads"]]).mean(-1)
            ent = entropy(prob)
            dest_error = np.linalg.norm(pred - bbox_center(dboxes), axis=-1)
            for i, r in enumerate(rs):
                is_valid = bool(valid[i])
                rec = dict(id=r["id"], unit=r.get("unit"), suite=r.get("suite"), stage=r.get("stage"),
                           condition=r.get("condition"), group_id=group(r), query=query, policy=p["name"],
                           valid=is_valid, query_available=bool(available[i, q]), hit=is_valid and bool(error[i] <= threshold),
                           x=finite(pred[i, 0]) if is_valid else None, y=finite(pred[i, 1]) if is_valid else None,
                           error_px=finite(error[i]) if is_valid else None,
                           visual_mass=finite(mean_vis[i]) if is_valid else None,
                           conditional_region_mass=finite(fg[i]) if is_valid else None,
                           uniform_region_mass=finite(area[i].mean()),
                           region_enrichment=finite(fg[i] / area[i].mean()) if is_valid and area[i].mean()>0 else None,
                           raw_region_mass=finite(true_fg[i]) if is_valid else None,
                           entropy=finite(ent[i]) if is_valid else None,
                           destination_hit=(bool(dest_error[i]<=threshold) if is_valid and query=="destination" and deligible[i] else None))
                yield rec, prob[i] if p["name"] == "shared_primary" or (p["query"], p["name"]) == (primary["query"], primary["name"]) else None


def save_head_batch(path, rs, methods, maps, mass, available):
    # Reduce to per-head scalars, coordinates and mass. No token attention matrix is written.
    shape = maps.shape[:4]
    region, true_region, ent = (np.zeros(shape, np.float16) for _ in range(3))
    coords = np.zeros((*shape, 2), np.float16)
    uniform = np.zeros(shape[:2], np.float32)
    for q, name in enumerate(methods):
        x = normalize_maps(maps[:, q], mass[:, q])
        gt, _ = row_boxes(rs, name, destination_labels=False)
        area = patch_overlap(gt)
        region[:, q] = np.einsum("nlhp,np->nlh", x, area)
        uniform[:, q] = area.mean(-1)
        true_region[:, q] = region[:, q].astype(np.float32) * mass[:, q]
        ent[:, q] = entropy(x)
        coords[:, q] = decode_maps(x, "local3")
    np.savez_compressed(path, sample_ids=[r["id"] for r in rs], methods=methods, valid=available,
                        visual_mass=mass, conditional_region_mass=region, raw_region_mass=true_region,
                        entropy=ent, center_xy=coords, uniform_region_mass=uniform)


def baseline_csv(items, test):
    result = {}
    expected = {r["id"] for r in test}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name or name in result:
            raise ValueError("--baseline uses a unique NAME=CSV")
        with Path(path).open(newline="") as f:
            parsed = list(csv.DictReader(f))
        preds = {r["id"]: r for r in parsed}
        if len(preds) != len(parsed) or set(preds) != expected:
            raise ValueError(f"Baseline sample IDs differ: {name}")
        for r in test:
            expected_gt = np.array(r["current_bbox"])
            actual = preds[r["id"]]
            if not np.array_equal(np.array(json.loads(actual["current_gt_box"])), expected_gt):
                raise ValueError(f"Baseline GT differs: {name}/{r['id']}")
            for field in ("unit", "suite", "condition", "prompt", "image"):
                if str(r.get(field) or "") != actual.get(field, ""):
                    raise ValueError(f"Baseline {field} differs: {name}/{r['id']}")
        result[name] = dict(path=path, sha256=digest(path), predictions=preds)
    return result


def stat(pred):
    n = len(pred)
    hits = sum(p["hit"] for p in pred)
    valid = sum(p["valid"] for p in pred)
    d = dict(hits=hits, planned=n, valid=valid, hit_rate=hits / n if n else None,
             eligible=sum(p.get("query_available", p["valid"]) for p in pred),
             hit_rate_valid=hits / valid if valid else None,
             coverage=valid / n if n else None)
    d["hit_rate_ci"] = clustered([float(p["hit"]) for p in pred], [p["group_id"] for p in pred])
    for name in ("error_px", "visual_mass", "entropy", "conditional_region_mass", "raw_region_mass", "region_enrichment"):
        vals = [p[name] for p in pred if p.get(name) is not None]
        d[f"median_{name}"] = finite(np.median(vals)) if vals else None
    destination = [p["destination_hit"] for p in pred if p.get("destination_hit") is not None]
    if destination:
        d.update(destination_hits=sum(destination), destination_valid=len(destination))
    return d


def all_groups(pred):
    d = defaultdict(list)
    for p in pred:
        for suite, unit, condition in (("all","all","all"), (p["suite"], "all","all"),
                                       ("all", p["unit"],"all"), (p["suite"],p["unit"],"all"),
                                       ("all",p["unit"],p["condition"]), (p["suite"],p["unit"],p["condition"])):
            d[(suite, unit, condition)].append(p)
    return [dict(suite=s, unit=u, condition=c, **stat(group)) for (s,u,c), group in sorted(d.items())]


def paired_metrics(test, primary, base):
    by_id = {r["id"]: r for r in test}
    pmap = {r["id"]: r for r in primary}
    details = []
    for r in test:
        related = by_id.get(r.get("paired_id"))
        if related is None:
            continue
        task = r["unit"]
        if not ((task=="A" and r["condition"]=="target_moved") or
                (task=="B" and r["condition"]=="other_query") or task=="D"):
            continue
        q, p = pmap[r["id"]], pmap[related["id"]]
        gt = bbox_center(np.array(r["current_bbox"])) - bbox_center(np.array(related["current_bbox"]))
        row = dict(id=r["id"], paired_id=related["id"], unit=task, suite=r["suite"], condition=r["condition"],
                   group_id=group(r), valid_pair=p["valid"] and q["valid"],
                   primary_pair_hit=p["hit"] and q["hit"], anchor_hit=p["hit"], intervention_hit=q["hit"],
                   gt_dx=float(gt[0]), gt_dy=float(gt[1]), gt_shift_px=float(np.linalg.norm(gt)))
        center_fixed_delta = np.linalg.norm(bbox_center(np.array(r["current_bbox"])) - SIZE/2) <= 10
        center_fixed_anchor = np.linalg.norm(bbox_center(np.array(related["current_bbox"])) - SIZE/2) <= 10
        row["fixed_center_pair_hit"] = bool(center_fixed_anchor and center_fixed_delta)
        if row["valid_pair"]:
            move = np.array([q["x"]-p["x"], q["y"]-p["y"]])
            error = np.linalg.norm(move-gt)
            row.update(pred_dx=float(move[0]), pred_dy=float(move[1]), pred_shift_px=float(np.linalg.norm(move)),
                       displacement_error_px=float(error),
                       displacement_gain_vs_no_move=float(np.linalg.norm(gt)-error),
                       vector_cosine=finite(np.dot(move, gt) / (np.linalg.norm(move)*np.linalg.norm(gt)))
                       if np.linalg.norm(move)*np.linalg.norm(gt) > 0 else None)
            if r.get("reference_bbox") is not None:
                old = bbox_center(np.array(r["reference_bbox"]))
                row["reference_only_hit"] = bool(not q["hit"] and np.linalg.norm(np.array([q["x"], q["y"]])-old) <= 10)
        for name, item in base.items():
            before, after = item["predictions"][related["id"]], item["predictions"][r["id"]]
            row[name + "_pair_hit"] = before["current_hit"] == after["current_hit"] == "True"
        details.append(row)
    summary = []
    for key in sorted({(r["unit"], r["suite"], r["condition"]) for r in details}
                      | {(r["unit"], "all", r["condition"]) for r in details}):
        u, s, c = key
        choose = [r for r in details if r["unit"] == u and r["condition"] == c and
                  (s == "all" or r["suite"] == s)]
        gs = [r["group_id"] for r in choose]
        valid = sum(r["valid_pair"] for r in choose)
        pair_hits = sum(r["primary_pair_hit"] for r in choose)
        results = dict(unit=u, suite=s, condition=c, planned_pairs=len(choose), valid_pairs=valid, pair_hits=pair_hits,
                       pair_hit_ci=clustered([int(r["primary_pair_hit"]) for r in choose], gs),
                       delta_pair_vs_fixed_center=clustered([int(r["primary_pair_hit"])-int(r["fixed_center_pair_hit"]) for r in choose], gs),
                       displacement_gain_vs_no_move=clustered([r.get("displacement_gain_vs_no_move") for r in choose],gs))
        for field in ("gt_shift_px", "pred_shift_px", "displacement_error_px", "vector_cosine"):
            vals = [r[field] for r in choose if r.get(field) is not None]
            results["median_" + field] = finite(np.median(vals)) if vals else None
        results["baseline_pair_hits"] = {k: sum(r[k+"_pair_hit"] for r in choose) for k in base}
        results["delta_pair_vs_baseline"] = {k: clustered([int(r["primary_pair_hit"])-int(r[k+"_pair_hit"])
                                                            for r in choose],gs) for k in base}
        summary.append(results)
    return details, summary


def strata(test, primary, baseline):
    result = []
    for sample, pred in zip(test, primary):
        box = np.array(sample["current_bbox"])
        area = np.prod(box[2:] - box[:2])
        xy = bbox_center(box)
        category = sample.get("category") if sample.get("condition") == "added_query" else sample.get("target")
        category = str(category or "unknown").rsplit("_", 1)[0] if str(category).rsplit("_", 1)[-1].isdigit() else category
        reference = sample.get("reference_bbox")
        offset = float(np.linalg.norm(xy-bbox_center(np.array(reference)))) if sample.get("reference_kind") == "old_position" and reference else None
        move = None if offset is None else ("<32" if offset < 32 else "32–63" if offset < 64 else "≥64")
        for dim, value in (("size", "<16px" if np.sqrt(area) < 16 else "16–31px" if np.sqrt(area) < 32 else "≥32px"),
                           ("category", category), ("image_quadrant", ("upper" if xy[1] < 128 else "lower") + "_" + ("left" if xy[0] < 128 else "right")),
                           ("position_change_px", move), ("stage", sample.get("stage")), ("suite", sample.get("suite"))):
            if value is not None:
                result.append((dim, str(value), sample, pred))
    output = []
    for d, v in sorted({(a,b) for a,b,_,_ in result}):
        subset = [(r,p) for a,b,r,p in result if (a,b)==(d,v)]
        record = dict(slice=d, value=v, **stat([p for _,p in subset]))
        for model, b in baseline.items():
            record[model + "_hits"] = sum(b["predictions"][r["id"]]["current_hit"]=="True" for r,_ in subset)
        output.append(record)
    return output


def evaluate(run, baseline_items, diagnostics):
    lock_path = run / "selection.json"
    lock = read(lock_path)
    if lock["test_used"] is not False or lock["dev_manifest_sha256"] != digest(run / "dev_rows.json"):
        raise ValueError("Invalid dev selection lock")
    sha_lock = digest(lock_path)
    test_path = run / "test_rows.json"
    test = rows(test_path)
    baseline = baseline_csv(baseline_items, test)
    ids = {r["id"] for r in test}
    test_groups = {r["id"]: group(r) for r in test}
    primary = lock["primary"]
    methods = lock["methods"]
    threshold = lock["hit_threshold_px"]
    specs = {q: [p for p in lock["policies"] if p["query"] == q] for q in methods}
    for q in methods:
        if q != primary["query"]:
            specs[q].append(dict(primary, query=q, name="shared_primary"))
    out = run / "results"
    out.mkdir(exist_ok=True, parents=True)
    if diagnostics:
        (out / "per_head").mkdir(exist_ok=True)
    prediction_buckets = defaultdict(list)
    selected_maps = defaultdict(dict)
    captures = batches(run, "test", test)
    with gzip.open(out / "all_predictions.csv.gz", "wt", newline="") as f:
        writer = None
        for i, (rs, names, mm, vis, valid) in enumerate(captures):
            if names != methods or list(mm.shape[2:4]) != [lock["layers"],lock["heads"]]:
                raise ValueError("Capture schema differs from selection")
            if diagnostics:
                save_head_batch(out / "per_head" / f"test_{i:04d}.npz", rs, names, mm, vis, valid)
            for row, heat in score_batch(rs, names, mm, vis, valid, specs, primary, threshold):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                prediction_buckets[(row["query"],row["policy"])].append(row)
                if heat is not None:
                    selected_maps[row["query"]][row["id"]] = heat.astype(np.float16)
    n = len(ids)
    assert all(len(bb) == n for bb in prediction_buckets.values())
    pp = {(r["id"]):r for r in prediction_buckets[(primary["query"],primary["name"])]}
    primary_predictions, joined = [pp[r["id"]] for r in test], []
    for sample, pred in zip(test, primary_predictions):
        record = dict(**pred, task=sample.get("task"), target=sample.get("target"), prompt=sample.get("prompt"),
                      reference_kind=sample.get("reference_kind"), paired_id=sample.get("paired_id"),
                      gt_x=float(bbox_center(np.array(sample["current_bbox"]))[0]),
                      gt_y=float(bbox_center(np.array(sample["current_bbox"]))[1]))
        for model, value in baseline.items():
            orig = value["predictions"][sample["id"]]
            record[model+"_hit"] = orig["current_hit"]=="True"
            coords = json.loads(orig["current_center"]) if orig["current_center"] else None
            record[model+"_x"] = coords[0] if coords is not None else None
            record[model+"_y"] = coords[1] if coords is not None else None
        joined.append(record)
    write_csv(out/"primary_vs_baselines.csv", joined)
    compare_groups, comparisons = defaultdict(list), []
    for r in joined:
        for key in (("all", "all", "all"), ("all",r["unit"],"all"), (r["suite"],r["unit"],r["condition"]),
                    ("all",r["unit"],r["condition"])):
            compare_groups[key].append(r)
    for (s,u,c), rs in sorted(compare_groups.items()):
        gs = [test_groups[r["id"]] for r in rs]
        fixed = [np.linalg.norm(np.array([r["gt_x"], r["gt_y"]]) - SIZE/2) <= threshold for r in rs]
        item = dict(suite=s,unit=u,condition=c,primary=stat(rs),baseline={},
                    fixed_center=dict(hits=int(sum(fixed)),planned=len(rs)),
                    delta_vs_fixed_center=clustered([int(r["hit"])-int(h) for r,h in zip(rs,fixed)],gs))
        for k in baseline:
            hit = sum(r[k+"_hit"] for r in rs)
            item["baseline"][k] = dict(hits=hit,planned=len(rs),rate=hit/len(rs),
                                       delta_paired_ci=clustered([int(r["hit"])-int(r[k+"_hit"]) for r in rs],gs))
        comparisons.append(item)
    table = [dict(query=q,policy=p,**v) for (q,p), bb in prediction_buckets.items() for v in all_groups(bb)]
    flat = [{k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in r.items()} for r in table]
    write_csv(out/"policies.csv",flat)
    props = [{"id":r["id"],"query":primary["query"],"policy":primary["name"]} for r in test]
    np.savez_compressed(out/"heatmaps.npz", sample_ids=[r["id"] for r in props], methods=methods,
                        maps=np.stack([[selected_maps[q][r["id"]] for r in props] for q in methods],axis=1))
    query_compare = []
    main = {r["id"]: r for r in primary_predictions}
    for q in methods:
        if q == primary["query"]:
            continue
        for unit in ("all","A","B","C","D"):
            match = [(r,main[r["id"]]) for r in prediction_buckets[(q,"shared_primary")] if r["valid"]
                     and main[r["id"]]["valid"] and (unit == "all" or r["unit"] == unit)]
            gains = [int(x["hit"])-int(y["hit"]) for x,y in match]
            query_compare.append(dict(query=q,against=primary["query"],unit=unit,shared_heads=True,overlap=len(match),
                                      alternative_hits=sum(x["hit"] for x,_ in match), primary_hits=sum(y["hit"] for _,y in match),
                                      hit_delta=clustered(gains,[x["group_id"] for x,_ in match])))
    details, pairs = paired_metrics(test,primary_predictions,baseline)
    write_csv(out/"paired_predictions.csv",details)
    write_csv(out/"failure_slices.csv",strata(test,primary_predictions,baseline))
    for r in test:
        if r.get("paired_id") and r["paired_id"] not in ids:
            raise ValueError(f"Unknown frozen pair ID {r['paired_id']}")
    nearest = np.min(np.linalg.norm(CENTERS[None]-bbox_center(np.array([r["current_bbox"] for r in test]))[:,None],axis=-1),axis=-1)
    checks = [r for r in pairs if r["suite"]=="all" and r["unit"]=="A" and r["condition"]=="target_moved"]
    dev_ci = lock["dev_primary_delta_vs_fixed_center"]
    move_ci = checks[0]["displacement_gain_vs_no_move"] if checks else None
    text_cis = [r["delta_pair_vs_fixed_center"] for r in pairs
                if r["suite"]!="all" and r["unit"]=="B" and r["condition"]=="other_query"]
    gates = dict(dev_beats_fixed_center=bool(dev_ci and (dev_ci["low"] or 0)>0),
                 movement_beats_no_move=bool(move_ci and (move_ci["low"] or 0)>0),
                 any_suite_pair_beats_fixed_center=any(c and (c["low"] or 0)>0 for c in text_cis))
    report = dict(frozen_rows_sha256=digest(test_path), selector_sha256=sha_lock,
                  sample_count=n, primary=primary, dev=lock, models={k:{f:v for f,v in b.items() if f!="predictions"}
                                                                   for k,b in baseline.items()},
                  primary_and_baselines=comparisons, paired=pairs, queries_fixed_heads=query_compare,
                  patch_center_oracle=dict(hits=int(np.sum(nearest<=threshold)),planned=n),
                  stages={s:sum(r.get("stage")==s for r in test) for s in sorted({r.get("stage") for r in test})},
                  continuation_checks=gates, all_policy_rows=len(table))
    if sha_lock != digest(lock_path):
        raise ValueError("Selection lock changed during scoring")
    write(out/"summary.json",report)
    print(f"E0 scored {n} frozen samples; primary {sum(r['hit'] for r in joined)}/{n}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    sub.add_parser("select").add_argument("run", type=Path)
    test = sub.add_parser("evaluate")
    test.add_argument("run", type=Path)
    test.add_argument("--baseline", action="append", default=[], help="NAME=frozen baseline per-sample CSV")
    test.add_argument("--skip-head-diagnostics", action="store_true")
    args = cli.parse_args()
    if args.command == "select":
        select(args.run)
    else:
        evaluate(args.run, args.baseline, not args.skip_head_diagnostics)


if __name__ == "__main__":
    main()
