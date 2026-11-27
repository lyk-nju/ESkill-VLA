from collections import Counter
from pathlib import Path

from .assets import (
    KLAL,
    MOTION,
    NATIVE,
    SPLIT,
    VISUAL,
    audit_images,
    digest,
    locked_paths,
    read,
    rgb,
    write_new,
)
from .motion import RUNS


def verify_locked(b):
    split = read(b.file(SPLIT))
    train, val = set(split["train"]), set(split["val"])
    m = read(b.file(MOTION))
    if len(train) != 1518 or len(val) != 175 or train & val or {e["parent"] for e in m["episodes"]} != val:
        raise ValueError("Parent trajectory split differs")
    if m["references"]["split"]["sha256"] != b.files[SPLIT]["sha256"]:
        raise ValueError("Motion split identity differs")
    dev_path = b.file(NATIVE + "inputs/dev.json")
    dev, test = read(dev_path), read(b.file(NATIVE + "inputs/test.json"))
    selection = read(b.file(NATIVE + "selection.json"))
    parents = {s: {r["parent_id"] for r in dev if r["split"] == s} for s in ("fit", "check")}
    if (
        parents["fit"] & parents["check"]
        or not (parents["fit"] | parents["check"]) <= train
        or [len(parents[s]) for s in ("fit", "check")] != [136, 68]
        or selection["test_used"]
        or selection["dev_manifest_sha256"] != digest(dev_path)
    ):
        raise ValueError("Native development lock differs")
    vis = read(b.file(VISUAL))["records"]
    klal = read(b.file(KLAL))
    if klal["source_manifest_sha256"] != b.files[VISUAL]["sha256"] or len({r["id"] for r in vis}) != 1307:
        raise ValueError("Stage 1 lock differs")
    for r, t, k in zip(vis, test, klal["rows"], strict=True):
        for key in (
            "id",
            "current_bbox",
            "subject_bbox",
            "target",
            "rgb_sha256",
            "prompt",
            "unit",
            "stage",
            "initial_state_index",
            "frame",
            "episode_id",
        ):
            if r.get(key) != t.get(key):
                raise ValueError(f"{r['id']}: Native test differs on {key}")
        if (
            r["id"],
            r["prompt"],
            r["target"],
            r["current_bbox"],
            r["subject_bbox"],
            r["rgb_sha256"],
        ) != (
            k["id"],
            k["task"],
            k["target"],
            k["boxes"][k["target"]],
            k["subject_box"],
            Path(k["image"]).stem,
        ):
            raise ValueError(f"{r['id']}: KLAL row differs")
    return dict(
        train=len(train),
        val=len(val),
        fit_parents=len(parents["fit"]),
        check_parents=len(parents["check"]),
        native_dev=len(dev),
        visual_queries=dict(Counter(r["unit"] for r in vis), E=0),
        visual_unique_rgb=len({r["rgb_sha256"] for r in vis}),
        motion_pairs=len(m["pairs"]),
        offsets=m["spec"]["offsets"],
    )


def visual_regression(b):
    locks = verify_locked(b)
    stage = {}
    for run in ("run_001", "run_002", "run_003"):
        prefix = VISUAL.rsplit("/", 1)[0].replace("run_001", run)
        if b.files[prefix + "/manifest.json"]["sha256"] != b.files[VISUAL]["sha256"]:
            raise ValueError("Legacy models evaluated different queries")
        report = read(b.file(prefix + "/results/summary.json"))
        for name, model in report["models"].items():
            fields = (
                "samples",
                "missing",
                "current_hits",
                "subject_hits",
                "current_unreadable_or_missing",
                "subject_unreadable_or_missing",
            )
            counts = {k: sum(r[k] for r in model["task"]) for k in fields}
            if (
                counts["samples"] != sum(locks["visual_queries"].values())
                or counts["missing"] != model["missing"]
            ):
                raise ValueError(f"{run}/{name}: task counts differ from locked population")
            stage.setdefault(str(model["identity"]["step"]), {})[name] = counts
    e0 = read(b.file(NATIVE + "results/summary.json"))
    total = {
        k: sum(
            r["primary"][k]
            for r in e0["primary_and_baselines"]
            if r["suite"] == "all" and r["condition"] == "all" and r["unit"] in ("A", "B", "C", "D")
        )
        for k in ("planned", "valid", "hits")
    }
    ref = next(
        r["primary"]
        for r in e0["primary_and_baselines"]
        if (r["suite"], r["condition"], r["unit"]) == ("all", "all", "all")
    )
    if total != {k: ref[k] for k in total}:
        raise ValueError("Native E0 A–D totals differ from the historical global table")
    routing = read(b.file(NATIVE + "e1_routing/summary.json"))
    group = next(r for r in routing["groups"] if r["suite"] is None and r["stage"] is None)
    e1 = {
        k: {v: group[k][v] for v in ("planned", "available", "hits")}
        for k in ("oracle_phrase", "oracle_stage", "predicted_stage")
    }
    probe = read(
        b.file("reference/visual/vlm_capability_diagnosis/frozen_representation_probe/run_001/selection.json")
    )
    return dict(
        visual_queries=locks["visual_queries"],
        stage1_task_totals=stage,
        e0_formal_by_unit=total,
        e0_development=e0["primary"],
        e1_development_report=e1,
        probe_development_selection=probe["winner"],
    )


def inventory(s):
    b = s.bundle
    result = dict(locks=verify_locked(b), files={}, images={})

    def identify(name, path, expected=None):
        path = Path(path)
        found = digest(path) if path.is_file() and expected else None
        status = (
            "missing"
            if not path.exists()
            else "hash_mismatch"
            if expected and found != expected
            else "verified"
            if expected
            else "present"
        )
        result["files"][name] = dict(path=str(path), status=status, expected=expected, actual=found)

    identify("base_for_formal_klal", s.paths["base"], s.expected.get("klal_base_sha256"))
    identify("tokenizer", s.paths["tokenizer"], s.expected.get("tokenizer_sha256"))
    for name in (
        "vae",
        "sim_python",
        "plus_python",
        "libero",
        "libero_plus",
        "composition",
    ):
        identify(name, s.paths[name])
    motion, _ = locked_paths(s, MOTION)
    for key in ("data_index", "split"):
        r = motion["references"][key]
        identify(key, r["path"], r["sha256"])
    ep = Counter()
    for e in motion["episodes"]:
        p = Path(e["path"])
        ep[
            "missing"
            if not p.is_file()
            else "verified"
            if digest(p) == e["metadata_sha256"]
            else "hash_mismatch"
        ] += 1
    result["motion_episode_metadata"] = dict(ep)
    for name in RUNS:
        r = read(b.file("reference/motion/" + name + "/run.json"))
        identify("motion_" + name, s.relocate(r["checkpoint"]), r["checkpoint_sha256"])
    for run in ("run_001", "run_002", "run_003"):
        summary = read(b.file(f"reference/visual/stage1_visual_localization/{run}/results/summary.json"))
        for name, entry in summary["models"].items():
            i = entry["identity"]
            identify(
                name if run != "run_003" else name + "_10k",
                s.relocate(i["checkpoint"]),
                i["checkpoint_sha256"],
            )
    if "klal" in s.paths:
        identify("klal", s.paths["klal"], s.expected.get("klal_sha256"))
    for key in (
        VISUAL,
        NATIVE + "inputs/dev.json",
        NATIVE + "inputs/test.json",
        MOTION,
    ):
        doc, _ = locked_paths(s, key)
        rows = (
            doc
            if isinstance(doc, list)
            else doc["records"]
            if "records" in doc
            else [c[side] for c in doc["pairs"] for side in ("a", "b")]
        )
        result["images"][key] = audit_images(rows)
    return result


def restore_dev(settings, destination):
    """Recover recorded dev RGB from the same episode/frame; never synthesize held-out renders."""
    from collections import defaultdict
    import hashlib
    import numpy as np
    from src.openpi.policies.eskill_dataset import EpisodeReader

    b = settings.bundle
    verify_locked(b)
    key = NATIVE + "inputs/dev.json"
    rows, changes = locked_paths(settings, key)
    root = Path(destination)
    if (root / "mapping.json").exists():
        raise FileExistsError(root / "mapping.json")
    root.mkdir(parents=True, exist_ok=True)
    rgb_dir = root / "dev_rgb"
    rgb_dir.mkdir(exist_ok=True)
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[row["episode_ref"]].append(row)
    hashes = {}
    for path, group in by_episode.items():
        reader = EpisodeReader(path)
        for row in group:
            dst = rgb_dir / Path(row["rgb_path"]).name
            sha = row["rgb_sha256"]
            if reader.meta["episode_id"] != row["episode_id"]:
                raise ValueError(f"Wrong parent for {row['id']}")
            if dst not in hashes:
                image = reader.rgb("image", row["frame"])
                if image is None or hashlib.sha256(image.tobytes()).hexdigest() != sha:
                    raise ValueError(f"Changed RGB at {row['id']}")
                if dst.exists():
                    rgb(dst, expected=sha)
                else:
                    np.savez_compressed(dst, main=image)
                hashes[dst] = sha
            elif hashes[dst] != sha:
                raise ValueError(f"Conflicting pixel identity for {dst}")
            row["image"] = row["rgb_path"] = str(dst)
    if (root / "dev.json").exists():
        if read(root / "dev.json") != rows:
            raise ValueError("Existing recovered manifest differs")
    else:
        write_new(root / "dev.json", rows)
    write_new(
        root / "mapping.json",
        dict(
            original=key,
            original_sha256=b.files[key]["sha256"],
            local_sha256=digest(root / "dev.json"),
            frames=len(hashes),
            queries=len(rows),
            paths={
                src: str(rgb_dir / Path(dst).name) if dst.endswith(".npz") else dst
                for src, dst in changes.items()
            },
        ),
    )
    return dict(frames=len(hashes), queries=len(rows), dev_rgb=str(rgb_dir))
