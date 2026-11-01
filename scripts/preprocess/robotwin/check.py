"""Fixed sampling, field checks and replay; human decisions remain separate."""
import argparse
import csv
import json
from pathlib import Path
import random

from .source import (SOURCE_ROOT, digest, freeze_entry, inventory, load_config,
                    load_episode, read_json, validate_episode, verify_entry, write_json)


def sample(config, source_report=None):
    root = Path(config["check_root"])
    path = root / "manifest.json"
    if path.exists():
        raise FileExistsError(f"Sample already frozen: {path}; reuse it or choose a new check_root")
    entries = inventory(config)
    count, seed = int(config["sample_count"]), int(config["seed"])
    if not 0 < count <= len(entries):
        raise ValueError("Sample count outside source population")
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry["task"], []).append(entry)
    rng = random.Random(seed)
    if source_report:
        previous = read_json(source_report)
        selected = [{"task": s["task"], "episode": s["episode"], "hdf5": s["hdf5_path"],
                     "events": s["events_path"], "format": "xpolicylab-v1"}
                    for s in previous["samples"]]
        if config["source_format"] != "xpolicylab-v1" or len(selected) != count:
            raise ValueError("Historical report count/format differs from configuration")
        rule = previous["sampling"]
    elif count <= len(grouped):
        selected = [rng.choice(grouped[task]) for task in rng.sample(sorted(grouped), count)]
        rule = "uniform distinct tasks, one uniformly selected episode per task"
    else:
        selected = rng.sample(entries, count)
        rule = "uniform episodes without replacement"
    manifest = {"format": "eskill.robotwin.check.v1", "config": config,
                "sampling": rule, "population": len(entries),
                "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                "samples": [freeze_entry(entry) for entry in selected]}
    if source_report:
        manifest["historical_report"] = {"path": str(Path(source_report).resolve()),
                                         "sha256": digest(source_report)}
    write_json(path, manifest)
    with (root / "manual_review.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "episode", "decision", "notes"])
        writer.writeheader()
        writer.writerows({"task": e["task"], "episode": e["episode"], "decision": "pending"} for e in selected)
    return path


def run_check(manifest_path, *, render=False, annotations=None, output=None):
    manifest = read_json(manifest_path)
    root = Path(output or Path(manifest_path).parent)
    results = []
    for entry in manifest["samples"]:
        item = {"task": entry["task"], "episode": entry["episode"]}
        try:
            verify_entry(entry)
            episode = load_episode(entry)
            item.update(validate_episode(episode))
            annotation = None
            if annotations:
                from .annotate import validate_annotation
                annotation = read_json(Path(annotations) / entry["task"] / f"{entry['episode']}.json")
                validate_annotation(episode, annotation)
            if render and not item["errors"]:
                from .render import render_episode
                destination = root / "video" / f"{entry['task']}__{entry['episode']}.mp4"
                item.update(render_episode(episode, destination,
                                          cameras=manifest["config"]["cameras"], annotation=annotation))
        except Exception as exc:
            item["errors"] = [f"{type(exc).__name__}: {exc}"]
        results.append(item)
        print(f"{entry['task']}/{entry['episode']}: {item.get('errors') or 'technical checks passed'}")
    write_json(root / ("render_report.json" if render else "report.json"),
               {"manifest": str(Path(manifest_path).resolve()), "manifest_sha256": digest(manifest_path),
                "annotations": str(annotations) if annotations else None,
                "results": results, "manual_review": "pending"})
    return 1 if any(item.get("errors") for item in results) else 0


def field_table(manifest_path, output):
    manifest = read_json(manifest_path)
    rows = []
    for entry in manifest["samples"]:
        verify_entry(entry)
        episode = load_episode(entry)
        rows.append({"rgb": bool(episode.cameras), "text": bool(episode.instructions),
                     "state": any("joint" in k or k.startswith("state/") for k in episode.signals),
                     "pose": bool(episode.poses), "intervals": bool(episode.evidence.get("subtasks")),
                     "calibration": any(episode.calibration.values())})
    def status(key):
        return "有" if all(r[key] for r in rows) else "部分" if any(r[key] for r in rows) else "无"
    table = [
        ("RGB 图像", status("rgb"), "终点是否完整见逐条 report；有字段不代表验收。"),
        ("任务指令", status("text"), "源任务文本。"),
        ("机器人状态", "部分" if any(r["state"] for r in rows) else "无", "命令／实测字段分开，历史语义待核验。"),
        ("机器人动作与运动记录", "部分", "源命令／动作与末端参考点需明确，见报告。"),
        ("每臂 Event 边界", "部分" if any(r["intervals"] for r in rows) else "无", "primitive 是来源证据，不直接当语义 Event。"),
        ("子事件描述与技能标签", "部分" if any(r["intervals"] for r in rows) else "无", "Pick／Place／Hold 映射未自动生成。"),
        ("执行器视觉区域", "待核实", "本命令不将原生分割／源字段自动解释为夹爪框。"),
        ("当前对象视觉区域", "待核实", "需检查标注中的同对象区域。"),
        ("Event 终点对象区域", "部分", "由确认的 Event 结束帧和同对象区域读取。"),
        ("时间与同步信息", "部分", "帧数与频率由 validate 检查，语义对齐需人工核验。"),
        ("坐标与相机信息", "部分" if any(r["calibration"] for r in rows) else "待核实", "矩阵约定和末端参考点需确认。"),
    ]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("对照 docs/data.md；仅统计此 manifest，人工验收待完成。\n\n"
                    "| 我们需要什么 | 有没有 | 备注 |\n| --- | --- | --- |\n"
                    + "".join(f"| {a} | {b} | {c} |\n" for a, b, c in table))


def collection_summary(source_root, *, render=False):
    """Summarize this clean run; optionally freeze and render ten distinct episodes."""
    import collections
    import h5py
    source_root = Path(source_root).resolve()
    run = read_json(source_root / "collection_run.json")
    source_roots = [source_root]
    parent_identity = read_json(source_root / "collection.json")["fingerprint"]
    supplement = source_root / "supplement"
    if (supplement / "collection.json").exists():
        child = read_json(supplement / "collection.json")
        if (child["config"].get("resume_from") != str(source_root) or
                child["config"].get("parent_fingerprint") != parent_identity):
            raise ValueError("Supplement source does not belong to this collection")
        source_roots.append(supplement)
    fingerprints = {root: read_json(root / "collection.json")["fingerprint"] for root in source_roots}
    records, tasks = [], []
    for task in run["tasks"]:
        folders = [root / task / "aloha_agilex" for root in source_roots]
        attempts = [json.loads(line) for folder in folders if (folder / "attempts.jsonl").exists()
                    for line in (folder / "attempts.jsonl").read_text().splitlines()]
        statuses = dict(collections.Counter(row["status"] for row in attempts))
        entries = []
        for path in sorted(path for folder in folders for path in (folder / "raw").glob("episode_*.hdf5")):
            with h5py.File(path, "r") as f:
                if not f.attrs.get("complete") or not f.attrs.get("task_success") or f.attrs.get("condition") != "clean":
                    raise ValueError(f"Invalid clean source: {path}")
                if f.attrs.get("collection_fingerprint") != fingerprints[path.parents[3]]:
                    raise ValueError(f"Source fingerprint mismatch: {path}")
                entries.append({"task": task, "episode": path.stem, "hdf5": str(path),
                                "format": "robotwin.raw.v2", "seed": int(f.attrs["seed"])})
        if len({entry["episode"] for entry in entries}) != len(entries):
            raise ValueError(f"Duplicate episode identity across collection versions: {task}")
        records.extend(entries)
        tasks.append({"task": task, "target": run["episodes_per_task"], "collected": len(entries),
                      "attempts": len(attempts), "attempt_statuses": statuses})
    all_complete = all(row["collected"] == row["target"] for row in tasks)
    summary = {"source_root": str(source_root), "source_versions": {str(k): v for k, v in fingerprints.items()},
               "condition": "clean", "tasks": tasks,
               "target": len(tasks)*run["episodes_per_task"], "collected": len(records),
               "all_complete": all_complete, "manual_review": "pending"}
    write_json(SOURCE_ROOT / "collection_clean/summary.json", summary)
    coverage_path = SOURCE_ROOT / "collection_clean/primitive_coverage.csv"
    if coverage_path.exists():
        with coverage_path.open() as stream:
            reader = csv.DictReader(stream)
            fields, coverage = reader.fieldnames, list(reader)
        counts = {row["task"]: row["collected"] for row in tasks}
        with coverage_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in coverage:
                count = counts.get(row["task"], 0)
                row["runtime_check"] = f"{count} collected and source-validated" if count else "pending"
                writer.writerow(row)
    if not render or (len(source_roots) > 1 and not all_complete):
        return summary
    if len({row["task"] for row in records}) < 10:
        raise ValueError("Fewer than ten collected tasks available for review sampling")
    processed = SOURCE_ROOT / "processed"
    path = processed / "clean_manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["source_root"] != str(source_root):
            raise ValueError("Existing clean review belongs to another source root")
    else:
        grouped = {}
        for record in records:
            grouped.setdefault(record["task"], []).append(record)
        rng = random.Random(42)
        selected = [rng.choice(grouped[task]) for task in rng.sample(sorted(grouped), 10)]
        manifest = {"source_root": str(source_root), "population": len(records), "seed": 42,
                    "sampling": "uniform distinct tasks, one uniformly selected episode per task",
                    "collection_complete_at_sampling": all_complete,
                    "code_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
                    "samples": [freeze_entry(entry) for entry in selected]}
        write_json(path, manifest)
        with (processed / "clean_manual_review.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["task", "episode", "seed", "decision", "notes"])
            writer.writeheader()
            writer.writerows({k: entry[k] for k in ("task", "episode", "seed")} | {"decision": "pending"}
                             for entry in selected)
    from .render import render_collected_episode
    results = []
    for entry in manifest["samples"]:
        verify_entry(entry)
        video = processed / "video/clean" / f"{entry['task']}__{entry['episode']}__seed_{entry['seed']}__clean.mp4"
        report_path = video.with_suffix(".json")
        if video.exists() and report_path.exists():
            result = read_json(report_path)
        else:
            result = render_collected_episode(entry["hdf5"], video)
            write_json(report_path, result)
        results.append(result)
        print(f"Rendered clean source: {entry['task']}/{entry['episode']}", flush=True)
    summary.update(review_manifest=str(path), rendered_videos=results)
    write_json(SOURCE_ROOT / "collection_clean/summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "fields", "validate", "render", "collection"))
    parser.add_argument("--config", type=Path, default=SOURCE_ROOT / "config.yaml")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--from-report", type=Path, help="Preserve identities from historical source_inspection.json")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--render", action="store_true", help="Render ten frozen clean collection samples")
    args = parser.parse_args()
    if args.command == "collection":
        if args.source_root is None:
            parser.error("collection requires --source-root")
        result = collection_summary(args.source_root, render=args.render)
        print(f"clean collection: {result['collected']}/{result['target']}, complete={result['all_complete']}")
        return 0
    if args.command == "sample":
        print(sample(load_config(args.config), args.from_report))
        return 0
    if args.manifest is None:
        parser.error("--manifest is required; checks do not resample implicitly")
    if args.command == "fields":
        field_table(args.manifest, args.output or args.manifest.parent / "field_check.md")
        return 0
    return run_check(args.manifest, render=args.command == "render",
                     annotations=args.annotations, output=args.output)


if __name__ == "__main__":
    raise SystemExit(main())
