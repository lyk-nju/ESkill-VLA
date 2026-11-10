"""Audit ESkill timing readiness, Event-local query coverage and action-boundary crossings.

Reads saved numeric labels only, without RGB decoding, models or simulator imports.
An optional timing manifest supplies evidenced clocks and gripper command semantics;
this tool never interprets playback fps or invents missing physical timestamps.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader, episode_timestamps, gripper_semantics


def audit(index_path, *, timing_path=None, offsets_s=tuple(np.arange(1, 9) * .25),
          action_horizon=10, execute_prefix=5, episode_ids=None):
    index_path = Path(index_path)
    timing = json.loads(Path(timing_path).read_text()) if timing_path else {}
    offsets = np.asarray(offsets_s)
    skill_counts, problems = Counter(), Counter()
    covered = np.zeros(len(offsets), np.int64)
    total_rows = no_query = short_events = crossed_h = crossed_prefix = 0
    records = []
    for entry in json.loads(index_path.read_text())["episodes"]:
        if episode_ids is not None and entry["episode_id"] not in episode_ids:
            continue
        reader = EpisodeReader(index_path.parent / entry["path"])
        meta = reader.meta
        record = {**timing.get("defaults", {}), **timing.get("episodes", {}).get(meta["episode_id"], {})}
        times, reason = None, None
        try:
            times = episode_timestamps(reader, record)
            gripper_semantics(meta, record)
        except ValueError as error:
            reason = str(error)
            problems["physical_clock_or_command_semantics_missing"] += 1
        action_frames = reader.array(meta["actions"]["frame_indices"])
        for event in meta["events"]:
            if event.get("skill_valid") and event.get("skill"):
                skill_counts[event["skill"]] += 1
            if times is not None:
                short_events += int(times[event["stop_frame"] - 1] - times[event["start_frame"]] < offsets[0])
        sample_rows = 0
        for arm in meta["arm_ids"]:
            events = reader.array(meta["event_index"][arm])[action_frames]
            for row, event_i in enumerate(events):
                if event_i < 0:
                    continue
                sample_rows += 1
                crossed_h += int(np.any(events[row:row + action_horizon] != event_i))
                crossed_prefix += int(np.any(events[row:row + execute_prefix] != event_i))
                if times is None:
                    continue
                event = meta["events"][int(event_i)]
                requested = times[action_frames[row]] + offsets
                future_rows = np.searchsorted(times, requested)
                take = np.minimum(future_rows, len(times) - 1)
                valid = ((future_rows < len(times))
                         & np.isclose(times[take], requested, rtol=0, atol=1e-6)
                         & (take >= event["start_frame"]) & (take < event["stop_frame"]))
                covered += valid
                no_query += int(not valid.any())
        total_rows += sample_rows
        records.append(dict(episode_id=meta["episode_id"], temporal_ready=times is not None,
                            command_ready=reason is None, samples=sample_rows, issue=reason))
    timed_rows = sum(r["samples"] for r in records if r["temporal_ready"])
    return dict(episodes=len(records), samples=total_rows, timed_samples=timed_rows,
                physical_time_ready_episodes=sum(r["temporal_ready"] for r in records),
                command_ready_episodes=sum(r["command_ready"] for r in records),
                offsets_s=offsets.tolist(), matched_same_event_by_offset=covered.tolist() if timed_rows else None,
                query_time_coverage_by_offset=(covered / timed_rows).tolist() if timed_rows else None,
                samples_without_same_event_query=no_query if timed_rows else None,
                events_shorter_than_first_query=short_events if timed_rows else None,
                action_horizon=action_horizon, crossing_action_chunks=crossed_h,
                execute_prefix=execute_prefix, crossing_execute_prefixes=crossed_prefix,
                skill_event_counts=dict(skill_counts), issues=dict(problems), episodes_detail=records,
                coverage_scope="time/event coverage only; projection and command-value validity are separate",
                manual_review="pending")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--subset", choices=("train", "val", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ids = None
    if args.subset != "all":
        if args.split is None:
            parser.error("--subset requires the fixed parent split via --split")
        split = json.loads(args.split.read_text())
        parents = set(split[args.subset])
        ids = set()
        for entry in json.loads(args.index.read_text())["episodes"]:
            meta = json.loads((args.index.parent / entry["path"]).read_text())
            if meta.get("source", {}).get("parent_trajectory_id", meta["episode_id"]) in parents:
                ids.add(meta["episode_id"])
    report = audit(args.index, timing_path=args.timing, episode_ids=ids)
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    print(json.dumps({k: v for k, v in report.items() if k != "episodes_detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
