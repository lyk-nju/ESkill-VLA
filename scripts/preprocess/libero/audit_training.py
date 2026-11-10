"""Count Events, lengths, commands and actual ESkill supervision without decoding RGB.

Motion queries count saved main-camera frames, not seconds. Physical timestamps
are optional metadata. Training and audit share the same Event-local targets.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import (
    ARM_SLOTS, EpisodeReader, episode_timestamps, event_motion_targets, gripper_semantics,
)


def distribution(values):
    if not values:
        return None
    q = np.percentile(values, [0, 10, 50, 90, 100])
    return dict(count=len(values), mean=float(np.mean(values)), **dict(zip(
        ('min', 'p10', 'median', 'p90', 'max'), map(float, q))))


def event_summary(rows):
    lengths = [r['length_frames'] for r in rows]
    histogram = Counter()
    for length in lengths:
        label = next((f'{a}-{b}' for a, b in ((1, 5), (6, 10), (11, 20), (21, 40),
                      (41, 80), (81, 160)) if a <= length <= b), '161+')
        histogram[label] += 1
    return dict(events=len(rows), episodes=len({r['episode_id'] for r in rows}),
                action_rows=sum(r['action_rows'] for r in rows),
                event_length_frames=distribution(lengths), length_histogram_frames=dict(histogram),
                observed_span_s=distribution([r['observed_span_s'] for r in rows if r['observed_span_s'] is not None]),
                **{k: sum(r[k] for r in rows) for k in ('open', 'close', 'invalid_command', 'command_changes')})


def coverage_summary(records, offsets):
    if not records:
        return None
    result = dict(samples=len(records))
    for name in ('trace', 'gripper'):
        counts = np.array([r[name] for r in records], dtype=bool)
        per_sample = counts.sum(1)
        result[name] = dict(valid_by_offset=counts.sum(0).tolist(),
                            coverage_by_offset=counts.mean(0).tolist(),
                            valid_points_per_sample=np.bincount(per_sample, minlength=len(offsets) + 1).tolist(),
                            samples_without_targets=int((per_sample == 0).sum()))
    return result


def audit(index_path, *, timing_path=None, motion_stride=5, motion_queries=8, action_horizon=10, execute_prefix=5,
          episode_ids=None, split_path=None, subset='all'):
    index_path = Path(index_path)
    timing = json.loads(Path(timing_path).read_text()) if timing_path else {}
    membership = {}
    if split_path:
        split = json.loads(Path(split_path).read_text())
        membership = {parent: part for part in ('train', 'val') for parent in split[part]}
        if len(membership) != len(split['train']) + len(split['val']):
            raise ValueError('Parent trajectory occurs in both train and val')
    elif subset != 'all':
        raise ValueError('subset requires the fixed parent split')
    offsets = np.arange(1, motion_queries + 1) * motion_stride
    problems, skill_counts = Counter(), Counter()
    covered = np.zeros(len(offsets), np.int64)
    total_rows = no_query = short_events = crossed_h = crossed_prefix = 0
    records, events, coverage = [], [], []
    for entry in json.loads(index_path.read_text())['episodes']:
        if episode_ids is not None and entry['episode_id'] not in episode_ids:
            continue
        reader = EpisodeReader(index_path.parent / entry['path'])
        meta = reader.meta
        parent = meta['source'].get('parent_trajectory_id') or meta['episode_id']
        part = membership[parent] if split_path else 'unspecified'
        if subset != 'all' and part != subset:
            continue
        task = meta['instructions'][0]
        record = {**timing.get('defaults', {}), **timing.get('episodes', {}).get(meta['episode_id'], {})}
        times, convention, issues = None, None, {}
        # Command readiness is independent of clock readiness.
        for name, read in (('clock', lambda: episode_timestamps(reader, record)),
                           ('command', lambda: gripper_semantics(meta, record))):
            try:
                value = read()
                if name == 'clock':
                    times = value
                else:
                    convention = value
            except ValueError as error:
                issues[name] = str(error)
                problems[name + '_missing'] += 1
        action_frames = reader.array(meta['actions']['frame_indices'])
        commands = reader.array(meta['actions']['values'])
        command_valid = reader.array(meta['actions']['valid'])
        for ei, event in enumerate(meta['events']):
            skill = event['skill'] if event.get('skill_valid') and event.get('skill') else 'unknown'
            skill_counts[skill] += 1
            start, stop = event['start_frame'], event['stop_frame']
            in_event = reader.array(meta['event_index'][event['arm']])[action_frames] == ei
            rows = np.flatnonzero(in_event)
            counts = dict(open=0, close=0, invalid_command=len(rows), command_changes=0)
            if convention is not None:
                col = convention['channels'][event['arm']]
                raw = commands[rows, col]
                valid = command_valid[rows, col] & np.isfinite(raw) & (raw != 0)
                closed = raw * convention['close_sign'] > 0
                changes = (valid[1:] & valid[:-1] & (closed[1:] != closed[:-1]) & (np.diff(rows) == 1))
                counts = dict(open=int((valid & ~closed).sum()), close=int((valid & closed).sum()),
                              invalid_command=int((~valid).sum()), command_changes=int(changes.sum()))
            span = None if times is None else float(times[stop - 1] - times[start])
            short_events += int(stop - start <= offsets[0])
            events.append(dict(episode_id=meta['episode_id'], event_id=event['event_id'], arm=event['arm'],
                               split=part, task=task, skill=skill, boundary_valid=event['boundary_valid'],
                               start_frame=start, stop_frame=stop, length_frames=stop - start,
                               observed_span_s=span, action_rows=len(rows), **counts))
        # Read decision-time frames from the actual main-camera pairing, as Dataset does.
        camera = 'image'
        paired = reader.array(meta['actions']['conditioning_image_indices'][camera])
        if np.any(paired < 0):
            raise ValueError(f"{meta['episode_id']}: missing main-camera pairing")
        current_frames = reader.array(meta['cameras'][camera]['frame_indices'])[paired]
        sample_rows = 0
        for row, current in enumerate(current_frames):
            actual = None
            if convention is not None:
                _, actual, _ = event_motion_targets(reader, current, convention,
                                                    offsets_frames=offsets, camera=camera)
            for arm in meta['arm_ids']:
                event_stream = reader.array(meta['event_index'][arm])
                event_i = int(event_stream[current])
                if event_i < 0:
                    continue
                sample_rows += 1
                crossed_h += int(np.any(event_stream[action_frames[row:row + action_horizon]] != event_i))
                crossed_prefix += int(np.any(event_stream[action_frames[row:row + execute_prefix]] != event_i))
                event = meta['events'][event_i]
                camera_frames = reader.array(meta['cameras'][camera]['frame_indices'])
                requested = paired[row] + offsets
                matched = requested < len(camera_frames)
                take = camera_frames[np.minimum(requested, len(camera_frames) - 1)]
                valid = (matched & event['boundary_valid']
                         & (take >= event['start_frame']) & (take < event['stop_frame']))
                covered += valid
                no_query += int(not valid.any())
                if actual is not None:
                    slot = ARM_SLOTS[arm]
                    remaining = int(np.searchsorted(camera_frames, event['stop_frame']) - paired[row] - 1)
                    limits = motion_stride * np.array([1, 2, 4, motion_queries])
                    bucket = next((f'<= {limit}f' for limit in limits if remaining <= limit), f'> {offsets[-1]}f')
                    coverage.append(dict(skill=event['skill'], remaining=bucket,
                                         trace=actual['trace_valid'][slot], gripper=actual['gripper_valid'][slot]))
        total_rows += sample_rows
        records.append(dict(episode_id=meta['episode_id'], split=part, task=task, events=len(meta['events']),
                            temporal_ready=times is not None, command_ready=convention is not None,
                            samples=sample_rows, action_rows=len(action_frames), issues=issues))
    timed_rows = sum(r['samples'] for r in records if r['temporal_ready'])
    groups = {}
    for key in ('skill', 'task', 'split'):
        grouped = defaultdict(list)
        for event in events:
            grouped[event[key]].append(event)
        groups['by_' + key] = {name: event_summary(rows) for name, rows in sorted(grouped.items())}
    effective = coverage_summary(coverage, offsets)
    if effective:
        for key in ('skill', 'remaining'):
            effective['by_' + key] = {name: coverage_summary([r for r in coverage if r[key] == name], offsets)
                                      for name in sorted({r[key] for r in coverage})}
    return dict(episodes=len(records), tasks=len({r['task'] for r in records}), samples=total_rows,
                action_rows=sum(r['action_rows'] for r in records), timed_samples=timed_rows,
                physical_time_ready_episodes=sum(r['temporal_ready'] for r in records),
                command_ready_episodes=sum(r['command_ready'] for r in records),
                offsets_frames=offsets.tolist(), matched_same_event_by_offset=covered.tolist(),
                query_frame_coverage_by_offset=(covered / total_rows).tolist() if total_rows else None,
                samples_without_same_event_query=no_query,
                events_shorter_than_first_query=short_events,
                action_horizon=action_horizon, crossing_action_chunks=crossed_h,
                execute_prefix=execute_prefix, crossing_execute_prefixes=crossed_prefix,
                skill_event_counts=dict(skill_counts), event_statistics=event_summary(events),
                events_per_episode=distribution([r['events'] for r in records]), **groups,
                effective_supervision=effective, issues=dict(problems), episodes_detail=records, events_detail=events,
                coverage_scope='saved main-camera frames; effective coverage: command-ready sample-arms with an Event',
                duration_definition='saved length=stop-start; observed span=timestamps[stop-1]-timestamps[start], not inferred control duration',
                manual_review='pending')


def markdown(report):
    lines = ['# LIBERO-40 Event 与夹爪统计', '',
             f"轨迹 {report['episodes']} 条；任务 {report['tasks']} 个；动作行 {report['action_rows']:,}；"
             f"Event {report['event_statistics']['events']:,} 个。", '',
             '长度以保存帧数计算（stop−start），不是物理控制步数或秒数。开／闭统计是命令，不是实际开度。', '',
             '| 技能 | Event 数 | 动作行 | 长度中位数 | P10–P90 | 打开命令 | 闭合命令 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for skill, row in report['by_skill'].items():
        d = row['event_length_frames']
        lines.append(f"| {skill} | {row['events']} | {row['action_rows']} | {d['median']:g} | "
                     f"{d['p10']:g}–{d['p90']:g} | {row['open']} | {row['close']} |")
    lines += ['', '| 划分 | 轨迹 | Event | 动作行 |', '| --- | ---: | ---: | ---: |']
    for part, row in report['by_split'].items():
        lines.append(f"| {part} | {row['episodes']} | {row['events']} | {row['action_rows']} |")
    lines += ['', f"夹爪语义就绪 {report['command_ready_episodes']}/{report['episodes']}；"
              f"物理时钟就绪 {report['physical_time_ready_episodes']}/{report['episodes']}。"]
    if report['effective_supervision'] is None:
        lines += ['没有夹爪语义就绪的样本，无法计算有效监督覆盖率。']
    else:
        offset_text = '、'.join(map(str, report['offsets_frames']))
        lines += ['', f'Motion 固定查询 +{offset_text} 个保存主相机帧；不要求物理时间戳。', '',
                  f"| 监督 | 统计样本臂 | 无有效点 | 0…{len(report['offsets_frames'])} 有效点数量 |",
                  '| --- | ---: | ---: | --- |']
        s = report['effective_supervision']
        for name in ('trace', 'gripper'):
            lines.append(f"| {name} | {s['samples']} | {s[name]['samples_without_targets']} | "
                         f"{s[name]['valid_points_per_sample']} |")
    lines += ['', f"H={report['action_horizon']} 跨 Event：{report['crossing_action_chunks']:,}；"
              f"执行前缀 {report['execute_prefix']} 步跨 Event：{report['crossing_execute_prefixes']:,}。",
              '', 'JSON 包含逐任务、逐技能、逐划分及每个 Event 明细；同名 CSV 可筛选逐 Event。',
              '统计检查不等于视频人工验收。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--timing', type=Path, help='Optional verified clock for duration statistics only')
    parser.add_argument('--motion-stride', type=int, default=5)
    parser.add_argument('--split', type=Path)
    parser.add_argument('--subset', choices=('train', 'val', 'all'), default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.index, timing_path=args.timing, split_path=args.split, subset=args.subset, motion_stride=args.motion_stride)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        args.output.with_suffix('.md').write_text(markdown(report))
        with args.output.with_suffix('.csv').open('w') as f:
            if report['events_detail']:
                writer = csv.DictWriter(f, fieldnames=list(report['events_detail'][0]))
                writer.writeheader()
                writer.writerows(report['events_detail'])
    print(json.dumps({k: report[k] for k in ('episodes', 'tasks', 'action_rows', 'skill_event_counts',
                     'physical_time_ready_episodes', 'command_ready_episodes', 'event_statistics')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
