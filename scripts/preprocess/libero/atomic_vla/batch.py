"""Generate all-frame native regions, save episodes, and prepare LIBERO review videos."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
from multiprocessing import get_context
import os
from pathlib import Path
import random
import time

import numpy as np

from openpi.policies.eskill_dataset import load_episode
from scripts.preprocess import PROJECT_ROOT
from scripts.preprocess.artifacts import digest
from scripts.preprocess.export import validate_episode, write_index
from .export import Inputs, export_episode, read
from scripts.preprocess import DATA_ROOT
from .source import source_data

CHECK = DATA_ROOT / 'libero/atomic-vla/check/test'
OUT = CHECK / 'full_frame_v1'
STORAGE = Path('/data1/yuankai/robot-research/data/external/atomic_vla/full_frame_v1')
# Exported Episodes before calibration. calibrate.py reads this staging index and writes the
# training data; never point this at the live training directory.
EPISODES = Path('/data1/yuankai/robot-research/data/atomicdata/staging/libero40_regions/episodes')
VIDEO = CHECK / 'video/full_frame_v1'
_inputs = None  # set by main -> prepare


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def inputs():
    inp = _inputs
    return inp.samples(), inp.labels(), inp.manifest()


def prepare():
    samples, labels, original = inputs()
    tasks, suites = defaultdict(list), defaultdict(list)
    for sample in samples:
        tasks[sample['task']].append(sample)
        suites[sample['suite']].append(sample)
    rng = random.Random(42)
    reasons = defaultdict(list)
    for task in sorted(tasks):
        reasons[rng.choice(tasks[task])['annotation_id']].append('one_per_task_seed42')
    for sample in samples:
        if any(e['status'] == 'manual_bound' for e in labels[str(sample['annotation_id'])]['events']):
            reasons[sample['annotation_id']].append('assistant_instance_annotation')
    for suite, members in suites.items():
        worst = max(members, key=lambda s: read(_inputs.audit(s['annotation_id']))
                    ['robot_alignment']['same_row']['eef_max_m'])
        reasons[worst['annotation_id']].append('largest_state_alignment_residual_in_suite')
    for task, members in tasks.items():
        push = [s for s in members if any(e['skill'] == 'Push' for e in labels[str(s['annotation_id'])]['events'])]
        if push and not any(s['annotation_id'] in reasons for s in push):
            reasons[rng.choice(push)['annotation_id']].append('user_confirmed_push_task')
    source_files = [_inputs.sources / 'sources.json', _inputs.sources / 'manifest.json', _inputs.bindings]
    code_files = [Path(__file__), Path(__file__).with_name('review_native.py'),
                  Path(__file__).with_name('export.py'), Path(__file__).with_name('source.py'),
                  PROJECT_ROOT / 'scripts/preprocess/export.py',
                  PROJECT_ROOT / 'src/openpi/policies/eskill_dataset.py',
                  Path(__file__).parents[1] / 'view.py', Path(__file__).parents[1] / 'render.py']
    config = dict(scope='AtomicVLA LIBERO-40 full-frame native regions and saved-label review',
        trajectories=len(samples), frames=sum(s['frames'] for s in samples),
        events=sum(len(e['events']) for e in labels.values()), tasks=len(tasks),
        storage=str(STORAGE), episodes=str(EPISODES), region_extent='full_in_frame', camera='image',
        input_sha256={str(p.resolve()): digest(p) for p in source_files},
        runtime_sha256={str(p.resolve()): digest(p) for p in code_files},
        binding_release=original['rules'], review_annotation_ids=sorted(reasons),
        review_selection={str(k): v for k, v in sorted(reasons.items())},
        budget=dict(download_bytes=0, workers=4, storage_gib=10, expected_wall_hours=2),
        manual_review='pending')
    destination = OUT / 'manifest.json'
    if destination.exists():
        if read(destination) != config:
            raise ValueError('Batch inputs or code changed; use a new batch directory')
        return config
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['regions', 'video']:
        (STORAGE / name).mkdir(parents=True, exist_ok=True)
    EPISODES.mkdir(parents=True, exist_ok=True)
    (OUT / 'episodes').symlink_to(EPISODES, target_is_directory=True)
    (OUT / 'index.json').symlink_to(EPISODES.parent / 'index.json')
    VIDEO.parent.mkdir(parents=True, exist_ok=True)
    VIDEO.symlink_to(STORAGE / 'video', target_is_directory=True)
    save(destination, config)
    (OUT / 'README.md').write_text(
        '# LIBERO-40 全量逐帧区域与人工验收\n\n'
        f"范围：{len(samples):,} 条、{config['frames']:,} 帧、{config['events']:,} 个 Event、{len(tasks)} 个任务。\n\n"
        '复用固定 RLDS RGB／关节／动作、官方动作匹配场景、已规范化的事件与合并实例标签。'
        '夹爪使用 RLDS 关节，操作实例／部件使用对应官方场景状态；每个目标独立渲染，'
        '忽略外部遮挡并裁掉屏外部分，从 mask 求紧致 bbox。完全出画保留有效空 mask，bbox 无效。\n\n'
        '方法：逐条生成区域后保存 JSON＋NPZ，回读核对源状态、动作、XYZ、所有 mask／bbox、'
        '逐帧事件归属与各 Event 的同对象终点。已有区域仅在身份、事件、来源和像素检查通过后复用。'
        '绑定 rules 指向既有标签发布版；当前执行代码的真实 SHA256 单列于 manifest.runtime_sha256。\n\n'
        f"人工视频预选 {len(reasons)} 条：seed=42 每任务一条、全部 12 条助手实例补充、"
        '各套件末端偏差最大的一条，并保证覆盖 Push 任务；去重后的作者 ID 在 manifest 中固定。'
        '这些是验收样本，不是未见任务的泛化评测。\n\n'
        '预算：不下载，不运行 VLM／SAM 或策略；4 个工作进程，预计 2 小时内、数据盘新增低于 10 GiB。'
        '视频沿用原 1920×1340 布局，每个源步对应一视频帧，10 fps 仅为展示。\n\n'
        '路径：`episodes/` 指向数据盘标签，`regions/` 为逐条原生区域元数据，'
        '`checks/` 为保存回读检查，`index.json` 为全量标签索引；'
        '视频在 `../video/full_frame_v1/`，从保存标签回读生成。'
        '结束后查看 `index.html`、`manual_review.csv` 和 `report.json`。\n\n'
        '状态：执行中；用户人工验收 pending。740 条原状态对齐警示继续保留，'
        '逐帧生成成功不表示区域已与原 RGB 人工对齐通过。\n')
    return config


def check_saved(reader, sample, labels, region):
    """Compare saved arrays with native extraction and exact source values."""
    validate_episode(reader)
    meta, n = reader.meta, sample['frames']
    data = source_data(sample)
    for sequence, expected in [(meta['state'], np.concatenate([data['joints'], data['state']], axis=1)),
                               (meta['actions'], data['actions']),
                               (meta['effector_positions']['single'], data['state'][:, :3])]:
        np.testing.assert_array_equal(reader.array(sequence['values']), expected)
    if 'image_bytes' in data:
        main = meta['cameras']['image']['rgb']
        assert np.array_equal(reader._raw(main['uri'], main['key']), data['image_bytes'])
        assert np.array_equal(reader._raw(main['uri'], main['offsets_key']), data['image_offsets'])
    assert meta['quality']['manual_review'] == 'pending'
    assert ('state_alignment_exceeds_10mm' in meta['quality']['warnings']) == labels['state_alignment_exceeds_10mm']
    with np.load(region['archive'], allow_pickle=False) as archive:
        boxes, bits, available = (archive[k] for k in ['boxes', 'mask_bits', 'mask_valid'])
    coverage = np.zeros(n, np.int8)
    tracks = {0: meta['effector_regions']['single']['image']}
    goals = []
    empty_objects = []
    by_event = {e['source_id']: e for e in region['bindings']}
    for event, source in zip(meta['events'], labels['events']):
        assert event['source_annotation'] == source
        coverage[event['start_frame']:event['stop_frame']] += 1
        track = by_event[event['event_id']]['region_track'] + 1
        sequence = event['object_regions']['image']
        if track in tracks:
            assert sequence == tracks[track], 'Events for the same physical track reference different saved regions'
        tracks[track] = sequence
        goal = reader.goal(event['event_id'], 'image', include_mask=True)
        assert goal['frame'] == event['goal_frame'] and goal['rgb'] is not None
        assert goal['region'] is not None and goal['region']['mask'] is not None
        goals.append(bool(goal['region']['bbox'] is not None))
        for frame in range(event['start_frame'], event['stop_frame']):
            assert reader.event_at('single', frame)['event_id'] == event['event_id']
            if not np.isfinite(boxes[track, frame]).all():
                empty_objects.append(frame)
    assert len(meta['events']) == len(labels['events']) and np.all(coverage == 1)
    for track, sequence in tracks.items():
        np.testing.assert_array_equal(reader.array(sequence['boxes_xyxy']), boxes[track])
        np.testing.assert_array_equal(reader.array(sequence['masks_valid']), available[track])
        np.testing.assert_array_equal(reader.array(sequence['boxes_valid']),
                                      available[track] & np.isfinite(boxes[track]).all(axis=1))
        masks = reader.array(sequence['masks'])
        np.testing.assert_array_equal(np.packbits(masks.reshape(n, -1), axis=1), bits[track])
    return dict(frames=n, events=len(goals), mask_bbox_pairs=int(available.sum()),
                gripper_nonempty_frames=int(np.isfinite(boxes[0]).all(axis=1).sum()),
                active_object_nonempty_frames=n-len(empty_objects),
                active_object_empty_frames=empty_objects, nonempty_goal_boxes=sum(goals),
                saved_arrays_equal_native=True, saved_numeric_equal_source=True,
                source_rgb='original JPEG bytes referenced without resampling',
                manual_review='pending')


def build_one(job):
    from .review_native import make_regions
    sample, labels, manifest = job
    identity = sample['annotation_id']
    result_path = OUT / 'checks' / f'{identity:04d}.json'
    episode_path = EPISODES / f'{identity:04d}' / 'episode.json'
    previous = read(result_path) if result_path.exists() else None
    if previous is not None:
        if (digest(episode_path) != previous['episode_sha256'] or
                digest(episode_path.parent / 'arrays.npz') != previous['arrays_sha256']):
            raise ValueError('Previously completed saved episode changed')
    started = time.monotonic()
    region_path = OUT / 'regions' / (sample['key'] + '.json')
    reused = previous['reused_regions'] if previous is not None else None
    if not region_path.exists():
        for folder in [_inputs.bindings.parent / 'regions', _inputs.sources / 'review10/regions']:
            old = folder / region_path.name
            if old.exists():
                region_path.parent.mkdir(parents=True, exist_ok=True)
                region_path.write_bytes(old.read_bytes())
                reused = str(old)
                break
    provenance = manifest['rules']
    if region_path.exists():
        cached = read(region_path)
        provenance = cached['provenance']
        if reused is None and Path(cached['archive']).parent != STORAGE / 'regions':
            reused = cached['archive']
        if any(provenance.get(key) != value for key, value in manifest['rules'].items()):
            raise ValueError('Cached regions use a different binding release')
        if ('manual_annotations' in provenance and provenance['manual_annotations'] !=
                digest(_inputs.bindings.parent / 'annotations.json')):
            raise ValueError('Cached manual instance annotations changed')
    region = make_regions((sample, labels['source_scene']), out=OUT, cache=STORAGE,
        bindings=labels['events'], version='full frame v1', provenance=provenance,
        state_source_label='official LIBERO action-matched states; alignment warnings retained')
    if not episode_path.exists():
        export_episode(sample, labels, region_path, episode_path.parent, manifest=manifest)
    result = check_saved(load_episode(episode_path), sample, labels, region)
    result.update(annotation_id=identity, key=sample['key'], task=sample['task'], suite=sample['suite'],
        episode=str(episode_path), episode_sha256=digest(episode_path),
        arrays_sha256=digest(episode_path.parent / 'arrays.npz'),
        region_record=str(region_path), reused_regions=reused,
        state_alignment_exceeds_10mm=labels['state_alignment_exceeds_10mm'],
        assistant_binding=any(e['status'] == 'manual_bound' for e in labels['events']),
        seconds=round(time.monotonic()-started, 3))
    save(result_path, result)
    return result


def review_one(identity):
    from scripts.preprocess.libero.view import render_saved
    import av
    result_path = OUT / 'renders' / f'{identity:04d}.json'
    if result_path.exists():
        result = read(result_path)
        if digest(result['video']) != result['sha256']:
            raise ValueError('Completed review video changed')
        return result
    episode_path = EPISODES / f'{identity:04d}' / 'episode.json'
    video = VIDEO / f'{identity:04d}_review.mp4'
    result = render_saved(episode_path, video)
    with av.open(str(video)) as container:
        count = 0
        for count, frame in enumerate(container.decode(video=0), 1):
            if ((frame.width, frame.height) != (1920, 1340) or
                    abs(float(frame.pts*frame.time_base)-(count-1)/10) > 1e-6):
                raise ValueError('Review video resolution or timestamps changed')
    if count != result['frames']:
        raise ValueError('Review video does not contain every source frame')
    result.update(annotation_id=identity, video=str(video), sha256=digest(video),
                  all_frames_decoded=True, display_timestamps_checked=True)
    save(result_path, result)
    return result


def report():
    config = read(OUT / 'manifest.json')
    samples, labels, _ = inputs()
    checks = {int(p.stem): read(p) for p in (OUT / 'checks').glob('*.json')}
    videos = {int(p.stem): read(p) for p in (OUT / 'renders').glob('*.json')}
    complete = len(checks) == len(samples)
    if complete:
        write_index(OUT / 'index.json', [checks[s['annotation_id']]['episode'] for s in samples])
    fields = ['annotation_id', 'suite', 'task', 'frames', 'saved', 'state_alignment_warning',
              'assistant_binding', 'empty_object_frames', 'review_video', 'binding', 'gripper_bbox',
              'object_bbox', 'event_goal', 'decision', 'notes']
    review_path = OUT / 'manual_review.csv'
    previous = {}
    if review_path.exists():
        with review_path.open() as stream:
            previous = {int(row['annotation_id']): row for row in csv.DictReader(stream)}
    with review_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            identity = sample['annotation_id']
            checked = checks.get(identity, {})
            video = videos.get(identity, {}).get('video', '')
            row = dict(annotation_id=identity, suite=sample['suite'], task=sample['task'],
                frames=sample['frames'], saved=identity in checks,
                state_alignment_warning=labels[str(identity)]['state_alignment_exceeds_10mm'],
                assistant_binding=any(e['status'] == 'manual_bound' for e in labels[str(identity)]['events']),
                empty_object_frames=len(checked.get('active_object_empty_frames', [])),
                review_video=os.path.relpath(video, OUT) if video else '', decision='pending')
            row.update({field: previous[identity][field] for field in
                        ['binding', 'gripper_bbox', 'object_bbox', 'event_goal', 'decision', 'notes']
                        if identity in previous})
            writer.writerow(row)
    selected = config['review_annotation_ids']
    summary = dict(trajectories=len(samples), tasks=config['tasks'], saved_episodes=len(checks),
        frames=sum(r['frames'] for r in checks.values()), events=sum(r['events'] for r in checks.values()),
        mask_bbox_pairs=sum(r['mask_bbox_pairs'] for r in checks.values()),
        active_object_empty_frames=sum(len(r['active_object_empty_frames']) for r in checks.values()),
        nonempty_goal_boxes=sum(r['nonempty_goal_boxes'] for r in checks.values()),
        state_alignment_warnings=sum(r['state_alignment_exceeds_10mm'] for r in checks.values()),
        assistant_binding_episodes=sum(r['assistant_binding'] for r in checks.values()),
        reused_region_episodes=sum(r['reused_regions'] is not None for r in checks.values()),
        review_videos=len(videos), review_frames=sum(r['frames'] for r in videos.values()),
        pending_review_videos=[i for i in selected if i not in videos],
        all_frame_export_complete=complete, manual_review='pending')
    save(OUT / 'report.json', summary)
    rows = []
    for sample in sorted((s for s in samples if s['annotation_id'] in selected), key=lambda s:(s['suite'], s['task'], s['annotation_id'])):
        identity = sample['annotation_id']
        clip = videos.get(identity)
        relative = os.path.relpath(clip['video'], OUT) if clip else ''
        link = f'<a href="{html.escape(relative)}" onclick="document.querySelector(\'video\').src=this.href;return false">{identity:04d}</a>' if clip else f'{identity:04d} / rendering pending'
        flags = ', '.join(config['review_selection'][str(identity)])
        if labels[str(identity)]['state_alignment_exceeds_10mm']:
            flags += ', state alignment warning'
        rows.append(f'<tr><td>{link}</td><td>{html.escape(sample["suite"])}</td><td>{html.escape(sample["task"])}</td><td>{html.escape(flags)}</td></tr>')
    (OUT / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>LIBERO-40 bbox review</title>'
        '<h1>LIBERO-40 all-frame bbox / manual review pending</h1>'
        f'<p>{len(checks)} / {len(samples)} saved episodes; {len(videos)} / {len(selected)} selected review videos.</p>'
        '<p>Blue: gripper; green: current object; yellow: same object at Event endpoint. '
        'Full in-frame geometry; external occlusion ignored. State residual is not bbox error ground truth.</p>'
        '<p><a href="manual_review.csv">All episodes: manual review CSV</a> | <a href="report.json">Technical report</a> | '
        '<a href="index.json">Saved label index</a></p>'
        '<video controls preload="none" width="960"></video><table border="1" cellpadding="6">'
        '<tr><th>Author ID / video</th><th>Suite</th><th>Task</th><th>Review reason</th></tr>' + ''.join(rows) + '</table>')
    print(json.dumps(summary), flush=True)
    return summary


def main():
    global _inputs
    parser = argparse.ArgumentParser(description=__doc__)
    Inputs.add_arguments(parser)
    parser.add_argument('stage', choices=['prepare', 'build', 'review', 'report'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--annotation-id', type=int)
    parser.add_argument('--all', action='store_true', help='Render every saved episode instead of the fixed review selection')
    args = parser.parse_args()
    _inputs = Inputs(args.sources, args.bindings)
    config = prepare()
    if args.stage == 'prepare':
        print(json.dumps(config), flush=True)
        return
    if args.stage == 'report':
        report()
        return
    samples, labels, manifest = inputs()
    manifest = {key: manifest[key] for key in
                ['annotation_path', 'annotation_sha256', 'revision', 'annotation_revision', 'rules']}
    if args.stage == 'build':
        selected = [s for s in samples if args.annotation_id is None or s['annotation_id'] == args.annotation_id]
        jobs = [(s, labels[str(s['annotation_id'])], manifest) for s in selected]
        function = build_one
    else:
        jobs = [s['annotation_id'] for s in samples] if args.all else config['review_annotation_ids']
        if args.annotation_id is not None:
            jobs = [args.annotation_id]
        function = review_one
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        pending = {pool.submit(function, job): job for job in jobs}
        failures = []
        for index, future in enumerate(as_completed(pending), 1):
            job = pending[future]
            identity = job[0]['annotation_id'] if args.stage == 'build' else job
            try:
                result = future.result()
            except Exception as error:
                failure = dict(stage=args.stage, annotation_id=identity, error=repr(error))
                save(OUT / 'failures' / f'{args.stage}_{identity:04d}.json', failure)
                failures.append(failure)
                print(json.dumps(failure), flush=True)
                continue
            print(json.dumps(dict(stage=args.stage, completed=index, total=len(jobs),
                                  annotation_id=result['annotation_id'], frames=result['frames'])), flush=True)
    report()
    if failures:
        raise RuntimeError(f'{len(failures)} episode jobs failed; see the batch failures directory')


if __name__ == '__main__':
    main()
