"""Native regions, current shared renderer, and ordered concatenation for the fixed cohort."""
import argparse
from collections import Counter
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from fractions import Fraction
from functools import lru_cache
import html
import json
import os
from pathlib import Path
import sys
import time

import av
import numpy as np
from PIL import Image

from .review_source import ROOT, OUT, CACHE, sha, save, source_data, rgb
from .source import source_intervals
from .bind_instances import bind_events, resolve_geometry, RULES_PATH
from .native_gripper import bbox

LIBERO = Path('/data1/yuankai/robot-research/workspace/third_party/libero-native')
VIDEO = ROOT / 'check/test/video/review100_v1'
MERGED = VIDEO.parent / 'atomicvla_100_native_rule_v1.mp4'


def manifest():
    value = json.loads((OUT/'manifest.json').read_text())
    for name, digest in value['frozen_code'].items():
        if sha(name) != digest:
            raise ValueError(f'Frozen v1 rule / confirmed renderer changed: {name}')
    return value


def make_regions(job, *, out=OUT, cache=CACHE, bindings=None, version='v1', provenance=None,
                 save_scene_rgb=False, state_source_label='action-matched scene mirror'):
    sample, match = job
    destination = out / 'regions' / (sample['key']+'.json')
    if destination.exists():
        record = json.loads(destination.read_text())
        if provenance is not None and record.get('provenance') != provenance:
            raise ValueError(f'Region report uses different binding code: {destination}')
        return record
    started = time.monotonic()
    sys.path.insert(0, str(LIBERO))
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    import mujoco
    import robosuite.macros as macros
    data = source_data(sample)
    n = sample['frames']
    file = Path(sample['metadata']['file_path'])
    bddl = LIBERO / 'libero/libero/bddl_files' / file.parent.name / file.name.replace('_demo.hdf5','.bddl')
    scene = robosuite_parse_problem(str(bddl))
    annotation_error = None
    try:
        events = source_intervals(sample['annotation'], n) if sample['annotation'] else []
        if sample['annotation'] is None:
            annotation_error = 'Source annotation linkage is unresolved'
    except (ValueError, KeyError, TypeError) as error:
        events = []; annotation_error = str(error)
    bindings = (bind_events(scene, events, sample['task'], json.loads(RULES_PATH.read_text()))
                if bindings is None else deepcopy(bindings))
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=['agentview'],
        camera_heights=256, camera_widths=256, render_gpu_device_id=(sample['number']-1)%4, hard_reset=False)
    env.seed(0); env.reset()
    sim, robot = env.sim, env.robots[0]
    context = sim._render_context_offscreen
    original_groups = sim.model.geom_group.copy()
    original_options = context.vopt.geomgroup.copy()
    context.vopt.sitegroup[:] = 0
    gripper_names = list(robot.gripper.visual_geoms)
    gripper_ids = [sim.model.geom_name2id(name) for name in gripper_names]
    targets = []
    for event in bindings:
        if event['status'] not in {'rule_bound', 'manual_bound'}:
            continue
        try:
            event['geometry'] = resolve_geometry(env, {**event, 'status':'rule_bound'})
        except (ValueError, KeyError) as error:
            event['geometry'] = {'status': 'unresolved', 'reason': str(error)}
            continue
        names = event['geometry']['visual_geom_names']
        existing = next((i for i,t in enumerate(targets) if t['geom_names'] == names), None)
        if existing is None:
            existing = len(targets)
            targets.append({'instance': event['instance'], 'part': event['scope']['part'], 'geom_names': names,
                'frames': set(), 'geom_ids': [sim.model.geom_name2id(name) for name in names]})
        event['region_track'] = existing
        targets[existing]['frames'].update(range(event['start_frame'], event['stop_frame']))
        targets[existing]['frames'].add(event['goal_frame'])
    states = None
    if match and match['status'] == 'actions_exact_unique_match':
        if sha(match['excerpt_path']) != match['excerpt_sha256']:
            raise ValueError('Matched scene excerpt changed')
        with np.load(match['excerpt_path']) as archive:
            states = archive['states'].copy()
        if states.shape != (n, len(sim.get_state().flatten())):
            raise ValueError('Matched scene states do not fit the compiled model')
    count = len(targets)+1
    boxes = np.full((count,n,4), np.nan, np.float32)
    masks = np.zeros((count,n,256*256//8), np.uint8)
    valid = np.zeros((count,n), bool)
    alignment, eef_errors, scene_rgb, scene_frames = [], [], [], []

    def oriented(value):
        convention = 1 if macros.IMAGE_CONVENTION == 'opengl' else -1
        return value[::convention][::-1,::-1].copy()

    def region(index, frame, selected):
        sim.model.geom_group[:] = 5; sim.model.geom_group[selected] = 1
        context.vopt.geomgroup[:] = 0; context.vopt.geomgroup[1] = 1
        seg = oriented(sim.render(width=256,height=256,camera_name='agentview',segmentation=True))
        mask = (seg[...,0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,1],selected)
        box = bbox(mask)
        if box is not None:
            boxes[index,frame] = box
        masks[index,frame] = np.packbits(mask.reshape(-1))
        valid[index,frame] = True

    try:
        for frame in range(n):
            if states is not None:
                sim.set_state_from_flattened(states[frame]); sim.forward()
                if frame in {0,n//2,n-1}:
                    sim.model.geom_group[:] = original_groups; context.vopt.geomgroup[:] = original_options
                    restored = oriented(sim.render(width=256,height=256,camera_name='agentview'))
                    alignment.append({'frame': frame, 'scene_rgb_mae_vs_source': float(np.abs(restored.astype(float)-rgb(data,frame)).mean())})
                    if save_scene_rgb:
                        scene_frames.append(frame); scene_rgb.append(restored)
                for index, target in enumerate(targets, 1):
                    if frame in target['frames']:
                        region(index,frame,target['geom_ids'])
            # Independent gripper pass uses exact RLDS joints, not the mirror's arm observation.
            sim.data.qpos[robot._ref_joint_pos_indexes] = data['joints'][frame]
            sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = data['state'][frame,6:]
            sim.data.qvel[:] = 0; sim.forward()
            eef_errors.append(float(np.linalg.norm(sim.data.site_xpos[robot.eef_site_id]-data['state'][frame,:3])))
            region(0,frame,gripper_ids)
    finally:
        sim.model.geom_group[:] = original_groups; context.vopt.geomgroup[:] = original_options
        env.close()
    path = cache / 'regions' / (sample['key']+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
    extra = {'scene_frames':np.asarray(scene_frames,dtype=np.int64), 'scene_rgb':np.asarray(scene_rgb,dtype=np.uint8)} if save_scene_rgb else {}
    np.savez_compressed(path, boxes=boxes, mask_bits=masks, mask_valid=valid, **extra)
    for target in targets:
        target['frames'] = sorted(target['frames'])
        del target['geom_ids']
    result = {'number': sample['number'], 'key': sample['key'], 'frames': n, 'bindings': bindings,
        'annotation_error': annotation_error, 'targets': targets, 'gripper_geom_names': gripper_names,
        'region_extent': 'full_in_frame', 'gripper_pose_source': 'source RLDS joint_state and finger qpos',
        'object_pose_source': state_source_label if states is not None else 'missing',
        'source_cache_sha256': sample['source_cache_sha256'], 'bddl_sha256': sha(bddl), 'scene_match': match,
        'event_coverage': dict(Counter(sum(e['start_frame']<=i<e['stop_frame'] for e in events) for i in range(n))),
        'rules_sha256': sha(RULES_PATH), 'binder_sha256': sha(Path(__file__).with_name('bind_instances.py')),
        'script_sha256': sha(__file__), 'archive': str(path), 'archive_sha256': sha(path),
        'scene_alignment': alignment, 'max_eef_error_m': max(eef_errors),
        'gripper_boxes': int(np.isfinite(boxes[0]).all(axis=1).sum()),
        'object_boxes': int(np.isfinite(boxes[1:]).all(axis=2).sum()),
        'seconds': round(time.monotonic()-started,3), 'manual_review': 'pending'}
    if provenance is not None:
        result.update(version=version, provenance=provenance,
                      rules_sha256=provenance['catalog'], binder_sha256=provenance['binder'])
    save(destination,result)
    return result


def make_video(sample, *, out=OUT, video=VIDEO, version='v1', total=100):
    from scripts.preprocess.libero.render import Episode, Event, Region, Renderer, render_video
    report_path = out / 'renders' / (sample['key']+'.json')
    if report_path.exists():
        return json.loads(report_path.read_text())
    started = time.monotonic()
    record = json.loads((out/'regions'/(sample['key']+'.json')).read_text())
    data = source_data(sample)
    if sha(record['archive']) != record['archive_sha256']:
        raise ValueError('Native region archive changed')
    with np.load(record['archive']) as archive:
        boxes, bits, valid = archive['boxes'].copy(), archive['mask_bits'].copy(), archive['mask_valid'].copy()
    @lru_cache(maxsize=None)
    def image(frame):
        return rgb(data,frame)
    def lookup(target,frame):
        if not 0 <= frame < sample['frames'] or not valid[target,frame]:
            return None
        box = boxes[target,frame].tolist() if np.isfinite(boxes[target,frame]).all() else None
        mask = np.unpackbits(bits[target,frame]).reshape(256,256).astype(bool)
        return Region(box,mask)
    events = []
    for event in record['bindings']:
        track = event.get('region_track')
        name = event['instance']
        if name and event['scope']['part'] != 'whole':
            name += ' / ' + event['scope']['part']
        events.append(Event(event['source_id'],event['start_frame'],event['stop_frame'],event['goal_frame'],
            event['skill'],name or 'UNRESOLVED',
            (lambda frame, i=track+1: lookup(i,frame)) if track is not None else None))
    view = Episode(key=f"{sample['number']:03d}/{total} | {sample['key']}",frame_count=sample['frames'],rgb=image,
        source=f"AtomicVLA | {sample['number']:03d}/{total}",task=sample['task'],events=events,
        gripper_state=data['state'][:,6:],actions=data['actions'],eef_xyz=data['state'][:,:3],
        xyz_note='SOURCE EEF XYZ / state[:3] / NO RESAMPLING',gripper_region=lambda frame:lookup(0,frame),
        playback_fps=10.0,source_fps=None,annotation_label=f'NATIVE FULL-IN-FRAME / RULE {version.upper()}',review_status='pending',
        note='SCENE MATCH / RESIDUALS RECORDED' if record['object_pose_source'] != 'missing' else 'SCENE STATES MISSING')
    if any(e['status']=='manual_bound' for e in record['bindings']):
        view.annotation_label = 'NATIVE FULL-IN-FRAME / ASSISTANT INSTANCE REVIEW'
        view.note = 'MANUAL INSTANCES / USER REVIEW PENDING'
    unresolved = sum(e['status'] not in {'rule_bound','manual_bound'} for e in record['bindings'])
    if unresolved:
        view.note = f'{unresolved} EVENT BINDINGS UNRESOLVED'
    conflicts = [e for e in record['bindings'] if e.get('skill_check') == 'conflict']
    if conflicts:
        view.annotation_label = f'RULE {version.upper()} | SKILL CONFLICT: ' + ', '.join(f"{e['skill']} / {e['parsed']['action']}" for e in conflicts)
        view.note = 'SOURCE SKILL NEEDS REVIEW'
    destination = video / f"{sample['number']:03d}_{sample['key']}.mp4"
    result = render_video(view,destination)
    preview = out/'previews'/f"{sample['number']:03d}.jpg"; preview.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(Renderer(view).frame(sample['frames']//2)).save(preview,quality=90)
    result.update(number=sample['number'],key=sample['key'],video=str(destination.resolve()),sha256=sha(destination),
        renderer_sha256=sha(Path(__file__).parents[1]/'render.py'),regions_sha256=sha(out/'regions'/(sample['key']+'.json')),
        seconds=round(time.monotonic()-started,3))
    save(report_path,result)
    return result


def make_complete(job):
    make_regions(job)
    return make_video(job[0])


def batch(stage,workers,selected=None):
    data = manifest()
    samples = [s for s in data['samples'] if selected is None or s['number']==selected]
    if stage=='ready':
        matches = [row for path in (OUT/'scene_matches').glob('*.json')
                   for row in json.loads(path.read_text())['samples']]
        by_key = {s['key']:s for s in matches if s['status']=='actions_exact_unique_match'}
        function = make_complete
        jobs = [(s,by_key[s['key']]) for s in samples if s['key'] in by_key
                and not (OUT/'renders'/(s['key']+'.json')).exists()]
    elif stage=='regions':
        matches = json.loads((OUT/'scene_summary.json').read_text())
        by_key = {s['key']:s for s in matches['samples']}
        function, jobs = make_regions, [(s,by_key.get(s['key'])) for s in samples]
    else:
        function,jobs = make_video,samples
    done = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function,job) for job in jobs]
        for future in as_completed(futures):
            result = future.result(); done.append(result)
            print(json.dumps({'stage':stage,'complete':len(done),'total':len(jobs),'number':result['number'],
                              'frames':result['frames'],'seconds':result['seconds']}),flush=True)
    save(OUT/(stage+'_summary.json'),{'samples':sorted(done,key=lambda r:r['number']),
        'manifest_sha256':sha(OUT/'manifest.json'),'manual_review':'pending'})


def merge():
    data = manifest()
    clips = [json.loads((OUT/'renders'/(s['key']+'.json')).read_text()) for s in data['samples']]
    if len(clips)!=100 or [r['number'] for r in clips]!=list(range(1,101)):
        raise ValueError('Need all 100 clips in fixed manifest order')
    if MERGED.exists():
        raise FileExistsError(MERGED)
    temporary = MERGED.with_name(MERGED.stem+'.partial.mp4')
    chapters, offset = [], Fraction(0)
    with av.open(str(temporary),'w',options={'movflags':'+faststart'}) as output:
        with av.open(clips[0]['video']) as first:
            template = first.streams.video[0]
            stream = output.add_stream_from_template(template)
            configuration = template.codec_context.extradata
        output.metadata['title']='AtomicVLA | 100 full trajectories | native rule v1 | review pending'
        cursor, embedded = 0, []
        for sample,clip in zip(data['samples'],clips):
            embedded.append({'id':sample['number'],'start':cursor,'end':cursor+clip['frames'],
                'time_base':Fraction(1,10),'metadata':{'title':f"{sample['number']:03d}/100 | {sample['task']}"}})
            cursor += clip['frames']
        output.set_chapters(embedded)
        for sample,clip in zip(data['samples'],clips):
            if sha(clip['video'])!=clip['sha256']:
                raise ValueError('Rendered clip changed before merging')
            start = offset
            with av.open(clip['video']) as incoming:
                source = incoming.streams.video[0]
                if source.codec_context.extradata!=configuration:
                    raise ValueError('Clip codec configuration differs; cannot concatenate unchanged packets')
                for packet in incoming.demux(source):
                    if packet.dts is None:
                        continue
                    shift = offset / packet.time_base
                    if shift.denominator != 1:
                        raise ValueError('Non-integral chapter packet timestamp')
                    packet.pts += int(shift); packet.dts += int(shift); packet.stream=stream
                    output.mux(packet)
            offset += Fraction(clip['frames'],10)
            chapters.append({'number':sample['number'],'key':sample['key'],'task':sample['task'],
                'start_seconds':float(start),'end_seconds':float(offset),'frames':clip['frames'],
                'start_output_frame':int(start*10),'stop_output_frame':int(offset*10)})
    count=0
    with av.open(str(temporary)) as container:
        if len(container.chapters())!=100:
            raise ValueError('Merged video chapter metadata is incomplete')
        for frame in container.decode(video=0):
            if (frame.width,frame.height)!=(1920,1340) or abs(float(frame.pts*frame.time_base)-count/10)>1e-6:
                raise ValueError('Merged frame resolution / continuous timestamps mismatch')
            count+=1
    if count!=sum(c['frames'] for c in clips):
        raise ValueError('Merged video lost or duplicated source frames')
    temporary.replace(MERGED)
    save(OUT/'chapters.json',chapters)
    with (OUT/'manual_review.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['number','key','start_seconds','binding','gripper_bbox','object_bbox','event_goal','decision','notes'])
        writer.writeheader()
        for chapter in chapters:
            writer.writerow({k:chapter[k] for k in ['number','key','start_seconds']} | {'decision':'pending'})
    video_relative=os.path.relpath(MERGED,OUT)
    items=''.join(f"<tr><td>{c['number']:03d}</td><td><a href='#' onclick=\"document.querySelector('video').currentTime={c['start_seconds']};return false\">{c['start_seconds']:.1f}s</a></td><td>{html.escape(c['task'])}</td></tr>" for c in chapters)
    (OUT/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>AtomicVLA 100 review</title><h1>100 trajectories / manual review pending</h1><video controls preload="metadata" width="1280" src="{video_relative}"></video><table>{items}</table>')
    regions=[json.loads((OUT/'regions'/(s['key']+'.json')).read_text()) for s in data['samples']]
    statuses=Counter(e['status'] for r in regions for e in r['bindings'])
    with (OUT/'event_check.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['number','key','event','video_start_seconds','skill','description','instance','part','status','reason','manual_review'])
        writer.writeheader()
        for record,chapter in zip(regions,chapters):
            for event in record['bindings']:
                writer.writerow({'number':record['number'],'key':record['key'],'event':event['source_id'],
                    'video_start_seconds':chapter['start_seconds']+event['start_frame']/10,
                    'skill':event['skill'],'description':event['description'],'instance':event['instance'],
                    'part':event['scope']['part'] if event['scope'] else '', 'status':event['status'],
                    'reason':event.get('reason',''),'manual_review':'pending'})
    result={'video':str(MERGED.resolve()),'sha256':sha(MERGED),'bytes':MERGED.stat().st_size,'trajectories':100,
        'frames':count,'duration_seconds':float(offset),'width':1920,'height':1340,'playback_fps':10,
        'full_decode_verified':True,'source_frame_order_preserved':True,'embedded_chapters':100,'chapters':chapters,
        'binding_statuses':dict(statuses),'scene_missing':sum(r['object_pose_source']=='missing' for r in regions),
        'source_annotation_errors':sum(r['annotation_error'] is not None for r in regions),
        'manifest_sha256':sha(OUT/'manifest.json'),'frozen_code':data['frozen_code'],'manual_review':'pending'}
    save(OUT/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'chapters','frozen_code'}}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['ready','regions','render','merge'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    merge() if args.stage=='merge' else batch(args.stage,args.workers,args.sample)


if __name__=='__main__':
    main()
