"""Compare official states to the fixed RLDS cohort and render review artifacts."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import importlib
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw

from . import source

native=importlib.import_module('scripts.preprocess.libero.atomic_vla.review_native')
binding=importlib.import_module('scripts.preprocess.libero.atomic_vla.binding_v2')
motion=importlib.import_module('scripts.preprocess.libero.atomic_vla.binding_motion')
intervals=importlib.import_module('scripts.preprocess.libero.atomic_vla.source').source_intervals
sha,save=source.sha,source.save
OUT=source.OUT/'test'
VIDEO=OUT/'video'


def provenance():
    return {'binder':sha(binding.__file__),'catalog':sha(binding.CATALOG_PATH),'motion':sha(motion.__file__),
            'official_manifest':sha(source.OUT/'manifest.json'),'state_adapter':sha(__file__),
            'event_adapter':sha(intervals.__code__.co_filename)}


def model_check(env, match):
    """Confirm state ordering and explicit camera parameters before interpreting qpos."""
    info=match.get('model_file')
    if not info or not info['contains_xml']:
        return {'status':'xml_unavailable'}
    xml=ET.parse(info['path']).getroot()
    body=xml.find('worldbody')
    joint_names=[e.attrib['name'] for e in body.iter() if e.tag in {'joint','freejoint'} and 'name' in e.attrib]
    same_joints=joint_names==list(env.sim.model.joint_names)
    camera=body.find(".//camera[@name='agentview']")
    checks={}
    if camera is not None:
        cid=env.sim.model.camera_name2id('agentview')
        for key,actual in [('pos',env.sim.model.cam_pos[cid]),('quat',env.sim.model.cam_quat[cid]),('fovy',env.sim.model.cam_fovy[cid])]:
            if key in camera.attrib:
                expected=np.fromstring(camera.attrib[key],sep=' ')
                equal=np.allclose(expected,actual,atol=1e-6,rtol=0)
                if key=='quat':
                    equal |= np.allclose(expected,-actual,atol=1e-6,rtol=0)
                checks[key]={'official':expected.tolist(),'compiled':np.asarray(actual).tolist(),'equal':bool(equal)}
    return {'status':'compatible' if same_joints and all(c['equal'] for c in checks.values()) else 'mismatch',
            'joint_order_equal':same_joints,'joint_count':len(joint_names),'agentview':checks,'xml_sha256':info['sha256']}


def restore_track(env, states, bodies):
    sim,robot=env.sim,env.robots[0]
    joints,positions,objects,rotations=[],[],[],[]
    body_ids=[sim.model.body_name2id(name) for name in bodies]
    for state in states:
        sim.set_state_from_flattened(state); sim.forward()
        joints.append(sim.data.qpos[robot._ref_joint_pos_indexes].copy())
        positions.append(sim.data.site_xpos[robot.eef_site_id].copy())
        objects.append(sim.data.body_xpos[body_ids].copy())
        rotations.append(sim.data.body_xmat[body_ids].reshape(-1,3,3).copy())
    return {'joints':np.asarray(joints),'eef':np.asarray(positions),
            'objects':np.asarray(objects),'rotations':np.asarray(rotations)}


def robot_metrics(track, data):
    result={}
    n=len(data['state'])
    for offset in [-1,0,1]:
        start,stop=max(0,-offset),min(n,n-offset)
        joint=np.abs(track['joints'][start+offset:stop+offset]-data['joints'][start:stop])
        eef=np.linalg.norm(track['eef'][start+offset:stop+offset]-data['state'][start:stop,:3],axis=1)
        result[str(offset)]={'frames':stop-start,'joint_abs_max_rad':float(joint.max()),
                             'joint_abs_mean_rad':float(joint.mean()),'eef_mean_m':float(eef.mean()),
                             'eef_median_m':float(np.median(eef)),'eef_p95_m':float(np.percentile(eef,95)),
                             'eef_max_m':float(eef.max())}
    return {'same_row':result['0'],'offset_checks':result,'offset_applied':0}


def audit(job):
    sample,match=job
    started=time.monotonic()
    sys.path.insert(0,str(native.LIBERO))
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    file=Path(sample['metadata']['file_path'])
    bddl=native.LIBERO/'libero/libero/bddl_files'/file.parent.name/file.name.replace('_demo.hdf5','.bddl')
    scene=robosuite_parse_problem(str(bddl))
    events=intervals(sample['annotation'],sample['frames'])
    rows=binding.bind_events(scene,events,sample['task'])
    env=ControlEnv(bddl_file_name=str(bddl),use_camera_obs=False,has_offscreen_renderer=False,hard_reset=False)
    record={'number':sample['number'],'key':sample['key'],'task':sample['task'],'frames':sample['frames'],
            'match':match,'provenance':provenance(),'bddl_sha256':sha(bddl),'manual_review':'pending'}
    try:
        if match['status']=='actions_exact_unique_match':
            if sha(match['excerpt_path'])!=match['excerpt_sha256']:
                raise ValueError('Official state excerpt changed')
            model=model_check(env,match)
            record['model_check']=model
            if model['status']=='mismatch':
                raise ValueError(f"Official model ordering/camera mismatch in {sample['key']}: {model}")
            with np.load(match['excerpt_path']) as archive:
                states=archive['states'].copy()
                mapping=archive['source_to_official_indices'].copy()
            prior_path=source.atomic.CACHE/'raw_replay'/(sample['key']+'_states.npz')
            if prior_path.exists():
                with np.load(prior_path) as prior:
                    record['previous_raw_candidate_comparison']={
                        'path':str(prior_path),'sha256':sha(prior_path),
                        'states_identical':bool(np.array_equal(states,prior['states'])),
                        'mapping_identical':bool(np.array_equal(mapping,prior['source_to_raw_indices']))}
            if states.shape!=(sample['frames'],len(env.sim.get_state().flatten())):
                raise ValueError('Official state shape does not fit source frames / compiled model')
            data=source.atomic.source_data(sample)
            measured=motion.collect(env,scene,rows,states,data)
            rows=binding.bind_events(scene,events,sample['task'],motion=measured)
            for row in rows:
                if row['status']=='rule_bound':
                    row['geometry']=binding.resolve_geometry(env,row)
            bodies=sorted({e['geometry']['root_body'] for e in rows if e['geometry']['status']=='resolved'})
            track=restore_track(env,states,bodies)
            record.update(official_robot=robot_metrics(track,data),
                          mapping={'first_raw_frame':int(mapping[0]),'last_raw_frame':int(mapping[-1]),
                                   'removed_rows_between':int(mapping[-1]-mapping[0]+1-len(mapping)),
                                   'strictly_increasing':bool(np.all(np.diff(mapping)>0))})
            old_matches=json.loads((source.atomic.OUT/'scene_summary.json').read_text())['samples']
            old=next(m for m in old_matches if m['key']==sample['key'])
            if old['status']=='actions_exact_unique_match':
                with np.load(old['excerpt_path']) as archive:
                    old_track=restore_track(env,archive['states'],bodies)
                differences={}
                for i,name in enumerate(bodies):
                    distances=np.linalg.norm(track['objects'][:,i]-old_track['objects'][:,i],axis=1)
                    delta=np.einsum('nji,njk->nik',old_track['rotations'][:,i],track['rotations'][:,i])
                    angles=np.arccos(np.clip((np.trace(delta,axis1=1,axis2=2)-1)/2,-1,1))
                    differences[name]={'position_mean_m':float(distances.mean()),'position_max_m':float(distances.max()),
                                       'rotation_max_rad':float(angles.max())}
                record.update(existing_mirror_robot=robot_metrics(old_track,data),
                              official_vs_existing_mirror_objects=differences)
        else:
            for row in rows:
                row['motion']={'status':'no_matched_states','candidates':[]}
        record['bindings']=rows
    finally:
        env.close()
    record['seconds']=round(time.monotonic()-started,3)
    save(OUT/'alignments'/(sample['key']+'.json'),record)
    return record


def regions(job):
    sample,match=job
    record=json.loads((OUT/'alignments'/(sample['key']+'.json')).read_text())
    if record['provenance']!=provenance():
        raise ValueError('Run official state audit after code changes')
    return native.make_regions(job,out=OUT,cache=Path(source.manifest()['cache']),bindings=record['bindings'],
                               version='v2 official',provenance=record['provenance'],save_scene_rgb=True,
                               state_source_label='official LIBERO HDF5; image alignment pending review')


def comparison(sample):
    from scripts.preprocess.libero.render import Region, overlay, font, BLUE, GREEN
    record=json.loads((OUT/'regions'/(sample['key']+'.json')).read_text())
    data=source.atomic.source_data(sample)
    match=record['scene_match']
    if match['status']!='actions_exact_unique_match':
        return None
    with np.load(match['excerpt_path']) as archive:
        stored=archive['official_rgb'].copy(); mapping=archive['source_to_official_indices'].copy()
        stored_frames=archive['frames'].copy()
    with np.load(record['archive']) as archive:
        frames=archive['scene_frames'].copy(); restored=archive['scene_rgb'].copy()
        boxes=archive['boxes'].copy()
    assert np.array_equal(frames,stored_frames)
    canvas=Image.new('RGB',(1108,1000),'#171d23'); draw=ImageDraw.Draw(canvas)
    draw.text((16,12),f"{sample['number']:03d} | official LIBERO states vs source RLDS | review pending",font=font(17),fill='white')
    draw.text((16,40),sample['task'][:116],font=font(15),fill='white')
    for i,label in enumerate(['SOURCE RLDS RGB','OFFICIAL STORED OBS (128→256)','OFFICIAL STATE RESTORED','SOURCE + FULL BBOX']):
        draw.text((16+274*i,74),label,font=font(12),fill='#8ad8bc')
    measurements=[]
    for i,frame in enumerate(frames):
        original=source.atomic.rgb(data,int(frame))
        raw=np.asarray(Image.fromarray(stored[i][::-1,::-1]).resize((256,256),Image.Resampling.BILINEAR))
        shown=overlay(original,Region(boxes[0,frame]),BLUE)
        active=[e for e in record['bindings'] if e['start_frame']<=frame<e['stop_frame']]
        if len(active)==1 and active[0].get('region_track') is not None:
            shown=overlay(np.asarray(shown),Region(boxes[active[0]['region_track']+1,frame]),GREEN)
        for column,pixels in enumerate([original,raw,restored[i],shown]):
            canvas.paste(Image.fromarray(np.asarray(pixels)),(16+274*column,120+292*i))
        metric={'frame':int(frame),'official_frame':int(mapping[frame]),
                'restored_rgb_mae':float(np.abs(restored[i].astype(float)-original).mean()),
                'stored_observation_rgb_mae_after_rotate_resize':float(np.abs(raw.astype(float)-original).mean())}
        measurements.append(metric)
        draw.text((16,98+292*i),f"source {frame} ↔ official {mapping[frame]} | restored RGB MAE {metric['restored_rgb_mae']:.2f} / 255",font=font(14),fill='white')
    destination=OUT/'comparisons'/f"{sample['number']:03d}.jpg"; destination.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(destination,quality=92)
    result={'number':sample['number'],'key':sample['key'],'image':str(destination),
            'comparisons':measurements,'orientation':'official stored RGB rotated 180 degrees; resized for display only',
            'note':'Stored obs and states may use different timing; no time shift or geometric correction applied',
            'manual_review':'pending'}
    save(OUT/'comparisons'/(sample['key']+'.json'),result)
    return result


def render(job):
    return native.make_video(job[0],out=OUT,video=VIDEO,version='v2 official')


def report():
    manifest=source.manifest()
    matches=json.loads((source.OUT/'scene_summary.json').read_text())
    audits=[json.loads(p.read_text()) for p in sorted((OUT/'alignments').glob('*.json'))]
    regions=[json.loads(p.read_text()) for p in sorted((OUT/'regions').glob('*.json'))]
    images=[json.loads(p.read_text()) for p in sorted((OUT/'comparisons').glob('*.json'))]
    videos=[json.loads(p.read_text()) for p in sorted((OUT/'renders').glob('*.json'))]
    region_by_key={r['key']:r for r in regions}
    image_by_key={r['key']:r for r in images}
    video_by_key={r['key']:r for r in videos}
    goal_boxes,checked_frames=0,0
    complete_regions=[]
    for record in regions:
        with np.load(record['archive']) as archive:
            boxes,valid,masks=archive['boxes'],archive['mask_valid'],archive['mask_bits']
            assert boxes.shape[1]==record['frames'] and valid[0].all()
            for target,frame in zip(*np.where(valid)):
                mask=np.unpackbits(masks[target,frame]).reshape(256,256).astype(bool)
                box=native.bbox(mask)
                if box is None:
                    assert np.isnan(boxes[target,frame]).all()
                else:
                    np.testing.assert_array_equal(boxes[target,frame],box)
                checked_frames+=1
            complete=True
            for event in record['bindings']:
                index=event.get('region_track')
                available=index is not None and valid[index+1,event['goal_frame']]
                if index is not None:
                    target=record['targets'][index]
                    assert (target['instance'],target['part'])==(event['instance'],event['scope']['part'])
                goal_boxes+=bool(available and np.isfinite(boxes[index+1,event['goal_frame']]).all())
                complete &= bool(available and valid[index+1,event['start_frame']:event['stop_frame']].all())
            if complete:
                complete_regions.append(record['number'])
    for video in videos:
        assert sha(video['video'])==video['sha256']
    rows=[]
    for audit in sorted(audits,key=lambda r:r['number']):
        match=audit['match']; region=region_by_key.get(audit['key'])
        metrics=audit.get('official_robot',{}).get('same_row',{})
        image=image_by_key.get(audit['key'])
        row={'number':audit['number'],'key':audit['key'],'task':audit['task'],
             'suite':match['remote_file'].split('/')[0],'demo':match.get('demo'),
             'match_status':match['status'],'match_method':match.get('match_method'),
             'frames':audit['frames'],'eef_max_mm':metrics['eef_max_m']*1000 if metrics else None,
             'eef_median_mm':metrics['eef_median_m']*1000 if metrics else None,
             'joint_max_deg':float(np.rad2deg(metrics['joint_abs_max_rad'])) if metrics else None,
             'three_frame_rgb_mae':float(np.mean([c['restored_rgb_mae'] for c in image['comparisons']])) if image else None,
             'bound_events':sum(e['status']=='rule_bound' for e in audit['bindings']),
             'events':len(audit['bindings']),'regions_complete':audit['number'] in complete_regions,
             'video':video_by_key.get(audit['key'],{}).get('video',''),'manual_review':'pending'}
        if 'existing_mirror_robot' in audit:
            row['existing_mirror_eef_max_mm']=audit['existing_mirror_robot']['same_row']['eef_max_m']*1000
        rows.append(row)
    fields=['number','key','task','suite','demo','match_status','match_method','frames','eef_max_mm','eef_median_mm',
            'joint_max_deg','existing_mirror_eef_max_mm','three_frame_rgb_mae','bound_events','events','regions_complete','video','manual_review']
    with (source.OUT/'alignment.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    result={'repository':manifest['repository'],'revision':manifest['revision'],'tasks_fetched':matches['tasks'],
            'trajectories':len(rows),'frames':sum(r['frames'] for r in rows),
            'action_matches':dict(Counter(r['match_status'] for r in rows)),
            'matching_methods':dict(Counter(r['match_method'] for r in rows)),
            'suite_counts':dict(Counter(r['suite'] for r in rows)),
            'range_bytes':matches['range_bytes'],'full_remote_hdf5_hash_verified':False,
            'model_checks':dict(Counter(a.get('model_check',{}).get('status','not_checked') for a in audits)),
            'official_robot_eef_max_mm':max((r['eef_max_mm'] for r in rows if r['eef_max_mm'] is not None),default=None),
            'bindings':dict(Counter(e['status'] for a in audits for e in a['bindings'])),
            'bound_complete_trajectories':sum(r['bound_events']==r['events'] for r in rows),
            'complete_region_trajectories':complete_regions,'available_goal_boxes':goal_boxes,
            'mask_bbox_pairs_verified':checked_frames,'comparison_sheets':len(images),
            'review_samples':manifest['review_samples'],'rendered_samples':sorted(v['number'] for v in videos),
            'rendered_frames':sum(v['frames'] for v in videos),
            'focus_samples':[r for r in rows if r['number'] in {11,44,67}],
            'trajectory_checks':rows,'provenance':provenance(),'manual_review':'pending',
            'interpretation':'Exact actions establish correspondence, not pixel-identical replay. State/RGB residuals remain measured evidence, not an automatic acceptance threshold.'}
    save(source.OUT/'report.json',result)
    table=[]
    for row in rows:
        image=f"test/comparisons/{row['number']:03d}.jpg"
        link=f"<a href='{image}'>三帧对照</a>" if row['key'] in image_by_key else 'MISSING'
        if row['video']:
            video_path=os.path.relpath(row['video'],source.OUT)
            link+=f" · <a href='{video_path}' onclick=\"document.getElementById('review').src=this.href;return false\">视频</a>"
        values=[f'{row[k]:.2f}' if row[k] is not None else 'MISSING' for k in ['eef_max_mm','three_frame_rgb_mae']]
        table.append(f"<tr><td>{row['number']:03d}</td><td>{html.escape(row['task'])}</td><td>{row['demo']}</td>"
                     f"<td>{values[0]}</td><td>{values[1]}</td><td>{link}</td><td>pending</td></tr>")
    first=os.path.relpath(sorted(videos,key=lambda r:r['number'])[0]['video'],source.OUT) if videos else ''
    (source.OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>官方 LIBERO-40 对齐核验</title>'
        '<style>body{font:16px sans-serif;margin:24px;background:#f4f5f6;color:#17212b}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbc4ca}video{max-width:100%}</style>'
        '<h1>官方 LIBERO-40 → 当前 AtomicVLA / RLDS</h1>'
        '<p>原 RGB 与事件来自当前 RLDS / AtomicVLA；物体区域来自本轮官方 HDF5 状态，夹爪仍使用源关节恢复。人工待核验。</p>'
        '<p>蓝色：完整屏内夹爪；绿色：当前对象；黄色：同一 Event 对象的终点区域。动作匹配不等于像素一致；MAE 不是 bbox 正确率。</p>'
        f'<video id="review" controls preload="metadata" width="1100" src="{first}"></video>'
        '<table><tr><th>编号</th><th>任务</th><th>官方示范</th><th>末端最大偏差 mm</th><th>三帧 RGB MAE / 255</th><th>核验入口</th><th>人工</th></tr>'+''.join(table)+'</table>')
    print(json.dumps({k:v for k,v in result.items() if k not in {'trajectory_checks','provenance','complete_region_trajectories'}},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','regions','compare','render','report'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    if args.stage=='report':
        report(); return
    manifest=source.manifest()
    matches={s['key']:s for p in (source.OUT/'matches').glob('*.json') for s in json.loads(p.read_text())['samples']}
    samples=[s for s in manifest['samples'] if args.sample is None or s['number']==args.sample]
    if args.stage=='render' and args.sample is None:
        samples=[s for s in samples if s['number'] in manifest['review_samples']]
    if args.stage=='compare':
        for sample in samples:
            comparison(sample)
        return
    function={'audit':audit,'regions':regions,'render':render}[args.stage]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(function,(s,matches[s['key']])) for s in samples]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            print(json.dumps({'stage':args.stage,'done':i,'total':len(futures),'number':result['number'],
                              'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
