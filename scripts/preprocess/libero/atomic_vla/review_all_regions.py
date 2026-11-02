"""Native keyframe checks and ten full review videos for the full-dataset audit."""
from collections import defaultdict
import json
import sys
import time

import numpy as np
from PIL import Image,ImageDraw

from .review_all import OUT,CACHE,VIDEO,LIBERO,manifest,provenance,raw,signals
from . import review_native as native


def keyframes(sample,events):
    frames={0,sample['frames']//2,sample['frames']-1}
    for event in events:frames.update([event['start_frame'],(event['start_frame']+event['stop_frame']-1)//2,event['goal_frame']])
    return sorted(frames)


def regions_task(job):
    sys.path.insert(0,str(LIBERO))
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    import robosuite.macros as macros
    bddl,jobs=job
    env=OffScreenRenderEnv(bddl_file_name=bddl,camera_names=['agentview'],camera_heights=256,camera_widths=256,
        render_gpu_device_id=(jobs[0][0]['number']-1)%4,hard_reset=False)
    env.seed(0);env.reset();sim,robot=env.sim,env.robots[0]
    context=sim._render_context_offscreen;original_groups=sim.model.geom_group.copy()
    original_options=context.vopt.geomgroup.copy();context.vopt.sitegroup[:]=0
    gripper_ids=[sim.model.geom_name2id(n) for n in robot.gripper.visual_geoms]

    def oriented(value):
        convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
        return value[::convention][::-1,::-1].copy()

    def mask_for(ids):
        sim.model.geom_group[:]=5;sim.model.geom_group[ids]=1
        context.vopt.geomgroup[:]=0;context.vopt.geomgroup[1]=1
        seg=oriented(sim.render(width=256,height=256,camera_name='agentview',segmentation=True))
        return (seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,1],ids)

    try:
        for sample,match in jobs:
            destination=OUT/'keyframe_regions'/f"{sample['annotation_id']}.json"
            if destination.exists():continue
            started=time.monotonic();audit=json.loads((OUT/'audits'/f"{sample['annotation_id']}.json").read_text())
            record={'annotation_id':sample['annotation_id'],'key':sample['key'],'rules':provenance(),
                'region_extent':'full_in_frame','scope':'Event start/middle/goal and three episode RGB diagnostics',
                'manual_review':'pending'}
            try:
                data=raw.source_data(sample);rows=audit['bindings'];frames=keyframes(sample,rows)
                states=None
                if (sample['source_link']=='linked' and match and match['status']=='actions_exact_unique_match'
                        and audit.get('model_check',{}).get('status')=='compatible' and 'audit_error' not in audit):
                    with np.load(match['excerpt_path']) as archive:states=archive['states'].copy()
                requests=defaultdict(dict)
                if states is not None:
                    for event in rows:
                        if event['status']!='rule_bound' or event['geometry']['status']!='resolved':continue
                        target=(event['instance'],event['scope']['part'])
                        selected={event['start_frame'],(event['start_frame']+event['stop_frame']-1)//2,event['goal_frame']}
                        selected.update(f for f in [0,sample['frames']//2,sample['frames']-1] if event['start_frame']<=f<event['stop_frame'])
                        ids=[sim.model.geom_name2id(n) for n in event['geometry']['visual_geom_names']]
                        for frame in selected:requests[frame][target]=(event['source_id'],ids)
                checks,bits,boxes,images,errors=[],[],[],[],[]
                lookup={};diagnostics={0,sample['frames']//2,sample['frames']-1}
                def keep(role,frame,mask,event=None,target=None):
                    box=native.bbox(mask);i=len(checks)
                    checks.append({'role':role,'frame':frame,'event':event,'target':list(target) if target else None,
                        'bbox':box.tolist() if isinstance(box,np.ndarray) else box,
                        'area_pixels':int(mask.sum()),'mask_index':i})
                    boxes.append(box if box is not None else [np.nan]*4);bits.append(np.packbits(mask.reshape(-1)))
                    lookup[role,frame,target]=i
                for frame in frames:
                    if states is not None:
                        sim.set_state_from_flattened(states[frame]);sim.forward()
                        if frame in diagnostics:
                            sim.model.geom_group[:]=original_groups;context.vopt.geomgroup[:]=original_options
                            restored=oriented(sim.render(width=256,height=256,camera_name='agentview'))
                            images.append({'frame':frame,'rgb':restored,
                                'rgb_mae':float(np.abs(restored.astype(float)-raw.rgb(data,frame)).mean())})
                        for target,(event,ids) in requests[frame].items():keep('object',frame,mask_for(ids),event,target)
                    sim.data.qpos[robot._ref_joint_pos_indexes]=data['joints'][frame]
                    sim.data.qpos[robot._ref_gripper_joint_pos_indexes]=data['state'][frame,6:]
                    sim.data.qvel[:]=0;sim.forward()
                    errors.append(float(np.linalg.norm(sim.data.site_xpos[robot.eef_site_id]-data['state'][frame,:3])))
                    keep('gripper',frame,mask_for(gripper_ids))
                goals=[]
                for event in rows:
                    target=(event['instance'],event['scope']['part']) if event['scope'] else None
                    i=lookup.get(('object',event['goal_frame'],target))
                    goals.append({'event':event['source_id'],'available':i is not None,
                        'nonempty':i is not None and checks[i]['bbox'] is not None})
                path=CACHE/'keyframe_regions'/f"{sample['key']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,boxes=np.asarray(boxes,np.float32),mask_bits=np.asarray(bits,np.uint8),
                    scene_frames=np.asarray([p['frame'] for p in images],np.int64),
                    scene_rgb=np.asarray([p['rgb'] for p in images],np.uint8))
                record.update(checks=checks,goals=goals,all_event_goals_available=all(g['available'] for g in goals),
                    archive=str(path),archive_sha256=raw.sha(path),gripper_eef_max_m=max(errors),
                    scene_rgb_checks=[{k:v for k,v in p.items() if k!='rgb'} for p in images])
                comparison(sample,audit,record,data,images,lookup)
            except Exception as error:record['region_error']=repr(error)
            record['seconds']=round(time.monotonic()-started,3);raw.save(destination,record)
    finally:
        sim.model.geom_group[:]=original_groups;context.vopt.geomgroup[:]=original_options;env.close()
    return {'file':bddl,'trajectories':len(jobs)}


def comparison(sample,audit,record,data,images,lookup):
    from scripts.preprocess.libero.render import Region,overlay,font,BLUE,GREEN
    if len(images)!=3:return
    canvas=Image.new('RGB',(838,1010),'#171d23');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),f"annotation {sample['annotation_id']} | native full-in-frame | review pending",font=font(16),fill='white')
    draw.text((12,37),sample['task'][:96],font=font(14),fill='white')
    for col,label in enumerate(['SOURCE RLDS','OFFICIAL STATE RESTORED','SOURCE + KEYFRAME BBOX']):
        draw.text((12+274*col,70),label,font=font(12),fill='#8ad8bc')
    for i,image in enumerate(images):
        frame=image['frame'];original=raw.rgb(data,frame)
        gi=lookup['gripper',frame,None];shown=overlay(original,Region(record['checks'][gi]['bbox']),BLUE)
        active=[e for e in audit['bindings'] if e['start_frame']<=frame<e['stop_frame']]
        if len(active)==1 and active[0]['scope']:
            event=active[0];target=(event['instance'],event['scope']['part']);oi=lookup.get(('object',frame,target))
            if oi is not None:shown=overlay(np.asarray(shown),Region(record['checks'][oi]['bbox']),GREEN)
        draw.text((12,99+292*i),f"frame {frame} | restored RGB MAE {image['rgb_mae']:.2f} / 255",font=font(14),fill='white')
        for col,pixels in enumerate([original,image['rgb'],shown]):
            canvas.paste(Image.fromarray(np.asarray(pixels)),(12+274*col,122+292*i))
    path=OUT/'comparisons'/f"{sample['annotation_id']:04d}.jpg";path.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(path,quality=90);record['comparison']=str(path)


def full_video(job):
    sample,match=job;audit=json.loads((OUT/'audits'/f"{sample['annotation_id']}.json").read_text())
    out=OUT/'review10'
    native.make_regions(job,out=out,cache=CACHE/'review10',bindings=audit['bindings'],
        version='all rules v1',provenance=manifest()['rules'],state_source_label='official LIBERO; alignment diagnostics recorded')
    result=native.make_video(sample,out=out,video=VIDEO,version='all rules v1',total=len(manifest()['samples']))
    return {'file':result['video'],'trajectories':1,'annotation_id':sample['annotation_id']}
