"""Evidence sheets and explicit instance annotations for the twelve reviewed episodes."""
import argparse
from collections import defaultdict
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from fractions import Fraction
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

from . import review_all as dataset,review_native as native

OUT=dataset.ROOT/'check/test/manual_binding12'
CACHE=dataset.CACHE.parent/'manual_binding12'
VIDEO=dataset.ROOT/'check/test/video/manual_binding12'
COLORS=('#d78aff','#ffb65c')


def selected():
    dataset.manifest()
    with (dataset.OUT/'binding_issues.csv').open() as stream:
        ids={int(r['annotation_id']) for r in csv.DictReader(stream)}
    return [s for s in dataset.sample_list() if s['annotation_id'] in ids]


def inspection_frames(events):
    rows=[]
    for event in events:
        fractions=[0,.5,.75,.9,1] if event['skill']=='Pick' else [0,.25,.5,.75,1]
        rows.append([event['start_frame']+round((event['stop_frame']-event['start_frame']-1)*x) for x in fractions])
    return rows


def inspect():
    sys.path.insert(0,str(dataset.LIBERO))
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from scripts.preprocess.libero.render import font,overlay,Region
    import mujoco
    import robosuite.macros as macros
    samples=selected();assert len({s['bddl'] for s in samples})==1
    env=OffScreenRenderEnv(bddl_file_name=samples[0]['bddl'],camera_names=['agentview'],camera_heights=256,camera_widths=256,
        render_gpu_device_id=0,hard_reset=False)
    env.seed(0);env.reset();sim,robot=env.sim,env.robots[0]
    scene=robosuite_parse_problem(samples[0]['bddl'])
    instances=sorted(dataset.binding.SceneIndex(scene).entities('moka pot'));assert len(instances)==2
    objects=env.env.objects_dict
    roots=[sim.model.body_name2id(objects[n].root_body) for n in instances]
    visual=[[sim.model.geom_name2id(g) for g in objects[n].visual_geoms] for n in instances]
    collision=[{sim.model.geom_name2id(g) for g in objects[n].contact_geoms} for n in instances]
    fingers=[{sim.model.geom_name2id(g) for g in robot.gripper.important_geoms[side]} for side in ['left_finger','right_finger']]
    gripper={sim.model.geom_name2id(g) for g in robot.gripper.contact_geoms}
    context=sim._render_context_offscreen;original_groups=sim.model.geom_group.copy()
    original_options=context.vopt.geomgroup.copy();context.vopt.sitegroup[:]=0

    def oriented(value):
        convention=1 if macros.IMAGE_CONVENTION=='opengl' else -1
        return value[::convention][::-1,::-1].copy()

    try:
        for sample in samples:
            identity=sample['annotation_id'];audit=json.loads((dataset.OUT/'audits'/f'{identity}.json').read_text())
            events=audit['bindings'];frames_by_event=inspection_frames(events)
            wanted={f for row in frames_by_event for f in row};images={};boxes={};native_first=None
            data=dataset.raw.source_data(sample);match=audit['match']
            with np.load(match['excerpt_path']) as archive:states=archive['states'].copy()
            positions=[];contacts=[];bilateral=[];eef_errors=[]
            for frame,state in enumerate(states):
                sim.set_state_from_flattened(state);sim.forward()
                positions.append(sim.data.body_xpos[roots].copy())
                eef_errors.append(float(np.linalg.norm(sim.data.site_xpos[robot.eef_site_id]-data['state'][frame,:3])))
                pairs=[(int(c.geom1),int(c.geom2)) for c in sim.data.contact[:sim.data.ncon]]
                touches=[{a for a,b in pairs if b in geoms}|{b for a,b in pairs if a in geoms} for geoms in collision]
                contacts.append([bool(t&gripper) for t in touches]);bilateral.append([all(bool(t&f) for f in fingers) for t in touches])
                if frame not in wanted:continue
                if frame==0:
                    sim.model.geom_group[:]=original_groups;context.vopt.geomgroup[:]=original_options
                    native_first=oriented(sim.render(width=256,height=256,camera_name='agentview'))
                image=Image.fromarray(dataset.raw.rgb(data,frame))
                frame_boxes=[]
                for ids,color in zip(visual,COLORS,strict=True):
                    sim.model.geom_group[:]=5;sim.model.geom_group[ids]=1
                    context.vopt.geomgroup[:]=0;context.vopt.geomgroup[1]=1
                    seg=oriented(sim.render(width=256,height=256,camera_name='agentview',segmentation=True))
                    mask=(seg[...,0]==int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[...,1],ids)
                    box=native.bbox(mask);frame_boxes.append(box)
                    image=overlay(np.asarray(image),Region(box),color)
                draw=ImageDraw.Draw(image)
                for number,(box,color) in enumerate(zip(frame_boxes,COLORS),1):
                    if box:draw.text((box[0],max(0,box[1]-19)),str(number),font=font(16),fill=color,stroke_width=1,stroke_fill='black')
                images[frame]=image;boxes[frame]=frame_boxes
            positions=np.asarray(positions);contacts=np.asarray(contacts);bilateral=np.asarray(bilateral)
            metrics=[]
            for event in events:
                left,right=event['start_frame'],event['stop_frame']
                metrics.append({'event':event['source_id'],'skill':event['skill'],'description':event['description'],
                    'candidates':[{'instance':name,'raw_contact_frames':int(contacts[left:right,j].sum()),
                        'raw_bilateral_frames':int(bilateral[left:right,j].sum()),
                        'max_displacement_m':float(np.linalg.norm(positions[left:right,j]-positions[left,j],axis=1).max())}
                        for j,name in enumerate(instances)],
                    'eef_error_max_m':max(eef_errors[left:right])})
            canvas=Image.new('RGB',(1320,1320),'#111922');draw=ImageDraw.Draw(canvas)
            draw.text((12,10),f"annotation {identity} | SOURCE RGB + NATIVE CANDIDATES | assistant review",font=font(21),fill='white')
            draw.text((12,40),'1 = moka_pot_1 (purple) | 2 = moka_pot_2 (orange) | not final labels',font=font(19),fill='white')
            for row,(event,frames) in enumerate(zip(events,frames_by_event,strict=True)):
                top=76+row*308
                draw.text((12,top),f"{event['source_id']} | {event['skill']} | [{event['start_frame']},{event['stop_frame']}) | {event['description']}",font=font(16),fill='white')
                for col,frame in enumerate(frames):
                    draw.text((12+260*col,top+23),f'frame {frame}',font=font(15),fill='#9fd4c7')
                    canvas.paste(images[frame],(12+260*col,top+44))
            sheet=OUT/'evidence'/f'{identity:04d}.jpg';sheet.parent.mkdir(parents=True,exist_ok=True);canvas.save(sheet,quality=94)
            anchor=Image.new('RGB',(800,300),'#111922');ad=ImageDraw.Draw(anchor)
            for col,(label,pixels) in enumerate([('SOURCE FRAME 0',dataset.raw.rgb(data,0)),('SOURCE + INSTANCE IDS',images[0]),('RESTORED FRAME 0',native_first)]):
                ad.text((10+264*col,8),label,font=font(14),fill='white');anchor.paste(Image.fromarray(np.asarray(pixels)),(10+264*col,34))
            anchor_path=OUT/'anchors'/f'{identity:04d}.jpg';anchor_path.parent.mkdir(parents=True,exist_ok=True);anchor.save(anchor_path,quality=94)
            track=CACHE/'tracks'/f'{identity}.npz';track.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(track,positions=positions,contact=contacts,bilateral_contact=bilateral,eef_error=np.asarray(eef_errors))
            dataset.raw.save(OUT/'evidence'/f'{identity}.json',{'annotation_id':identity,'key':sample['key'],
                'source_cache_sha256':sample['source_cache_sha256'],'official_state_sha256':match['excerpt_sha256'],
                'instances':instances,'frames_by_event':frames_by_event,'initial_boxes':boxes[0],
                'sheet':str(sheet),'anchors':str(anchor_path),'track':str(track),'track_sha256':dataset.raw.sha(track),
                'metrics':metrics,'note':'Raw contacts and body motion are supporting evidence; official-state alignment residuals remain.'})
            print(json.dumps({'inspection_sheet':identity,'frames':sample['frames']}),flush=True)
    finally:
        sim.model.geom_group[:]=original_groups;context.vopt.geomgroup[:]=original_options;env.close()


def apply_manual(sample,automatic,annotations):
    """Apply explicit reviewed entries without changing automatic decisions or source fields."""
    rows=deepcopy(automatic);by_event={e['source_id']:e for e in rows}
    for patch in annotations['events']:
        if patch['annotation_id']!=sample['annotation_id']:continue
        if patch['key']!=sample['key'] or patch['source_cache_sha256']!=sample['source_cache_sha256']:
            raise ValueError('Manual annotation belongs to a different source episode')
        row=by_event[patch['event']]
        if row['status']=='rule_bound':raise ValueError('Manual entries only fill unresolved automatic events')
        if any(row[k]!=patch[k] for k in ['start_frame','stop_frame','goal_frame','skill']):
            raise ValueError('Manual annotation event bounds or skill changed')
        previous={k:row.get(k) for k in ['status','reason','instance','scope']}
        row.update(status='manual_bound',instance=patch['instance'],scope=patch['scope'],
            geometry=patch['geometry'],reason='assistant_visual_instance_annotation',
            automatic_result=previous,manual_assignment={
                'reviewer':'assistant_visual_review','evidence_frames':patch['evidence_frames'],
                'observation':patch['observation'],'annotation_record':'manual_binding12/annotations.json',
                'user_review':'pending'})
        row['evidence']=[*row.get('evidence',[]),{'method':'assistant_visual_review',
            'source':'source RGB + native instance anchors + raw contact/body traces',
            'target':[patch['instance'],patch['scope']['part']],'frames':patch['evidence_frames']}]
    return rows


def record():
    sys.path.insert(0,str(dataset.LIBERO))
    from libero.libero.envs.env_wrapper import ControlEnv
    samples=selected();manual=json.loads((OUT/'visual_decisions.json').read_text())
    decisions={r['annotation_id']:r for r in manual['decisions']}
    assert set(decisions)=={s['annotation_id'] for s in samples}
    env=ControlEnv(bddl_file_name=samples[0]['bddl'],use_camera_obs=False,has_offscreen_renderer=False,hard_reset=False)
    patches=[]
    try:
        for sample in samples:
            identity=sample['annotation_id'];decision=decisions[identity]
            audit=json.loads((dataset.OUT/'audits'/f'{identity}.json').read_text())
            evidence=json.loads((OUT/'evidence'/f'{identity}.json').read_text())
            assert evidence['source_cache_sha256']==sample['source_cache_sha256']
            assert len(decision['event_instances'])==len(audit['bindings'])==4
            for i,(event,instance) in enumerate(zip(audit['bindings'],decision['event_instances'],strict=True)):
                assert instance in evidence['instances']
                if event['status']=='rule_bound':
                    assert event['instance']==instance and event['scope']['part']=='whole'
                    continue
                pair=i//2;frames=decision['evidence_frames'][pair]
                left=audit['bindings'][pair*2]['start_frame'];right=audit['bindings'][pair*2+1]['stop_frame']
                assert all(left<=f<right for f in frames)
                scope={'part':'whole','body_suffix':None}
                geometry=native.resolve_geometry(env,{'status':'rule_bound','instance':instance,'scope':scope})
                patches.append({'annotation_id':identity,'key':sample['key'],'event':event['source_id'],
                    'start_frame':event['start_frame'],'stop_frame':event['stop_frame'],'goal_frame':event['goal_frame'],
                    'skill':event['skill'],'source_cache_sha256':sample['source_cache_sha256'],
                    'official_state_sha256':audit['match']['excerpt_sha256'],'instance':instance,'scope':scope,
                    'geometry':geometry,'evidence_frames':frames,'observation':decision['observation'],
                    'evidence_sheet_sha256':dataset.raw.sha(evidence['sheet']),
                    'initial_anchor_sha256':dataset.raw.sha(evidence['anchors'])})
    finally:env.close()
    assert len(patches)==34
    m=dataset.manifest()
    annotations={'format':'eskill.atomic.manual_instances.v1','reviewer':'assistant_visual_review',
        'source_annotation_sha256':m['annotation_sha256'],
        'automatic_manifest_sha256':dataset.raw.sha(dataset.OUT/'manifest.json'),
        'visual_decisions_sha256':dataset.raw.sha(OUT/'visual_decisions.json'),
        'scope':'Only the 34 unresolved events in the requested twelve episodes',
        'events':patches,'user_review':'pending'}
    dataset.raw.save(OUT/'annotations.json',annotations)
    episodes={};automatic_count=0;manual_count=0
    keep=['source_id','arm','start_frame','stop_frame','goal_frame','skill','description','source_skill',
        'source_start_frame','source_end_frame','source_kind','status','instance','scope','geometry',
        'boundary_correction','skill_correction','manual_assignment','automatic_result']
    for sample in dataset.sample_list():
        identity=sample['annotation_id'];audit=json.loads((dataset.OUT/'audits'/f'{identity}.json').read_text())
        rows=apply_manual(sample,audit['bindings'],annotations)
        assert all(e['status'] in {'rule_bound','manual_bound'} for e in rows)
        automatic_count+=sum(e['status']=='rule_bound' for e in rows);manual_count+=sum(e['status']=='manual_bound' for e in rows)
        episodes[str(identity)]={'key':sample['key'],'frames':sample['frames'],
            'source_cache':sample['source_cache'],'source_cache_sha256':sample['source_cache_sha256'],
            'source_scene':audit['match'],
            'state_alignment_exceeds_10mm':audit['robot_alignment']['same_row']['eef_max_m']>.01,
            'events':[{k:e[k] for k in keep if k in e} for e in rows]}
        if identity in decisions:
            dataset.raw.save(OUT/'bindings'/f'{identity}.json',{'annotation_id':identity,
                'bindings':rows,'automatic_audit':str(dataset.OUT/'audits'/f'{identity}.json'),
                'reviewer':'assistant_visual_review','user_review':'pending'})
    dataset.raw.save(OUT/'merged_bindings.json',{'source_annotation_sha256':m['annotation_sha256'],
        'automatic_rules':m['rules'],'manual_annotations_sha256':dataset.raw.sha(OUT/'annotations.json'),
        'episodes':episodes,'automatic_events':automatic_count,'manual_events':manual_count,'user_review':'pending'})
    summary={'trajectories':len(episodes),'events':automatic_count+manual_count,'automatic_events':automatic_count,
        'assistant_annotated_events':manual_count,'assistant_reviewed_trajectories':len(samples),
        'remaining_instance_gaps':0,'state_alignment_warnings_unchanged':740,'user_review':'pending'}
    dataset.raw.save(OUT/'label_report.json',summary);print(json.dumps(summary),flush=True)


def render_one(sample):
    identity=sample['annotation_id']
    audit=json.loads((dataset.OUT/'audits'/f'{identity}.json').read_text())
    annotations=json.loads((OUT/'annotations.json').read_text())
    rows=apply_manual(sample,audit['bindings'],annotations)
    provenance={**dataset.manifest()['rules'],'manual_annotations':dataset.raw.sha(OUT/'annotations.json')}
    native.make_regions((sample,audit['match']),out=OUT,cache=CACHE,bindings=rows,
        version='manual instances 12',provenance=provenance,state_source_label='official LIBERO; alignment risks remain')
    result=native.make_video(sample,out=OUT,video=VIDEO,version='manual instances 12',total=1693)
    return result


def render(workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(render_one,s) for s in selected()]
        for n,f in enumerate(as_completed(futures),1):
            r=f.result();print(json.dumps({'rendered':n,'total':len(futures),'key':r['key']}),flush=True)


def merge():
    import av
    samples=selected();clips=[json.loads((OUT/'renders'/f"{s['key']}.json").read_text()) for s in samples]
    merged=VIDEO.parent/'atomicvla_manual_binding12.mp4';offset=Fraction(0);chapters=[]
    with av.open(str(merged),'w',options={'movflags':'+faststart'}) as output:
        with av.open(clips[0]['video']) as first:
            stream=output.add_stream_from_template(first.streams.video[0]);configuration=first.streams.video[0].codec_context.extradata
        cursor=0;embedded=[]
        for sample,clip in zip(samples,clips,strict=True):
            embedded.append({'id':sample['annotation_id'],'start':cursor,'end':cursor+clip['frames'],
                'time_base':Fraction(1,10),'metadata':{'title':f"annotation {sample['annotation_id']} | assistant instance review"}})
            cursor+=clip['frames']
        output.set_chapters(embedded)
        for sample,clip in zip(samples,clips,strict=True):
            assert dataset.raw.sha(clip['video'])==clip['sha256']
            with av.open(clip['video']) as incoming:
                source=incoming.streams.video[0];assert source.codec_context.extradata==configuration
                for packet in incoming.demux(source):
                    if packet.dts is None:continue
                    shift=offset/packet.time_base;assert shift.denominator==1
                    packet.pts+=int(shift);packet.dts+=int(shift);packet.stream=stream;output.mux(packet)
            chapters.append({'annotation_id':sample['annotation_id'],'key':sample['key'],'start_seconds':float(offset),
                'frames':clip['frames'],'video':clip['video']});offset+=Fraction(clip['frames'],10)
    count=0
    with av.open(str(merged)) as incoming:
        assert len(incoming.chapters())==12
        for frame in incoming.decode(video=0):
            assert (frame.width,frame.height)==(1920,1340) and abs(float(frame.pts*frame.time_base)-count/10)<1e-6
            count+=1
    assert count==4982
    dataset.raw.save(OUT/'chapters.json',chapters)
    dataset.raw.save(OUT/'video_report.json',{'trajectories':12,'frames':count,'duration_seconds':float(offset),
        'video':str(merged.resolve()),'sha256':dataset.raw.sha(merged),'full_decode_verified':True,
        'reviewer':'assistant_visual_review','user_review':'pending'})
    annotations=json.loads((OUT/'annotations.json').read_text());items=[];review=[]
    for chapter in chapters:
        identity=chapter['annotation_id'];patches=[p for p in annotations['events'] if p['annotation_id']==identity]
        bindings=json.loads((OUT/'bindings'/f'{identity}.json').read_text())['bindings']
        instances=[e['instance'] for e in bindings]
        sequence=' → '.join(n.removeprefix('moka_pot_') for n in instances)
        items.append(f"<tr><td>{identity}</td><td>{html.escape(sequence)}</td><td>{len(patches)}</td>"
            f"<td><a href='#' onclick=\"document.querySelector('video').currentTime={chapter['start_seconds']};return false\">{chapter['start_seconds']:.1f}s</a>"
            f" · <a href='{os.path.relpath(chapter['video'],OUT)}'>单条视频</a> · <a href='evidence/{identity:04d}.jpg'>检查图</a></td><td>pending</td></tr>")
        review.append({'annotation_id':identity,'instances':'; '.join(instances),
            'assistant_added_events':len(patches),'decision':'pending','notes':''})
    path=OUT/'user_review.csv'
    if not path.exists():dataset.table(path,review,list(review[0]))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>12 条实例补充核验</title>'
        '<style>body{font:16px sans-serif;margin:24px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}video{max-width:100%}</style>'
        '<h1>12 条助手视觉补充：实例身份待用户核验</h1><p>当前新增34个manual_bound，原14个自动绑定保留。此表中的2/1是moka_pot_2/moka_pot_1，不是执行顺序编号。</p>'
        '<p>检查图紫色为候选实例1，橙色为候选实例2；最终视频仍用蓝色夹爪、绿色当前对象、黄色终点对象。原有状态对齐风险保留。</p>'
        f'<video controls preload="metadata" width="1200" src="{os.path.relpath(merged,OUT)}"></video>'
        '<p><a href="annotations.json">34 个独立补充标注</a> · <a href="merged_bindings.json">全量合并实例标签</a></p>'
        '<table><tr><th>标注 ID</th><th>四个 Event 的实例</th><th>本次补充数</th><th>核验材料</th><th>用户</th></tr>'+''.join(items)+'</table>')
    print(json.dumps({'trajectories':12,'frames':count,'seconds':float(offset),'video':str(merged)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['inspect','record','render','merge'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='inspect':inspect()
    elif args.stage=='record':record()
    elif args.stage=='render':render(args.workers)
    elif args.stage=='merge':merge()


if __name__=='__main__':main()
