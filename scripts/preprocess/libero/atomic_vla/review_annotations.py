"""Apply the approved boundary convention; render source skill conflicts for review."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import csv
from fractions import Fraction
import hashlib
import html
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from . import review_source as raw
from .source import source_intervals

OUT = raw.ROOT/'check/test/annotation_review'
VIDEO = raw.ROOT/'check/test/video/skill_conflicts'
MERGED = VIDEO.parent/'atomicvla_skill_conflicts_33.mp4'
CACHE = raw.CACHE.parent/'annotation_review/source'
VERBS = {'pick up':'Pick','place':'Place','push':'Push','open':'Open','close':'Close','turn':'Turn'}


def paths(reviewed=False):
    if reviewed:
        return OUT/'push_reviewed',VIDEO.parent/'push_reviewed',VIDEO.parent/'atomicvla_push_reviewed_33.mp4'
    return OUT,VIDEO,MERGED


def table(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def prepare():
    from scripts.preprocess.libero.check_raw import atomic_positions
    source_manifest=raw.OUT/'manifest.json'
    original=json.loads(source_manifest.read_text())
    annotation_path=raw.RAW/'libero_lerobot.json'
    assert raw.sha(annotation_path)==original['source_annotation_sha256']
    annotations=json.loads(annotation_path.read_text())
    normalized,changes,conflicts={},[],[]
    checked_frames=0
    for identity,annotation in annotations.items():
        n=annotation['all_frames']
        events=source_intervals(annotation,n,apply_reviewed_skills=False)
        coverage=np.zeros(n,np.int8)
        for event in events:
            coverage[event['start_frame']:event['stop_frame']]+=1
            before=event.get('boundary_correction')
            if before:
                changes.append({'annotation_id':int(identity),'event':event['source_id'],
                    'start_frame':event['start_frame'],'source_end_frame':event['source_end_frame'],
                    'previous_stop':before['source_stop_frame'],'stop_frame':event['stop_frame'],
                    'previous_goal':before['source_goal_frame'],'goal_frame':event['goal_frame']})
            verb=re.match(r'^(pick up|place|push|open|close|turn)\b',event['description'],re.I)
            described=VERBS[verb[1].lower()] if verb else None
            if described and described!=event['skill']:
                conflicts.append({'annotation_id':int(identity),'event':event['source_id'],
                    'source_skill':event['skill'],'description_action':described,
                    'description':event['description'],'manual_review':'pending'})
        if not np.all(coverage==1):
            raise ValueError(f'Non-unique or missing frame ownership in annotation {identity}')
        checked_frames+=n
        normalized[identity]={'frames':n,'events':events}
    positions={v:k for k,v in atomic_positions(raw.RAW).items()}
    repository=json.loads((raw.RAW/'repository.json').read_text())
    shard_files={}
    for file in repository['siblings']:
        match=re.search(r'tfrecord-(\d+)-of-',file['rfilename'])
        if match: shard_files[file['rfilename'].split('/')[0],int(match[1])]=file
    cohort={s['annotation_candidate_id']:s['number'] for s in original['samples']}
    samples=[]
    for number,identity in enumerate(sorted({c['annotation_id'] for c in conflicts}),1):
        suite,shard,row=positions[identity]
        annotation=annotations[str(identity)]
        samples.append({'number':number,'key':f'{suite}_s{shard:03d}_r{row:03d}',
            'annotation_candidate_id':identity,'cohort_number':cohort.get(identity),
            'suite':suite,'shard':shard_files[suite,shard]['rfilename'],'record_index':row,
            'annotation':annotation,'frames':annotation['all_frames'],
            'task':annotation['segments'][0]['action_name'],
            'conflicts':[c for c in conflicts if c['annotation_id']==identity],
            'source_cache':str(CACHE/f'{identity}.npz')})
    selected={s['shard'] for s in samples}
    files=[v for v in repository['siblings'] if v['rfilename'] in selected]
    additional=sum(f['size'] for f in files if not (raw.RAW/f['rfilename']).exists())
    if additional>1.5*1024**3: raise ValueError('Conflict source download exceeds the declared budget')
    manifest={'annotation_revision':'33d2f98574a52838b9db34b334bd5cc6cbe57f50',
        'annotation_path':str(annotation_path),'annotation_sha256':raw.sha(annotation_path),
        'repository':repository['id'],'revision':repository['sha'],'files':files,'samples':samples,
        'selection':'Every event whose explicit description action differs from its source skill; no subsampling',
        'boundary_policy':'[start,stop); shared boundary belongs to following Event; goal=stop-1',
        'boundary_adapter_sha256':raw.sha(Path(__file__).with_name('source.py')),
        'script_sha256':raw.sha(__file__),'additional_download_bytes':additional,'manual_review':'pending'}
    raw.save(OUT/'events.json',{'source_sha256':manifest['annotation_sha256'],
        'boundary_policy':manifest['boundary_policy'],'episodes':normalized})
    raw.save(OUT/'manifest.json',manifest)
    table(OUT/'boundary_changes.csv',changes,['annotation_id','event','start_frame','source_end_frame',
        'previous_stop','stop_frame','previous_goal','goal_frame'])
    table(OUT/'skill_conflicts.csv',conflicts,['annotation_id','event','source_skill','description_action','description','manual_review'])
    summary={'episodes':len(annotations),'events':sum(len(v['events']) for v in normalized.values()),
        'corrected_boundaries':len(changes),'checked_frame_ownership':checked_frames,
        'remaining_overlapping_frames':0,'missing_event_frames':0,
        'skill_conflicts':len(conflicts),'conflict_trajectories':len(samples),
        'skill_counts':dict(Counter(e['skill'] for v in normalized.values() for e in v['events'])),
        'skills_changed':0,'source_annotation_sha256':manifest['annotation_sha256'],
        'boundary_adapter_sha256':manifest['boundary_adapter_sha256'],'manual_review':'pending'}
    raw.save(OUT/'boundary_report.json',summary)
    print(json.dumps(summary),flush=True)


def approve():
    """Record the user's group-level skill decision and export corrected labels."""
    original=json.loads((OUT/'manifest.json').read_text())
    assert raw.sha(original['annotation_path'])==original['annotation_sha256']
    annotations=json.loads(Path(original['annotation_path']).read_text())
    episodes,changes={},[]
    for identity,annotation in annotations.items():
        n=annotation['all_frames'];events=source_intervals(annotation,n)
        before=source_intervals(annotation,n,apply_reviewed_skills=False)
        for event,previous in zip(events,before,strict=True):
            assert all(event[k]==previous[k] for k in ['start_frame','stop_frame','goal_frame'])
            if event['skill']!=previous['skill']:
                changes.append({'annotation_id':int(identity),'event':event['source_id'],
                    'source_skill':previous['skill'],'skill':event['skill'],
                    'reason':event['skill_correction']['reason']})
        episodes[identity]={'frames':n,'events':events}
    expected={(c['annotation_id'],c['event']) for s in original['samples'] for c in s['conflicts']}
    assert {(c['annotation_id'],c['event']) for c in changes}==expected
    review={'reviewed_at':'2026-11-01','scope':'Skill category of the delivered 33-conflict group; no bbox or scene-alignment acceptance',
        'reviewer':'user_in_conversation','decision':'Push',
        'user_statement':'我看了确实是push技能，他是压住碗然后用摩擦力往外移动',
        'reason':'压住操作物体，利用摩擦力沿支撑面平移',
        'source_annotation_sha256':original['annotation_sha256'],
        'source_manifest_sha256':raw.sha(OUT/'manifest.json'),
        'reviewed_video_sha256':raw.sha(MERGED),'corrections':changes,
        'skill_review':'user_confirmed'}
    raw.save(OUT/'skill_review.json',review)
    out,_,_=paths(True)
    manifest={**original,'parent_manifest_sha256':raw.sha(OUT/'manifest.json'),
        'boundary_adapter_sha256':raw.sha(Path(__file__).with_name('source.py')),
        'script_sha256':raw.sha(__file__),'skill_review_sha256':raw.sha(OUT/'skill_review.json'),
        'manual_review':'skill_only_confirmed'}
    raw.save(out/'manifest.json',manifest)
    raw.save(out/'events.json',{'source_sha256':original['annotation_sha256'],
        'boundary_policy':original['boundary_policy'],'skill_review':review,'episodes':episodes})
    table(out/'skill_changes.csv',changes,['annotation_id','event','source_skill','skill','reason'])
    for name in ['manual_review.csv','skill_conflicts.csv']:
        path=OUT/name
        with path.open(newline='') as stream:
            reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
        for row in rows:
            if int(row['annotation_id']) in {c['annotation_id'] for c in changes}:
                if 'decision' in row:
                    row.update(decision='Push',notes='用户观看合并视频后确认本组技能为 Push：压住操作物体，通过摩擦力平移。')
                if 'manual_review' in row: row['manual_review']='user_confirmed_Push'
        table(path,rows,fields)
    counts=Counter(e['skill'] for episode in episodes.values() for e in episode['events'])
    report={'episodes':len(episodes),'events':sum(counts.values()),'skills_changed':len(changes),
        'skill_counts':dict(counts),'boundaries_changed_by_skill_review':0,
        'source_annotation_sha256':original['annotation_sha256'],
        'skill_review_sha256':manifest['skill_review_sha256'],'manual_review':'skill_only_confirmed'}
    raw.save(out/'label_report.json',report)
    print(json.dumps(report),flush=True)


def fetch(workers):
    manifest=json.loads((OUT/'manifest.json').read_text())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda file:raw.download(file,manifest['revision']),manifest['files']))


def extract():
    from scripts.preprocess.libero.check_raw import arrays,metadata,normalized
    from scripts.preprocess.libero.rlds import load_records
    manifest=json.loads((OUT/'manifest.json').read_text())
    for file in manifest['files']:
        assert raw.sha(raw.RAW/file['rfilename'])==file['lfs']['sha256']
    for sample,data in load_records(raw.RAW,manifest['samples']):
        state,actions,joints=arrays(data)
        tasks={normalized(t.decode()) for t in data['steps/language_instruction']}
        assert tasks=={normalized(sample['task'])} and len(state)==sample['frames']
        assert np.isfinite(state).all() and np.isfinite(actions).all() and np.isfinite(joints).all()
        images=data['steps/observation/image']
        assert len(images)==sample['frames']
        path=Path(sample['source_cache']);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,state=state,actions=actions,joints=joints,
            image_bytes=np.frombuffer(b''.join(images),np.uint8),
            image_offsets=np.r_[0,np.cumsum([len(v) for v in images])].astype(np.int64))
        sample.update(metadata=metadata(data),source_cache_sha256=raw.sha(path))
    assert all('source_cache_sha256' in s for s in manifest['samples'])
    fingerprints=[hashlib.sha256(raw.source_data(s)['image_bytes']).hexdigest() for s in manifest['samples']]
    assert len(set(fingerprints))==len(fingerprints)
    manifest['source_identity_check']='Exact task and length, deterministic RLDS position, full shard SHA256; independent main RGB streams'
    raw.save(OUT/'manifest.json',manifest)
    print(json.dumps({'extracted_trajectories':len(manifest['samples']),
        'frames':sum(s['frames'] for s in manifest['samples'])}),flush=True)


def render_one(sample, reviewed=False):
    from functools import lru_cache
    from PIL import Image
    from scripts.preprocess.libero.render import Episode,Event,Renderer,render_video
    started=time.monotonic();data=raw.source_data(sample)
    events=source_intervals(sample['annotation'],sample['frames'],apply_reviewed_skills=reviewed)
    @lru_cache(maxsize=None)
    def rgb(frame):return raw.rgb(data,frame)
    identity=sample['annotation_candidate_id']
    view=Episode(key=f"{sample['number']:02d}/33 | annotation {identity} | {sample['key']}",
        frame_count=sample['frames'],rgb=rgb,task=sample['task'],
        events=[Event(e['source_id'],e['start_frame'],e['stop_frame'],e['goal_frame'],e['skill']) for e in events],
        source=f'AtomicVLA annotation {identity}',gripper_state=data['state'][:,6:],
        actions=data['actions'],eef_xyz=data['state'][:,:3],xyz_note='SOURCE EEF XYZ / NO RESAMPLING',
        annotation_label='REVIEWED PUSH / ORIGINAL PICK' if reviewed else 'SOURCE PICK / TEXT PUSH / REVIEW PENDING',
        note='USER CONFIRMED PUSH SKILL' if reviewed else 'SKILL CONFLICT; SOURCE LABEL RETAINED',
        review_status='skill confirmed' if reviewed else 'pending')
    out,video,_=paths(reviewed)
    destination=video/f"{sample['number']:02d}_annotation_{identity}.mp4"
    result=render_video(view,destination)
    preview=out/'previews'/f'{identity}.jpg';preview.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(Renderer(view).frame(sample['frames']//2)).save(preview,quality=90)
    result.update(number=sample['number'],annotation_id=identity,key=sample['key'],
        video=str(destination.resolve()),sha256=raw.sha(destination),
        source_cache_sha256=sample['source_cache_sha256'],source_skill='Pick',description_action='Push',
        skill=events[0]['skill'],skill_review='user_confirmed' if reviewed else 'pending',
        renderer_sha256=raw.sha(Path(__file__).parents[1]/'render.py'),
        boundary_adapter_sha256=raw.sha(Path(__file__).with_name('source.py')),
        seconds=round(time.monotonic()-started,3))
    raw.save(out/'renders'/f'{identity}.json',result)
    return result


def render(workers, reviewed=False):
    out,_,_=paths(reviewed)
    manifest=json.loads((out/'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(render_one,s,reviewed) for s in manifest['samples']]
        for n,future in enumerate(as_completed(futures),1):
            result=future.result()
            print(json.dumps({'rendered':n,'total':len(futures),'annotation_id':result['annotation_id'],
                'seconds':result['seconds']}),flush=True)


def merge(reviewed=False):
    import av
    out,_,merged=paths(reviewed)
    manifest=json.loads((out/'manifest.json').read_text())
    clips=[json.loads((out/'renders'/f"{s['annotation_candidate_id']}.json").read_text()) for s in manifest['samples']]
    chapters=[];offset=Fraction(0)
    with av.open(str(merged),'w',options={'movflags':'+faststart'}) as output:
        with av.open(clips[0]['video']) as first:
            stream=output.add_stream_from_template(first.streams.video[0])
            configuration=first.streams.video[0].codec_context.extradata
        embedded=[];cursor=0
        for clip in clips:
            embedded.append({'id':clip['number'],'start':cursor,'end':cursor+clip['frames'],
                'time_base':Fraction(1,10),'metadata':{'title':f"{clip['number']:02d}/33 | annotation {clip['annotation_id']} | "+('reviewed Push' if reviewed else 'Pick vs Push')}})
            cursor+=clip['frames']
        output.set_chapters(embedded)
        for clip in clips:
            assert raw.sha(clip['video'])==clip['sha256']
            with av.open(clip['video']) as incoming:
                source=incoming.streams.video[0]
                assert source.codec_context.extradata==configuration
                for packet in incoming.demux(source):
                    if packet.dts is None:continue
                    shift=offset/packet.time_base
                    assert shift.denominator==1
                    packet.pts+=int(shift);packet.dts+=int(shift);packet.stream=stream;output.mux(packet)
            chapters.append({'number':clip['number'],'annotation_id':clip['annotation_id'],
                'start_seconds':float(offset),'frames':clip['frames']})
            offset+=Fraction(clip['frames'],10)
    count=0
    with av.open(str(merged)) as video:
        assert len(video.chapters())==len(clips)
        for frame in video.decode(video=0):
            assert (frame.width,frame.height)==(1920,1340)
            assert abs(float(frame.pts*frame.time_base)-count/10)<1e-6
            count+=1
    assert count==sum(c['frames'] for c in clips)
    raw.save(out/'chapters.json',chapters)
    report={'trajectories':len(clips),'frames':count,'duration_seconds':float(offset),
        'video':str(merged.resolve()),'sha256':raw.sha(merged),'full_decode_verified':True,
        'source_frame_order_preserved':True,'width':1920,'height':1340,'playback_fps':10,
        'source_fps':None,'skills_changed':len(clips) if reviewed else 0,
        'source_skill':'Pick','skill':'Push' if reviewed else 'Pick','description_action':'Push',
        'manifest_sha256':raw.sha(out/'manifest.json'),'manual_review':'skill_only_confirmed' if reviewed else 'pending'}
    raw.save(out/'video_report.json',report)
    review_path=out/'manual_review.csv'
    if not review_path.exists():
        table(review_path,[{'annotation_id':c['annotation_id'],'start_seconds':c['start_seconds'],
            'source_skill':'Pick','description_action':'Push','decision':'Push' if reviewed else 'pending',
            'notes':'用户确认压住物体通过摩擦力平移' if reviewed else ''} for c in chapters],
            ['annotation_id','start_seconds','source_skill','description_action','decision','notes'])
    rows=[]
    decision='用户确认 Push' if reviewed else 'pending'
    for chapter,clip in zip(chapters,clips,strict=True):
        rows.append(f"<tr><td>{chapter['number']:02d}</td><td>{chapter['annotation_id']}</td>"
            f"<td><a href='#' onclick=\"document.querySelector('video').currentTime={chapter['start_seconds']};return false\">{chapter['start_seconds']:.1f}s</a></td>"
            f"<td><a href='{html.escape(os.path.relpath(clip['video'],out))}'>单条视频</a></td><td>Pick</td><td>Push</td><td>{decision}</td></tr>")
    title='33 条技能已人工确认为 Push' if reviewed else '33 条 Pick / Push 冲突：人工待核验'
    note='当前标签为 Push，原始 Pick 保留为来源记录。下压物体后利用摩擦力沿支撑面平移。' if reviewed else '源标签保持 Pick；任务及事件描述写 Push。'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AtomicVLA 技能核验</title>'
        '<style>body{font:16px sans-serif;margin:24px}td,th{padding:8px;border:1px solid #bbb}table{border-collapse:collapse}video{max-width:100%}</style>'
        f'<h1>{title}</h1><p>{note}全部源帧按序保留，10 fps 仅为展示。</p>'
        f'<video controls preload="metadata" width="1200" src="{os.path.relpath(merged,out)}"></video>'
        '<table><tr><th>顺序</th><th>标注 ID</th><th>合并视频起点</th><th>独立视频</th><th>原技能</th><th>描述动作</th><th>人工</th></tr>'+''.join(rows)+'</table>')
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','approve','fetch','extract','render','merge'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--reviewed',action='store_true',help='Render/merge the user-confirmed Push labels')
    args=parser.parse_args()
    if args.stage=='render': render(args.workers,args.reviewed)
    elif args.stage=='merge': merge(args.reviewed)
    elif args.stage=='fetch': fetch(args.workers)
    else:{'prepare':prepare,'approve':approve,'extract':extract}[args.stage]()


if __name__=='__main__':main()
