"""Exploratory Qwen3-VL + SAM3 grounding on the fixed AtomicVLA raw sample."""
import argparse
from datetime import datetime
import csv
import html
import json
import re
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from scripts.preprocess.artifacts import artifact, digest, episodes, read_json, write_json
from scripts.preprocess.visual_grounding.qwen import Qwen, pixel_box
from scripts.preprocess.visual_grounding.sam import Sam, SamDetector
from scripts.preprocess.visual_grounding.annotate import keyframes, target_catalog, track_episode
from scripts.preprocess.visual_grounding.render import Regions
from scripts.preprocess.libero.render import render_video
from scripts.preprocess.libero.view import from_episode

SOURCE = 'atomic-vla'
from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"
RAW = ROOT/'check/raw/manifest.json'
OUT = ROOT/'check/test'


def enlarged(rgb):
    return np.asarray(Image.fromarray(rgb).resize((768, 768)))


def object_records(episode):
    """Read noun phrases already present in the published task chain, preserving skills."""
    events = [{**e, 'object_id': None, 'status': 'source_label'} for e in episode.evidence['intervals']]
    objects, held = {}, None
    for event in events:
        verb = (event.get('skill') or event['source_skill']).lower()
        description = event['description'].strip().rstrip('.')
        if verb == 'place' and held is not None:
            name = held
        else:
            prefixes = {'pick': r'^pick(?: up)? ', 'push': r'^push ', 'open': r'^open ', 'close': r'^close ', 'turn': r'^turn(?: on| off)? '}
            name = re.sub(prefixes.get(verb, r'^$'), '', description, flags=re.I)
            name = re.sub(r'^(?:the|a|an) ', '', name, flags=re.I)
            if verb == 'turn' and name == 'stove':
                name = 'stove knob'
        if verb == 'pick':
            held = name
        elif verb == 'push':
            held = None
        identity = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
        if identity not in objects:
            objects[identity] = {'id': identity, 'label': name,
                                 'description': name+'. Source task: '+episode.instructions[0], 'event_ids': []}
        objects[identity]['event_ids'].append(event['source_id'])
        event['object_id'] = identity
    return {'format':'eskill.subtasks.draft.v1','source':episode.entry,'frame_count':episode.count,
            'events':events,'objects':list(objects.values()),'interval_convention':episode.evidence['endpoint_rule'],
            'association_method':'source noun phrases; Pick to Place carries the same object; Turn stove maps to control knob',
            'manual_review':'pending'}


def source_objects_stage(config):
    for episode in episodes(SOURCE, RAW):
        write_json(artifact(OUT,'text',episode.entry['key']),object_records(episode))
        print('source_objects: '+episode.entry['key'],flush=True)


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return intersection/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)


def propose_stage(config):
    detector = SamDetector(config['sam'])
    for episode in episodes(SOURCE, RAW):
        text = read_json(artifact(OUT,'text',episode.entry['key']))
        targets = target_catalog(episode,text)
        for target in targets:
            if target['role']=='gripper':
                target['queries']=['robot gripper','robot hand','robot fingers']
            else:
                obj=next(o for o in text['objects'] if o['id']==target['id'])
                target['queries']=[obj['label']]
        frames=[]
        for target in targets:
            for frame in keyframes(episode,target,text['events'],3):
                values=detector.propose(episode.rgb('image',frame),target['queries'])
                chosen=[]
                for value in values:
                    if not any(iou(value['bbox'],c['bbox'])>0.9 for c in chosen):
                        chosen.append({k:v for k,v in value.items() if k!='mask'})
                    if len(chosen)>=config['max_candidates']:
                        break
                frames.append({'target_id':target['id'],'frame_index':frame,'candidates':chosen,
                               'all_candidate_count':len(values)})
        write_json(artifact(OUT,'proposals',episode.entry['key']),
                   {'source':episode.entry,'targets':targets,'frames':frames,'events':text['events'],
                    'text_sha256':digest(artifact(OUT,'text',episode.entry['key']))})
        print('propose: '+episode.entry['key'],flush=True)


def choose_target(episode,target,frames,client):
    images=[]
    for group in frames:
        image=Image.fromarray(episode.rgb('image',group['frame_index'])).resize((768,768))
        draw=ImageDraw.Draw(image)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
        for index,candidate in enumerate(group['candidates']):
            x1,y1,x2,y2=[v*3 for v in candidate['bbox']]
            draw.rectangle((x1,y1,x2,y2),outline='lime',width=2)
            draw.text((x1,max(0,y1-25)),str(index),font=font,fill='red',stroke_width=1,stroke_fill='white')
        images.append((f"frame_index={group['frame_index']} candidates numbered 0..{len(group['candidates'])-1}",np.asarray(image)))
    prompt=('Choose the region of ONE specific physical instance from the numbered SAM3 candidates. '
            'Task: '+episode.instructions[0]+'\nTarget: '+target['description']+
            '\nReturn JSON {"visible":true,"frame_index":0,"candidate_id":0,"reason":"visible evidence"}. '
            'Choose only an actually supplied frame and candidate ID. Use visible:false if none depicts the intended object. '
            'Do not substitute a container/destination. ')
    if target['role']=='gripper':
        prompt+=('The gripper is the short white rectangular hand housing and two dark fingers at the FREE end '
                 'of the Franka Panda arm. It is not a rounded cylindrical arm joint or the robot base. '
                 'Candidate boxes may be too large: additionally return bbox_1000=[x1,y1,x2,y2] tightly around '
                 'the gripper housing and visible fingers, excluding the long arm and held object, normalized to '
                 '0..1000 in the selected WHOLE image. candidate_id may be null only when refining the clearly '
                 'visible gripper outside the candidates. Partial housing visibility is allowed.')
    raw=None
    try:
        answer,raw=client.ask(prompt,images)
        assert type(answer['visible']) is bool
        if answer['visible']:
            assert type(answer['frame_index']) is int
            group=next(g for g in frames if g['frame_index']==answer['frame_index'])
            if target['role']=='gripper':
                answer['bbox']=pixel_box(answer['bbox_1000'],256,256)
                answer['geometry_source']='qwen_gripper_refinement'
            else:
                assert type(answer['candidate_id']) is int and 0<=answer['candidate_id']<len(group['candidates'])
                candidate=group['candidates'][answer['candidate_id']]
                answer['bbox']=candidate['bbox']
                answer['geometry_source']='sam3_text_mask'
    except (ValueError, KeyError, AssertionError, TypeError, StopIteration, IndexError) as error:
        answer={'visible':False,'reason':'Invalid instance selection: '+str(error)}
        raw=client.last_raw_response
    return {'camera':'image','target_id':target['id'],'supplied_frames':[g['frame_index'] for g in frames],
            'answer':answer,'raw_response':raw,'prompt':prompt}


def select_stage(config):
    client=Qwen(config['qwen'])
    for episode in episodes(SOURCE,RAW):
        proposal=read_json(artifact(OUT,'proposals',episode.entry['key']))
        assert proposal['text_sha256']==digest(artifact(OUT,'text',episode.entry['key']))
        prompts=[]
        for target in proposal['targets']:
            groups=[g for g in proposal['frames'] if g['target_id']==target['id']]
            prompts.append(choose_target(episode,target,groups,client))
        write_json(artifact(OUT,'prompts',episode.entry['key']),
                   {'format':'eskill.grounding.prompts.v1','source':episode.entry,'frame_count':episode.count,
                    'targets':proposal['targets'],'events':proposal['events'],'prompts':prompts,
                    'text_sha256':proposal['text_sha256'],'model':client.metadata,'manual_review':'pending'})
        print('select: '+episode.entry['key'],flush=True)


def track_stage(config):
    tracker=Sam(config['sam'])
    for episode in episodes(SOURCE,RAW):
        path=artifact(OUT,'prompts',episode.entry['key'])
        prompts=read_json(path)
        assert prompts['text_sha256']==digest(artifact(OUT,'text',episode.entry['key']))
        record=track_episode(episode,prompts,tracker,OUT)
        record['prompts_sha256']=digest(path)
        write_json(artifact(OUT,'regions',episode.entry['key']),record)
        print('track: '+episode.entry['key'],flush=True)


def render_stage(config):
    rows=[]
    for episode in episodes(SOURCE,RAW):
        record=read_json(artifact(OUT,'regions',episode.entry['key']))
        regions=Regions(OUT,record)
        path=OUT/'video'/f"{episode.entry['number']:02d}_{episode.entry['key']}_test.mp4"
        view=from_episode(episode, record, regions, source='AtomicVLA / MODEL PROPOSALS')
        rendered=render_video(view, path)
        decoded=rendered['frames']
        prompts=read_json(artifact(OUT,'prompts',episode.entry['key']))
        for r in record['regions']:
            if r['bbox']:
                assert all(0<=v<=256 for v in r['bbox'])
                assert regions.mask(r).shape==(256,256)
        row={'key':episode.entry['key'],'number':episode.entry['number'],'video':str(path.relative_to(OUT)),
             'frames':decoded,'sha256':digest(path),'manual_review':'pending',
             'targets':len(record['targets']),'localized':sum(p['answer']['visible'] for p in prompts['prompts']),
             'gripper_localized':next(p['answer']['visible'] for p in prompts['prompts'] if p['target_id']=='gripper_single'),
             'object_targets':sum(t['role']=='object' for t in record['targets']),
             'object_localized':sum(p['answer']['visible'] for p in prompts['prompts'] if p['target_id']!='gripper_single'),
             'bbox_frames':sum(r['bbox'] is not None for r in record['regions']), 'region_frames':len(record['regions'])}
        rows.append(row)
        print('render: '+episode.entry['key'],flush=True)
    rows.sort(key=lambda row:row['number'])
    write_json(OUT/'report.json',{'samples':rows,'manual_review':'pending'})
    with (OUT/'manual_review.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['key','video','gripper','object','identity','event_end','decision','notes'])
        writer.writeheader()
        for row in rows:writer.writerow({'key':row['key'],'video':row['video'],'decision':'pending'})
    items=''.join(f"<h2>{r['number']:02d} {html.escape(r['key'])}</h2><video controls preload='none' width='960' src='{r['video']}'></video>" for r in sorted(rows,key=lambda r:r['number']))
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>AtomicVLA grounding TEST — review pending</h1>'+items)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['source_objects','propose','select','track','render'])
    args=parser.parse_args()
    config=yaml.safe_load((OUT/'config.yaml').read_text())
    start=time.monotonic()
    record={'stage':args.stage,'started_at':datetime.now().astimezone().isoformat(),
            'config_sha256':digest(OUT/'config.yaml'),'source_manifest_sha256':digest(RAW),
            'code_sha256':{str(p):digest(p) for p in [Path(__file__),Path(__file__).with_name('source.py')]},'status':'running'}
    common=Path(__file__).parents[2]/'visual_grounding'
    record['common_code_sha256']={p.name:digest(p) for p in common.glob('*.py')}
    if args.stage=='render':
        code_root = Path(__file__).parents[1]
        record['renderer_code_sha256']={p.name:digest(p) for p in [code_root/'render.py',code_root/'view.py']}
    try:
        globals()[args.stage+'_stage'](config)
        record['status']='completed'
    except Exception as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        record['seconds']=round(time.monotonic()-start,3)
        write_json(OUT/'stages'/(args.stage+'.json'),record)


if __name__=='__main__':main()
