"""Localize grounding failures without replacing the original test predictions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from scripts.preprocess.artifacts import episodes, digest, read_json, write_json
from scripts.preprocess.visual_grounding.qwen import Qwen, pixel_box
from scripts.preprocess.visual_grounding.sam import Sam, SamDetector, mask_box
from scripts.preprocess.visual_grounding.render import Regions
from .grounding import choose_target

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"/'check/test'
RAW = ROOT.parent/'raw/manifest.json'
OUT = ROOT/'diagnosis'
CONFIG = yaml.safe_load((ROOT/'config.yaml').read_text())


def case(number):
    return next(e for e in episodes('atomic-vla', RAW) if e.entry['number']==number)


def audit():
    rows=[]
    for episode in episodes('atomic-vla', RAW):
        key=episode.entry['key']
        proposal=read_json(ROOT/'proposals'/f'{key}.json')
        prompts=read_json(ROOT/'prompts'/f'{key}.json')
        regions=read_json(ROOT/'regions'/f'{key}.json')
        for prompt in prompts['prompts']:
            target=prompt['target_id'];answer=prompt['answer']
            groups=[g for g in proposal['frames'] if g['target_id']==target]
            row={'case':episode.entry['number'],'target':target,'frames':prompt['supplied_frames'],
                 'candidate_counts':[len(g['candidates']) for g in groups],
                 'visible_claim':answer['visible'],'answer':answer,'raw_response':prompt['raw_response']}
            if answer['visible']:
                frame=answer['frame_index'];group=next(g for g in groups if g['frame_index']==frame)
                if target=='gripper_single':
                    assert np.allclose(pixel_box(answer['bbox_1000'],256,256),answer['bbox'])
                else:
                    assert group['candidates'][answer['candidate_id']]['bbox']==answer['bbox']
                seed=next(r for r in regions['regions'] if r['target_id']==target and r['frame_index']==frame)
                row['seed_mask_bbox']=seed['bbox']
                row['video_prompt_box_expected']=[16+answer['bbox'][0]*2,112+answer['bbox'][1]*2,
                                                16+answer['bbox'][2]*2,112+answer['bbox'][3]*2]
            rows.append(row)
    write_json(OUT/'audit.json',{'rows':rows,'coordinate_conversion_consistent':True,
                                'object_selection_uses_sam_bbox':True,
                                'gripper_selection_uses_unconstrained_qwen_bbox':True})
    print(json.dumps({'rows':len(rows),'coordinate_checks':'passed'}),flush=True)


def order():
    episode=case(1)
    proposal=read_json(ROOT/'proposals'/f"{episode.entry['key']}.json")
    target=next(t for t in proposal['targets'] if t['id']=='black_bowl')
    frames=[g for g in proposal['frames'] if g['target_id']==target['id']]
    client=Qwen(CONFIG['qwen']);results=[]
    for name,groups in [('original_order',frames),('reversed_order',deepcopy(frames))]:
        if name=='reversed_order':
            for group in groups:group['candidates'].reverse()
        answer=choose_target(episode,target,groups,client)
        results.append({'condition':name,'prediction':answer,
                        'candidate_boxes':{str(g['frame_index']):[c['bbox'] for c in g['candidates']] for g in groups}})
        print(name+': '+json.dumps(answer['answer']),flush=True)
    write_json(OUT/'candidate_order.json',{'case':1,'results':results,'only_numbering_order_changed':True})


def concepts():
    detector=SamDetector(CONFIG['sam']);results=[]
    specs=[(5,'moka_pot',['moka pot','coffee pot','coffee maker']),
           (10,'cream_cheese_box',['cream cheese box','food package','box']),
           (10,'butter',['butter','food package','box'])]
    for number,target,queries in specs:
        episode=case(number)
        proposal=read_json(ROOT/'proposals'/f"{episode.entry['key']}.json")
        group=next(g for g in proposal['frames'] if g['target_id']==target)
        frame=group['frame_index'];rgb=episode.rgb('image',frame)
        values=detector.propose(rgb,queries)
        results.append({'case':number,'target':target,'frame':frame,'queries':queries,
                        'original_candidate_count':len(group['candidates']),
                        'counts':{q:sum(v['query']==q for v in values) for q in queries},
                        'proposals':[{k:v for k,v in row.items() if k!='mask'} for row in values]})
        sheet=Image.new('RGB',(3*384,420),(24,27,34));draw=ImageDraw.Draw(sheet)
        for column,query in enumerate(queries):
            image=Image.fromarray(rgb).resize((384,384));d=ImageDraw.Draw(image)
            for value in values:
                if value['query']==query:d.rectangle([x*1.5 for x in value['bbox']],outline='lime',width=2)
            sheet.paste(image,(column*384,30));draw.text((column*384+5,8),query,fill='white')
        sheet.save(OUT/f'concept_case{number:02d}_{target}.png')
        print(json.dumps(results[-1]['counts']),flush=True)
    write_json(OUT/'concept_queries.json',{'results':results,'confidence_threshold':CONFIG['sam']['confidence_threshold']})


def seeds():
    import torch
    episode=case(1);model=Sam(CONFIG['sam']);model._load()
    proposal=read_json(ROOT/'proposals'/f"{episode.entry['key']}.json")
    alternative=next(g for g in proposal['frames'] if g['target_id']=='black_bowl' and g['frame_index']==0)['candidates'][1]['bbox']
    conditions=[
        {'name':'object_candidate1','target':'black_bowl','seed_frame':0,'bbox':alternative,
         'hint_source':'existing SAM3 candidate1, assistant visual selection'},
        {'name':'gripper_box_f96','target':'gripper_single','seed_frame':96,'bbox':[0,147,72,192],
         'hint_source':'assistant visual approximate reference; not ground truth'},
        {'name':'gripper_box_points_f96','target':'gripper_single','seed_frame':96,'bbox':[0,147,72,192],
         'points':[[18,167],[24,124],[54,181]],'labels':[1,0,0],
         'hint_source':'assistant visible palm positive, joint/bowl negative; not ground truth'},
        {'name':'gripper_box_f48','target':'gripper_single','seed_frame':48,'bbox':[63,147,143,197],
         'hint_source':'assistant visual approximate reference, different visibility frame; not ground truth'},
    ]
    with tempfile.TemporaryDirectory(prefix='eskill-diagnosis-') as temporary:
        directory=Path(temporary)
        for i in range(episode.count):Image.fromarray(episode.rgb('image',i)).save(directory/f'{i:06d}.jpg',quality=100,subsampling=0)
        for condition in conditions:
            with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                state=model.predictor.init_state(video_path=str(directory),offload_video_to_cpu=True,offload_state_to_cpu=False)
                kwargs={}
                if 'points' in condition:
                    kwargs={'points':np.asarray(condition['points'],np.float32)/256,'labels':condition['labels']}
                output=model.predictor.add_new_points_or_box(inference_state=state,frame_idx=condition['seed_frame'],obj_id=1,
                            box=np.asarray(condition['bbox'],np.float32).reshape(1,4)/256,rel_coordinates=True,**kwargs)
                initial=(output[3][list(output[1]).index(1)]>0).squeeze(0).cpu().numpy().astype(bool)
                values={}
                for reverse in (False,True):
                    if reverse and condition['seed_frame']==0:continue
                    for frame,ids,_,masks,_ in model.predictor.propagate_in_video(state,start_frame_idx=condition['seed_frame'],
                            max_frame_num_to_track=episode.count,reverse=reverse,propagate_preflight=True):
                        if reverse and frame==condition['seed_frame']:continue
                        values[int(frame)]=(masks[list(ids).index(1)]>0).squeeze(0).cpu().numpy().astype(bool)
                assert set(values)==set(range(episode.count))
                masks=np.stack([values[i] for i in range(episode.count)])
                condition['initial_mask_bbox']=mask_box(initial)
                condition['propagated_seed_mask_bbox']=mask_box(values[condition['seed_frame']])
                x1,y1,x2,y2=condition['bbox'];inside=np.zeros((256,256),bool);inside[y1:y2,x1:x2]=True
                condition['initial_mask_fraction_outside_reference_box']=float((initial&~inside).sum()/max(1,initial.sum()))
                condition['nonempty']=sum(mask_box(v) is not None for v in masks)
                path=OUT/'masks'/f"{condition['name']}.npz";path.parent.mkdir(exist_ok=True)
                with path.open('xb') as stream:np.savez_compressed(stream,bits=np.packbits(masks.reshape(episode.count,-1),axis=1),shape=[256,256],frames=np.arange(episode.count))
                condition['mask']=str(path.relative_to(OUT))
                del state
            print(json.dumps({k:v for k,v in condition.items() if k not in ['hint_source']}),flush=True)
    write_json(OUT/'seed_controls.json',{'case':1,'conditions':conditions,'model':model.metadata,
                                       'manual_hint_results_not_automatic_label_quality':True})


def unpack(path):
    with np.load(path) as d:return np.unpackbits(d['bits'],axis=1,count=256*256).reshape(-1,256,256).astype(bool)


def render():
    episode=case(1);record=read_json(ROOT/'regions'/f"{episode.entry['key']}.json");regions=Regions(ROOT,record)
    corrected=read_json(OUT/'seed_controls.json')['conditions']
    panels=[('AUTO gripper / seed96','gripper_single',None),
            ('REF box gripper / seed96','gripper_single',corrected[1]),
            ('REF box+points / seed96','gripper_single',corrected[2]),
            ('REF box gripper / seed48','gripper_single',corrected[3]),
            ('AUTO bowl / candidate0','black_bowl',None),
            ('ALT bowl / candidate1','black_bowl',corrected[0])]
    arrays={c['name']:unpack(OUT/c['mask']) for c in corrected}
    path=ROOT/'video/diagnosis_case01_seed_controls.mp4'
    if path.exists():raise FileExistsError(path)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    snapshots=[]
    with av.open(str(path),'w') as container:
        stream=container.add_stream('libx264',rate=10);stream.width,stream.height,stream.pix_fmt=960,760,'yuv420p'
        stream.options={'crf':'20','preset':'fast','threads':'2'}
        for frame in range(episode.count):
            canvas=Image.new('RGB',(960,760),(24,27,34));draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f'Case01 seed diagnosis | frame {frame}/96 | 10fps DISPLAY | NOT accepted labels',font=font,fill='white')
            draw.text((10,34),'REF = assistant visual hints, not ground truth. Same image / weights / tracker.',font=font,fill='yellow')
            for index,(label,target,condition) in enumerate(panels):
                if condition:mask=arrays[condition['name']][frame]
                else:mask=regions.mask(regions.index[('image',target,frame)])
                rgb=episode.rgb('image',frame).copy();color=np.array([80,165,255] if target=='gripper_single' else [80,235,130])
                rgb[mask]=(.6*rgb[mask]+.4*color).astype(np.uint8);image=Image.fromarray(rgb);d=ImageDraw.Draw(image);box=mask_box(mask)
                if box:d.rectangle(box,outline=tuple(color),width=2)
                x,y=index%3*320,index//3*344+74
                canvas.paste(image.resize((300,300)),(x+10,y+28));draw.text((x+10,y),label,font=font,fill='white')
            if frame in [0,48,96]:canvas.save(OUT/f'controls_frame{frame:03d}.png')
            for packet in stream.encode(av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24')):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    with av.open(str(path)) as container:
        count=0
        for frame in container.decode(video=0):
            assert abs(float(frame.pts*frame.time_base)-count/10)<1e-6;count+=1
        assert count==97
    write_json(OUT/'video_check.json',{'video':str(path.relative_to(ROOT)),'frames':97,'sha256':digest(path),'manual_review':'pending'})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['audit','order','concepts','seeds','render']);args=parser.parse_args()
    start=time.monotonic();record={'stage':args.stage,'config_sha256':digest(ROOT/'config.yaml'),'script_sha256':digest(Path(__file__))}
    try:globals()[args.stage]();record['status']='completed'
    except Exception as error:record.update(status='failed',error=repr(error));raise
    finally:record['seconds']=round(time.monotonic()-start,3);write_json(OUT/'stages'/f'{args.stage}.json',record)


if __name__=='__main__':main()
