"""Run the frozen binding rules on every published AtomicVLA LIBERO episode."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
import hashlib
import html
import importlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time

import numpy as np

from . import review_source as raw,binding_v2 as binding,binding_motion as motion
from .source import source_intervals

official=importlib.import_module('scripts.preprocess.libero.libero_40.source')
alignment=importlib.import_module('scripts.preprocess.libero.libero_40.align')
ROOT=raw.ROOT
OUT=ROOT/'check/test/all_rules_v1'
CACHE=raw.CACHE.parent/'all_rules_v1'
LIBERO=Path('/data1/yuankai/robot-research/workspace/third_party/libero-native')
VIDEO=ROOT/'check/test/video/all_rules_v1'


def provenance():
    return {'binder':raw.sha(binding.__file__),'catalog':raw.sha(binding.CATALOG_PATH),
            'motion':raw.sha(motion.__file__),'event_adapter':raw.sha(Path(__file__).with_name('source.py'))}


def manifest():
    value=json.loads((OUT/'manifest.json').read_text())
    if value['rules']!=provenance():raise ValueError('Rules changed during the full-data check')
    return value


def sample_list():
    return json.loads((OUT/'sources.json').read_text())['samples']


def signals(sample):
    if raw.sha(sample['source_cache'])!=sample['source_cache_sha256']:
        raise ValueError('Source cache changed')
    with np.load(sample['source_cache']) as archive:
        return {k:archive[k].copy() for k in ['state','actions','joints']}


def table(path,rows,fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def save_match(path,value):
    temporary=path.with_name(path.name+'.tmp')
    raw.save(temporary,value);temporary.replace(path)


def prepare():
    sys.path.insert(0,str(LIBERO))
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from scripts.preprocess.libero.check_raw import atomic_positions
    old=json.loads((raw.OUT/'manifest.json').read_text())
    annotation_path=raw.RAW/'libero_lerobot.json'
    assert raw.sha(annotation_path)==old['source_annotation_sha256']
    annotations=json.loads(annotation_path.read_text())
    by_task={binding.normal(s['task']):s for s in old['samples']}
    scenes={}
    for task,sample in by_task.items():
        file=Path(sample['metadata']['file_path'])
        bddl=LIBERO/'libero/libero/bddl_files'/file.parent.name/file.name.replace('_demo.hdf5','.bddl')
        scenes[task]=(bddl,robosuite_parse_problem(str(bddl)))
    assert len(scenes)==40
    positions={v:k for k,v in atomic_positions(raw.RAW).items()}
    files=json.loads((OUT/'source_files.json').read_text())['files']
    shard_by_index={(f['rfilename'].split('/')[0],int(re.search(r'tfrecord-(\d+)-',f['rfilename'])[1])):f for f in files}
    existing={s['key']:s for s in old['samples']}
    push=json.loads((ROOT/'check/test/annotation_review/manifest.json').read_text())
    existing.update({s['key']:s for s in push['samples']})
    cohort={s['annotation_candidate_id']:s['number'] for s in old['samples']}
    samples=[]
    for number,(identity,annotation) in enumerate(sorted(annotations.items(),key=lambda x:int(x[0])),1):
        identity=int(identity);suite,shard,row=positions[identity]
        task=annotation['segments'][0]['action_name'];original=by_task[binding.normal(task)]
        bddl,scene=scenes[binding.normal(task)]
        events=source_intervals(annotation,annotation['all_frames'])
        coverage=np.zeros(annotation['all_frames'],np.int8)
        for event in events:coverage[event['start_frame']:event['stop_frame']]+=1
        assert np.all(coverage==1)
        static=binding.bind_events(scene,events,task)
        key=f'{suite}_s{shard:03d}_r{row:03d}'
        sample={'number':number,'annotation_id':identity,'annotation_candidate_id':identity,
            'key':key,'suite':suite,'shard':shard_by_index[suite,shard]['rfilename'],
            'record_index':row,'task':task,'frames':annotation['all_frames'],'annotation':annotation,
            'metadata':original['metadata'],'bddl':str(bddl),'bddl_sha256':raw.sha(bddl),
            'cohort_number':cohort.get(identity),'source_cache':str(CACHE/'source'/f'{key}.npz')}
        if key in existing:
            sample['reusable_source']={k:existing[key][k] for k in ['source_cache','source_cache_sha256']}
        raw.save(OUT/'static'/f'{identity}.json',{'annotation_id':identity,'task':task,'bindings':static,'rules':provenance()})
        samples.append(sample)
    rng=random.Random(42);selected=[];used=set()
    reasons=sorted({e.get('reason') for s in samples for e in json.loads((OUT/'static'/f"{s['annotation_id']}.json").read_text())['bindings'] if e['status']!='rule_bound'})
    for reason in reasons:
        pool=[s for s in samples if s not in selected and any(e.get('reason')==reason for e in json.loads((OUT/'static'/f"{s['annotation_id']}.json").read_text())['bindings'])]
        chosen=rng.choice(pool);selected.append(chosen);used.add(chosen['bddl'])
    for suite,count in [('libero_10_no_noops',3),('libero_goal_no_noops',2),('libero_object_no_noops',2),('libero_spatial_no_noops',3)]:
        pool=[s for s in samples if s['suite']==suite];rng.shuffle(pool)
        for sample in pool:
            if sum(s['suite']==suite for s in selected)>=count:break
            if sample['bddl'] not in used:selected.append(sample);used.add(sample['bddl'])
    assert len(selected)==10
    source_files=json.loads((OUT/'source_files.json').read_text())
    official_manifest=official.manifest()
    value={'repository':source_files['repository'],'revision':source_files['revision'],
        'annotation_revision':'33d2f98574a52838b9db34b334bd5cc6cbe57f50',
        'annotation_path':str(annotation_path),'annotation_sha256':raw.sha(annotation_path),
        'samples':samples,'rules':provenance(),'official_repository':official.REPOSITORY,
        'official_revision':official_manifest['revision'],'official_files':official_manifest['files'],
        'review_annotation_ids':sorted(s['annotation_id'] for s in selected),
        'review_selection':'seed42; one episode per static failure reason, then distinct tasks to suite quotas 3/2/2/3; shared failure task may occur twice',
        'manual_review':'pending','script_sha256':raw.sha(__file__)}
    raw.save(OUT/'manifest.json',value)
    print(json.dumps({'trajectories':len(samples),'events':sum(len(s['annotation']['segments']) for s in samples),
        'frames':sum(s['frames'] for s in samples),'review_ids':value['review_annotation_ids']}),flush=True)


def extract_shard(job):
    from scripts.preprocess.libero.rlds import records
    from scripts.preprocess.libero.check_raw import arrays,metadata,normalized
    file,samples=job
    destination=OUT/'source_shards'/(file['rfilename'].replace('/','__')+'.json')
    if destination.exists():return json.loads(destination.read_text())
    assert raw.sha(raw.RAW/file['rfilename'])==file['lfs']['sha256']
    by_row={s['record_index']:s for s in samples};output=[]
    for row,offset,data in records(raw.RAW/file['rfilename']):
        if row not in by_row:continue
        sample={**by_row[row]};state,actions,joints=arrays(data)
        assert all(np.isfinite(v).all() for v in [state,actions,joints])
        task=data['steps/language_instruction'][0].decode();meta=metadata(data)
        same_task=normalized(task)==normalized(sample['task'])
        same_length=len(state)==sample['frames']
        same_file=Path(meta['file_path']).name==Path(sample['metadata']['file_path']).name
        sample.update(source_link='linked' if same_task and same_length and same_file else 'mismatch',
            source_task=task,source_frames=len(state),metadata=meta,byte_offset=offset,
            source_link_checks={'task':same_task,'frames':same_length,'scene_file':same_file})
        reusable=sample.pop('reusable_source',None)
        if reusable:
            cached=signals(reusable)
            for k,v in [('state',state),('actions',actions),('joints',joints)]:np.testing.assert_array_equal(cached[k],v)
            sample.update(reusable)
        else:
            path=Path(sample['source_cache']);path.parent.mkdir(parents=True,exist_ok=True)
            images=data['steps/observation/image']
            np.savez_compressed(path,state=state,actions=actions,joints=joints,
                image_bytes=np.frombuffer(b''.join(images),np.uint8),
                image_offsets=np.r_[0,np.cumsum([len(v) for v in images])].astype(np.int64))
            sample['source_cache_sha256']=raw.sha(path)
        fingerprint=hashlib.sha256(state.tobytes()+actions.tobytes())
        for image in data['steps/observation/image']:fingerprint.update(image)
        sample['main_state_action_sha256']=fingerprint.hexdigest();output.append(sample)
    assert len(output)==len(samples)
    result={'file':file['rfilename'],'sha256':file['lfs']['sha256'],'samples':output}
    raw.save(destination,result);return result


def extract(workers):
    m=manifest();groups=defaultdict(list)
    for sample in m['samples']:groups[sample['shard']].append(sample)
    files=json.loads((OUT/'source_files.json').read_text())['files']
    jobs=[(file,groups[file['rfilename']]) for file in files if file['rfilename'] in groups]
    run_jobs(extract_shard,jobs,workers,'extract')
    samples=[s for p in (OUT/'source_shards').glob('*.json') for s in json.loads(p.read_text())['samples']]
    samples.sort(key=lambda s:s['annotation_id']);assert len(samples)==len(m['samples'])
    result={'samples':samples,'trajectories':len(samples),'source_links':dict(Counter(s['source_link'] for s in samples)),
        'independent_fingerprints':len({s['main_state_action_sha256'] for s in samples})}
    raw.save(OUT/'sources.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='samples'}),flush=True)


def match_task(job):
    import h5py
    file,samples,revision=job;name=file['rfilename'].replace('/','__').replace('.hdf5','.json')
    destination=OUT/'scene_matches'/name
    if destination.exists():return json.loads(destination.read_text())
    sources={s['key']:signals(s) for s in samples};old={r['key']:r for r in json.loads((official.OUT/'scene_summary.json').read_text())['samples']}
    for attempt in range(3):
        spent=sum(json.loads(p.read_text())['range_bytes'] for folder in ['scene_matches','scene_errors'] for p in (OUT/folder).glob('*.json'))
        if spent+4*256*1024**2>12*1024**3:raise RuntimeError('Insufficient remaining Range budget for four workers')
        remote=official.RangeFile(f'https://huggingface.co/datasets/{official.REPOSITORY}/resolve/{revision}/{file["rfilename"]}',
            file['size'],budget=256*1024**2,block_size=1024**2)
        try:
            matches=defaultdict(list);rows=[]
            with h5py.File(remote,'r') as h:
                for demo,group in h['data'].items():
                    actions=group['actions'][()].astype(np.float32)
                    for sample in samples:
                        indices,method=official.match_indices(actions,sources[sample['key']]['actions'])
                        if indices is not None:matches[sample['key']].append((demo,indices,method))
                for sample in samples:
                    candidates=matches[sample['key']]
                    row={'annotation_id':sample['annotation_id'],'key':sample['key'],'frames':sample['frames'],
                        'repository':official.REPOSITORY,'revision':revision,'remote_file':file['rfilename'],
                        'status':'unmatched' if not candidates else 'ambiguous_actions',
                        'matches':[c[0] for c in candidates],'manual_review':'pending'}
                    if len(candidates)==1:
                        demo,indices,method=candidates[0]
                        row.update(status='actions_exact_unique_match',demo=demo,match_method=method)
                        previous=old.get(sample['key'])
                        if previous and previous['status']=='actions_exact_unique_match':
                            assert previous['demo']==demo and raw.sha(previous['excerpt_path'])==previous['excerpt_sha256']
                            row.update({k:previous[k] for k in ['excerpt_path','excerpt_sha256','model_file','states_shape']})
                            row['cache_origin']='previous official fixed100 excerpt'
                        else:
                            group=h['data'][demo];states=group['states'][indices]
                            path=CACHE/'scenes'/f"{sample['key']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
                            np.savez_compressed(path,states=states,source_to_official_indices=indices)
                            xml=group.attrs['model_file'];xml=xml.decode() if isinstance(xml,bytes) else str(xml)
                            xml_path=path.with_suffix('.model.txt');xml_path.write_text(xml)
                            row.update(excerpt_path=str(path),excerpt_sha256=raw.sha(path),states_shape=list(states.shape),
                                model_file={'path':str(xml_path),'sha256':raw.sha(xml_path),'contains_xml':xml.lstrip().startswith('<')})
                    rows.append(row)
            result={'file':file['rfilename'],'samples':rows,'range_bytes':remote.transferred,
                'range_requests':remote.requests,'revision':revision,'full_remote_hash_verified':False}
            save_match(destination,result);return result
        except Exception as error:
            save_match(OUT/'scene_errors'/f'{name}.{attempt}.json',{'file':file['rfilename'],
                'range_bytes':remote.transferred,'error':repr(error)})
            if attempt==2:raise
        finally:remote.session.close()


def states(workers):
    m=manifest();groups=defaultdict(list)
    for sample in sample_list():
        if sample['source_link']=='linked':groups[official.relative_file(sample)].append(sample)
    jobs=[(file,groups[file['rfilename']],m['official_revision']) for file in m['official_files'] if file['rfilename'] in groups]
    run_jobs(match_task,jobs,workers,'states')
    records=[json.loads(p.read_text()) for p in (OUT/'scene_matches').glob('*.json')]
    errors=[json.loads(p.read_text()) for p in (OUT/'scene_errors').glob('*.json')]
    transferred=sum(r['range_bytes'] for r in records+errors)
    assert transferred<=12*1024**3
    rows=sorted([s for r in records for s in r['samples']],key=lambda s:s['annotation_id'])
    raw.save(OUT/'scenes.json',{'samples':rows,'range_bytes':transferred,
        'matches':dict(Counter(s['status'] for s in rows)),'full_remote_hash_verified':False})


def task_jobs():
    groups=defaultdict(list)
    matches={r['key']:r for r in json.loads((OUT/'scenes.json').read_text())['samples']}
    for sample in sample_list():groups[sample['bddl']].append((sample,matches.get(sample['key'])))
    return [(bddl,jobs) for bddl,jobs in sorted(groups.items())]


def audit_task(job):
    sys.path.insert(0,str(LIBERO))
    from libero.libero.envs.env_wrapper import ControlEnv
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    bddl,jobs=job;scene=robosuite_parse_problem(bddl)
    env=ControlEnv(bddl_file_name=bddl,use_camera_obs=False,has_offscreen_renderer=False,hard_reset=False)
    try:
        for sample,match in jobs:
            destination=OUT/'audits'/f"{sample['annotation_id']}.json"
            if destination.exists():continue
            started=time.monotonic()
            static=json.loads((OUT/'static'/f"{sample['annotation_id']}.json").read_text())['bindings']
            record={'annotation_id':sample['annotation_id'],'key':sample['key'],'task':sample['task'],
                'cohort_number':sample['cohort_number'],'frames':sample['frames'],'source_link':sample['source_link'],
                'match':match,'rules':provenance(),'static_bindings':static,'manual_review':'pending'}
            rows=static
            try:
                if sample['source_link']=='linked' and match and match['status']=='actions_exact_unique_match':
                    assert raw.sha(match['excerpt_path'])==match['excerpt_sha256']
                    model=alignment.model_check(env,match);record['model_check']=model
                    if model['status']!='compatible':raise ValueError('Official model compatibility failed')
                    with np.load(match['excerpt_path']) as archive:states=archive['states'].copy()
                    data=signals(sample)
                    measured=motion.collect(env,scene,static,states,data)
                    rows=binding.bind_events(scene,source_intervals(sample['annotation'],sample['frames']),sample['task'],motion=measured)
                    track=alignment.restore_track(env,states,[])
                    record['robot_alignment']=alignment.robot_metrics(track,data)
                else:record['audit_gap']='source_link_or_matched_states_missing'
                for row in rows:
                    if row['status']=='rule_bound':
                        try:row['geometry']=binding.resolve_geometry(env,row)
                        except (KeyError,ValueError) as error:row['geometry']={'status':'unresolved','reason':str(error)}
            except Exception as error:record['audit_error']=repr(error)
            record.update(bindings=rows,seconds=round(time.monotonic()-started,3))
            raw.save(destination,record)
    finally:env.close()
    return {'file':bddl,'trajectories':len(jobs)}


def run_jobs(function,jobs,workers,stage):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(function,job) for job in jobs]
        for n,future in enumerate(as_completed(futures),1):
            result=future.result()
            print(json.dumps({'stage':stage,'done':n,'total':len(futures),
                'file':result.get('file'),'trajectories':result.get('trajectories',len(result.get('samples',[])))}),flush=True)


def report():
    from .review_native import bbox
    m=manifest();samples=sample_list();rows=[];audits=[];regions=[];event_rows=[]
    verified_masks=0
    for sample in samples:
        identity=sample['annotation_id']
        audit=json.loads((OUT/'audits'/f'{identity}.json').read_text());audits.append(audit)
        region=json.loads((OUT/'keyframe_regions'/f'{identity}.json').read_text());regions.append(region)
        assert audit['rules']==region['rules']==m['rules']
        if 'archive' in region:
            assert raw.sha(region['archive'])==region['archive_sha256']
            with np.load(region['archive']) as archive:
                boxes,bits=archive['boxes'],archive['mask_bits']
                assert len(boxes)==len(bits)==len(region['checks'])
                for check,box,packed in zip(region['checks'],boxes,bits,strict=True):
                    mask=np.unpackbits(packed).reshape(256,256).astype(bool);expected=bbox(mask)
                    if expected is None:assert np.isnan(box).all() and check['bbox'] is None
                    else:np.testing.assert_array_equal(box,expected);assert check['bbox']==list(expected)
                    assert int(mask.sum())==check['area_pixels'];verified_masks+=1
        final=audit['bindings'];static=audit['static_bindings'];match=audit['match']
        errors=[]
        if sample['source_link']!='linked':errors.append('source_link')
        if not match or match['status']!='actions_exact_unique_match':errors.append('scene_match')
        if audit.get('model_check',{}).get('status')!='compatible':errors.append('model_or_missing_states')
        if 'audit_error' in audit:errors.append('audit_error')
        unbound=[e for e in final if e['status']!='rule_bound']
        geometry_bad=[e for e in final if e['status']=='rule_bound' and e['geometry']['status']!='resolved']
        if unbound:errors.append('binding')
        if geometry_bad:errors.append('geometry')
        if 'region_error' in region:errors.append('native_render')
        alignment_metrics=audit.get('robot_alignment',{}).get('same_row')
        risk=bool(alignment_metrics and alignment_metrics['eef_max_m']>binding.CONFIG['motion']['max_scene_eef_error_m'])
        mae=[r['rgb_mae'] for r in region.get('scene_rgb_checks',[])]
        ready=not errors and region.get('all_event_goals_available',False)
        row={'annotation_id':identity,'key':sample['key'],'suite':sample['suite'],'task':sample['task'],
            'cohort_number':sample['cohort_number'],'frames':sample['frames'],'events':len(final),
            'static_bound':sum(e['status']=='rule_bound' for e in static),
            'final_bound':sum(e['status']=='rule_bound' for e in final),'unbound_events':len(unbound),
            'geometry_failures':len(geometry_bad),'source_link':sample['source_link'],
            'scene_match':match['status'] if match else 'missing','model_check':audit.get('model_check',{}).get('status','not_checked'),
            'eef_max_mm':alignment_metrics['eef_max_m']*1000 if alignment_metrics else None,
            'eef_median_mm':alignment_metrics['eef_median_m']*1000 if alignment_metrics else None,
            'alignment_exceeds_10mm':risk,'three_frame_rgb_mae':float(np.mean(mae)) if mae else None,
            'keyframe_goals_available':sum(g['available'] for g in region.get('goals',[])),
            'nonempty_goal_boxes':sum(g['nonempty'] for g in region.get('goals',[])),
            'gripper_eef_max_mm':region.get('gripper_eef_max_m',0)*1000 if 'gripper_eef_max_m' in region else None,
            'structural_ready':ready,'issues':';'.join(errors),'manual_review':'pending'}
        rows.append(row)
        for event in final:
            event_rows.append({'annotation_id':identity,'key':sample['key'],'event':event['source_id'],
                'skill':event['skill'],'source_skill':event['source_skill'],
                'skill_review':event.get('skill_correction',{}).get('skill_review','not_reviewed'),
                'skill_check':event['skill_check'],'description':event['description'],
                'start_frame':event['start_frame'],'stop_frame':event['stop_frame'],'goal_frame':event['goal_frame'],
                'status':event['status'],'reason':event.get('reason',''),'instance':event['instance'],
                'part':event['scope']['part'] if event['scope'] else '',
                'motion':event['motion']['status'],'geometry':event['geometry']['status'],'manual_review':'pending'})
    assert len(rows)==len(m['samples'])==1693 and len(event_rows)==3873
    review_records=[json.loads(p.read_text()) for p in (OUT/'review10/renders').glob('*.json')]
    selected={s['key'] for s in samples if s['annotation_id'] in m['review_annotation_ids']}
    assert {r['key'] for r in review_records}==selected
    for r in review_records:assert raw.sha(r['video'])==r['sha256']
    fields=list(rows[0]);table(OUT/'trajectories.csv',rows,fields)
    table(OUT/'events.csv',event_rows,list(event_rows[0]))
    binding_issues=[r for r in rows if r['unbound_events'] or r['geometry_failures']]
    table(OUT/'binding_issues.csv',binding_issues,fields)
    risks=[r for r in rows if r['alignment_exceeds_10mm']];table(OUT/'alignment_risks.csv',risks,fields)
    inputs=[r for r in rows if any(t in r['issues'].split(';') for t in ['source_link','scene_match','model_or_missing_states','audit_error','native_render'])]
    table(OUT/'input_issues.csv',inputs,fields)
    by_task=[]
    for task in sorted({r['task'] for r in rows}):
        group=[r for r in rows if r['task']==task]
        by_task.append({'task':task,'trajectories':len(group),'binding_issues':sum(bool(r['unbound_events'] or r['geometry_failures']) for r in group),
            'alignment_risks':sum(r['alignment_exceeds_10mm'] for r in group),'structural_ready':sum(r['structural_ready'] for r in group)})
    regression={'trajectories':0,'previous_bound_events':0,'current_bound_events':0,
        'changed_previously_bound_targets':[],'lost_bindings':[],'gained_bindings':[],
        'updated_boundaries_or_skills':[]}
    for audit in audits:
        if audit['cohort_number'] is None:continue
        previous=json.loads((official.OUT/'test/alignments'/f"{audit['key']}.json").read_text())
        regression['trajectories']+=1
        for before,after in zip(previous['bindings'],audit['bindings'],strict=True):
            old_bound=before['status']=='rule_bound';new_bound=after['status']=='rule_bound'
            regression['previous_bound_events']+=old_bound;regression['current_bound_events']+=new_bound
            identity={'cohort_number':audit['cohort_number'],'event':after['source_id']}
            if old_bound and new_bound and any(before[k]!=after[k] for k in ['instance','scope']):
                regression['changed_previously_bound_targets'].append(identity)
            if old_bound and not new_bound:regression['lost_bindings'].append(identity)
            if not old_bound and new_bound:regression['gained_bindings'].append(identity)
            if any(before[k]!=after[k] for k in ['start_frame','stop_frame','goal_frame','skill']):
                regression['updated_boundaries_or_skills'].append(identity)
    summary={'trajectories':len(rows),'events':len(event_rows),'frames':sum(r['frames'] for r in rows),
        'source_links':dict(Counter(r['source_link'] for r in rows)),
        'independent_fingerprints':len({s['main_state_action_sha256'] for s in samples}),
        'scene_matches':dict(Counter(r['scene_match'] for r in rows)),
        'matching_methods':dict(Counter(a['match'].get('match_method') for a in audits if a['match'])),
        'model_checks':dict(Counter(r['model_check'] for r in rows)),
        'static_binding_statuses':dict(Counter(e['status'] for a in audits for e in a['static_bindings'])),
        'final_binding_statuses':dict(Counter(r['status'] for r in event_rows)),
        'all_events_bound_trajectories':sum(r['final_bound']==r['events'] for r in rows),
        'binding_issue_trajectories':len(binding_issues),'input_issue_trajectories':len(inputs),
        'structural_ready_trajectories':sum(r['structural_ready'] for r in rows),
        'remaining_reasons':dict(Counter(r['reason'] for r in event_rows if r['status']!='rule_bound')),
        'skill_checks':dict(Counter(r['skill_check'] for r in event_rows)),
        'skill_counts':dict(Counter(r['skill'] for r in event_rows)),
        'motion_statuses':dict(Counter(r['motion'] for r in event_rows)),
        'alignment_exceeds_10mm_trajectories':len(risks),
        'official_state_eef_max_mm':max((r['eef_max_mm'] for r in rows if r['eef_max_mm'] is not None),default=None),
        'three_frame_rgb_mae_median':float(np.median([r['three_frame_rgb_mae'] for r in rows if r['three_frame_rgb_mae'] is not None])),
        'keyframe_mask_bbox_pairs_verified':verified_masks,
        'keyframe_goal_regions_available':sum(r['keyframe_goals_available'] for r in rows),
        'nonempty_keyframe_goal_boxes':sum(r['nonempty_goal_boxes'] for r in rows),
        'source_gripper_eef_max_mm':max((r['gripper_eef_max_mm'] for r in rows if r['gripper_eef_max_mm'] is not None),default=None),
        'range_bytes':json.loads((OUT/'scenes.json').read_text())['range_bytes'],
        'review_annotation_ids':m['review_annotation_ids'],'rendered_frames':sum(r['frames'] for r in review_records),
        'by_task':by_task,'fixed100_regression':regression,'rules':m['rules'],
        'workflow_sha256':raw.sha(__file__),
        'region_workflow_sha256':raw.sha(Path(__file__).with_name('review_all_regions.py')),'manual_review':'pending',
        'interpretation':'Binding coverage, motion support, source alignment, native keyframe checks and manual correctness are separate. An EEF residual above 10mm is an alignment diagnostic, not bbox error ground truth.'}
    raw.save(OUT/'report.json',summary)
    review_by_key={r['key']:r for r in review_records}
    items=[]
    for row in rows:
        image=OUT/'comparisons'/f"{row['annotation_id']:04d}.jpg"
        links=f"<a href='{os.path.relpath(image,OUT)}'>三帧对照</a>" if image.exists() else 'MISSING'
        if row['key'] in review_by_key:
            url=os.path.relpath(review_by_key[row['key']]['video'],OUT)
            links+=f" · <a href='{url}' onclick=\"document.querySelector('video').src=this.href;return false\">完整视频</a>"
        value=f"{row['eef_max_mm']:.2f}" if row['eef_max_mm'] is not None else 'MISSING'
        items.append(f"<tr data-issue='{int(bool(row['issues']) or row['alignment_exceeds_10mm'])}'><td>{row['annotation_id']}</td>"
            f"<td>{html.escape(row['task'])}</td><td>{row['final_bound']}/{row['events']}</td><td>{html.escape(row['issues']) or '—'}</td>"
            f"<td>{value}</td><td>{links}</td><td>pending</td></tr>")
    first=os.path.relpath(sorted(review_records,key=lambda r:r['number'])[0]['video'],OUT)
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AtomicVLA 全量规则核验</title>'
        '<style>body{font:16px sans-serif;margin:24px;color:#18232d}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbc4ca}video{max-width:100%}button{margin:12px;padding:8px}</style>'
        '<h1>AtomicVLA 全量通用规则检查</h1>'
        f"<p>1,693 条／3,873 Event；{summary['all_events_bound_trajectories']} 条全部事件有唯一候选，{len(binding_issues)} 条有绑定／几何缺口。人工待核验。</p>"
        '<p>末端误差比较官方场景状态与 RLDS 状态，不是 bbox 正确率；关键帧区域不代表整套逐帧区域已验收。</p>'
        f'<video controls preload="metadata" width="1150" src="{first}"></video>'
        '<p><a href="binding_issues.csv">绑定问题</a> · <a href="alignment_risks.csv">对齐风险</a> · <a href="input_issues.csv">来源／执行问题</a> · <a href="events.csv">全部事件</a></p>'
        '<button onclick="document.querySelectorAll(\'tr[data-issue]\').forEach(r=>r.hidden=r.dataset.issue===\'0\')">仅看需检查</button>'
        '<button onclick="document.querySelectorAll(\'tr[data-issue]\').forEach(r=>r.hidden=false)">显示全部</button>'
        '<table><tr><th>标注 ID</th><th>任务</th><th>唯一候选</th><th>问题</th><th>末端最大偏差 mm</th><th>核验入口</th><th>人工</th></tr>'+''.join(items)+'</table>')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'by_task','rules'}},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','extract','states','audit','regions','render','report'])
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.stage=='prepare':prepare()
    elif args.stage=='extract':extract(args.workers)
    elif args.stage=='states':states(args.workers)
    elif args.stage=='audit':run_jobs(audit_task,task_jobs(),args.workers,'audit')
    elif args.stage=='regions':
        from .review_all_regions import regions_task
        run_jobs(regions_task,task_jobs(),args.workers,'regions')
    elif args.stage=='render':
        from .review_all_regions import full_video
        selected=set(manifest()['review_annotation_ids'])
        jobs=[job for _,group in task_jobs() for job in group if job[0]['annotation_id'] in selected]
        run_jobs(full_video,jobs,args.workers,'render')
    elif args.stage=='report':report()


if __name__=='__main__':main()
