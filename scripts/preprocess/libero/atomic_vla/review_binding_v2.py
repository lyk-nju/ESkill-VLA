"""Fixed-100 v2 regression, native regions and changed-trajectory review videos."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from . import binding_v2, binding_motion, review_native
from .review_source import ROOT, OUT as BASE, CACHE as BASE_CACHE, sha, save, source_data
from .source import source_intervals

OUT = ROOT/'check/test/binding_v2'
CACHE = BASE_CACHE.parent/'binding_v2'
VIDEO = ROOT/'check/test/video/binding_v2'


def provenance():
    return {'binder':sha(binding_v2.__file__), 'catalog':sha(binding_v2.CATALOG_PATH),
            'motion':sha(binding_motion.__file__), 'source_adapter':sha(Path(__file__).with_name('source.py')),
            'manifest':sha(BASE/'manifest.json')}


def audit(job):
    sample, match = job
    started = time.monotonic()
    sys.path.insert(0, str(review_native.LIBERO))
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    from libero.libero.envs.env_wrapper import ControlEnv
    file = Path(sample['metadata']['file_path'])
    bddl = review_native.LIBERO/'libero/libero/bddl_files'/file.parent.name/file.name.replace('_demo.hdf5','.bddl')
    scene = robosuite_parse_problem(str(bddl))
    events = source_intervals(sample['annotation'], sample['frames'])
    initial = binding_v2.bind_events(scene, events, sample['task'])
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False, hard_reset=False)
    try:
        if match and match['status'] == 'actions_exact_unique_match':
            if sha(match['excerpt_path']) != match['excerpt_sha256']:
                raise ValueError('Matched scene excerpt changed')
            with np.load(match['excerpt_path']) as archive:
                motion = binding_motion.collect(env, scene, initial, archive['states'], source_data(sample))
        else:
            motion = {e['source_id']:{'status':'no_matched_states','candidates':[]} for e in events}
        rows = binding_v2.bind_events(scene, events, sample['task'], motion=motion)
        for row in rows:
            if row['status'] == 'rule_bound':
                row['geometry'] = binding_v2.resolve_geometry(env, row)
    finally:
        env.close()
    old = json.loads((BASE/'regions'/(sample['key']+'.json')).read_text())['bindings']
    changes = []
    for before, after in zip(old, rows, strict=True):
        if any(before[k] != after[k] for k in ['source_id','start_frame','stop_frame','goal_frame','skill','description']):
            raise ValueError('v2 changed source event semantics')
        fields = ['instance','scope','status']
        if any(before[k] != after[k] for k in fields):
            changes.append({'event':after['source_id'], 'before':{k:before[k] for k in fields},
                            'after':{k:after[k] for k in fields}, 'reason':after.get('reason'),
                            'previously_bound':before['status']=='rule_bound'})
    record = {'number':sample['number'], 'key':sample['key'], 'task':sample['task'], 'frames':sample['frames'],
              'bindings':rows, 'changes':changes, 'provenance':provenance(), 'scene_match':match,
              'bddl_path':str(bddl), 'bddl_sha256':sha(bddl),
              'event_coverage':dict(Counter(sum(e['start_frame']<=i<e['stop_frame'] for e in events) for i in range(sample['frames']))),
              'seconds':round(time.monotonic()-started,3), 'manual_review':'pending'}
    save(OUT/'bindings'/(sample['key']+'.json'), record)
    return record


def regions(job):
    record = json.loads((OUT/'bindings'/(job[0]['key']+'.json')).read_text())
    if record['provenance'] != provenance():
        raise ValueError('Re-run audit after binding code changes')
    return review_native.make_regions(job, out=OUT, cache=CACHE, bindings=record['bindings'],
                                      version='v2', provenance=record['provenance'])


def video(job):
    return review_native.make_video(job[0], out=OUT, video=VIDEO, version='v2')


def report(samples):
    records = [json.loads((OUT/'bindings'/(s['key']+'.json')).read_text()) for s in samples]
    rows = [e for r in records for e in r['bindings']]
    changed = [r for r in records if r['changes']]
    region_records = [json.loads(p.read_text()) for s in samples if (p:=OUT/'regions'/(s['key']+'.json')).exists()]
    rendered = [json.loads(p.read_text()) for s in samples if (p:=OUT/'renders'/(s['key']+'.json')).exists()]
    checked = []
    for record in region_records:
        with np.load(record['archive']) as archive:
            boxes, masks, valid = archive['boxes'], archive['mask_bits'], archive['mask_valid']
            assert boxes.shape[1] == record['frames'] and valid[0].all()
            # Verify every stored bbox against the independently rendered full-in-frame mask.
            for target, frame in zip(*np.where(valid)):
                mask = np.unpackbits(masks[target,frame]).reshape(256,256).astype(bool)
                box = review_native.bbox(mask)
                if box is None:
                    assert np.isnan(boxes[target,frame]).all()
                else:
                    np.testing.assert_array_equal(boxes[target,frame],box)
            goals, goal_boxes, complete = 0, 0, True
            for event in record['bindings']:
                index = event.get('region_track')
                if index is not None:
                    target = record['targets'][index]
                    assert (target['instance'],target['part']) == (event['instance'],event['scope']['part'])
                available = index is not None and valid[index+1,event['goal_frame']]
                goals += bool(available)
                goal_boxes += bool(available and np.isfinite(boxes[index+1,event['goal_frame']]).all())
                complete &= bool(available and valid[index+1,event['start_frame']:event['stop_frame']].all())
            checked.append({'number':record['number'], 'goals':goals, 'goal_boxes':goal_boxes, 'complete_regions':bool(complete),
                            'unique_event_coverage':set(record['event_coverage']) == {'1'}})
    result = {'trajectories':len(records), 'events':len(rows), 'binding_statuses':dict(Counter(e['status'] for e in rows)),
              'all_events_bound_trajectories':sum(all(e['status']=='rule_bound' for e in r['bindings']) for r in records),
              'geometry_resolved':sum(e['geometry']['status']=='resolved' for e in rows),
              'skill_conflicts':[{'number':r['number'],'event':e['source_id'],'skill':e['skill'],'description':e['description']}
                                 for r in records for e in r['bindings'] if e['skill_check']=='conflict'],
              'motion_statuses':dict(Counter(e['motion']['status'] for e in rows)),
              'changed_trajectories':[r['number'] for r in changed],
              'newly_bound_events':sum(c['after']['status']=='rule_bound' and not c['previously_bound'] for r in records for c in r['changes']),
              'previously_bound_changes':[{'number':r['number'],**c} for r in records for c in r['changes'] if c['previously_bound']],
              'regions_checked':len(checked), 'goal_regions_available':sum(c['goals'] for c in checked),
              'nonempty_goal_boxes':sum(c['goal_boxes'] for c in checked),
              'gripper_boxes':sum(r['gripper_boxes'] for r in region_records),
              'scene_missing':[r['number'] for r in region_records if r['object_pose_source']=='missing'],
              'source_event_overlap':[r['number'] for r in records if any(int(k)>1 for k in r['event_coverage'])],
              'complete_region_trajectories':sum(c['complete_regions'] for c in checked),
              'complete_regions_and_unique_coverage':sum(c['complete_regions'] and c['unique_event_coverage'] for c in checked),
              'rendered_trajectories':[r['number'] for r in rendered], 'rendered_frames':sum(r['frames'] for r in rendered),
              'provenance':provenance(), 'manual_review':'pending'}
    save(OUT/'report.json',result)
    fields=['number','key','event','skill','skill_check','description','status','instance','part','reason','motion','changed','manual_review']
    with (OUT/'events.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for record in records:
            for row in record['bindings']:
                writer.writerow({k:row[k] for k in ['skill','skill_check','description','status','instance']} |
                    {'number':record['number'],'key':record['key'],'event':row['source_id'],
                     'part':row['scope']['part'] if row['scope'] else '', 'reason':row.get('reason',''),
                     'motion':row['motion']['status'],'changed':any(c['event']==row['source_id'] for c in record['changes']),
                     'manual_review':'pending'})
    items=[]
    for record in changed:
        destination=VIDEO/f"{record['number']:03d}_{record['key']}.mp4"
        link=os.path.relpath(destination,OUT)
        items.append(f"<tr><td>{record['number']:03d}</td><td><a href='{link}'>{html.escape(record['task'])}</a></td><td>pending</td></tr>")
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>AtomicVLA binding v2 review</title>'
        '<h1>v2：新增或变化的轨迹</h1><p>蓝色：完整夹爪；绿色：当前操作对象。终点仍为同一 Event 对象。人工待核验。</p>'
        '<p>源技能冲突不会自动更改标签；014 的源事件重叠仍保留。</p><table>'+''.join(items)+'</table>')
    print(json.dumps(result,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','regions','render','report'])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--sample',type=int)
    args=parser.parse_args()
    samples=review_native.manifest()['samples']
    if args.stage=='report':
        report(samples); return
    matches={m['key']:m for m in json.loads((BASE/'scene_summary.json').read_text())['samples']}
    selected=[s for s in samples if args.sample is None or s['number']==args.sample]
    if args.stage=='render' and args.sample is None:
        selected=[s for s in selected if json.loads((OUT/'bindings'/(s['key']+'.json')).read_text())['changes']]
    function={'audit':audit,'regions':regions,'render':video}[args.stage]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(function,(s,matches.get(s['key']))) for s in selected]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            print(json.dumps({'stage':args.stage,'done':i,'total':len(futures),'number':result['number'],
                              'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
