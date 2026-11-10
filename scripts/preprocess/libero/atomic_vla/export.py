"""Export existing full-frame AtomicVLA labels as self-contained Episodes; render saved labels.

No simulator or instance inference is run here. The batch region producer can call
export_episode() immediately after completing a trajectory's native regions. Each Episode
holds both cameras' original JPEG bytes in images.npz; Motion is added later by calibrate.py.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import SCHEMA_VERSION, load_episode
from scripts.preprocess.export import EpisodeWriter, validate_episode
from scripts.preprocess.artifacts import digest as sha
from .source import RAW, encoded_frames, source_data, wrist_frames
from .gripper import gripper_convention

class Inputs:
    """Confirmed export inputs, given explicitly on the command line.

    sources: directory with the source manifest (sources.json, manifest.json) and the per-trajectory
    scene audits (audits/<annotation_id>.json). bindings: merged instance labels. Neither is a
    review-output directory; check/ only receives reports.
    """
    def __init__(self, sources, bindings):
        self.sources, self.bindings = Path(sources).resolve(), Path(bindings).resolve()
        for path in (self.sources/'sources.json', self.sources/'manifest.json', self.bindings):
            if not path.is_file():
                raise FileNotFoundError(path)

    def samples(self):
        return read(self.sources/'sources.json')['samples']

    def manifest(self):
        return read(self.sources/'manifest.json')

    def labels(self):
        return read(self.bindings)['episodes']

    def audit(self, annotation_id):
        return self.sources/'audits'/f'{annotation_id}.json'

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--sources', type=Path, required=True,
                            help='source manifest directory (sources.json, manifest.json, audits/)')
        parser.add_argument('--bindings', type=Path, required=True, help='merged instance labels JSON')


def read(path):
    return json.loads(Path(path).read_text())


def evidence(method, description, *references, review='pending'):
    return dict(method=method,description=description,references=list(map(str,references)),review_status=review)


def effector_semantics(source_cache):
    """Verified geometry of the fixed OpenVLA LIBERO RLDS release; values stay unchanged."""
    return dict(unit='m', coordinate_frame='world', reference_point='gripper0_grip_site',
        evidence=evidence('source', 'RLDS state[:3] preserves robot0_eef_pos: Panda grip_site world position in meters',
            source_cache,
            'https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py#L164',
            'https://github.com/moojink/rlds_dataset_builder/blob/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_10/LIBERO_10_dataset_builder.py#L48',
            'https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/robots/single_arm.py#L215',
            'https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/assets/grippers/panda_gripper.xml#L20'))


def export_episode(sample, labels, region_path, destination, *, manifest, labels_path, wrist=None):
    """All-frame labels -> one EpisodeData bundle; reject sparse or stale region input."""
    region_path = Path(region_path).resolve()
    record = read(region_path)
    n = sample['frames']
    if (record['key'] != sample['key'] or record['frames'] != n or
        record['source_cache_sha256'] != sample['source_cache_sha256'] or
        labels['source_cache_sha256'] != sample['source_cache_sha256'] or
        record['scene_match']['excerpt_sha256'] != labels['source_scene']['excerpt_sha256'] or
        record['region_extent'] != 'full_in_frame'):
        raise ValueError('Regions, identity labels and source episode must describe the same data')
    if record['binder_sha256'] != manifest['rules']['binder'] or record['rules_sha256'] != manifest['rules']['catalog']:
        raise ValueError('Region binding rule version differs from the checked release')
    if sha(record['archive']) != record['archive_sha256']:
        raise ValueError('Region archive changed')
    data = source_data(sample)
    with np.load(record['archive'],allow_pickle=False) as archive:
        boxes,bits,available = (archive[k] for k in ['boxes','mask_bits','mask_valid'])
    # Native extraction uses the unmodified 256x256 main camera in this release.
    h = w = 256
    if boxes.shape != (len(record['targets'])+1,n,4) or bits.shape != (len(boxes),n,h*w//8):
        raise ValueError('Expected full-frame region arrays; keyframe excerpts are not a full export')
    if available.shape != boxes.shape[:2] or not available[0].all():
        raise ValueError('Gripper region extraction is incomplete')
    rows = {e['source_id']:e for e in record['bindings']}
    writer = EpisodeWriter()
    frame_indices = writer.add('frame_indices',np.arange(n,dtype=np.int64))
    all_valid = writer.add('observations_valid',np.ones(n,dtype=bool))
    main = encoded_frames(data)
    if wrist is None:
        wrist, rlds_main = wrist_frames(sample)
        if rlds_main != main:
            raise ValueError('RLDS record main camera differs from the source cache; wrist view is not paired')
    if len(main) != n or len(wrist) != n:
        raise ValueError('Both cameras must have one frame per source step')
    cameras = {name: dict(rgb=writer.add_encoded(name, frames, h, w), frame_indices=frame_indices,
                          valid=all_valid, height=h, width=w, role=role, calibration=None)
               for name, frames, role in (('image', main, 'external'), ('wrist_image', wrist, 'wrist'))}

    def numeric(prefix,values):
        return dict(values=writer.add(prefix,values),frame_indices=frame_indices,
                    valid=writer.add(prefix+'_valid',np.isfinite(values)))

    def region(track):
        prefix = f'region_{track:03d}'
        mask = np.unpackbits(bits[track],axis=1,count=h*w,bitorder='big').reshape(n,h,w).astype(bool)
        valid_boxes = available[track] & np.isfinite(boxes[track]).all(axis=1)
        # Native full masks and boxes must describe exactly the same pixels.
        for t in np.flatnonzero(available[track]):
            y,x = np.nonzero(mask[t])
            if bool(len(x)) != bool(valid_boxes[t]):
                raise ValueError('Empty mask / bbox availability mismatch')
            if len(x) and not np.array_equal(boxes[track,t],[x.min(),y.min(),x.max()+1,y.max()+1]):
                raise ValueError('BBox is not the tight bound of the saved full mask')
        return dict(frame_indices=frame_indices,boxes_xyxy=writer.add(prefix+'_boxes',boxes[track]),
            boxes_valid=writer.add(prefix+'_boxes_valid',valid_boxes),
            masks=writer.add(prefix+'_masks',mask,pack_mask=True),
            masks_valid=writer.add(prefix+'_masks_valid',available[track]),visibility=None,
            region_extent='full_in_frame',evidence=evidence('native',
                record['gripper_pose_source'] if track == 0 else record['object_pose_source'],region_path))

    regions = {0:region(0)}
    events = []
    for source in labels['events']:
        old = rows[source['source_id']]
        fields = ['start_frame','stop_frame','goal_frame','skill','instance','scope','status']
        if any(old[k] != source[k] for k in fields) or old.get('region_track') is None:
            raise ValueError('Event regions do not match the current merged instance labels')
        track = old['region_track']+1
        target = record['targets'][track-1]
        if (target['instance'] != source['instance'] or target['part'] != source['scope']['part'] or
            target['geom_names'] != source['geometry']['visual_geom_names']):
            raise ValueError('Region track is for a different physical target')
        required = np.r_[np.arange(source['start_frame'],source['stop_frame']),source['goal_frame']]
        if not available[track,required].all():
            raise ValueError('An active/goal object frame is missing; complete native extraction first')
        if track not in regions:
            regions[track] = region(track)
        corrected = 'skill_correction' in source
        events.append(dict(event_id=source['source_id'],arm='single',start_frame=source['start_frame'],
            stop_frame=source['stop_frame'],goal_frame=source['goal_frame'],boundary_valid=True,
            description=source['description'],skill=source['skill'],skill_valid=True,object_present=True,
            object_regions={'image':regions[track]},cooperative_group_id=None,
            source_annotation=deepcopy(source),evidence={
                'boundary':evidence('rule','Source intervals; shared boundary belongs to following event',labels_path),
                'skill':evidence('human' if corrected else 'source',
                    'User-confirmed Push correction' if corrected else 'Published AtomicVLA skill',
                    labels_path,review='accepted' if corrected else 'pending'),
                'object_binding':evidence('assistant' if source['status']=='manual_bound' else 'rule',
                    source['status'],labels_path)}))

    channels = [f'joint_{i}' for i in range(7)]+['eef_x','eef_y','eef_z','eef_rotation_0',
                'eef_rotation_1','eef_rotation_2','gripper_finger_0','gripper_finger_1']
    state = dict(**numeric('state',np.concatenate([data['joints'],data['state']],axis=1)),
        channels=[dict(name=name,arm='single',unit=None) for name in channels],
        semantics='Exact source joint_state[7] + state[8], unchanged physical values; channel units need upstream confirmation')
    action = dict(**numeric('actions',data['actions']),
        channels=[dict(name=f'source_action_{i}',arm='single',unit=None) for i in range(data['actions'].shape[1])],
        representation=None,coordinate_frame=None,rotation_convention=None,gripper_convention=gripper_convention(),
        alignment='Source RLDS step pairing; no shift or resampling',
        conditioning_image_indices={'image':frame_indices,'wrist_image':frame_indices},step_duration_s=None)
    pose = dict(**numeric('eef_xyz',data['state'][:,:3]), **effector_semantics(sample['source_cache']))
    source_paths = [sample['source_cache'],manifest['annotation_path'],str(labels_path.resolve()),str(region_path),
                    record['archive'],labels['source_scene']['excerpt_path']]
    checksums = dict(zip(source_paths,[sample['source_cache_sha256'],manifest['annotation_sha256'],sha(labels_path),
        sha(region_path),record['archive_sha256'],labels['source_scene']['excerpt_sha256']]))
    # Wrist JPEGs come from the same RLDS record; its main-camera bytes were checked against the cache.
    source_paths.append(f"{RAW/sample['shard']}@{sample['byte_offset']}")
    warnings = ['physical_timestamps_unavailable','action_semantics_pending']
    if labels['state_alignment_exceeds_10mm']:
        warnings.append('state_alignment_exceeds_10mm')
    metadata = dict(schema_version=SCHEMA_VERSION,episode_id=sample['key'],
        source=dict(dataset='AtomicVLA / openvla/modified_libero_rlds',revision=manifest['revision'],
            episode_id=str(sample['annotation_id']),parent_trajectory_id=sample['key'],
            artifacts=[dict(uri=p,key=None,shape=None,dtype=None) for p in source_paths],checksums=checksums),
        instructions=[sample['task']],embodiment='libero_panda',arm_ids=['single'],source_arm_mapping={'single':'single'},
        frame_count=n,timeline='source_steps',timestamps_s=None,nominal_fps=None,
        alignment='Exact RLDS source rows; annotation converted explicitly to [start,stop) + goal; no physical timestamps',
        cameras=cameras,state=state,actions=action,
        effector_positions={'single':pose},effector_regions={'single':{'image':regions[0]}},
        events=events,cooperative_goal_groups=[],quality=dict(manual_review='pending',warnings=warnings,
            full_frame_regions=True,gripper_eef_max_error_m=record['max_eef_error_m'],
            scene_alignment=record['scene_alignment'],object_state_source=labels['source_scene'],
            rules=manifest['rules'],annotation_revision=manifest['annotation_revision']))
    return writer.save(destination,metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    export = commands.add_parser('export')
    Inputs.add_arguments(export)
    export.add_argument('--annotation-id',type=int,required=True,help='Author annotation ID, zero based')
    export.add_argument('--regions',type=Path,required=True,help='Existing native full-frame region JSON')
    export.add_argument('--output',type=Path,required=True,help='New directory for this episode')
    render = commands.add_parser('render')
    render.add_argument('--episode',type=Path,required=True)
    render.add_argument('--video',type=Path,required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        inputs = Inputs(args.sources, args.bindings)
        sample = next(s for s in inputs.samples() if s['annotation_id']==args.annotation_id)
        path = export_episode(sample,inputs.labels()[str(args.annotation_id)],args.regions,args.output,
                              manifest=inputs.manifest(),labels_path=inputs.bindings)
        print(json.dumps(dict(episode=str(path),**validate_episode(load_episode(path)))),flush=True)
    else:
        from scripts.preprocess.libero.view import render_saved
        render_saved(args.episode, args.video)


if __name__ == '__main__':
    main()
