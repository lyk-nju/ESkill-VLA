"""Export RoboTwin raw v2 to self-contained Episodes; semantic annotations remain pending."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from openpi.policies.eskill_dataset import SCHEMA_VERSION
from scripts.preprocess.export import EpisodeWriter, write_index

CAMERAS = dict(image='head_camera', left_wrist_image='left_camera',
               right_wrist_image='right_camera', front_image='front_camera')
ARMS = ('left', 'right')


def observation_rows(physical_steps):
    """Keep the last saved row at each physical instant, without inventing observations."""
    steps = np.asarray(physical_steps)
    if steps.ndim != 1 or len(steps) < 2 or np.any(np.diff(steps) < 0):
        raise ValueError('Expected monotone physical steps and at least two source observations')
    rows = np.r_[np.flatnonzero(np.diff(steps) != 0), len(steps) - 1]
    if len(rows) < 2:
        raise ValueError('Need two distinct observation times for next-observation targets')
    return rows


def channels(kind):
    return [dict(name=f'{arm}_{name}', arm=arm, unit='rad' if i < 6 else 'normalized',
                 semantics=kind if i < 6 else 'gripper_command_0_closed_1_open')
            for arm in ARMS for i, name in enumerate([f'joint_{j}' for j in range(6)] + ['gripper'])]


def export_episode(path, output):
    path, output = Path(path).resolve(), Path(output)
    writer = EpisodeWriter()
    with h5py.File(path, 'r') as source:
        if source.attrs.get('format') != 'robotwin.raw.v2' or not source.attrs.get('complete'):
            raise ValueError(f'Expected complete raw v2: {path}')
        task, revision = str(source.attrs['task']), str(source.attrs['collection_fingerprint'])
        identity = f'{task}_{path.stem}'
        obs = source['observations']
        rows = observation_rows(obs['physical_step'][:])
        n = len(rows)
        times = source['timestamp'][:][rows]
        if not np.all(np.diff(times) > 0):
            raise ValueError(f'{identity}: observation timestamps do not match physical ordering')
        frames = writer.add('frames', np.arange(n, dtype=np.int64))
        action_frames = writer.add('action_frames', np.arange(n - 1, dtype=np.int64))
        all_valid = writer.add('rgb_valid', np.ones(n, dtype=bool))

        def numeric(key, values, indices=frames, **metadata):
            values = np.asarray(values)
            return dict(frame_indices=indices, values=writer.add(key, values),
                        valid=writer.add(key + '_valid', np.isfinite(values)), **metadata)

        cameras = {}
        for camera, source_camera in CAMERAS.items():
            if source_camera not in obs['observation']:
                continue
            group = obs['observation'][source_camera]
            height, width = group['rgb'].shape[1:3]
            rgb_values = group["rgb"][:]
            encoded = []
            # Source RGB is already decoded. PNG preserves every source pixel exactly.
            for row in rows:
                buffer = BytesIO()
                Image.fromarray(rgb_values[int(row)]).save(buffer, format='PNG', compress_level=1)
                encoded.append(buffer.getvalue())
            intrinsic = group['intrinsic_cv'][:][rows]
            extrinsic = group['extrinsic_cv'][:][rows]
            if extrinsic.shape[-2:] == (3, 4):
                expanded = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
                expanded[:, :3] = extrinsic
                extrinsic = expanded
            # Moving wrist calibration is retained, but not used for fixed-view Motion labels.
            if np.array_equal(intrinsic, np.broadcast_to(intrinsic[0], intrinsic.shape)):
                intrinsic = intrinsic[0]
            if np.array_equal(extrinsic, np.broadcast_to(extrinsic[0], extrinsic.shape)):
                extrinsic = extrinsic[0]
            calibration = dict(intrinsics=writer.add(camera + '_intrinsics', intrinsic),
                               extrinsics=writer.add(camera + '_extrinsics', extrinsic),
                               image_transform=writer.add(camera + '_pixel_transform', np.eye(3)),
                               extrinsics_convention='world_to_camera_cv', world_frame='world')
            cameras[camera] = dict(frame_indices=frames, valid=all_valid, height=height, width=width,
                                   rgb=writer.add_encoded(camera, encoded, height, width),
                                   role='wrist' if 'wrist' in camera else 'external', calibration=calibration)
        if not {'image', 'left_wrist_image', 'right_wrist_image'} <= cameras.keys():
            raise ValueError('Clean training episodes require main and both wrist views')
        state = np.concatenate([obs[f'joint_state/{arm}_{part}'][:].reshape(-1, size)[rows]
                                for arm in ARMS for part, size in (('arm', 6), ('gripper_command', 1))], axis=1)
        commands = obs['joint_action/vector'][:][rows]
        control = source['control_trace']
        control_steps = control['physical_step'][:]
        target_control_rows = np.searchsorted(control_steps, obs['physical_step'][:][rows[1:]], side='left') - 1
        if np.any(target_control_rows < 0):
            raise ValueError('Next-observation command has no preceding physical control')
        executed = control['action'][:][target_control_rows]
        if not np.allclose(commands[1:], executed, atol=1e-7, rtol=0):
            raise ValueError(f'{identity}: next-observation command differs from last applied control')
        actions = numeric('actions', commands[1:], action_frames,
                          channels=channels('absolute_joint_drive_target'), representation='absolute_joint_targets',
                          coordinate_frame='joint', rotation_convention=None, gripper_convention='0_closed_1_open',
                          alignment='current unique observation -> next observation drive target; no second shift',
                          conditioning_image_indices={camera: action_frames for camera in cameras},
                          step_duration_s=writer.add('action_duration_s', np.diff(times)))

        def region(key, group, *, full):
            prefix = 'full_' if full else ''
            return dict(frame_indices=frames,
                        boxes_xyxy=writer.add(key + '_boxes', group[prefix + 'boxes_xyxy'][:][rows]),
                        boxes_valid=writer.add(key + '_valid', group[prefix + 'boxes_valid'][:][rows]),
                        masks=None, masks_valid=None, visibility=None,
                        region_extent='full_in_frame' if full else 'visible',
                        evidence={'method': 'native_instance_render', 'review_status': 'pending'})

        regions = obs['native_regions/regions']
        effectors = {arm: {'image': region(arm, regions[arm + '_gripper'], full=True)} for arm in ARMS}
        reference = str(source.attrs.get('ee_reference', ''))
        confirmed = reference == 'world_wxyz_get_ee_pose_is_endpose_false_tcp_offset_local_x_0.12m'
        poses = {arm: obs[f'endpose/{arm}_endpose'][:][rows] for arm in ARMS}
        positions = {arm: numeric(arm + '_eef_xyz', poses[arm][:, :3],
                                 unit='m' if confirmed else None, coordinate_frame='world' if confirmed else None,
                                 reference_point=reference or None) for arm in ARMS}
        # Visible objects and unreviewed primitives are evidence, not valid Grounding/skill labels.
        objects = {name: {'image': region('object_' + name, group, full=False)}
                   for name, group in regions.items() if name not in ('left_gripper', 'right_gripper')}
        warnings = ['event_semantics_pending', 'object_full_in_frame_pending',
                    'gripper_state_is_command', 'variable_observation_intervals']
        if not confirmed:
            warnings.append('eef_semantics_pending')
        meta = dict(
            schema_version=SCHEMA_VERSION, episode_id=identity, task=task,
            source=dict(dataset='robotwin_clean', revision=revision, episode_id=path.stem,
                        parent_trajectory_id=identity, hdf5=str(path), seed=int(source.attrs['seed'])),
            instructions=json.loads(source['instructions'][()]), embodiment='aloha_agilex',
            arm_ids=list(ARMS), frame_count=n, timeline='timestamps', timestamps_s=writer.add('timestamp_s', times),
            nominal_fps=None, alignment='unique physical observation times; keep last source row at each time',
            cameras=cameras, state=numeric('state', state, channels=channels('measured_joint_qpos')),
            actions=actions, effector_positions=positions, effector_regions=effectors,
            events=[], cooperative_goal_groups=[],
            source_annotations=dict(evidence=json.loads(source['evidence'][()]),
                                    entities=json.loads(source.attrs['native_entities']),
                                    visible_object_regions=objects,
                                    original_frame_count=int(source.attrs['frame_count'])),
            alignment_metadata=dict(source_rows=writer.add('source_rows', rows),
                                    physical_steps=writer.add('physical_steps', obs['physical_step'][:][rows]),
                                    target_control_rows=writer.add('target_control_rows', target_control_rows),
                                    source_frame_count=int(source.attrs['frame_count']),
                                    action_sampling='next_unique_observation_target_not_250Hz_motor_commands'),
            effector_poses={arm: writer.add(arm + '_pose_world_wxyz', pose) for arm, pose in poses.items()},
            quality=dict(manual_review='pending', warnings=warnings))
        saved = writer.save(output / 'episodes' / identity, meta)
    return str(saved)


def export_dataset(paths, output, workers=2):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(export_episode, p, output) for p in paths]
        saved = []
        for job in jobs:
            saved.append(job.result())
            print(json.dumps(dict(completed=len(saved), total=len(paths), episode=saved[-1])), flush=True)
    return write_index(output / 'index.json', saved)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--manifest', type=Path, help='Fixed source samples with hdf5 paths')
    inputs.add_argument('--source-root', type=Path, help='Full clean collection, including supplement')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    if args.manifest:
        paths = [Path(e['hdf5']) for e in json.loads(args.manifest.read_text())['samples']]
    else:
        paths = sorted(args.source_root.glob('*/aloha_agilex/raw/episode_*.hdf5'))
        paths += sorted((args.source_root / 'supplement').glob('*/aloha_agilex/raw/episode_*.hdf5'))
    identities = [(p.parents[2].name, p.stem) for p in paths]
    if not paths or len(set(identities)) != len(paths):
        parser.error('Need nonempty source episodes with unique task/episode identities')
    export_dataset(paths, args.output, args.workers)


if __name__ == '__main__':
    main()
