"""Export calibrated training episodes and fixed-time 2D EEF review labels.

Run ``python -m scripts.preprocess.libero.atomic_vla.motion_2d --help``.
The source episodes stay unchanged. Labels are saved before videos read them back.
"""
import argparse
from collections import Counter
import csv
from fractions import Fraction
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

import av
import numpy as np
from PIL import Image, ImageDraw

from openpi.policies.eskill_dataset import load_episode
from openpi.transforms import future_targets, project_points, resize_image, resize_points
from scripts.preprocess import DATA_ROOT
from scripts.preprocess.artifacts import digest
from scripts.preprocess.export import validate_episode, write_index
from scripts.preprocess.libero.render import BG, BORDER, PANEL, WHITE, MUTED, Renderer, font, render_video
from scripts.preprocess.libero.view import from_saved

CHECK = DATA_ROOT / 'libero/atomic-vla/check/test'
COLORS = ('#54cde8', '#62dda6', '#e4d65b', '#f8a251', '#ed6b83', '#c694f6', '#83a4ff', '#f5b9db')
OFFSETS = (4, 8, 12, 16, 20, 24, 28, 32)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def camera_parameters(sim, height, width, convention):
    """MuJoCo GL camera -> CV -> released OpenVLA RGB (pixel-center coordinates)."""
    camera = sim.model.camera_name2id('agentview')
    f = .5 * height / np.tan(np.deg2rad(sim.model.cam_fovy[camera]) / 2)
    intrinsic = np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = sim.data.cam_xmat[camera].reshape(3, 3) @ np.diag([1, -1, -1])
    camera_to_world[:3, 3] = sim.data.cam_xpos[camera]
    # Raw OpenGL rows run bottom to top. The RLDS conversion then rotates 180 degrees.
    image_transform = np.array([[-1, 0, width-1], [0, 1, 0], [0, 0, 1]])
    if convention == 'opencv':
        image_transform[1] = [0, -1, height-1]
    elif convention != 'opengl':
        raise ValueError(f'Unknown robosuite image convention: {convention}')
    return dict(intrinsic=intrinsic.tolist(), world_to_camera=np.linalg.inv(camera_to_world).tolist(),
                image_transform=image_transform.tolist(), image_shape=[height, width],
                camera_name='agentview', source_camera='image', reference='current_main_view',
                static=True, fovy_degrees=float(sim.model.cam_fovy[camera]),
                convention=convention, pixel_coordinates='top-left pixel center (0,0); edges at -0.5 and W/H-0.5',
                rgb_transform='robosuite observable convention, then OpenVLA 180-degree rotation')


def project(xyz, calibration):
    return project_points(xyz, **{k: calibration[k] for k in
        ('intrinsic', 'world_to_camera', 'image_transform', 'image_shape')})


def open_scene(sample, libero_root, gpu_device):
    sys.path.insert(0, str(libero_root))
    from libero.libero.envs.env_wrapper import ControlEnv
    env = ControlEnv(bddl_file_name=sample['bddl'], use_camera_obs=False, has_offscreen_renderer=True,
                     hard_reset=False, camera_names=['agentview'], camera_heights=256, camera_widths=256,
                     render_gpu_device_id=gpu_device)
    env.seed(0)
    env.reset()
    return env


def native_check(reader, sample, destination, libero_root, gpu_device, *, env=None, raster_reference=None):
    """Check every source joint pose, plus three independently rasterized site markers."""
    sys.path.insert(0, str(libero_root))
    import mujoco
    import robosuite.macros as macros
    from scripts.preprocess.libero.libero_40.align import model_check

    m = reader.meta
    state = reader.array(m['state']['values'])
    xyz = reader.array(m['effector_positions']['single']['values'])
    assert state.shape == (len(xyz), 15) and m['arm_ids'] == ['single']
    pose = m['effector_positions']['single']
    assert (pose['unit'], pose['coordinate_frame'], pose['reference_point']) == ('m', 'world', 'gripper0_grip_site')
    height, width = m['cameras']['image']['height'], m['cameras']['image']['width']
    assert (height, width) == (256, 256), 'This pilot uses the released square main camera'
    audit_path = CHECK/'all_rules_v1/audits'/f"{sample['annotation_id']}.json"
    audit = read(audit_path)
    owns_env = env is None
    if owns_env:
        env = open_scene(sample, libero_root, gpu_device)
    sim, robot = env.sim, env.robots[0]
    try:
        checked_model = model_check(env, audit['match'])
        if (checked_model['status'] != 'compatible'
                or digest(audit['match']['model_file']['path']) != checked_model['xml_sha256']):
            raise ValueError(f"Scene/camera mismatch: {sample['key']}")
        calibration = camera_parameters(sim, height, width, macros.IMAGE_CONVENTION)
        if raster_reference is not None:
            reference = read(raster_reference)
            if (reference['bddl_sha256'] != digest(sample['bddl']) or reference['camera'] != calibration):
                raise ValueError('Shared scene has different camera parameters')
        uv, valid = project(xyz, calibration)
        context = sim._render_context_offscreen
        site = robot.eef_site_id
        sim.model.site_group[:] = 0
        sim.model.site_group[site] = 4
        sim.model.site_type[site] = int(mujoco.mjtGeom.mjGEOM_SPHERE)
        sim.model.site_size[site] = [.007] * 3
        sim.model.site_rgba[site] = [1, 0, 1, 1]
        geom_ids = [sim.model.geom_name2id(name) for name in robot.gripper.visual_geoms]
        sim.model.geom_group[:] = 5
        sim.model.geom_group[geom_ids] = 1
        # A clipped sphere has a biased centroid; choose three fully in-frame markers.
        marker_frames = np.flatnonzero(valid & (uv >= 8).all(-1) & (uv < [width-8, height-8]).all(-1))
        if raster_reference is None and not len(marker_frames):
            raise ValueError('No in-frame EEF site available for a raster projection check')
        keyframes = (set(marker_frames[[0, len(marker_frames)//2, -1]].tolist())
                     if raster_reference is None else set())
        reconstructed, marker_checks = [], []
        camera_drift = 0.

        def oriented(image):
            convention = 1 if macros.IMAGE_CONVENTION == 'opengl' else -1
            return image[::convention][::-1, ::-1].copy()

        for t in range(len(xyz)):
            sim.data.qpos[robot._ref_joint_pos_indexes] = state[t, :7]
            sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = state[t, -2:]
            sim.data.qvel[:] = 0
            sim.forward()
            reconstructed.append(sim.data.site_xpos[site].copy())
            current_camera = camera_parameters(sim, height, width, macros.IMAGE_CONVENTION)
            camera_drift = max(camera_drift, float(np.abs(np.asarray(current_camera['world_to_camera'])
                                                         - calibration['world_to_camera']).max()))
            if t not in keyframes:
                continue
            context.vopt.geomgroup[:] = 0
            context.vopt.sitegroup[:] = 0
            context.vopt.sitegroup[4] = 1
            seg = oriented(sim.render(width=width, height=height, camera_name='agentview', segmentation=True))
            mask = (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_SITE)) & (seg[..., 1] == site)
            ys, xs = np.nonzero(mask)
            center = np.array([xs.mean(), ys.mean()]) if len(xs) else None
            error = float(np.linalg.norm(center-uv[t])) if center is not None else None
            marker_checks.append(dict(frame=t, pixels=len(xs), source_uv=uv[t].tolist(),
                                      raster_center=center.tolist() if center is not None else None, error_px=error))
            context.vopt.sitegroup[:] = 0
            context.vopt.geomgroup[1] = 1
            native = oriented(sim.render(width=width, height=height, camera_name='agentview'))
            board = Image.new('RGB', (3*width, height+48), BG)
            marker = np.zeros((height, width, 3), np.uint8)
            marker[mask] = [255, 0, 255]
            for col, (rgb, title) in enumerate(zip([reader.rgb('image', t), native, marker],
                    ['SOURCE RGB + EEF', 'SOURCE JOINT GRIPPER', 'RASTERIZED EEF SITE'])):
                view = Image.fromarray(rgb)
                draw = ImageDraw.Draw(view)
                if valid[t]:
                    u, v = uv[t]
                    draw.line((u-5, v, u+5, v), fill=WHITE, width=1)
                    draw.line((u, v-5, u, v+5), fill=WHITE, width=1)
                board.paste(view, (col*width, 48))
                ImageDraw.Draw(board).text((col*width+6, 5), title, font=font(13), fill=WHITE)
            error_text = f'{error:.3f} px' if error is not None else 'MISSING'
            ImageDraw.Draw(board).text((6, 25), f"{sample['annotation_id']:04d} / frame {t} / raster error {error_text}",
                                       font=font(13), fill=WHITE)
            board.save(destination/f'native_{t:04d}.png')
        reconstructed = np.asarray(reconstructed)
        fk_uv, fk_valid = project(reconstructed, calibration)
        metrics = dict(eef_fk_max_m=float(np.linalg.norm(reconstructed-xyz, axis=-1).max()),
                       eef_fk_projection_max_px=float(np.linalg.norm(fk_uv[valid & fk_valid]-uv[valid & fk_valid], axis=-1).max()),
                       static_camera_matrix_max_change=camera_drift,
                       marker_checks=marker_checks, model_check=checked_model)
        if raster_reference is not None:
            metrics['raster_reference'] = dict(path=str(raster_reference.resolve()), sha256=digest(raster_reference),
                method='same checked scene and camera; raster markers inherited, all source poses rechecked')
        if (metrics['eef_fk_max_m'] > .001 or metrics['eef_fk_projection_max_px'] > .5
                or camera_drift > 1e-9 or any(c['error_px'] is None or c['error_px'] > 1 for c in marker_checks)):
            raise ValueError(f"Native projection check failed: {metrics}")
        return calibration, uv, valid, metrics, audit_path
    finally:
        if owns_env:
            env.close()


class MotionRenderer(Renderer):
    """Use the shared LIBERO layout, replacing its 3D panel with the saved 2D targets."""
    def __init__(self, episode, labels, offsets, model_size, time_unit_frames=1):
        super().__init__(episode)
        self.labels, self.offsets, self.model_size = labels, offsets, model_size
        self.time_unit_frames = time_unit_frames

    @staticmethod
    def draw_points(image, origin, size, current, future, valid, padded=None):
        draw = ImageDraw.Draw(image)
        def pixel(point):
            return tuple(np.asarray(origin) + np.asarray(point)*size - .5)
        prior = pixel(current) if np.isfinite(current).all() and ((current >= 0) & (current < 1)).all() else None
        for j, good in enumerate(valid):
            if not good:
                prior = None
                continue
            p = pixel(future[j])
            if prior is not None:
                draw.line((*prior, *p), fill='#f8a251', width=2)
            draw.ellipse((p[0]-3, p[1]-3, p[0]+3, p[1]+3), fill=COLORS[j % len(COLORS)])
            prior = p
        if padded is not None and np.any(padded & valid):
            u, v = pixel(future[np.flatnonzero(padded & valid)[-1]])
            draw.ellipse((u-8, v-8, u+8, v+8), outline=WHITE, width=2)
        if np.isfinite(current).all() and ((current >= 0) & (current < 1)).all():
            u, v = pixel(current)
            draw.line((u-7, v, u+7, v), fill=WHITE, width=2)
            draw.line((u, v-7, u, v+7), fill=WHITE, width=2)

    def frame(self, index):
        canvas = Image.fromarray(super().frame(index))
        d, e = self.labels, self.episode
        current, future, valid = d['eef_model_xy'][index], d['future_model_xy'][index], d['future_valid'][index]
        padded = d['future_padded'][index] if 'future_padded' in d else np.zeros(len(valid), bool)
        source_frames = d['future_source_frame_indices'][index] if 'future_source_frame_indices' in d else index+np.asarray(self.offsets)
        # Source images are square 256x256; the existing main panel displays 518x518.
        self.draw_points(canvas, (88, 193), 518, current, future, valid, padded)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((28, 747, 1892, 789), fill=BG)
        draw.rectangle((28, 747, 970, 1268), fill=BG)
        draw.text((28, 753), 'EEF / FIXED FUTURE SOURCE STEPS', font=font(22, 'bold'), fill=WHITE)
        draw.text((1002, 759), 'SOURCE XYZ / gripper0_grip_site / world meters', font=font(15, 'mono'), fill=MUTED)
        draw.rounded_rectangle((28, 790, 966, 1268), radius=6, fill=PANEL, outline=BORDER)
        draw.text((48, 805), f'MODEL CANVAS {self.model_size}x{self.model_size} / SAVED TARGETS', font=font(22, 'bold'), fill=WHITE)
        draw.text((48, 837), f'1 TIME UNIT = {self.time_unit_frames} SOURCE FRAMES / OFFSETS SHOWN IN FRAMES',
                  font=font(12, 'mono'), fill=MUTED)
        model_image = Image.fromarray(resize_image(e.rgb(index), self.model_size))
        canvas.paste(model_image.resize((384, 384), Image.Resampling.NEAREST), (48, 854))
        self.draw_points(canvas, (48, 854), 384, current, future, valid, padded)
        draw.text((456, 854), '#   STEP   SOURCE    NORMALIZED (x,y)', font=font(16, 'mono'), fill=MUTED)
        for j, offset in enumerate(self.offsets):
            y = 895 + round(j * min(46, 245 / max(1, len(self.offsets) - 1)))
            draw.ellipse((456, y+5, 466, y+15), fill=COLORS[j % len(COLORS)])
            coordinates = f'{future[j,0]:.3f}, {future[j,1]:.3f}' if valid[j] else 'MASKED'
            source = f'{source_frames[j]:03d}' if source_frames[j] >= 0 else '---'
            suffix = ' END' if padded[j] else ''
            draw.text((476, y), f'{j+1}  +{offset:02d}   {source}    {coordinates}{suffix}',
                      font=font(15, 'mono'), fill=WHITE if valid[j] else MUTED)
        draw.text((456, 1180), 'WHITE + = CURRENT EEF / ARM: SINGLE', font=font(14, 'mono'), fill=WHITE)
        draw.text((456, 1207), 'END / WHITE RING = LAST-POSITION PADDING', font=font(13, 'mono'), fill=MUTED)
        draw.text((456, 1234), '10 FPS DISPLAY / TARGETS, NOT PREDICTIONS', font=font(13, 'mono'), fill=MUTED)
        return np.asarray(canvas)


def load_projection(path, reader):
    """Reuse checked per-frame pixels; verify source identity and arrays before resampling."""
    record = read(path)
    arrays = path.parent/record['arrays']
    if (record['episode_id'] != reader.meta['episode_id']
            or record['source_episode_sha256'] != digest(reader.path)
            or record['arrays_sha256'] != digest(arrays)):
        raise ValueError('Projection cache no longer matches its source episode or saved arrays')
    if not record['camera']['static']:
        raise ValueError('Per-frame projection reuse requires a checked static camera')
    with np.load(arrays) as archive:
        np.testing.assert_array_equal(archive['frame_indices'],
            reader.array(reader.meta['effector_positions']['single']['frame_indices']))
        uv, valid = archive['eef_uv'].copy(), archive['eef_valid'].copy()
    evidence = dict(metadata=str(path.resolve()), metadata_sha256=digest(path),
                    arrays=str(arrays.resolve()), arrays_sha256=record['arrays_sha256'],
                    method='reuse verified per-frame projection; native checks inherited, not rerun')
    return record['camera'], uv, valid, record['checks'], Path(record['audit']), evidence


def build_episode(sample, episode_path, output, offsets, model_size, libero_root, gpu_device,
                  *, reuse_projection=None, time_unit_frames=1, tail_policy='hold_last',
                  native_env=None, raster_reference=None):
    destination = output/'episodes'/f"{sample['annotation_id']:04d}"
    destination.mkdir(parents=True)
    reader = load_episode(episode_path)
    indices = reader.array(reader.meta['effector_positions']['single']['frame_indices'])
    np.testing.assert_array_equal(indices, np.arange(reader.meta['frame_count']))
    projection_evidence = None
    if reuse_projection is None:
        calibration, uv, valid, metrics, audit_path = native_check(
            reader, sample, destination, libero_root, gpu_device, env=native_env, raster_reference=raster_reference)
    else:
        previous = reuse_projection/'episodes'/f"{sample['annotation_id']:04d}"/'motion.json'
        calibration, uv, valid, metrics, audit_path, projection_evidence = load_projection(previous, reader)
    valid &= reader.array(reader.meta['effector_positions']['single']['valid']).all(-1)
    model_uv, model_valid = resize_points(uv, valid, calibration['image_shape'], model_size)
    pixel_targets = future_targets(indices, uv, valid, indices, offsets, tail_policy=tail_policy,
                                   last_frame=reader.meta['frame_count'] - 1)
    targets = future_targets(indices, model_uv, model_valid, indices, offsets, tail_policy=tail_policy,
                             last_frame=reader.meta['frame_count'] - 1)
    future_uv, future_valid = pixel_targets['trace'], pixel_targets['trace_valid']
    future_model_uv, future_model_valid = targets['trace'], targets['trace_valid']
    np.testing.assert_array_equal(future_valid, future_model_valid)
    event_ids = np.full(len(indices), -1, np.int64)
    for i, event in enumerate(reader.meta['events']):
        event_ids[event['start_frame']:event['stop_frame']] = i
    target_frames = targets['future_frame_indices']
    present = target_frames < len(indices)
    observed, padded = targets['future_observed'], targets['future_padded']
    source_frames = targets['future_source_frame_indices']
    crossed = present & (event_ids[np.minimum(target_frames, len(indices)-1)] != event_ids[:, None])
    stationary_errors = np.linalg.norm(future_model_uv-model_uv[:, None], axis=-1)*model_size
    arrays = dict(frame_indices=indices, eef_uv=uv, eef_valid=valid, eef_model_xy=model_uv,
                  future_frame_indices=target_frames, future_uv=future_uv, future_model_xy=future_model_uv,
                  future_valid=future_valid, future_observed=observed, future_padded=padded,
                  future_source_frame_indices=source_frames, crosses_event=crossed & future_valid)
    for key in ('intrinsic', 'world_to_camera', 'image_transform'):
        arrays['camera_' + key] = np.asarray(calibration[key], np.float64)
    array_path = destination/'motion.npz'
    np.savez_compressed(array_path, **arrays)
    count = future_valid.sum(0)
    report = dict(schema_version='eskill.motion2d.v2-pilot', episode_id=reader.meta['episode_id'],
                  annotation_id=sample['annotation_id'], suite=sample['suite'], task=reader.meta['instructions'][0],
                  source_episode=str(reader.path), source_episode_sha256=digest(reader.path),
                  source_cache=sample['source_cache'], source_cache_sha256=sample['source_cache_sha256'],
                  bddl=sample['bddl'], bddl_sha256=digest(sample['bddl']),
                  audit=str(audit_path), audit_sha256=digest(audit_path), frames=len(indices),
                  camera=calibration, arm='single', eef_reference='gripper0_grip_site',
                  timebase='source_steps', source_step_definition='rows of released no-noop RLDS; not physical seconds',
                  offsets=list(offsets), offset_units=[v//time_unit_frames for v in offsets],
                  time_unit_source_frames=time_unit_frames, anchor='paired_observation', model_size=model_size,
                  tail_policy=tail_policy, padded_points_are_supervised=tail_policy == 'hold_last',
                  normalized_coordinates='((source_uv + 0.5) * actual_resize_scale + padding) / model_size',
                  arrays='motion.npz', arrays_sha256=digest(array_path), checks=metrics,
                  existing_warnings=reader.meta['quality']['warnings'],
                  coverage=dict(eef_valid=int(valid.sum()), valid_by_offset=count.tolist(),
                                observed_valid_by_offset=observed.sum(0).tolist(),
                                padded_by_offset=padded.sum(0).tolist(),
                                invalid_by_offset=(~future_valid).sum(0).tolist(),
                                missing_future_by_offset=(~present).sum(0).tolist(),
                                invalid_projection_by_offset=(present & ~future_valid).sum(0).tolist(),
                                cross_event_points=int((crossed & future_valid).sum())),
                  stationary_baseline=dict(description='Copy current EEF location to all future times; no learned model',
                      pixel_error_sum_by_offset=np.where(future_valid, stationary_errors, 0).sum(0).tolist(),
                      mean_pixel_error_by_offset=(np.where(future_valid, stationary_errors, 0).sum(0)/np.maximum(count, 1)).tolist()),
                  manual_review='pending')
    if projection_evidence is not None:
        report['projection_source'] = projection_evidence
    training_episode = write_training_episode(reader, destination, arrays, report)
    report['training_episode'] = str(training_episode.resolve())
    save(destination/'motion.json', report)
    return destination/'motion.json'


def write_training_episode(reader, destination, arrays, record):
    """New metadata only: keep original arrays/RGB and attach checked camera calibration."""
    def rebase(value):
        if isinstance(value, dict):
            return {key: str((reader.path.parent / item).resolve()) if key == 'uri' and item is not None
                    else rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        return value
    meta = rebase(reader.meta)
    def ref(key):
        name = 'camera_' + key
        return dict(uri='motion.npz', key=name, shape=list(arrays[name].shape), dtype=str(arrays[name].dtype))
    camera = record['camera']['source_camera']
    meta['cameras'][camera]['calibration'] = dict(
        intrinsics=ref('intrinsic'), extrinsics=ref('world_to_camera'), image_transform=ref('image_transform'),
        extrinsics_convention='world_to_camera_cv', world_frame='world')
    meta['motion_projection'] = dict(source_episode=str(reader.path),
        source_episode_sha256=record['source_episode_sha256'], evidence='motion.json',
        reference_point=record['eef_reference'], manual_review='pending')
    path = destination/'episode.json'
    save(path, meta)
    validate_episode(load_episode(path))
    return path


def render_saved(path, video_path):
    record = read(path)
    reader = load_episode(record['source_episode'])
    if digest(reader.path) != record['source_episode_sha256'] or digest(path.parent/record['arrays']) != record['arrays_sha256']:
        raise ValueError('Saved inputs changed before rendering')
    with np.load(path.parent/record['arrays']) as archive:
        arrays = {k: archive[k] for k in archive.files}
    view = from_saved(reader)
    view.note = 'SAVED FUTURE EEF TARGETS / SOURCE STEPS / REVIEW PENDING'
    renderer = MotionRenderer(view, arrays, record['offsets'], record['model_size'],
                              record.get('time_unit_source_frames', 1))
    # Mid-trajectory preview, full video including every tail frame.
    Image.fromarray(renderer.frame(view.frame_count//2)).save(path.parent/'preview.jpg', quality=94)
    tail_frame = max(0, view.frame_count - max(record['offsets'])//2 - 1)
    Image.fromarray(renderer.frame(tail_frame)).save(path.parent/'tail_preview.jpg', quality=94)
    video = render_video(view, video_path, renderer=renderer)
    return {**record, 'label': str(path), 'video': str(video_path), 'video_sha256': digest(video_path),
            'video_check': video, 'tail_preview_frame': tail_frame}


def merge_videos(records, destination):
    """Same packet-copy concatenation as previous LIBERO reviews; no ffmpeg executable."""
    offset, chapters = Fraction(0), []
    with av.open(str(destination), 'w', options={'movflags': '+faststart'}) as output:
        with av.open(records[0]['video']) as first:
            source = first.streams.video[0]
            stream = output.add_stream_from_template(source)
            configuration = source.codec_context.extradata
        for r in records:
            chapters.append(dict(id=r['annotation_id'], start=int(offset*10), end=int(offset*10)+r['frames'],
                time_base=Fraction(1, 10), metadata={'title': r['task']}))
            with av.open(r['video']) as incoming:
                source = incoming.streams.video[0]
                assert source.codec_context.extradata == configuration
                for packet in incoming.demux(source):
                    if packet.dts is None:
                        continue
                    shift = offset / packet.time_base
                    assert shift.denominator == 1
                    packet.pts += int(shift)
                    packet.dts += int(shift)
                    packet.stream = stream
                    output.mux(packet)
            offset += Fraction(r['frames'], 10)
        output.set_chapters(chapters)
    count = 0
    with av.open(str(destination)) as incoming:
        assert len(incoming.chapters()) == len(records)
        for frame in incoming.decode(video=0):
            assert (frame.width, frame.height) == (1920, 1340)
            assert abs(float(frame.pts*frame.time_base) - count/10) < 1e-6
            count += 1
    assert count == sum(r['frames'] for r in records)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, default=Path('/data1/yuankai/robot-research/data/atomicdata/libero40/index.json'))
    parser.add_argument('--output', type=Path, default=CHECK/'motion_2d_8x4_v1')
    parser.add_argument('--video-dir', type=Path, default=CHECK/'video/motion_2d_8x4_v1')
    parser.add_argument('--libero-root', type=Path, default=Path('/data1/yuankai/robot-research/workspace/third_party/libero-native'))
    parser.add_argument('--offsets', nargs='+', type=int, default=list(OFFSETS), help='Future offsets measured in time units')
    parser.add_argument('--time-unit-frames', type=int, default=1, help='Source-frame steps per future-motion time unit')
    parser.add_argument('--tail-policy', choices=['hold_last', 'mask'], default='hold_last',
                        help='Extend the last valid terminal record, or mask targets past episode end')
    parser.add_argument('--reuse-projection', type=Path, help='Previous pilot directory with checked per-frame projections')
    parser.add_argument('--model-size', type=int, default=224)
    parser.add_argument('--gpu-device', type=int, default=1)
    parser.add_argument('--annotation-ids', nargs='+', type=int, help='Default: existing ten-trajectory four-suite review cohort')
    parser.add_argument('--all', action='store_true', help='Export every source episode')
    parser.add_argument('--skip-video', action='store_true', help='Export labels and training index without rendering')
    args = parser.parse_args()
    if args.offsets[0] <= 0 or any(b <= a for a, b in zip(args.offsets, args.offsets[1:])):
        parser.error('Offsets must be positive and strictly increasing')
    if args.time_unit_frames < 1:
        parser.error('Time unit must contain at least one source-frame step')
    offset_units = args.offsets
    args.offsets = [v*args.time_unit_frames for v in offset_units]
    args.output.mkdir(parents=True, exist_ok=False)
    if not args.skip_video:
        args.video_dir.mkdir(parents=True, exist_ok=False)
    source_path = CHECK/'all_rules_v1/sources.json'
    sources = read(source_path)['samples']
    if args.all and args.annotation_ids:
        parser.error('--all and --annotation-ids are mutually exclusive')
    selected = ([s['annotation_id'] for s in sources] if args.all else
                args.annotation_ids or read(CHECK/'all_rules_v1/manifest.json')['review_annotation_ids'])
    samples = [s for s in sources if s['annotation_id'] in selected]
    if len(samples) != len(set(selected)):
        raise ValueError('Requested source IDs are not all available')
    paths = {e['episode_id']: args.index.parent/e['path'] for e in read(args.index)['episodes']}
    save(args.output/'manifest.json', dict(annotation_ids=selected, offsets=args.offsets, timebase='source_steps',
         offset_units=offset_units, time_unit_source_frames=args.time_unit_frames,
         tail_policy=args.tail_policy,
         reuse_projection=str(args.reuse_projection.resolve()) if args.reuse_projection else None,
         model_size=args.model_size, source_index=str(args.index), source_index_sha256=digest(args.index),
         source_manifest=str(source_path), source_manifest_sha256=digest(source_path),
         script_sha256=digest(__file__), manual_review='pending', gpu_device=args.gpu_device,
         packages={p:version(p) for p in ['mujoco', 'robosuite', 'numpy', 'pillow', 'av']},
         runtime_sha256={str(p):digest(p) for p in [Path(__file__).parents[1]/'render.py',
             DATA_ROOT.parent/'src/openpi/transforms.py', DATA_ROOT.parent/'src/openpi/policies/eskill_dataset.py']}))
    records = []
    for i, sample in enumerate(samples, 1):
        path = build_episode(sample, paths[sample['key']], args.output, args.offsets, args.model_size,
                             args.libero_root, args.gpu_device, reuse_projection=args.reuse_projection,
                             time_unit_frames=args.time_unit_frames, tail_policy=args.tail_policy)
        record = read(path) if args.skip_video else render_saved(path, args.video_dir/f"{i:02d}_{sample['key']}.mp4")
        records.append(record)
        save(args.output/'progress.json', dict(completed=len(records), total=len(samples), episodes=records))
        print(json.dumps(dict(completed=len(records), total=len(samples), key=sample['key'],
              frames=record['frames'], fk_max_m=record['checks']['eef_fk_max_m'],
              marker_max_px=max(c['error_px'] for c in record['checks']['marker_checks']))), flush=True)
    write_index(args.output/'index.json', [r['training_episode'] for r in records])
    if args.skip_video:
        save(args.output/'report.json', dict(episodes=len(records), offsets=args.offsets,
             training_index=str((args.output/'index.json').resolve()), episodes_detail=records,
             manual_review='pending'))
        return
    merged = args.video_dir/f'merged{len(records):02d}_motion_2d.mp4'
    decoded = merge_videos(records, merged)
    counts = np.sum([r['coverage']['valid_by_offset'] for r in records], axis=0)
    error_sums = np.sum([r['stationary_baseline']['pixel_error_sum_by_offset'] for r in records], axis=0)
    summary = dict(episodes=len(records), frames=decoded, suites=dict(Counter(r['suite'] for r in records)),
        offsets=args.offsets, offset_units=offset_units, time_unit_source_frames=args.time_unit_frames,
        timebase='source_steps', model_size=args.model_size, tail_policy=args.tail_policy,
        native_checks_reused=args.reuse_projection is not None,
        valid_points_by_offset=counts.tolist(), valid_fraction_by_offset=(counts/decoded).tolist(),
        observed_points_by_offset=np.sum([r['coverage']['observed_valid_by_offset'] for r in records], axis=0).tolist(),
        padded_points_by_offset=np.sum([r['coverage']['padded_by_offset'] for r in records], axis=0).tolist(),
        stationary_mean_pixel_error_by_offset=(error_sums/np.maximum(counts, 1)).tolist(),
        eef_fk_max_m=max(r['checks']['eef_fk_max_m'] for r in records),
        eef_fk_projection_max_px=max(r['checks']['eef_fk_projection_max_px'] for r in records),
        marker_max_error_px=max(c['error_px'] for r in records for c in r['checks']['marker_checks']),
        cross_event_points=sum(r['coverage']['cross_event_points'] for r in records),
        merged_video=str(merged), merged_video_sha256=digest(merged), decoded_frames=decoded,
        manual_review='pending', episodes_detail=records)
    save(args.output/'report.json', summary)
    with (args.output/'manual_review.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['annotation_id', 'episode_id', 'projection', 'future_window', 'notes'])
        writer.writerows([r['annotation_id'], r['episode_id'], 'pending', 'pending', ''] for r in records)
    rows, cursor = [], 0
    for r in records:
        rows.append(f'<tr><td>{r["annotation_id"]:04d}</td><td>{html.escape(r["task"])}</td>'
            f'<td><a href="#" onclick="document.querySelector(\'video\').currentTime={cursor/10};return false">'
            f'{cursor/10:.1f}s</a></td><td><a href="{os.path.relpath(r["video"], args.output)}">单条视频</a>'
            f' · <a href="episodes/{r["annotation_id"]:04d}/tail_preview.jpg">末尾补齐预览</a></td></tr>')
        cursor += r['frames']
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>LIBERO 未来二维末端轨迹核验</title>'
        '<style>body{font-family:sans-serif;max-width:1280px;margin:24px auto;background:#101820;color:#e9f0f5}'
        'a{color:#79d9ad}td,th{padding:10px;text-align:left;border-bottom:1px solid #2b3b47}video{width:100%}</style>'
        f'<h1>{len(records)} 条轨迹 · 未来二维末端标签</h1>'
        f'<p>每时间单位 {args.time_unit_frames} 个来源帧，未来来源步：{args.offsets}；'
        f'末尾策略：{args.tail_policy}。白色十字为当前末端，{len(args.offsets)} 个彩色点为未来监督位置。'
        'END 和白色圆圈表示取轨迹最后一帧的位置补齐，属于训练约定，不是额外观察到的未来。'
        '蓝框为夹爪，绿框为当前操作物体，黄框为事件终点对象。遮挡不取消末端监督。'
        '视频 10 fps 仅用于展示；人工验收 pending。</p>'
        f'<video controls preload="metadata" src="{os.path.relpath(merged, args.output)}"></video>'
        '<table><tr><th>ID</th><th>任务</th><th>合并视频位置</th><th>单条</th></tr>' + ''.join(rows) + '</table>')
    print(json.dumps({k:v for k,v in summary.items() if k != 'episodes_detail'}), flush=True)


if __name__ == '__main__':
    main()
