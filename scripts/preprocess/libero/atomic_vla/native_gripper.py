"""Three-frame isolated Panda gripper rendering from the existing RLDS joints."""
import argparse
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from scripts.preprocess.libero.rlds import load_records
from scripts.preprocess.libero.render import Region, overlay, BLUE
from scripts.preprocess.artifacts import digest

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT / "libero/atomic-vla"
FRAMES = (0, 48, 96)


def bbox(mask):
    y, x = np.nonzero(mask)
    return [int(x.min()), int(y.min()), int(x.max())+1, int(y.max())+1] if len(x) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT/'check/test/native_gripper')
    args = parser.parse_args()
    started = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out/'report.json').exists():
        raise FileExistsError('Use a new output directory to preserve the previous test')
    sys.path.insert(0, str(args.libero_root))
    from libero.libero.envs import OffScreenRenderEnv
    import mujoco
    import robosuite.macros as macros
    import robosuite.utils.transform_utils as T
    from scipy.spatial.transform import Rotation

    manifest_path = ROOT/'check/raw/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sample = next(s for s in manifest['samples'] if s['number'] == 1)
    assert sample['key'] == 'libero_spatial_no_noops_s007_r022' and sample['frames'] == 97
    raw_root = Path(manifest['raw_root'])
    expected = next(f['sha256'] for f in manifest['files'] if f['path'] == sample['shard'])
    assert digest(raw_root/sample['shard']) == expected
    _, data = next(load_records(raw_root, [sample]))
    joints = np.asarray(data['steps/observation/joint_state'], np.float32).reshape(97, 7)
    state = np.asarray(data['steps/observation/state'], np.float32).reshape(97, 8)
    task = 'pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate'
    bddl = args.libero_root/'libero/libero/bddl_files/libero_spatial'/f'{task}.bddl'
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_names=['agentview'],
                             camera_heights=256, camera_widths=256, render_gpu_device_id=0)
    env.seed(0)
    env.reset()
    robot, sim = env.robots[0], env.sim
    model = sim.model
    geom_names = list(robot.gripper.visual_geoms)
    geom_ids = [model.geom_name2id(name) for name in geom_names]
    original_groups = model.geom_group.copy()
    camera_id = model.camera_name2id('agentview')
    camera = {'name': 'agentview', 'resolution': [256, 256],
              'position_model': model.cam_pos[camera_id].tolist(),
              'quaternion_wxyz_model': model.cam_quat[camera_id].tolist(),
              'fovy_degrees': float(model.cam_fovy[camera_id]),
              'world_position': sim.data.cam_xpos[camera_id].tolist(),
              'world_rotation': sim.data.cam_xmat[camera_id].reshape(3, 3).tolist()}
    # Keep the original model, poses and camera, but render only selected visual geoms.
    # This is a separate render pass and deliberately removes ALL external occluders.
    model.geom_group[:] = 5
    model.geom_group[geom_ids] = 1
    context = sim._render_context_offscreen
    context.vopt.geomgroup[:] = 0
    context.vopt.geomgroup[1] = 1
    context.vopt.sitegroup[:] = 0

    def to_rlds(image):
        # robosuite observable convention, then OpenVLA's documented 180-degree rotation.
        convention = 1 if macros.IMAGE_CONVENTION == 'opengl' else -1
        return image[::convention][::-1, ::-1].copy()

    rows, source_images, overlays, rendered_images = [], [], [], []
    for frame in FRAMES:
        sim.data.qpos[robot._ref_joint_pos_indexes] = joints[frame]
        sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = state[frame, 6:]
        sim.data.qvel[:] = 0
        sim.forward()
        xyz = sim.data.site_xpos[robot.eef_site_id].copy()
        # robosuite eef_pos uses grip_site, while eef_quat uses robot_model.eef_name.
        # These frames differ by the gripper mount rotation; do not compare them directly.
        rotation = sim.data.get_body_xmat(robot.robot_model.eef_name).reshape(3, 3).copy()
        site_rotation = sim.data.site_xmat[robot.eef_site_id].reshape(3, 3).copy()
        target_rotation = T.quat2mat(T.axisangle2quat(state[frame, 3:6]))
        xyz_error = float(np.linalg.norm(xyz-state[frame, :3]))
        angle_error = float(Rotation.from_matrix(rotation@target_rotation.T).magnitude())
        site_body_angle = float(Rotation.from_matrix(site_rotation@rotation.T).magnitude())
        seg = to_rlds(sim.render(width=256, height=256, camera_name='agentview', segmentation=True))
        foreground = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
        ids_present = np.unique(seg[..., 1][foreground]).tolist()
        assert set(ids_present).issubset(geom_ids), ids_present
        mask = foreground & np.isin(seg[..., 1], geom_ids)
        box = bbox(mask)
        isolated = to_rlds(sim.render(width=256, height=256, camera_name='agentview'))
        rgb = np.asarray(Image.open(BytesIO(data['steps/observation/image'][frame])).convert('RGB'))
        saved_input = ROOT/'check/test/keyframes/case01'/f'gripper_frame_{frame:03d}.png'
        assert np.array_equal(rgb, np.asarray(Image.open(saved_input)))
        preview = overlay(rgb, Region(box, mask), BLUE)
        Image.fromarray(mask.astype(np.uint8)*255).save(out/f'frame_{frame:03d}_mask.png')
        Image.fromarray(isolated).save(out/f'frame_{frame:03d}_isolated.png')
        preview.save(out/f'frame_{frame:03d}_overlay.png')
        source_images.append(Image.fromarray(rgb)); overlays.append(preview)
        rendered_images.append(Image.fromarray(isolated))
        row = {'frame_index': frame, 'joint_state': joints[frame].tolist(),
               'gripper_qpos': state[frame, 6:].tolist(), 'eef_xyz_source': state[frame, :3].tolist(),
               'eef_xyz_reconstructed': xyz.tolist(), 'eef_position_error_m': xyz_error,
               'eef_rotation_error_rad': angle_error, 'site_to_eef_body_rotation_rad': site_body_angle,
               'bbox': box, 'mask_pixels': int(mask.sum()),
               'geom_ids_present': ids_present, 'manual_review': 'pending'}
        rows.append(row)
        print(json.dumps(row), flush=True)
    model.geom_group[:] = original_groups
    env.close()

    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 21)
    board = Image.new('RGB', (1152, 1392), '#171b22')
    draw = ImageDraw.Draw(board)
    draw.text((12, 10), 'Case01 | native isolated gripper | external occlusion ignored | review pending', font=font, fill='white')
    for column, title in enumerate(['SOURCE MAIN RGB', 'NATIVE MASK + BBOX', 'ISOLATED GEOMETRY RGB']):
        draw.text((column*384+12, 46), title, font=font, fill='white')
    for row_index, frame in enumerate(FRAMES):
        y = 82+row_index*434
        for column, image in enumerate([source_images[row_index], overlays[row_index], rendered_images[row_index]]):
            board.paste(image.resize((384,384)), (column*384, y+30))
        draw.text((12, y), f'Frame {frame} | bbox {rows[row_index]["bbox"]}', font=font, fill='#58adf5')
    board.save(out/'overview.png')
    report = {'source_manifest': str(manifest_path), 'source_manifest_sha256': digest(manifest_path),
              'source_key': sample['key'], 'source_shard_sha256': expected, 'frames': list(FRAMES),
              'source_state_layout': 'EEF xyz / axis-angle (6) + gripper joint qpos (2)',
              'eef_position_reference': robot.gripper.important_sites['grip_site'],
              'eef_rotation_reference': robot.robot_model.eef_name,
              'libero_root': str(args.libero_root),
              'libero_commit': subprocess.check_output(['git','-C',str(args.libero_root),'rev-parse','HEAD'], text=True).strip(),
              'packages': {p:version(p) for p in ['mujoco','robosuite','numpy','scipy','pillow','bddl']},
              'script_sha256': digest(Path(__file__)), 'bddl_sha256': digest(bddl),
              'geom_names': geom_names, 'geom_ids': geom_ids, 'original_geom_groups': original_groups[geom_ids].tolist(),
              'camera': camera, 'robosuite_image_convention': macros.IMAGE_CONVENTION,
              'image_transform': 'robosuite observable convention, then 180-degree rotation for OpenVLA RLDS',
              'mask_semantics': 'isolated palm and fingers, ignoring external occlusion; clipped at image bounds',
              'scene_object_states_recovered': False, 'manual_calibration_applied': False,
              'results': rows, 'checks': {'all_nonempty': all(r['bbox'] for r in rows),
                  'eef_position_within_0.1mm': all(r['eef_position_error_m']<1e-4 for r in rows),
                  'eef_rotation_within_0.0001rad': all(r['eef_rotation_error_rad']<1e-4 for r in rows),
                  'rendered_geom_ids_only_selected_gripper': True, 'source_rgb_matches_previous_keyframes': True},
              'elapsed_seconds': round(time.monotonic()-started, 3), 'manual_review':'pending'}
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['checks']), flush=True)


if __name__ == '__main__':
    main()
