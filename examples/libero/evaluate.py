"""Shared OpenPi client loop for the pinned LIBERO and LIBERO-Plus benchmarks."""
import argparse
from collections import deque
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import time

import numpy as np

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
MAX_STEPS = dict(libero_spatial=220, libero_object=280, libero_goal=300, libero_10=520)
CATEGORIES = ('Objects Layout', 'Camera Viewpoints', 'Robot Initial States', 'Language Instructions',
              'Light Conditions', 'Background Textures', 'Sensor Noise')
WORKSPACE = Path(__file__).resolve().parents[3]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def task_ids(value, count):
    if value is None:
        return list(range(count))
    selected = set()
    for part in value.split(','):
        bounds = part.split('-')
        if len(bounds) == 1:
            selected.add(int(part))
        elif len(bounds) == 2:
            start, stop = map(int, bounds)
            if stop < start:
                raise ValueError('Task ID range must be increasing')
            selected.update(range(start, stop+1))
        else:
            raise ValueError('Use comma-separated IDs or inclusive ranges')
    if not selected or min(selected) < 0 or max(selected) >= count:
        raise ValueError('Task ID is outside the selected suite')
    return sorted(selected)


def select_tasks(tasks, ids=None, classification=None, category=None, difficulty=None):
    """Join Plus metadata by task name; its published IDs are one-based."""
    annotations = {}
    if classification is not None:
        annotations = {row['name']: row for row in classification}
        if len(annotations) != len(classification):
            raise ValueError('Duplicate task names in the benchmark classification')
    selected = []
    for identity in task_ids(ids, len(tasks)):
        task = tasks[identity]
        if classification is not None and task.name not in annotations:
            raise ValueError('Task is missing official perturbation metadata: '+task.name)
        details = annotations.get(task.name, {})
        if category is not None and details.get('category') != category:
            continue
        if difficulty is not None and details.get('difficulty_level') != difficulty:
            continue
        selected.append(dict(task_id=identity, task=task, variant_id=details.get('id'),
                             category=details.get('category'), difficulty=details.get('difficulty_level')))
    if not selected:
        raise ValueError('The selected suite has no tasks matching these filters')
    return selected


def quaternion_to_axisangle(quaternion):
    q = np.asarray(quaternion, dtype=float)
    w = np.clip(q[3], -1.0, 1.0)
    denominator = math.sqrt(1.0-w*w)
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return q[:3]*2.0*math.acos(w)/denominator


def state_vector(obs):
    state = np.concatenate((obs['robot0_eef_pos'], quaternion_to_axisangle(obs['robot0_eef_quat']),
                            obs['robot0_gripper_qpos'])).astype(np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise ValueError('LIBERO needs EEF xyz + axis-angle + two finger positions (8 values)')
    return state


def camera_image(obs, name='agentview_image'):
    # Matches OpenPi's LIBERO training/evaluation image convention.
    return np.ascontiguousarray(obs[name][::-1, ::-1])


def main_camera_parameters(env, image_shape, *, image_convention=None):
    """Live MuJoCo camera in the same pixel coordinates as camera_image()."""
    if image_convention is None:
        from robosuite import macros
        image_convention = macros.IMAGE_CONVENTION
    height, width = image_shape[:2]
    sim = env.sim
    camera = sim.model.camera_name2id('agentview')
    fovy = float(sim.model.cam_fovy[camera])
    if not 0 < fovy < 180:
        raise ValueError('Main camera fovy must be between 0 and 180 degrees')
    f = .5 * height / np.tan(np.deg2rad(fovy) / 2)
    intrinsic = np.array([[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1]])
    camera_to_world = np.eye(4)
    camera_to_world[:3, :3] = np.asarray(sim.data.cam_xmat[camera]).reshape(3, 3) @ np.diag([1, -1, -1])
    camera_to_world[:3, 3] = sim.data.cam_xpos[camera]
    # Robosuite's observable convention, followed by OpenVLA's 180-degree rotation.
    transform = np.array([[-1, 0, width-1], [0, 1, 0], [0, 0, 1]])
    if image_convention == 'opencv':
        transform[1] = [0, -1, height-1]
    elif image_convention != 'opengl':
        raise ValueError('Unknown robosuite image convention: '+str(image_convention))
    return dict(intrinsic=intrinsic, world_to_camera=np.linalg.inv(camera_to_world),
                image_transform=transform, image_shape=np.array([height, width], np.int64))


def policy_input(obs, instruction, resize_size=224, *, metadata=None, env=None):
    from openpi_client import image_tools
    metadata = metadata or {}
    def image(name):
        if metadata.get('image_resize_on_server', False):
            return camera_image(obs, name)
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(
            camera_image(obs, name), resize_size, resize_size))
    camera_names = dict(image='agentview_image', wrist_image='robot0_eye_in_hand_image')
    result = {'observation/' + camera: image(camera_names[camera])
              for camera in metadata.get('cameras', ['image', 'wrist_image'])}
    result['prompt'] = str(instruction)
    if metadata.get('requires_state', True):
        result['observation/state'] = state_vector(obs)
    if metadata.get('requires_camera_calibration', False):
        if env is None:
            raise ValueError('ESkill camera calibration requires the live simulator')
        if not metadata.get('image_resize_on_server', False):
            raise ValueError('Calibrated RGB must be resized with its projection on the policy server')
        result['observation/main_camera'] = main_camera_parameters(env, obs['agentview_image'].shape)
    return result


def action_chunk(response, replan_steps):
    actions = np.asarray(response['actions'])
    if (actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < replan_steps
            or not np.isfinite(actions).all()):
        raise ValueError('Policy must return finite [horizon,7] LIBERO actions; horizon >= replan_steps')
    return actions


def video_frame(obs, chunk, step, highlight):
    from PIL import Image, ImageDraw
    image = Image.fromarray(camera_image(obs)).copy()
    draw = ImageDraw.Draw(image)
    color = '#74e4bd' if highlight else '#ffffff'
    draw.rectangle((0, 0, image.width, 19), fill='#121c25')
    draw.text((5, 4), 'chunk %04d | step %04d' % (chunk, step), fill=color)
    if highlight:
        draw.rectangle((0, 0, image.width-1, image.height-1), outline=color, width=2)
    return np.asarray(image)


def rollout(env, client, initial_state, instruction, *, max_steps, wait_steps, replan_steps,
            resize_size=224, capture_video=True, policy_metadata=None):
    """A single fixed initial state. Each prediction is requested once per action chunk."""
    env.reset()
    obs = env.set_init_state(initial_state)
    client.reset()
    for _ in range(wait_steps):
        obs, _, _, _ = env.step([0.0]*6+[-1.0])
    queue = deque()
    executed, states, chunk_ids, predicted, requests, images = [], [], [], [], [], []
    recording = (policy_metadata or {}).get('prediction_recording')
    diagnostic_images = [camera_image(obs).copy()] if recording else []
    success, error = False, None
    chunk = -1
    within_chunk = 0
    try:
        for step in range(max_steps):
            if not queue:
                start = time.monotonic()
                response = client.infer(policy_input(obs, instruction, resize_size,
                                                    metadata=policy_metadata, env=env))
                values = action_chunk(response, replan_steps)
                requests.append(dict(step=step, latency_ms=(time.monotonic()-start)*1000,
                                     horizon=len(values)))
                if recording:
                    requests[-1]['diagnostic_id'] = str(response['diagnostic_id'])
                predicted.append(values.copy())
                queue.extend(values[:replan_steps])
                chunk += 1
                within_chunk = 0
            action = queue.popleft()
            state = state_vector(obs)
            obs, _, done, _ = env.step(action.tolist())
            states.append(state)
            executed.append(action.copy())
            chunk_ids.append(chunk)
            if recording:
                diagnostic_images.append(camera_image(obs).copy())
            if capture_video:
                images.append(video_frame(obs, chunk, len(executed), within_chunk < 6))
            within_chunk += 1
            if done:
                # LIBERO's sparse done is the environment's task-success predicate.
                success = bool(env.check_success())
                break
    except Exception as exception:
        logging.exception('Rollout failed')
        error = repr(exception)
    return dict(success=success, error=error, steps=len(executed), inference_requests=requests,
                actions=np.asarray(executed, dtype=np.float32).reshape(-1,7),
                states=np.asarray(states, dtype=np.float32).reshape(-1,8),
                chunk_ids=np.asarray(chunk_ids, dtype=np.int32), predictions=predicted, images=images,
                diagnostic_images=diagnostic_images)


def summarize(records, technical=False):
    groups = {}
    for field in ('suite', 'category'):
        groups[field] = {}
        for name in sorted({r[field] for r in records if r.get(field) is not None}):
            rows = [r for r in records if r.get(field) == name]
            groups[field][name] = dict(episodes=len(rows), successes=sum(r['success'] for r in rows),
                errors=sum(r['error'] is not None for r in rows),
                success_rate=None if technical else sum(r['success'] for r in rows)/len(rows))
    count = len(records)
    return dict(episodes=count, successes=sum(r['success'] for r in records),
        errors=sum(r['error'] is not None for r in records),
        success_rate=None if technical or count == 0 else sum(r['success'] for r in records)/count,
        averaging='per-episode micro average; each group reports its own denominator',
        by_suite=groups['suite'], by_category=groups['category'],
        purpose='integration_check' if technical else 'policy_evaluation')


def parser(benchmark):
    p = argparse.ArgumentParser(description='Evaluate a fine-tuned OpenPi LIBERO policy in '+benchmark)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--task-suite-name', choices=('all',)+SUITES, default='libero_spatial')
    p.add_argument('--task-ids', help='Zero-based IDs, e.g. 0,2,5-8; applied within each suite')
    p.add_argument('--num-trials-per-task', type=int, default=1 if benchmark == 'libero_plus' else 50)
    p.add_argument('--category', choices=CATEGORIES)
    p.add_argument('--difficulty', type=int)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--replan-steps', type=int, default=5)
    p.add_argument('--num-steps-wait', type=int, default=10)
    p.add_argument('--resize-size', type=int, default=224)
    p.add_argument('--max-steps', type=int, help='Override the pinned OpenPi suite horizon')
    p.add_argument('--render-gpu-device-id', type=int, default=0)
    p.add_argument('--video', choices=('all', 'failures', 'none'), default='all')
    p.add_argument('--output', type=Path, help='A new run directory, required for evaluation')
    p.add_argument('--checkpoint-tag', default='', help='Checkpoint identity to store with the run')
    p.add_argument('--libero-config', type=Path,
                   default=WORKSPACE/'workspace/envs/libero-evaluation'/benchmark,
                   help='Directory containing this benchmark installation\'s config.yaml')
    p.add_argument('--list-tasks', action='store_true', help='List selected task IDs and metadata without inference')
    p.add_argument('--technical', action='store_true', help='Connectivity check; suppress success-rate claims')
    return p


def main(benchmark='libero'):
    args = parser(benchmark).parse_args()
    if benchmark == 'libero' and (args.category is not None or args.difficulty is not None):
        raise ValueError('Perturbation filters require the LIBERO-Plus client')
    if args.replan_steps < 1 or args.num_trials_per_task < 1 or args.num_steps_wait < 0:
        raise ValueError('Invalid episode or action-window settings')
    os.environ['LIBERO_CONFIG_PATH'] = str(args.libero_config.resolve())
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    vendor = WORKSPACE/'workspace/envs/libero-evaluation/nvidia_egl.json'
    if os.environ['MUJOCO_GL'] == 'egl' and vendor.is_file():
        os.environ.setdefault('__EGL_VENDOR_LIBRARY_FILENAMES', str(vendor))
    os.environ.setdefault('NUMBA_CACHE_DIR', str(Path(tempfile.gettempdir())/'eskill-libero-eval-numba'))
    os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'eskill-libero-eval-mpl'))
    import libero.libero as libero_module
    from libero.libero import benchmark as benchmarks, get_libero_path
    classification = None
    classification_path = Path(get_libero_path('benchmark_root'))/'benchmark/task_classification.json' if benchmark == 'libero_plus' else None
    if classification_path is not None:
        classification = json.loads(classification_path.read_text())
    suite_names = SUITES if args.task_suite_name == 'all' else (args.task_suite_name,)
    random.seed(args.seed)
    np.random.seed(args.seed)
    selections = []
    for suite_name in suite_names:
        suite = benchmarks.get_benchmark_dict()[suite_name](task_order_index=0)
        tasks = [suite.get_task(i) for i in range(suite.n_tasks)]
        selected = select_tasks(tasks, args.task_ids, classification[suite_name] if classification is not None else None,
                                args.category, args.difficulty)
        selections.append((suite_name, suite, selected))
    if args.list_tasks:
        for suite_name, _, selected in selections:
            for item in selected:
                print(json.dumps({**{k:v for k,v in item.items() if k != 'task'},
                                  'suite':suite_name, 'name':item['task'].name,
                                  'instruction':item['task'].language}, ensure_ascii=False))
        return
    if args.output is None:
        raise ValueError('--output is required for a policy evaluation')
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import imageio
    args.output.mkdir(parents=True, exist_ok=False)
    client = WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    technical = args.technical or bool(metadata.get('technical_only', False))
    source = Path(libero_module.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(benchmark=benchmark, purpose='integration_check' if technical else 'policy_evaluation',
        environment_source=str(source), revision=revision, server_metadata=metadata,
        max_steps_by_suite=MAX_STEPS, control_frequency_hz=20, video_fps=20,
        state='eef_xyz + axis_angle + two finger positions',
        policy_state='eef_xyz + axis_angle + two finger positions', action='LIBERO OSC_POSE delta[6] + gripper',
        image_convention=('rotate both cameras by 180 degrees, then checkpoint resize/pad on server'
                          if metadata.get('image_resize_on_server', False) else
                          'rotate both cameras by 180 degrees, then OpenPi resize_with_pad'),
        main_camera_source='live simulator per request' if metadata.get('requires_camera_calibration') else None,
        initial_state_selection='published initial_states[trial], no modulo or random substitution')
    if classification_path is not None:
        config['classification_sha256'] = hashlib.sha256(classification_path.read_bytes()).hexdigest()
    write_json(args.output/'config.json', config)
    records = []
    for suite_name, suite, selected in selections:
        for item in selected:
            task = item['task']
            initial_states = suite.get_task_init_states(item['task_id'])
            if args.num_trials_per_task > len(initial_states):
                raise ValueError('Requested more trials than the published initial states')
            env = None
            try:
                env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
                    camera_heights=256, camera_widths=256, render_gpu_device_id=args.render_gpu_device_id)
                env.seed(args.seed)
                for trial in range(args.num_trials_per_task):
                    name = '%s_t%04d_r%03d' % (suite_name, item['task_id'], trial)
                    started = time.monotonic()
                    try:
                        result = rollout(env, client, initial_states[trial], task.language,
                            max_steps=args.max_steps if args.max_steps is not None else MAX_STEPS[suite_name],
                            wait_steps=args.num_steps_wait, replan_steps=args.replan_steps,
                            resize_size=args.resize_size, capture_video=args.video != 'none', policy_metadata=metadata)
                    except Exception as error:
                        logging.exception('Environment reset failed')
                        result = dict(success=False, error=repr(error), steps=0, inference_requests=[],
                                      images=[], predictions=[], actions=np.empty((0,7)), states=np.empty((0,8)),
                                      chunk_ids=np.empty(0,dtype=np.int32))
                    status = 'error' if result['error'] else 'success' if result['success'] else 'failure'
                    record = {k:v for k,v in item.items() if k != 'task'}
                    record.update(suite=suite_name, task_name=task.name, instruction=task.language,
                        trial=trial, initial_state_id=trial, seed=args.seed, success=result['success'],
                        error=result['error'], steps=result['steps'], seconds=time.monotonic()-started,
                        inference_requests=result['inference_requests'], purpose=config['purpose'], video=None)
                    if metadata.get('prediction_recording'):
                        record.update(prediction_recording=metadata['prediction_recording'], control_fps=20,
                                      replan_steps=args.replan_steps,
                                      diagnostic_frame_alignment='pre-control observations plus final observation')
                    record_path = args.output/'records'/(name+'.json')
                    record_path.parent.mkdir(parents=True, exist_ok=True)
                    arrays = dict(actions=result['actions'], states=result['states'], chunk_ids=result['chunk_ids'])
                    if result.get('diagnostic_images'):
                        arrays['diagnostic_images'] = np.asarray(result['diagnostic_images'])
                    arrays.update({'prediction_%04d'%i:values for i,values in enumerate(result['predictions'])})
                    np.savez_compressed(record_path.with_suffix('.npz'), **arrays)
                    if result['images'] and (args.video == 'all' or args.video == 'failures' and not result['success']):
                        video = args.output/'videos'/status/(name+'.mp4')
                        video.parent.mkdir(parents=True, exist_ok=True)
                        imageio.mimwrite(str(video), result['images'], fps=20, macro_block_size=1)
                        record['video'] = str(video.relative_to(args.output))
                    write_json(record_path, record)
                    records.append(record)
                    with (args.output/'episodes.jsonl').open('a') as output:
                        output.write(json.dumps(record, ensure_ascii=False)+'\n')
                    write_json(args.output/'summary.json', summarize(records, technical))
                    print(json.dumps(dict(suite=suite_name, task_id=item['task_id'], trial=trial,
                                          success=record['success'], steps=record['steps'], error=record['error'])), flush=True)
            finally:
                if env is not None:
                    env.close()
    client._ws.close()
    print(json.dumps(summarize(records, technical), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
