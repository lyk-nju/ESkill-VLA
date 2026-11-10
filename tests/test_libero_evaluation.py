"""Evaluation contracts: task identity, online-only observations, and action-block execution."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero import evaluate as evaluation
from scripts.serve_policy import command
from scripts import serve_policy
from openpi.policies.libero_policy import LiberoWirePolicy


def observation(step=0):
    image = np.arange(6*6*3, dtype=np.uint8).reshape(6,6,3)
    return dict(agentview_image=image, robot0_eye_in_hand_image=image.copy(),
                robot0_eef_pos=np.array([step,0,1.0]), robot0_eef_quat=np.array([0,0,0,1.0]),
                robot0_joint_pos=np.arange(7, dtype=np.float32) / 10 + step,
                robot0_gripper_qpos=np.array([0.04,0.04]), gt_future_bbox='must not be sent')


@pytest.fixture
def camera_env(monkeypatch):
    model = SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=np.array([90.]))
    data = SimpleNamespace(cam_xmat=np.eye(3).reshape(1, 9), cam_xpos=np.zeros((1, 3)))
    env = SimpleNamespace(sim=SimpleNamespace(model=model, data=data))
    original = evaluation.main_camera_parameters
    monkeypatch.setattr(evaluation, 'main_camera_parameters',
                        lambda env, shape: original(env, shape, image_convention='opengl'))
    return env


def test_plus_selection_joins_names_and_preserves_official_identity():
    tasks = [SimpleNamespace(name='b', language='rewritten instruction'), SimpleNamespace(name='a')]
    classification = [dict(id=1,name='a',category='Sensor Noise',difficulty_level=2),
                      dict(id=2,name='b',category='Language Instructions',difficulty_level=1)]
    rows = evaluation.select_tasks(tasks, classification=classification, category='Language Instructions')
    assert rows[0]['task_id'] == 0 and rows[0]['variant_id'] == 2
    assert rows[0]['task'].language == 'rewritten instruction'
    with pytest.raises(ValueError, match='missing official'):
        evaluation.select_tasks(tasks, classification=classification[:1])


@pytest.mark.parametrize('text', ['-1','4','3-1','0-2-3'])
def test_bad_task_ids_are_not_silently_dropped(text):
    with pytest.raises(ValueError):
        evaluation.task_ids(text,4)


def test_task_ranges_and_protocol_defaults():
    assert evaluation.task_ids('0,2-3,2',4) == [0,2,3]
    assert evaluation.parser('libero').parse_args([]).num_trials_per_task == 50
    assert evaluation.parser('libero_plus').parse_args([]).num_trials_per_task == 1


def test_online_observation_schema_and_rotation_are_explicit():
    obs = observation()
    angle = np.pi/3
    obs['robot0_eef_quat'] = np.array([0,0,np.sin(angle/2),np.cos(angle/2)])
    value = evaluation.policy_input(obs,'original instruction',resize_size=6)
    assert set(value) == {'observation/image','observation/wrist_image','observation/state','prompt'}
    np.testing.assert_array_equal(value['observation/image'],obs['agentview_image'][::-1,::-1])
    np.testing.assert_allclose(value['observation/state'],[0,0,1,0,0,angle,.04,.04],rtol=1e-6)
    assert value['observation/state'].dtype == np.float32
    assert value['prompt'] == 'original instruction'


@pytest.mark.parametrize('model_type,cameras', [
    ('pi05', ['image', 'wrist_image']),
    ('eskill', ['image']),
    ('eskill', ['image', 'wrist_image']),
])
def test_local_wire_matches_training_state_and_cameras(model_type, cameras, camera_env):
    captured = {}
    def infer(data, *, steps):
        captured.update(data, steps=steps)
        return {'actions': np.zeros((10, 7), np.float32)}
    config = dict(platform='libero', model_type=model_type, cameras=cameras)
    wire = LiberoWirePolicy(SimpleNamespace(infer=infer), config, steps=3)
    metadata = dict(cameras=wire.cameras, requires_camera_calibration=wire.requires_camera_calibration,
                    image_resize_on_server=wire.requires_camera_calibration)
    obs = observation(2)
    payload = evaluation.policy_input(obs, 'task', resize_size=6, metadata=metadata, env=camera_env)
    assert wire.cameras == cameras
    assert ('observation/main_camera' in payload) == (model_type == 'eskill')
    assert payload['observation/state'].shape == (8,)
    assert 'gt_future_bbox' not in payload
    result = wire.infer(payload)
    expected = evaluation.state_vector(obs)
    np.testing.assert_array_equal(captured['state'], expected)
    np.testing.assert_array_equal(captured['images'][0], obs['agentview_image'][::-1, ::-1])
    assert len(captured['images']) == len(cameras)
    assert captured['steps'] == 3 and result['actions'].shape == (10, 7)


def test_eskill_requires_current_camera_calibration():
    obs = observation()
    metadata = dict(cameras=['image'], requires_camera_calibration=True, image_resize_on_server=True)
    with pytest.raises(ValueError, match='live simulator'):
        evaluation.policy_input(obs, 'task', metadata=metadata)
    wire = LiberoWirePolicy(None, dict(model_type='eskill', cameras=['image']), steps=2)
    payload = {'observation/state': evaluation.state_vector(obs), 'observation/image': obs['agentview_image'],
               'prompt': 'task'}
    with pytest.raises(KeyError, match='observation/main_camera'):
        wire.infer(payload)


@pytest.mark.parametrize('cameras', [['image'], ['image', 'wrist_image']])
def test_pi05_service_uses_current_source_state(cameras):
    from openpi.policies.libero_policy import LiberoInputs
    captured = {}
    def infer(data, *, steps):
        data['state_valid'] = np.ones_like(data['state'], bool)
        captured.update(LiberoInputs(use_arm_state=False)(data))
        return {'actions': np.zeros((10, 7), np.float32)}
    wire = LiberoWirePolicy(SimpleNamespace(infer=infer), dict(model_type='pi05', cameras=cameras), steps=2)
    assert not wire.requires_camera_calibration and wire.cameras == cameras
    obs = observation()
    payload = evaluation.policy_input(obs, 'task', metadata=dict(cameras=wire.cameras))
    assert set(payload) == {'observation/' + camera for camera in cameras} | {'observation/state', 'prompt'}
    wire.infer(payload)
    assert captured['state'].shape == (8,) and 'arm_state' not in captured
    np.testing.assert_array_equal(captured['arm_presence'], [True, False])


class FakeEnvironment:
    def reset(self):
        self.steps = 0

    def set_init_state(self, state):
        assert state == 'published state'
        return observation()

    def step(self, action):
        assert len(action)==7
        self.steps += 1
        return observation(self.steps),0,self.steps>=9,{}

    def check_success(self):
        return self.steps>=9


class FakePolicy:
    def __init__(self):
        self.calls = 0
        self.resets = 0

    def reset(self):
        self.resets += 1

    def infer(self, obs):
        self.calls += 1
        assert obs['observation/state'].shape == (8,)
        return {'actions':np.tile([0,0,0,0,0,0,-1.0],(10,1))}


def test_identical_predictions_execute_once_per_chunk_and_clear_each_episode():
    env,client = FakeEnvironment(),FakePolicy()
    result = evaluation.rollout(env,client,'published state','task',max_steps=20,
        wait_steps=2,replan_steps=3,capture_video=False)
    assert result['success'] and result['steps']==7 and result['error'] is None
    assert client.calls==3 and client.resets==1
    np.testing.assert_array_equal(result['chunk_ids'],[0,0,0,1,1,1,2])
    np.testing.assert_array_equal(result['states'][:,0],np.arange(2,9))
    assert [r['step'] for r in result['inference_requests']] == [0,3,6]


def test_rollout_passes_server_requirements_and_live_camera_updates(camera_env):
    class CameraEnvironment(FakeEnvironment):
        sim = camera_env.sim

        def step(self, action):
            result = super().step(action)
            self.sim.data.cam_xpos[0, 0] = self.steps
            return result

    class CameraPolicy(FakePolicy):
        def infer(self, obs):
            assert 'observation/wrist_image' not in obs
            assert obs['observation/main_camera']['world_to_camera'][0, 3] == -(2 + self.calls * 3)
            return super().infer(obs)
    result = evaluation.rollout(CameraEnvironment(), CameraPolicy(), 'published state', 'task', max_steps=20,
        wait_steps=2, replan_steps=3, capture_video=False,
        policy_metadata=dict(cameras=['image'], requires_camera_calibration=True, image_resize_on_server=True))
    assert result['success'] and result['error'] is None


def test_failed_env_step_is_not_counted_as_completed_control():
    class Broken(FakeEnvironment):
        def step(self, action):
            if self.steps==2:
                raise RuntimeError('render failed')
            return super().step(action)
    result = evaluation.rollout(Broken(),FakePolicy(),'published state','task',max_steps=4,
        wait_steps=2,replan_steps=3,capture_video=False)
    assert result['steps']==0 and result['error'] is not None
    assert result['actions'].shape==(0,7) and result['states'].shape==(0,8)


@pytest.mark.parametrize('values', [np.zeros((10,14)),np.zeros((2,7)),np.full((10,7),np.nan)])
def test_wrong_embodiment_or_invalid_actions_fail(values):
    with pytest.raises(ValueError):
        evaluation.action_chunk({'actions':values},5)


def test_summary_denominators_and_integration_mode():
    rows=[dict(suite='a',category='x',success=True,error=None),
          dict(suite='a',category='x',success=False,error='failed'),
          dict(suite='b',category='y',success=True,error=None)]
    summary=evaluation.summarize(rows)
    assert summary['success_rate']==pytest.approx(2/3)
    assert summary['errors']==1 and summary['by_suite']['a']['episodes']==2
    assert evaluation.summarize(rows,technical=True)['success_rate'] is None


def test_checkpoint_launcher_uses_the_requested_runtime_and_config(tmp_path):
    args=SimpleNamespace(openpi_root=tmp_path,config='my_pi05_libero',checkpoint='/a/checkpoint with spaces',port=8001)
    argv=command(args)
    assert argv[1]==str(tmp_path/'scripts/serve_policy.py')
    assert argv[-3:]==['my_pi05_libero','--policy.dir','/a/checkpoint with spaces']


def test_policy_process_uses_isolated_openpi_and_pinned_lerobot(monkeypatch, tmp_path):
    runtime, lerobot = tmp_path/'openpi', tmp_path/'lerobot'
    monkeypatch.setenv('PYTHONPATH', '/caller/ESkill-VLA/src')
    monkeypatch.setattr(serve_policy.sys, 'argv', ['serve_policy.py', '--checkpoint', '/ckpt',
        '--openpi-root', str(runtime), '--lerobot-root', str(lerobot)])
    captured = {}
    monkeypatch.setattr(serve_policy.os, 'chdir', lambda path: captured.update(cwd=path))
    monkeypatch.setattr(serve_policy.os, 'execve', lambda exe, argv, env: captured.update(argv=argv, env=env))
    serve_policy.main()
    assert captured['cwd'] == runtime
    assert captured['env']['PYTHONPATH'].split(serve_policy.os.pathsep) == [
        str(runtime/'src'), str(runtime/'packages/openpi-client/src'), str(lerobot)]
