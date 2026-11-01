"""Evaluation contracts: task identity, online-only observations, and action-block execution."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero import evaluate as evaluation
from scripts.serve_policy import command
from scripts import serve_policy


def observation(step=0):
    image = np.arange(6*6*3, dtype=np.uint8).reshape(6,6,3)
    return dict(agentview_image=image, robot0_eye_in_hand_image=image.copy(),
                robot0_eef_pos=np.array([step,0,1.0]), robot0_eef_quat=np.array([0,0,0,1.0]),
                robot0_gripper_qpos=np.array([0.04,0.04]), gt_future_bbox='must not be sent')


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
