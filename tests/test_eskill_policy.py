"""ESkill online geometry and skill outputs use current observations only."""
from types import SimpleNamespace

import numpy as np
import pytest

from examples.libero import evaluate
from openpi.transforms import project_points, resize_points


def camera_env(position=(0, 0, 0)):
    model = SimpleNamespace(camera_name2id=lambda name: 0, cam_fovy=np.array([90.]))
    data = SimpleNamespace(cam_xmat=np.eye(3).reshape(1, 9),
                           cam_xpos=np.asarray(position, float)[None])
    return SimpleNamespace(sim=SimpleNamespace(model=model, data=data))


def observation():
    image = np.arange(4 * 8 * 3, dtype=np.uint8).reshape(4, 8, 3)
    return dict(agentview_image=image, robot0_eye_in_hand_image=image.copy(),
                robot0_eef_pos=np.array([.2, .2, -2.]), robot0_eef_quat=np.array([0, 0, 0, 1.]),
                robot0_gripper_qpos=np.array([.04, .04]), future_bbox='not an online input')


@pytest.mark.parametrize('convention,y_sign', [('opengl', -1), ('opencv', 1)])
def test_live_camera_projection_matches_rotated_source_pixels(convention, y_sign):
    env = camera_env()
    calibration = evaluate.main_camera_parameters(env, (4, 8), image_convention=convention)
    points, valid = project_points([[0, 0, -2], [.2, .2, -2]], **calibration)
    np.testing.assert_allclose(points[0], [3.5, 1.5])
    np.testing.assert_allclose(points[1], [3.3, 1.5 + y_sign * .2])
    assert valid.all()
    normalized, valid = resize_points(points, valid, (4, 8), 224)
    np.testing.assert_allclose(normalized[0], [.5, .5])
    # Camera perturbations are read again from this state's actual MuJoCo camera.
    env.sim.data.cam_xpos[0, 0] = .5
    changed = evaluate.main_camera_parameters(env, (4, 8), image_convention=convention)
    other, _ = project_points([[0, 0, -2]], **changed)
    assert other[0, 0] > points[0, 0]


def test_eskill_wire_payload_preserves_source_resolution_and_current_calibration(monkeypatch):
    original = evaluate.main_camera_parameters
    monkeypatch.setattr(evaluate, 'main_camera_parameters',
                        lambda env, shape: original(env, shape, image_convention='opengl'))
    metadata = dict(cameras=['image', 'wrist_image'], requires_camera_calibration=True,
                    image_resize_on_server=True)
    obs = observation()
    payload = evaluate.policy_input(obs, 'place the cup', metadata=metadata, env=camera_env())
    assert set(payload) == {'observation/image', 'observation/wrist_image', 'observation/state',
                            'observation/main_camera', 'prompt'}
    np.testing.assert_array_equal(payload['observation/image'], obs['agentview_image'][::-1, ::-1])
    assert payload['observation/image'].shape == (4, 8, 3)
    np.testing.assert_array_equal(payload['observation/main_camera']['image_shape'], [4, 8])
    np.testing.assert_allclose(payload['observation/state'][:3], obs['robot0_eef_pos'])
    with pytest.raises(ValueError, match='live simulator'):
        evaluate.policy_input(obs, 'task', metadata=metadata)
    with pytest.raises(ValueError, match='resized with its projection'):
        evaluate.policy_input(obs, 'task', metadata={**metadata, 'image_resize_on_server': False},
                              env=camera_env())


def test_policy_maps_active_route_ids():
    torch = pytest.importorskip('torch')
    from openpi.policies.policy import Policy

    class Model:
        config = SimpleNamespace(active_skill_ids=(0, 2, 5), skill_names=('Pick', 'Place', 'Open',
                                      'Close', 'Turn', 'Push', 'Hold', 'Idle', 'Return'))

        def eval(self):
            return self

        def sample_actions(self, inputs, *, noise, steps):
            return torch.zeros(1, 10, 7), {'skill_logits': torch.tensor([[[0., 1., 2.], [1., 0., 0.]]])}

    inputs = dict(arm_presence=np.array([True, False]))
    transforms = SimpleNamespace(inputs=lambda data: inputs)
    policy = Policy(Model(), lambda data: data, lambda action: action, transforms,
                    device='cpu')
    result = policy.infer(dict(state=np.zeros(8)))
    np.testing.assert_array_equal(result['current_skills'], [5, -1])
    assert result['actions'].shape == (10, 7)
    assert set(result) == {'actions', 'current_skills', 'skill_names'}
