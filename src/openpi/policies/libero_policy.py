"""LIBERO source-state and simulator-action adapters."""
import numpy as np


LIBERO_CAMERAS = ("image", "wrist_image")


class LiberoWirePolicy:
    """Map live simulator fields to the checkpoint's training input adapter."""
    def __init__(self, policy, config, *, steps):
        if config.get('platform', 'libero') != 'libero' or config['model_type'] not in ('pi05', 'eskill'):
            raise ValueError('Local LIBERO service supports pi05 or eskill checkpoints')
        self.policy, self.steps = policy, steps
        self.cameras = list(config['cameras'])
        if not self.cameras or any(camera not in LIBERO_CAMERAS for camera in self.cameras):
            raise ValueError(f'LIBERO service cameras must be chosen from {LIBERO_CAMERAS}')
        self.requires_camera_calibration = config['model_type'] == 'eskill'

    def infer(self, observation):
        state = np.asarray(observation['observation/state'], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('LIBERO wire state must contain 8 finite EEF/gripper values')
        data = dict(images=[observation['observation/' + camera] for camera in self.cameras],
                    state=state, prompt=observation['prompt'])
        if self.requires_camera_calibration:
            data['main_camera'] = observation['observation/main_camera']
        return self.policy.infer(data, steps=self.steps)


class LiberoInputs:
    """Map current source EEF pose and finger positions to one observed arm."""
    def __init__(self, *, use_arm_state=False):
        self.use_arm_state = use_arm_state

    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape not in ((15,), (8,)):
            raise ValueError("LIBERO state must contain 8 source values, optionally preceded by 7 saved joint values.")
        valid = np.asarray(data["state_valid"], bool)
        body_inputs = {}
        if self.use_arm_state:
            body = np.zeros((2, 8), np.float32)
            body_valid = np.zeros((2, 8), bool)
            body[0], body_valid[0] = state[-8:], valid[-8:]
            body_inputs = dict(arm_state=body, arm_state_valid=body_valid)
            if "effector_start" in data:
                body_inputs.update(effector_start=np.asarray(data["effector_start"], np.float32),
                                   effector_start_valid=np.asarray(data["effector_start_valid"], bool))
            else:
                # Calibration describes the actual RGB supplied below; transforms perform
                # the same resize/pad as offline calibrated EEF pixels.
                camera = data["main_camera"]
                from openpi.transforms import project_points
                point, point_valid = project_points(
                    state[-8:-5], camera["intrinsic"], camera["world_to_camera"],
                    camera["image_transform"], camera["image_shape"])
                body_inputs.update(effector_start_pixels=np.stack((point, np.zeros(2, np.float32))),
                                   effector_start_valid=np.array([point_valid, False]),
                                   effector_image_shape=np.asarray(camera["image_shape"], np.int64))
        images = list(data["images"])
        return dict(
            images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
            state=state[-8:], state_valid=valid[-8:],
            arm_presence=np.array([True, False]), **body_inputs,
        )


class LiberoOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        # Source OSC commands are passed through; do not apply a second delta transform.
        return self.normalize.apply("actions", np.asarray(actions)[..., :7], inverse=True)
