"""LIBERO source-state and simulator-action adapters."""
import numpy as np


LIBERO_CAMERAS = ("image", "wrist_image")


def camera_inputs(images):
    images = list(images)
    if not images or images[0] is None:
        raise ValueError('A valid main camera is required')
    valid = np.asarray([image is not None for image in images])
    return dict(images=[np.zeros_like(images[0]) if image is None else image for image in images],
                image_mask=valid)


class LiberoWirePolicy:
    """Map live simulator fields to the checkpoint's training input adapter."""
    def __init__(self, policy, config, *, steps):
        if config.get('platform', 'libero') != 'libero' or config['model_type'] not in ('pi05', 'eskill'):
            raise ValueError('Local LIBERO service supports pi05 or eskill checkpoints')
        self.policy, self.steps = policy, steps
        self.cameras = list(config['cameras'])
        if not self.cameras or any(camera not in LIBERO_CAMERAS for camera in self.cameras):
            raise ValueError(f'LIBERO service cameras must be chosen from {LIBERO_CAMERAS}')
        self.requires_state = (config['model_type'] != 'eskill'
                               and config.get('model', {}).get('discrete_state_input', True))
        self.requires_camera_calibration = False
        self.image_resize_on_server = config['model_type'] == 'eskill'

    def infer(self, observation):
        data = dict(images=[observation.get('observation/' + camera) for camera in self.cameras],
                    prompt=observation['prompt'])
        if self.requires_state:
            state = np.asarray(observation['observation/state'], np.float32)
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError('LIBERO wire state must contain 8 finite EEF/gripper values')
            data['state'] = state
        return self.policy.infer(data, steps=self.steps)


class LiberoInputs:
    """Current RGB/language and static arm identity; optional state tokens for pi05."""
    def __init__(self, *, include_state=True):
        self.include_state = include_state

    def __call__(self, data):
        images = list(data["images"])
        result = dict(**camera_inputs(images), prompt=data["prompt"],
                      arm_presence=np.array([True, False]))
        if self.include_state:
            state = np.asarray(data["state"], np.float32)
            if state.shape not in ((15,), (8,)):
                raise ValueError("LIBERO state must contain 8 source values, optionally preceded by 7 saved joint values.")
            valid = np.asarray(data.get("state_valid", np.ones_like(state, dtype=bool)), bool)
            result.update(state=state[-8:], state_valid=valid[-8:])
        return result


class LiberoOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        # Source OSC commands are passed through; do not apply a second delta transform.
        return self.normalize.apply("actions", np.asarray(actions)[..., :7], inverse=True)
