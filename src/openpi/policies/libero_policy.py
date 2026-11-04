"""LIBERO source-state and simulator-action adapters."""
import numpy as np


LIBERO_CAMERAS = ("image", "wrist_image")


def uses_arm_state(model_type, model):
    """Only ESkill has a body-state input; π0.5 sees the 8 source values alone."""
    settings = model if isinstance(model, dict) else vars(model)
    return model_type == "eskill" and settings.get("use_arm_state", True)


class LiberoWirePolicy:
    """Map live simulator fields to the checkpoint's training input adapter."""
    def __init__(self, policy, config, *, steps):
        if config.get('platform', 'libero') != 'libero' or config['model_type'] not in ('pi05', 'eskill'):
            raise ValueError('Local LIBERO service supports pi05 or eskill checkpoints')
        self.policy, self.steps = policy, steps
        self.cameras = list(config['cameras'])
        if not self.cameras or any(camera not in LIBERO_CAMERAS for camera in self.cameras):
            raise ValueError(f'LIBERO service cameras must be chosen from {LIBERO_CAMERAS}')
        self.requires_joint_state = uses_arm_state(config['model_type'], config.get('model', {}))

    def infer(self, observation):
        state = np.asarray(observation['observation/state'], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('LIBERO wire state must contain 8 finite EEF/gripper values')
        if self.requires_joint_state:
            joints = np.asarray(observation.get('observation/joint_state'), np.float32)
            if joints.shape != (7,) or not np.isfinite(joints).all():
                raise ValueError('Arm-state checkpoints require 7 finite, observed robot0_joint_pos values '
                                 'in observation/joint_state')
            state = np.concatenate((joints, state))
        data = dict(images=[observation['observation/' + camera] for camera in self.cameras],
                    state=state, prompt=observation['prompt'])
        return self.policy.infer(data, steps=self.steps)


class LiberoInputs:
    """Saved state is joint_state[7] + source state[8]; the model sees source state[8].

    With `use_arm_state`, the full 15 values also enter the model as the single arm's body state.
    """
    def __init__(self, *, use_arm_state=True):
        self.use_arm_state = use_arm_state

    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (15,) and (self.use_arm_state or state.shape != (8,)):
            raise ValueError("Saved LIBERO state must be joint_state[7] + source state[8].")
        valid = np.asarray(data["state_valid"], bool)
        body_inputs = {}
        if self.use_arm_state:
            body = np.zeros((2, 15), np.float32)
            body_valid = np.zeros((2, 15), bool)
            body[0], body_valid[0] = state, valid
            body_inputs = dict(arm_state=body, arm_state_valid=body_valid)
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
