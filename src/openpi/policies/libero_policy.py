"""LIBERO source-state and simulator-action adapters."""
import numpy as np


class LiberoWirePolicy:
    """Map live simulator fields to the checkpoint's training input adapter."""
    def __init__(self, policy, config, *, steps):
        recipe = (config['model_type'], config['data_format'])
        if config.get('platform', 'libero') != 'libero' or recipe not in (
                ('pi05', 'libero_sft'), ('eskill', 'episodes')):
            raise ValueError('Local LIBERO service supports pi05/libero_sft or eskill/episodes checkpoints')
        self.policy, self.steps = policy, steps
        eskill = config['model_type'] == 'eskill'
        self.requires_joint_state = eskill and config.get('model', {}).get('use_arm_state', True)
        self.cameras = [config.get('camera', 'image')] if eskill else ['image']
        if not eskill or config.get('sft_data_index'):
            self.cameras.append('wrist_image')
        if any(camera not in ('image', 'wrist_image') for camera in self.cameras):
            raise ValueError('LIBERO service cameras must be image or wrist_image')

    def infer(self, observation):
        state = np.asarray(observation['observation/state'], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('LIBERO wire state must contain 8 finite EEF/gripper values')
        if self.requires_joint_state:
            joints = np.asarray(observation.get('observation/joint_state'), np.float32)
            if joints.shape != (7,) or not np.isfinite(joints).all():
                raise ValueError('ESkill requires 7 finite, observed robot0_joint_pos values in observation/joint_state')
            state = np.concatenate((joints, state))
        data = dict(image=observation['observation/' + self.cameras[0]],
                    state=state, prompt=observation['prompt'])
        if len(self.cameras) == 2:
            data['wrist_image'] = observation['observation/' + self.cameras[1]]
        return self.policy.infer(data, steps=self.steps)


class LiberoSFTInputs:
    """Standard LIBERO views and source state; prompt/state use model config."""
    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (8,):
            raise ValueError("LIBERO SFT expects the original 8-dimensional state")
        return dict(images=[data["image"], data["wrist_image"]], image_mask=np.ones(2, bool),
                    state=state, state_valid=np.asarray(data["state_valid"], bool), prompt=data["prompt"])


class LiberoInputs:
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
        images = [data["image"]]
        if "wrist_image" in data:
            images.append(data["wrist_image"])
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
