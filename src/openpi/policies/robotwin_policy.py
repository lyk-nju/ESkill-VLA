"""RoboTwin absolute joint-target adapter for unified Episodes."""
import numpy as np


class RobotwinInputs:
    def __init__(self, *, use_arm_state=True):
        self.use_arm_state = use_arm_state

    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (14,):
            raise ValueError("RoboTwin adapter expects [left joints(6), gripper, right joints(6), gripper].")
        state_valid = np.asarray(data["state_valid"], bool)
        body_inputs = {}
        if self.use_arm_state:
            body = np.zeros((2, 15), np.float32)
            body[:, :7] = state.reshape(2, 7)
            valid = np.zeros_like(body, bool)
            valid[:, :7] = state_valid.reshape(2, 7)
            body_inputs = dict(arm_state=body, arm_state_valid=valid)
        images = list(data["images"])
        return dict(images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
                    state=state, state_valid=state_valid,
                    arm_presence=np.ones(2, bool), **body_inputs)


class RobotwinOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        return self.normalize.apply("actions", np.asarray(actions)[..., :14], inverse=True)
