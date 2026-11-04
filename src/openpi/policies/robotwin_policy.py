"""RoboTwin joint-space adapter; source export is connected separately."""
import numpy as np


class RobotwinInputs:
    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (14,):
            raise ValueError("RoboTwin adapter expects [left joints(6), gripper, right joints(6), gripper].")
        body = np.zeros((2, 15), np.float32)
        body[:, :7] = state.reshape(2, 7)
        valid = np.zeros_like(body, bool)
        state_valid = np.asarray(data["state_valid"], bool)
        valid[:, :7] = state_valid.reshape(2, 7)
        images = list(data["images"])
        return dict(images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
                    state=state, state_valid=state_valid, arm_state=body,
                    arm_state_valid=valid, arm_presence=np.ones(2, bool))


class RobotwinOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        return self.normalize.apply("actions", np.asarray(actions)[..., :14], inverse=True)
