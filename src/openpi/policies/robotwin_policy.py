"""RoboTwin absolute joint-target adapter for unified Episodes."""
import numpy as np


class RobotwinInputs:
    def __init__(self, *, use_arm_state=False):
        self.use_arm_state = use_arm_state

    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (14,):
            raise ValueError("RoboTwin adapter expects [left joints(6), gripper, right joints(6), gripper].")
        state_valid = np.asarray(data["state_valid"], bool)
        body_inputs = {}
        if self.use_arm_state:
            if "arm_state" not in data or "effector_start" not in data:
                raise ValueError("RoboTwin ESkill needs observed per-arm EEF pose/aperture in arm_state[2,8] "
                                 "and calibrated effector_start; joint targets alone are insufficient")
            body = np.asarray(data["arm_state"], np.float32)
            if body.shape != (2, 8):
                raise ValueError("RoboTwin ESkill arm_state must have shape [2,8]")
            body_inputs = dict(arm_state=body, arm_state_valid=np.asarray(data["arm_state_valid"], bool),
                               effector_start=np.asarray(data["effector_start"], np.float32),
                               effector_start_valid=np.asarray(data["effector_start_valid"], bool))
        images = list(data["images"])
        return dict(images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
                    state=state, state_valid=state_valid,
                    arm_presence=np.ones(2, bool), **body_inputs)


class RobotwinOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        return self.normalize.apply("actions", np.asarray(actions)[..., :14], inverse=True)
