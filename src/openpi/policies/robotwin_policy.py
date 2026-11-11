"""RoboTwin absolute joint-target adapter for unified Episodes."""
import numpy as np


class RobotwinInputs:
    """Current RGB/language and static dual-arm identity; optional pi05 state tokens."""
    def __init__(self, *, include_state=True):
        self.include_state = include_state

    def __call__(self, data):
        images = list(data["images"])
        result = dict(images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
                      arm_presence=np.ones(2, bool))
        if self.include_state:
            state = np.asarray(data["state"], np.float32)
            if state.shape != (14,):
                raise ValueError("RoboTwin adapter expects [left joints(6), gripper, right joints(6), gripper].")
            result.update(state=state, state_valid=np.asarray(
                data.get("state_valid", np.ones_like(state, dtype=bool)), bool))
        return result


class RobotwinOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        return self.normalize.apply("actions", np.asarray(actions)[..., :14], inverse=True)
