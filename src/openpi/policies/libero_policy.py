"""LIBERO source-state and simulator-action adapters."""
import numpy as np


class LiberoSFTInputs:
    """Standard LIBERO views and source state; prompt/state use model config."""
    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (8,):
            raise ValueError("LIBERO SFT expects the original 8-dimensional state")
        return dict(images=[data["image"], data["wrist_image"]], image_mask=np.ones(2, bool),
                    state=state, state_valid=np.asarray(data["state_valid"], bool), prompt=data["prompt"])


class LiberoInputs:
    def __call__(self, data):
        state = np.asarray(data["state"], np.float32)
        if state.shape != (15,):
            raise ValueError("Saved LIBERO state must be joint_state[7] + source state[8].")
        valid = np.asarray(data["state_valid"], bool)
        body = np.zeros((2, 15), np.float32)
        body_valid = np.zeros((2, 15), bool)
        body[0], body_valid[0] = state, valid
        images = [data["image"]]
        if "wrist_image" in data:
            images.append(data["wrist_image"])
        return dict(
            images=images, image_mask=np.ones(len(images), bool), prompt=data["prompt"],
            state=state[7:], state_valid=valid[7:],
            arm_state=body, arm_state_valid=body_valid, arm_presence=np.array([True, False]),
        )


class LiberoOutputs:
    def __init__(self, normalize):
        self.normalize = normalize

    def __call__(self, actions):
        # Source OSC commands are passed through; do not apply a second delta transform.
        return self.normalize.apply("actions", np.asarray(actions)[..., :7], inverse=True)
