"""Model sizes and the shared skill vocabulary; no data-loading dependencies."""
from dataclasses import dataclass, replace


SKILLS = (
    ("Pick", "Establish controlled grasp of an object and lift it."),
    ("Place", "Move a held object to a supporting location and release it."),
    ("Open", "Move an articulated object into an open state."),
    ("Close", "Move an articulated object into a closed state."),
    ("Turn", "Rotate an object or control to change its orientation or setting."),
    ("Push", "Move an object by sustained pushing contact."),
    ("Hold", "Maintain an established grasp or supporting contact."),
    ("Idle", "Remain available without active interaction."),
    ("Return", "Return the effector to its designated resting configuration."),
)


@dataclass(frozen=True)
class Pi0Config:
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 200
    image_size: int = 224
    vocab_size: int = 257152
    vlm_width: int = 2048
    expert_width: int = 1024
    depth: int = 18
    heads: int = 8
    kv_heads: int = 1
    head_dim: int = 256
    vlm_mlp: int = 16384
    expert_mlp: int = 4096
    vision_width: int = 1152
    vision_mlp: int = 4304
    vision_depth: int = 27
    vision_heads: int = 16
    patch_size: int = 14
    freeze_vlm: bool = True
    gradient_checkpointing: bool = True
    discrete_state_input: bool = True
    mask_action_loss: bool = True
    augment_images: bool = False


@dataclass(frozen=True)
class ESkillConfig(Pi0Config):
    hidden_dim: int = 256
    auxiliary_heads: int = 8
    auxiliary_depth: int = 2
    role_queries: int = 16
    motion_queries: int = 8
    motion_stride: int = 4
    motion_tail_policy: str = "hold_last"
    arm_state_dim: int = 15
    canvas_size: int = 256
    skill_names: tuple[str, ...] = tuple(name for name, _ in SKILLS)
    trace_scale: float = 1.0  # Retained for serialized SFT configs; unused by normalized 2D traces.
    trace_beta: float = 0.05
    foreground_weight: float = 1.0
    background_weight: float = 0.1

    @property
    def motion_offsets(self):
        return tuple(self.motion_stride * (j + 1) for j in range(self.motion_queries))

    @classmethod
    def tiny(cls, **overrides):
        """Same computation graph at CPU test scale; never a pretrained π0.5."""
        config = cls(
            action_horizon=8, max_token_len=96, image_size=32, vlm_width=32,
            expert_width=32, depth=2, heads=4, kv_heads=1, head_dim=8,
            vlm_mlp=64, expert_mlp=64, vision_width=32, vision_mlp=64,
            vision_depth=2, vision_heads=4, patch_size=8, hidden_dim=32,
            auxiliary_heads=4, auxiliary_depth=1, role_queries=4,
            canvas_size=32, gradient_checkpointing=False,
        )
        return replace(config, **overrides)
