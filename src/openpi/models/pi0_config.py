"""Model sizes and the shared skill vocabulary; no data-loading dependencies."""
from dataclasses import dataclass


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

    @classmethod
    def tiny(cls, **overrides):
        """Same computation graph at CPU test scale; never a pretrained π0.5."""
        sizes = dict(action_horizon=8, max_token_len=96, image_size=32, vlm_width=32,
                     expert_width=32, depth=2, heads=4, kv_heads=1, head_dim=8,
                     vlm_mlp=64, expert_mlp=64, vision_width=32, vision_mlp=64,
                     vision_depth=2, vision_heads=4, patch_size=8, gradient_checkpointing=False)
        return cls(**(sizes | overrides))


@dataclass(frozen=True)
class ESkillConfig(Pi0Config):
    """Continuous Skill -> object-conditioned Motion -> Action."""
    action_horizon: int = 10
    hidden_dim: int = 256
    auxiliary_heads: int = 8
    auxiliary_depth: int = 2
    role_queries: int = 16
    motion_queries: int = 8
    # v0 is checkpoint-compatible; v1/v2/v3 add text, sequential goal, goal-first Motion.
    visual_primitive_version: str = 'v0'
    # Optional gripper visual grounding for the dedicated grounding stage.
    grounding_subject: bool = False
    current_grounding: str = 'vae'
    subject_bbox: bool = False
    discrete_state_input: bool = False
    canvas_size: int = 256
    future_bbox_expand_px: int = 8
    skill_names: tuple[str, ...] = tuple(name for name, _ in SKILLS)
    skill_dim: int = 512
    skill_mlp: int = 2048
    skill_depth: int = 2
    skill_residual_scale: float = 0.1
    active_skill_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    motion_stride: int = 5
    motion_min_horizon_frames: int = 10
    # Derived 2D stop targets after the Event/minimum window; enabled by the V3 recipe.
    motion_terminal_hold: bool = False
    action_patch_mask_prob: tuple[float, ...] = (0., 0.)
    action_unmasked_prob: float = 0.
    trace_beta: float = 0.05
    foreground_weight: float = 1.0
    background_weight: float = 0.1

    def __post_init__(self):
        if self.current_grounding not in ('vae', 'bbox', 'bbox_gac'):
            raise ValueError('current_grounding must be vae, bbox, or bbox_gac')
        if self.current_grounding != 'vae' and not self.grounding_subject:
            raise ValueError('BBox supervision requires the dedicated Current/Subject model')
        if self.subject_bbox and self.current_grounding == 'vae':
            raise ValueError('Subject box regression requires a Current box recipe')
        for name in ('active_skill_ids', 'action_patch_mask_prob', 'skill_names'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.active_skill_ids != (0, 1, 2, 3, 4, 5):
            raise ValueError('ESkill currently supports the six recorded semantic skills (IDs 0..5)')
        if not isinstance(self.motion_stride, int) or self.motion_stride < 1 or self.motion_queries < 1:
            raise ValueError('Motion queries and stride must be positive frame counts')
        if not isinstance(self.motion_min_horizon_frames, int) or self.motion_min_horizon_frames < 0:
            raise ValueError('Motion minimum horizon must be a nonnegative frame count')
        if not isinstance(self.future_bbox_expand_px, int) or self.future_bbox_expand_px < 0:
            raise ValueError('Future bbox expansion must be a nonnegative canvas pixel count')
        if not self.action_patch_mask_prob or any(not 0 <= p <= 1 for p in self.action_patch_mask_prob):
            raise ValueError('Action patch mask probabilities must lie in [0, 1]')
        if not 0 <= self.action_unmasked_prob <= 1:
            raise ValueError('action_unmasked_prob must lie in [0, 1]')
        if self.visual_primitive_version not in ('v0', 'v1', 'v2', 'v3'):
            raise ValueError('visual_primitive_version must be v0, v1, v2 or v3')
        if self.discrete_state_input:
            raise ValueError('ESkill uses RGB and language only; discrete_state_input must be false')
        if self.augment_images:
            raise ValueError('ESkill geometric labels require model.augment_images=false')
        if self.skill_dim % self.auxiliary_heads or min(self.skill_mlp, self.skill_depth) <= 0:
            raise ValueError('Invalid Skill-MoE dimensions')

    @property
    def motion_offsets_frames(self):
        return tuple(self.motion_stride * i for i in range(1, self.motion_queries + 1))

    @classmethod
    def tiny(cls, **overrides):
        sizes = dict(action_horizon=10, hidden_dim=32, auxiliary_heads=4,
                     auxiliary_depth=1, role_queries=4, canvas_size=32,
                     skill_dim=32, skill_mlp=64, skill_depth=2)
        return super().tiny(**(sizes | overrides))
