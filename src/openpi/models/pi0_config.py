"""Model sizes and the shared skill vocabulary; no data-loading dependencies."""
from dataclasses import dataclass
import math


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
    attention_layers: tuple[int, ...] = tuple(range(10, 18))
    learned_alpha: bool = True
    gac_temperature: float = .07
    planner_joint: bool = False
    planner_version: str = 'v1'
    planner_frozen_vlm_layers: tuple[int, ...] = ()
    planner_prior_ratio: float = .3
    current_alignment_loss: str = 'kl'
    allocation_min_mass: float = .25
    allocation_min_patch_overlap: float = .25
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
        if self.current_grounding not in ('vae', 'vae_gac', 'bbox', 'bbox_gac', 'visual_primitive', 'motion_planner'):
            raise ValueError('Unknown current_grounding mode')
        object.__setattr__(self, 'attention_layers', tuple(self.attention_layers))
        object.__setattr__(self, 'planner_frozen_vlm_layers', tuple(self.planner_frozen_vlm_layers))
        if self.visual_primitive or self.motion_planner:
            if not self.freeze_vlm or self.subject_bbox:
                raise ValueError('Visual Primitive uses a frozen base and a spatial Subject decoder')
            if (not self.attention_layers or tuple(sorted(set(self.attention_layers))) != self.attention_layers
                    or any(type(i) is not int or not 0 <= i < self.depth for i in self.attention_layers)):
                raise ValueError('attention_layers must be increasing, distinct PaliGemma layer indices')
            if not math.isfinite(self.gac_temperature) or self.gac_temperature <= 0:
                raise ValueError('gac_temperature must be finite and positive')
            if self.current_alignment_loss not in ('kl', 'allocation'):
                raise ValueError('current_alignment_loss must be kl or allocation')
            if not 0 < self.allocation_min_mass <= 1 or not 0 <= self.allocation_min_patch_overlap < 1:
                raise ValueError('Attention allocation needs 0 < min_mass <= 1 and 0 <= min_patch_overlap < 1')
        if self.motion_planner:
            layers = self.planner_frozen_vlm_layers
            if tuple(sorted(set(layers))) != layers or any(type(i) is not int or not 0 <= i < self.depth for i in layers):
                raise ValueError('planner_frozen_vlm_layers must name distinct backbone layers in order')
            if self.planner_version not in ('v1', 'v2') or (self.planner_version == 'v2' and self.planner_joint):
                raise ValueError('Choose v1 or the frozen event-boundary planner v2')
            if self.grounding_subject or self.current_alignment_loss != 'kl' or not self.learned_alpha:
                raise ValueError('Motion Planner requires Current-only KLAL with learned alpha')
            if not 0 <= self.planner_prior_ratio <= 1 or self.hidden_dim % 4 or self.hidden_dim % self.auxiliary_heads:
                raise ValueError('Invalid Motion Planner prior ratio or dimensions')
            if self.motion_queries != 8:
                raise ValueError('Motion Planner uses eight future slots')
        if self.current_grounding != 'vae' and not self.motion_planner and not self.grounding_subject:
            raise ValueError('This grounding mode requires the dedicated Current/Subject model')
        if self.subject_bbox and not self.current_bbox:
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
    def visual_primitive(self):
        return self.current_grounding == 'visual_primitive'

    @property
    def motion_planner(self):
        return self.current_grounding == 'motion_planner'

    @property
    def current_bbox(self):
        return self.current_grounding in ('bbox', 'bbox_gac')

    @property
    def entity_contrast_enabled(self):
        return self.current_grounding in ('vae_gac', 'bbox_gac', 'visual_primitive')

    @property
    def motion_offsets_frames(self):
        return tuple(self.motion_stride * i for i in range(1, self.motion_queries + 1))

    @classmethod
    def tiny(cls, **overrides):
        sizes = dict(action_horizon=10, hidden_dim=32, auxiliary_heads=4,
                     auxiliary_depth=1, role_queries=4, canvas_size=32,
                     skill_dim=32, skill_mlp=64, skill_depth=2)
        if overrides.get('current_grounding') in ('visual_primitive', 'motion_planner'):
            sizes['attention_layers'] = tuple(range(overrides.get('depth', 2)))
        return super().tiny(**(sizes | overrides))
