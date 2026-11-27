"""Model sizes and the shared skill vocabulary; no data-loading dependencies."""
from dataclasses import dataclass
import math

from openpi.models import research


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
    """Model dimensions and available visual, planning and execution components."""
    action_horizon: int = 10
    hidden_dim: int = 256
    auxiliary_heads: int = 8
    auxiliary_depth: int = 2
    role_queries: int = 16
    motion_queries: int = 8
    # Optional gripper visual grounding for the dedicated grounding stage.
    grounding_subject: bool = False
    current_grounding: str = research.DEFAULT_GROUNDING
    subject_bbox: bool = False
    attention_layers: tuple[int, ...] = tuple(range(10, 18))
    finetune_vlm_layers: tuple[int, ...] = ()
    learned_alpha: bool = True
    gac_temperature: float = .07
    planner_joint: bool = False
    planner_version: str = research.DEFAULT_PLANNER
    planner_frozen_vlm_layers: tuple[int, ...] = ()
    planner_prior_ratio: float = .3
    action_motion_kv: bool = True
    action_current_ratio: float = .15
    action_motion_ratio: float = .15
    train_current: bool = False
    train_planner: bool = False
    train_action_expert: bool = True
    current_alignment_loss: str = 'kl'
    allocation_min_mass: float = .25
    allocation_min_patch_overlap: float = .25
    discrete_state_input: bool = False
    canvas_size: int = 256
    future_bbox_expand_px: int = 8
    skill_names: tuple[str, ...] = tuple(name for name, _ in SKILLS)
    active_skill_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    motion_stride: int = 5
    motion_min_horizon_frames: int = 10
    # Derived 2D stop targets after the Event/minimum window.
    motion_terminal_hold: bool = False
    trace_beta: float = 0.05
    foreground_weight: float = 1.0
    background_weight: float = 0.1

    def __post_init__(self):
        if self.current_grounding not in research.GROUNDING:
            raise ValueError(f'{self.current_grounding} is unavailable here; choose from {research.GROUNDING}')
        if self.grounding_subject and not research.SUBJECT:
            raise ValueError('Subject comparisons belong to visual-primitive')
        object.__setattr__(self, 'attention_layers', tuple(self.attention_layers))
        layers = tuple(self.finetune_vlm_layers)
        object.__setattr__(self, 'finetune_vlm_layers', layers)
        if layers and (not self.visual_primitive or tuple(sorted(set(layers))) != layers
                       or any(type(i) is not int or not 0 <= i < self.depth for i in layers)):
            raise ValueError('finetune_vlm_layers requires distinct ordered Visual Primitive backbone layers')
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
            if self.planner_version not in research.PLANNERS:
                raise ValueError(f'Available planners on this branch: {research.PLANNERS}')
            if self.planner_has_boundaries and self.planner_joint:
                raise ValueError('Event-boundary planners require a frozen Current (planner_joint=false)')
            if self.grounding_subject or self.current_alignment_loss != 'kl' or not self.learned_alpha:
                raise ValueError('Motion Planner requires Current-only KLAL with learned alpha')
            if not 0 <= self.planner_prior_ratio <= 1 or self.hidden_dim % 4 or self.hidden_dim % self.auxiliary_heads:
                raise ValueError('Invalid Motion Planner prior ratio or dimensions')
            if self.motion_queries != 8:
                raise ValueError('Motion Planner uses eight future slots')
        if self.action_execution:
            if self.planner_version != 'v2_joint':
                raise ValueError('Action execution needs the joint ten-slot planner')
            c, m = self.action_current_ratio, self.action_motion_ratio
            if not (0 <= c <= 1 and 0 <= m <= 1 and c + m <= 1):
                raise ValueError('Action prior weights must be nonnegative and sum to at most one')
            if not research.ACTION_ABLATIONS and (not self.action_motion_kv or c != .15 or m != .15):
                raise ValueError('The unified action path uses ten Motion K/Vs and 0.15 per prior')
        if self.current_only and (self.current_alignment_loss != 'kl' or not self.learned_alpha):
            raise ValueError('Current-only Learn to See uses KLAL and learned text weights')
        if self.current_grounding != 'vae' and not (self.motion_planner or self.visual_primitive or self.grounding_subject):
            raise ValueError('This grounding mode requires the dedicated Current/Subject model')
        if self.subject_bbox and not self.current_bbox:
            raise ValueError('Subject box regression requires a Current box recipe')
        for name in ('active_skill_ids', 'skill_names'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if self.active_skill_ids != (0, 1, 2, 3, 4, 5):
            raise ValueError('ESkill currently supports the six recorded semantic skills (IDs 0..5)')
        if not isinstance(self.motion_stride, int) or self.motion_stride < 1 or self.motion_queries < 1:
            raise ValueError('Motion queries and stride must be positive frame counts')
        if not isinstance(self.motion_min_horizon_frames, int) or self.motion_min_horizon_frames < 0:
            raise ValueError('Motion minimum horizon must be a nonnegative frame count')
        if not isinstance(self.future_bbox_expand_px, int) or self.future_bbox_expand_px < 0:
            raise ValueError('Future bbox expansion must be a nonnegative canvas pixel count')
        if self.discrete_state_input:
            raise ValueError('ESkill uses RGB and language only; discrete_state_input must be false')
        if self.augment_images:
            raise ValueError('ESkill geometric labels require model.augment_images=false')
        if self.auxiliary_heads < 1 or self.hidden_dim < 1 or self.hidden_dim % self.auxiliary_heads:
            raise ValueError('The query hidden dimension must be divisible by the number of attention heads')

    @property
    def visual_primitive(self):
        return self.current_grounding == 'visual_primitive'

    @property
    def current_only(self):
        return self.visual_primitive and not self.grounding_subject

    @property
    def motion_planner(self):
        return self.current_grounding in ('motion_planner', 'action_execution')

    @property
    def action_execution(self):
        return self.current_grounding == 'action_execution'

    @property
    def planner_has_boundaries(self):
        return self.planner_version in ('v2', 'v2_joint')

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
        if overrides.get('grounding_subject') and 'current_grounding' not in overrides and research.SUBJECT:
            overrides['current_grounding'] = 'vae'
        sizes = dict(action_horizon=10, hidden_dim=32, auxiliary_heads=4,
                     auxiliary_depth=1, role_queries=4, canvas_size=32)
        if overrides.get('current_grounding', research.DEFAULT_GROUNDING) in ('visual_primitive', 'motion_planner', 'action_execution'):
            sizes['attention_layers'] = tuple(range(overrides.get('depth', 2)))
        return super().tiny(**(sizes | overrides))
