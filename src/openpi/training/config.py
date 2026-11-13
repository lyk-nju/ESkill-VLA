"""Serializable training settings, with episode-level splits."""
from dataclasses import asdict, dataclass, field
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

from openpi.models.pi0_config import ESkillConfig, Pi0Config
from openpi.augmentation import ImageAugmentation

WANDB_CONFIG_PATH = Path(__file__).resolve().parents[3] / "wandb.json"


@dataclass
class WandbConfig:
    key: str = field(default="", repr=False)
    entity: str | None = None
    project: str = "EskillVLA"
    name: str | None = None
    mode: str = "online"
    id: str | None = None

    def __post_init__(self):
        if self.mode not in ("online", "offline", "disabled"):
            raise ValueError("wandb_info.mode must be online, offline or disabled")

    @classmethod
    def load(cls):
        return cls(**json.loads(WANDB_CONFIG_PATH.read_text()))


@dataclass
class TrainConfig:
    data_index: str
    tokenizer_path: str
    output_dir: str
    norm_stats: str
    pretrained_path: str | None = None
    init_checkpoint: str | None = None
    vae_path: str | None = None
    model_type: str = "eskill"
    lora_rank: int = 0
    lora_alpha: float = 16.
    backbone_dtype: str = "float32"
    platform: str = "libero"
    # Model input views in order; the first is the annotated view for regions and Motion.
    cameras: tuple[str, ...] = ("image",)
    # Tail past the last recorded action: "repeat_last" (kept valid) or "mask" (invalid).
    action_padding: str = "repeat_last"
    # Saved into every key of the statistics file and used by Normalize.
    norm_method: str = "quantile"
    model: Pi0Config | None = None
    batch_size: int = 8
    num_workers: int = 4
    num_steps: int = 30000
    save_every: int = 1000
    log_every: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    adam_betas: tuple[float, float] = (.9, .999)
    adam_fused: bool = False
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    validation_fraction: float = 0.1
    split_path: str | None = None
    val_interval: int = 1000
    val_max_batches: int = 32
    precision: str = "bfloat16"
    device: str = "cuda"
    smoke: bool = False
    frame_stride: int = 1
    skill_sampling: bool = False
    skill_sampling_max_weight: float = 3.0
    training_stage: str = 'joint'
    horizontal_flip_prob: float = 0.0
    image_augmentation: ImageAugmentation = field(default_factory=ImageAugmentation)
    motion_grad_interval: int = 0
    loss_weights: dict[str, float] | None = None
    wandb_info: WandbConfig | None = None
    run_name: str | None = None

    def __post_init__(self):
        if isinstance(self.image_augmentation, dict):
            self.image_augmentation = ImageAugmentation(**self.image_augmentation)
        if self.image_augmentation.enabled and self.model_type != 'eskill':
            raise ValueError('image_augmentation is for ESkill; pi05 uses its native model.augment_images')
        if self.model_type not in ("pi05", "eskill"):
            raise ValueError("model_type must be pi05 or eskill")
        if self.model is None:
            self.model = ESkillConfig() if self.model_type == "eskill" else Pi0Config()
        if self.training_stage not in ('joint', 'stage1', 'stage2'):
            raise ValueError('training_stage must be joint, stage1 or stage2')
        if self.training_stage != 'joint':
            if self.model_type != 'eskill' or self.lora_rank:
                raise ValueError('Two-stage training currently requires ESkill and rank 0')
            if self.training_stage == 'stage1' and not self.model.freeze_vlm:
                raise ValueError('Stage 1 requires a frozen VLM')
        if not 0 <= self.horizontal_flip_prob <= 1:
            raise ValueError('horizontal_flip_prob must lie in [0, 1]')
        if self.horizontal_flip_prob and self.training_stage != 'stage1':
            raise ValueError('Horizontal mirroring is only supervised in stage1; no mirrored action labels')
        if not isinstance(self.motion_grad_interval, int) or self.motion_grad_interval < 0:
            raise ValueError('motion_grad_interval must be a nonnegative integer')
        if self.model_type == "eskill" and not isinstance(self.model, ESkillConfig):
            raise ValueError("eskill requires ESkillConfig; use TrainConfig.from_dict for JSON configs")
        if self.loss_weights is None:
            self.loss_weights = {key: 1. for key in self.loss_keys}
        if self.model_type == "eskill":
            required = set(self.loss_keys)
            if set(self.loss_weights) != required:
                raise ValueError(f"eskill loss_weights must specify {sorted(required)}")
        if isinstance(self.wandb_info, dict):
            self.wandb_info = WandbConfig(**self.wandb_info)
        from openpi.policies.eskill_dataset import ACTION_PADDING
        self.cameras = tuple(self.cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError("cameras must be a nonempty list of distinct names")
        if self.model_type == "eskill" and self.platform == "libero" and self.cameras[0] != "image":
            raise ValueError("LIBERO grounding and EEF projection require image as the first camera")
        if self.action_padding not in ACTION_PADDING:
            raise ValueError(f"action_padding must be one of {ACTION_PADDING}")
        if self.action_padding == "mask" and not self.model.mask_action_loss:
            raise ValueError("action_padding=mask requires model.mask_action_loss=true")
        if self.norm_method not in ("quantile", "standard"):
            raise ValueError("norm_method must be quantile or standard")
        if not isinstance(self.skill_sampling, bool):
            raise ValueError('skill_sampling must be a boolean')
        if not math.isfinite(self.skill_sampling_max_weight) or self.skill_sampling_max_weight < 1:
            raise ValueError('skill_sampling_max_weight must be finite and at least 1')
        if self.model_type == "eskill" and self.model.augment_images:
            raise ValueError("ESkill does not support model.augment_images: region and Motion labels "
                             "are not transformed with the images")

    def validate_precision(self):
        if self.precision not in ('float32', 'bfloat16'):
            raise ValueError('precision must be float32 or bfloat16')
        if self.model_type == 'eskill' and self.backbone_dtype == 'bfloat16' and self.precision != 'bfloat16':
            raise ValueError("ESkill with backbone_dtype='bfloat16' requires precision='bfloat16'; "
                             'use a float32 backbone for float32 training')

    @property
    def loss_keys(self):
        auxiliary = ('skill', 'current', 'future_object', 'trace', 'gripper')
        if self.model_type != 'eskill':
            return ('action',)
        return auxiliary if self.training_stage == 'stage1' else ('action', *auxiliary)

    @property
    def validation_seed(self):
        return json.loads(Path(self.split_path).read_text())['seed'] if self.split_path else self.seed

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        model_class = ESkillConfig if data.get("model_type", "eskill") == "eskill" else Pi0Config
        data["model"] = model_class(**data.get("model", {}))
        return cls(**data)

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(path.read_text())
        else:
            data = json.loads(path.read_text())
        return cls.from_dict(data)

    def to_dict(self):
        """Serializable settings for logs/checkpoints; credentials stay in the input config."""
        data = asdict(self)
        if data["wandb_info"] is not None:
            data["wandb_info"]["key"] = ""
        return data

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n")


def _split_entries(index_path):
    index_path = Path(index_path)
    entries = []
    for entry in json.loads(index_path.read_text())["episodes"]:
        path = index_path.parent / entry["path"] if "path" in entry else None
        meta = json.loads(path.read_text()) if path is not None and path.suffix == ".json" else entry
        parent = meta.get("source", {}).get("parent_trajectory_id") or meta["episode_id"]
        task = meta.get('task') or (meta.get('instructions') or [''])[0]
        task = ' '.join(task.replace('_', ' ').split())
        if not task:
            raise ValueError(f'Task identity is missing for {entry["episode_id"]}')
        entries.append((entry['episode_id'], parent, task))
    return entries


def make_split(index_path, fraction=.1, seed=42):
    """Rank unique parents within each task; quotas count trajectories, not frames/views."""
    if not 0 <= fraction < 1:
        raise ValueError('validation_fraction must be in [0, 1)')
    groups, parent_tasks = defaultdict(set), {}
    for episode, parent, task in _split_entries(index_path):
        if parent in parent_tasks and parent_tasks[parent] != task:
            raise ValueError(f'Parent trajectory {parent} has conflicting task identities')
        parent_tasks[parent] = task
        groups[task].add(parent)
    result = dict(method='task_parent_rank_v1', seed=seed, validation_fraction=fraction,
                  source_index=str(Path(index_path).resolve()),
                  source_index_sha256=hashlib.sha256(Path(index_path).read_bytes()).hexdigest(),
                  train=[], val=[], tasks=[])
    for task, parents in sorted(groups.items()):
        ordered = sorted(parents, key=lambda parent: (hashlib.sha256(f'{seed}:{parent}'.encode()).hexdigest(), parent))
        count = 0 if fraction == 0 or len(ordered) < 2 else min(len(ordered) - 1, max(1, int(len(ordered) * fraction + .5)))
        result['val'].extend(ordered[:count])
        result['train'].extend(ordered[count:])
        result['tasks'].append(dict(task=task, total=len(ordered), train=len(ordered)-count, val=count))
    return result


def split_episodes(index_path, fraction=.1, seed=42, split_path=None):
    """Use the same saved parent assignments for SFT, event data and derived subsets."""
    if split_path is None and fraction == 0:
        return dict(train=[e['episode_id'] for e in json.loads(Path(index_path).read_text())['episodes']], val=[])
    split = json.loads(Path(split_path).read_text()) if split_path else make_split(index_path, fraction, seed)
    if split['validation_fraction'] != fraction:
        raise ValueError('Split manifest fraction differs from training config')
    membership = {parent: name for name in ('train', 'val') for parent in split[name]}
    if len(membership) != len(split['train']) + len(split['val']):
        raise ValueError('A parent trajectory appears in both split lists')
    result = dict(train=[], val=[])
    for episode, parent, task in _split_entries(index_path):
        result[membership[parent]].append(episode)
    return result


def create_model(config, *, training=True, initialize=True, init_checkpoint=None):
    """Construct training or inference modules; random weights require smoke mode."""
    if training:
        config.validate_precision()
    import torch
    from openpi.models.pi0_eskill import Pi0ESkill
    from openpi.models.pi0 import Pi0
    from openpi.models.lora import apply_lora

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. CPU smoke checks require an explicit tiny model configuration.")
    if config.smoke and config.model.vlm_width > 128:
        raise ValueError("Smoke runs require a tiny model; they do not train a random full-size π0.5.")
    if initialize and not config.smoke:
        if not config.pretrained_path or not Path(config.pretrained_path).exists():
            raise ValueError("Set pretrained_path to converted π0.5 model.safetensors, not JAX params.")
        checkpoint = Path(config.pretrained_path)
        if checkpoint.is_dir() and not (checkpoint / "model.safetensors").is_file():
            raise ValueError("π0.5 initialization requires model.safetensors; convert JAX weights first.")
    vae = None
    if training and config.model_type == "eskill":
        if config.vae_path:
            from openpi.models.grounding import load_vae
            vae = load_vae(config.vae_path)
        elif config.smoke:
            from diffusers import AutoencoderKL
            # Same frozen AutoencoderKL interface, random test weights; never a visual target for real training.
            vae = AutoencoderKL(
                in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32),
                layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=config.model.canvas_size)
        else:
            raise ValueError("Set vae_path to a local pretrained AutoencoderKL directory.")
    model_class = {"pi05": Pi0, "eskill": Pi0ESkill}[config.model_type]
    model = model_class(config.model) if config.model_type == "pi05" else model_class(config.model, vae=vae)
    if initialize and not config.smoke:
        model.load_base_weights(config.pretrained_path)
    if config.backbone_dtype == "bfloat16":
        for name in ("paligemma_with_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            getattr(model, name).to(torch.bfloat16)
    elif config.backbone_dtype != "float32":
        raise ValueError("backbone_dtype must be float32 or bfloat16")
    if config.lora_rank:
        apply_lora(model, config.lora_rank, config.lora_alpha)
    if init_checkpoint is not None:
        from openpi.training.checkpoints import load_sft_weights, load_stage1_weights
        (load_stage1_weights if config.training_stage == 'stage2' else load_sft_weights)(model, init_checkpoint, config)
    if training and config.training_stage == 'stage2' and initialize and init_checkpoint is None:
        raise ValueError('Stage 2 requires init_checkpoint from Stage 1; use --resume to continue Stage 2')
    if config.model_type == 'eskill':
        model.training_stage = config.training_stage
        if training:
            model.configure_stage(config.training_stage)
    model.to(config.device)
    return model
