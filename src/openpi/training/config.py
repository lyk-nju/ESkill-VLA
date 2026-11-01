"""Serializable training settings, with episode-level splits."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path

from openpi.models.pi0_config import ESkillConfig


@dataclass
class TrainConfig:
    data_index: str
    tokenizer_path: str
    output_dir: str
    norm_stats: str
    pretrained_path: str | None = None
    vae_path: str | None = None
    model_type: str = "eskill"
    data_format: str = "episodes"
    sample_manifest: str | None = None
    lora_rank: int = 0
    lora_alpha: float = 16.
    backbone_dtype: str = "float32"
    platform: str = "libero"
    camera: str = "image"
    model: ESkillConfig = field(default_factory=ESkillConfig)
    batch_size: int = 8
    num_workers: int = 4
    num_steps: int = 30000
    save_every: int = 1000
    log_every: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 0
    adam_betas: tuple[float, float] = (.9, .999)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42
    validation_fraction: float = 0.1
    precision: str = "bfloat16"
    device: str = "cuda"
    smoke: bool = False
    frame_stride: int = 1
    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "action": 1., "skill": .1, "future": .1, "grounding": .1, "trace": .1,
    })

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        data["model"] = ESkillConfig(**data.get("model", {}))
        return cls(**data)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n")


def split_episodes(index_path, fraction=.1, seed=42):
    """Parent trajectory hashing keeps all frames/views of an episode together."""
    index_path = Path(index_path)
    result = {"train": [], "val": []}
    for entry in json.loads(index_path.read_text())["episodes"]:
        path = index_path.parent / entry["path"] if "path" in entry else None
        meta = json.loads(path.read_text()) if path is not None and path.suffix == ".json" else entry
        parent = meta.get("source", {}).get("parent_trajectory_id") or meta["episode_id"]
        value = int(hashlib.sha256(f"{seed}:{parent}".encode()).hexdigest()[:16], 16) / 2**64
        result["val" if value < fraction else "train"].append(entry["episode_id"])
    return result


def create_model(config, *, training=True, initialize=True):
    """Construct training or inference modules; random weights require smoke mode."""
    import numpy as np
    import torch
    from diffusers import AutoencoderKL
    from openpi.models.pi0_config import SKILLS
    from openpi.models.pi0_eskill import Pi0ESkill
    from openpi.models.pi0 import Pi0
    from openpi.models.lora import apply_lora
    from openpi.transforms import TokenizePrompt

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
        if config.smoke:
            # Same frozen AutoencoderKL interface, random test weights; never a visual target for real training.
            vae = AutoencoderKL(
                in_channels=3, out_channels=3, down_block_types=("DownEncoderBlock2D",) * 2,
                up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32),
                layers_per_block=1, latent_channels=4, norm_num_groups=8, sample_size=config.model.canvas_size)
        else:
            if not config.vae_path:
                raise ValueError("Set vae_path to a local pretrained AutoencoderKL directory.")
            vae = AutoencoderKL.from_pretrained(config.vae_path, local_files_only=True)
    if config.model_type not in ("pi05", "eskill"):
        raise ValueError("model_type must be pi05 or eskill")
    model = Pi0ESkill(config.model, vae=vae) if config.model_type == "eskill" else Pi0(config.model)
    if initialize and not config.smoke:
        model.load_base_weights(config.pretrained_path)
    if config.backbone_dtype == "bfloat16":
        for name in ("paligemma_with_expert", "action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
            getattr(model, name).to(torch.bfloat16)
    elif config.backbone_dtype != "float32":
        raise ValueError("backbone_dtype must be float32 or bfloat16")
    if config.lora_rank:
        apply_lora(model, config.lora_rank, config.lora_alpha)
    model.to(config.device)
    if initialize and config.model_type == "eskill":
        tokenizer = TokenizePrompt(config.tokenizer_path, config.model.max_token_len)
        descriptions = dict(SKILLS)
        encoded = [tokenizer(descriptions[name]) for name in config.model.skill_names]
        tokens = torch.as_tensor(np.stack([x[0] for x in encoded]), device=config.device)
        mask = torch.as_tensor(np.stack([x[1] for x in encoded]), device=config.device)
        model.skill.text_prototypes.copy_(model.text_prototypes(tokens, mask))
    return model
