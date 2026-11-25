"""Training checkpoints, including per-rank RNG and compact LoRA weights."""
import hashlib
import random
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


RESOURCE_FIELDS = ("data_index", "split_path", "pretrained_path", "tokenizer_path", "vae_path")


def checkpoint_config(saved):
    """Resolve a Visual Primitive checkpoint's selectors from its saved model format."""
    settings = dict(saved["config"])
    if (saved.get("weights_kind") == "visual_adapter"
            or "grounding.subject_spatial.weight" in saved.get("model", {})):
        settings["training_stage"] = "visual_primitive"
        settings["model"] = dict(settings["model"], current_grounding="visual_primitive")
    return settings


def _hash_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def resource_identity(name, path):
    """Content identity of a training resource, independent of where it is stored.

    Data identity covers the index and every episode.json (labels, references, shapes), not
    the large arrays. Model identities hash all weight bytes; VAE identities also cover config.json.
    """
    # Unit-test configs use placeholder paths; they record no identity and cannot be resumed elsewhere.
    if path is None or not Path(path).exists():
        return None
    path = Path(path)
    if name == "data_index":
        entries = json.loads(path.read_text())["episodes"]
        return _hash_files([path, *(path.parent / e["path"] for e in entries)])
    if name in ("pretrained_path", "vae_path"):
        if path.is_dir():
            path = path / ("model.safetensors" if name == "pretrained_path" else "diffusion_pytorch_model.safetensors")
        paths = [path, path.parent / "config.json"] if name == "vae_path" else [path]
        return "sha256:" + _hash_files(paths)
    return _hash_files([path])


def resource_identities(config):
    return {name: resource_identity(name, getattr(config, name)) for name in RESOURCE_FIELDS}


def check_resources(saved, config, *, fields=RESOURCE_FIELDS, purpose="resume"):
    """Resources may move between machines; their content must stay the same."""
    recorded = saved.get("resources")
    if recorded is None:
        raise ValueError(f"Checkpoint has no resource identities; cannot {purpose}. Retrain with the current code.")
    for name in fields:
        path = getattr(config, name)
        if path is not None and not Path(path).exists() and not config.smoke:
            raise FileNotFoundError(f"Cannot {purpose}: {name} not found at {path}")
        expected = recorded.get(name)
        actual = resource_identity(name, path)
        if expected != actual:
            raise ValueError(f"Cannot {purpose}: {name} content differs from the checkpoint ({path})")


def check_sft_split(saved, config):
    """A validation run must reuse the SFT stage's split manifest (by content) and fraction."""
    if config.validation_fraction > 0:
        if not config.split_path or saved["config"].get("validation_fraction") != config.validation_fraction:
            raise ValueError('SFT initialization with validation requires the same fixed split_path and validation_fraction')
        check_resources(saved, config, fields=("split_path",), purpose="initialize from this SFT checkpoint")


def load_sft_weights(model, path, config):
    """Start ESkill from a trusted π0.5 SFT checkpoint, without optimizer/RNG/step."""
    from openpi.models.pi0_config import Pi0Config

    saved = torch.load(path, map_location="cpu", weights_only=False)
    source = saved["config"]
    if source["model_type"] != "pi05" or config.model_type != "eskill":
        raise ValueError("init_checkpoint expects a π0.5 SFT checkpoint and an ESkill target")
    check_sft_split(saved, config)
    for name in ("lora_rank", "lora_alpha", "backbone_dtype", "platform"):
        if source[name] != getattr(config, name):
            raise ValueError(f"SFT initialization requires matching {name}")
    check_resources(saved, config, fields=("tokenizer_path",), purpose="initialize from this SFT checkpoint")
    for field in fields(Pi0Config):
        if field.name in ("freeze_vlm", "gradient_checkpointing", "augment_images"):
            continue
        previous = source["model"].get(field.name, field.default)
        if previous != getattr(config.model, field.name):
            raise ValueError(f"SFT initialization requires matching model.{field.name}")
    stats = json.loads(Path(config.norm_stats).read_text())
    # ESkill is RGB/language only; state statistics are unused assets.
    if saved["norm_stats"]["actions"] != stats["actions"]:
        raise ValueError("SFT initialization requires unchanged normalization: actions")
    weights = saved["model"]
    if saved.get("weights_kind") == "lora":
        if not config.lora_rank:
            raise ValueError("SFT LoRA initialization requires lora_rank>0")
        check_resources(saved, config, fields=("pretrained_path",), purpose="initialize LoRA from this SFT checkpoint")
        required = {name for name, _ in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
        required.update(name for name, _ in model.named_buffers() if name in model._base_keys)
    else:
        if config.lora_rank:
            raise ValueError("Full SFT initialization requires lora_rank=0")
        required = model._base_keys
    if set(weights) != required:
        raise ValueError(f"Incomplete SFT weights: missing={sorted(required - weights.keys())}, "
                         f"unexpected={sorted(weights.keys() - required)}")
    model.load_state_dict(weights, strict=False)


def load_klal_weights(model, path, config):
    """Initialize Current and its native Q adapters; planner and optimizer start fresh."""
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    source = checkpoint_config(saved)
    kind = saved.get('weights_kind', 'full')
    if (source.get('training_stage') != 'visual_primitive' or kind not in ('visual_adapter', 'full')
            or source['model'].get('current_alignment_loss', 'kl') != 'kl'
            or source.get('loss_weights', {}).get('current_align', 0) <= 0):
        raise ValueError('Motion Planner init_checkpoint must be a Visual Primitive Current KL checkpoint')
    if config.planner_init_step is not None and saved['step'] != config.planner_init_step:
        raise ValueError(f"Expected KLAL source step {config.planner_init_step}, got {saved['step']}")
    for field in ('lora_rank', 'lora_alpha', 'backbone_dtype'):
        if source[field] != getattr(config, field):
            raise ValueError(f'KLAL initialization requires matching {field}')
    from openpi.models.pi0_config import ESkillConfig
    source_model = ESkillConfig(**source['model'])
    for field in ('attention_layers', 'learned_alpha', 'max_token_len', 'image_size', 'patch_size', 'depth',
                  'heads', 'kv_heads', 'head_dim', 'vlm_width', 'vlm_mlp'):
        if getattr(source_model, field) != getattr(config.model, field):
            raise ValueError(f'KLAL initialization requires matching model.{field}')
    check_sft_split(saved, config)
    check_resources(saved, config, fields=('pretrained_path', 'tokenizer_path'), purpose='initialize the planner')
    required = {k for k in model.state_dict() if k.startswith('grounding.alpha_scorer.') or '.q_proj.lora_' in k}
    if kind == 'full':
        required.update(model._base_keys)
    missing = required - saved['model'].keys()
    if missing:
        raise ValueError(f'Missing KLAL weights: {sorted(missing)}')
    model.load_state_dict({k: saved['model'][k] for k in required}, strict=False)
    model.planner_initialization = dict(checkpoint=str(path), source_step=saved['step'], weights_kind=kind)


def load_stage1_weights(model, path, config):
    """Start joint training from Stage 1, with a fresh optimizer, RNG and step."""
    from openpi.models.pi0_config import ESkillConfig
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    source = saved['config']
    if source.get('training_stage') != 'stage1' or saved.get('weights_kind') != 'full':
        raise ValueError('Stage 2 initialization requires a full Stage 1 checkpoint')
    check_sft_split(saved, config)
    check_resources(saved, config, fields=('data_index', 'tokenizer_path', 'vae_path'), purpose='initialize Stage 2')
    for name in ('platform', 'cameras', 'action_padding', 'norm_method', 'backbone_dtype'):
        if json.dumps(source[name]) != json.dumps(asdict(config)[name]):
            raise ValueError(f'Stage 2 initialization requires matching {name}')
    previous = asdict(ESkillConfig(**{'motion_min_horizon_frames': 0, **source['model']}))
    for name, value in asdict(config.model).items():
        if name not in ('freeze_vlm', 'gradient_checkpointing', 'action_patch_mask_prob', 'action_unmasked_prob') and previous[name] != value:
            raise ValueError(f'Stage 2 initialization requires matching model.{name}')
    stats = json.loads(Path(config.norm_stats).read_text())
    if saved['norm_stats']['actions'] != stats['actions']:
        raise ValueError('Stage 2 requires unchanged action normalization')
    model.load_state_dict(saved['model'], strict=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def random_state(device):
    return dict(torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state(device) if str(device).startswith("cuda") else [])


def restore_random_state(saved, device, rank=0):
    state = saved["rng_states"][rank] if "rng_states" in saved else saved
    torch.set_rng_state(state["torch_rng"])
    if str(device).startswith("cuda"):
        if isinstance(state["cuda_rng"], list):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        else:
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    np.random.set_state(state["numpy_rng"])
    random.setstate(state["python_rng"])


def load_weights(model, weights, *, adapter_only=False):
    if not adapter_only:
        model.load_state_dict(weights, strict=True)
        return
    if getattr(model.config, 'current_grounding', None) in ('visual_primitive', 'motion_planner'):
        expected = model.state_dict().keys() - model._base_keys
        if weights.keys() != expected:
            raise ValueError(f'Incomplete {model.config.current_grounding} adapter: missing={sorted(expected - weights.keys())}, '
                             f'unexpected={sorted(weights.keys() - expected)}')
        model.load_state_dict(weights, strict=False)
        return
    required = {name for name, p in model.named_parameters() if p.requires_grad}
    allowed = required | {name for name, _ in model.named_buffers()}
    if not required <= weights.keys():
        raise ValueError(f"Checkpoint is missing adapter parameters: {sorted(required - weights.keys())}")
    if weights.keys() - allowed:
        raise ValueError(f"Adapter checkpoint contains non-adapter weights: {sorted(weights.keys() - allowed)}")
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def save_checkpoint(path, *, model, optimizer, config, step, stats, resources=None):
    rank = dist.get_rank() if dist.is_initialized() else 0
    states = [random_state(config.device)]
    if dist.is_initialized():
        states = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(random_state(config.device), states, dst=0)
    if rank == 0:
        weights = model.state_dict()
        visual_stage = config.training_stage in ('visual_primitive', 'motion_planner')
        compact_visual = visual_stage and not config.smoke
        compact_lora = config.lora_rank and not visual_stage
        if compact_visual:
            weights = {k: v for k, v in weights.items() if k not in model._base_keys}
        elif compact_lora:
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            names.update(name for name, _ in model.named_buffers())
            weights = {k: v for k, v in weights.items() if k in names}
        payload = dict(model=weights, optimizer=optimizer.state_dict(), config=config.to_dict(),
                       resources=resources if resources is not None else resource_identities(config),
                       step=step, norm_stats=stats, rng_states=states,
                       world_size=dist.get_world_size() if dist.is_initialized() else 1,
                       weights_kind=("planner_adapter" if config.training_stage == "motion_planner" else "visual_adapter")
                       if compact_visual else "lora" if compact_lora else "full")
        if config.training_stage == "motion_planner":
            payload["planner_calibration"] = model.planner_calibration
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()
