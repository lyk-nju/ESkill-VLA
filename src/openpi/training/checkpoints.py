"""Training checkpoints, including per-rank RNG and compact LoRA weights."""
from dataclasses import asdict
import random

import numpy as np
import torch
import torch.distributed as dist


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
    required = {name for name, p in model.named_parameters() if p.requires_grad}
    allowed = required | {name for name, _ in model.named_buffers()}
    if not required <= weights.keys():
        raise ValueError(f"Checkpoint is missing adapter parameters: {sorted(required - weights.keys())}")
    if weights.keys() - allowed:
        raise ValueError(f"Adapter checkpoint contains non-adapter weights: {sorted(weights.keys() - allowed)}")
    result = model.load_state_dict(weights, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {result.unexpected_keys}")


def save_checkpoint(path, *, model, optimizer, config, step, stats):
    rank = dist.get_rank() if dist.is_initialized() else 0
    states = [random_state(config.device)]
    if dist.is_initialized():
        states = [None] * dist.get_world_size() if rank == 0 else None
        dist.gather_object(random_state(config.device), states, dst=0)
    if rank == 0:
        weights = model.state_dict()
        if config.lora_rank:
            names = {name for name, p in model.named_parameters() if p.requires_grad}
            names.update(name for name, _ in model.named_buffers())
            weights = {k: v for k, v in weights.items() if k in names}
        payload = dict(model=weights, optimizer=optimizer.state_dict(), config=asdict(config),
                       step=step, norm_stats=stats, rng_states=states,
                       world_size=dist.get_world_size() if dist.is_initialized() else 1,
                       weights_kind="lora" if config.lora_rank else "full")
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    if dist.is_initialized():
        dist.barrier()
