"""Train π0.5 or ESkill with one process or torchrun DDP."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.models.model import to_device
from openpi.training.checkpoints import load_weights, restore_random_state, save_checkpoint
from openpi.training.config import TrainConfig, create_model
from openpi.training.data_loader import create_data_loader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_epoch(loader, seed, epoch):
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    else:
        loader.sampler.generator.manual_seed(seed + epoch)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def train(config, *, resume=None):
    if config.precision not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_ddp and not dist.is_initialized():
        if config.device.startswith("cuda"):
            local_rank = int(os.environ["LOCAL_RANK"])
            config.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl" if config.device.startswith("cuda") else "gloo")
    rank = dist.get_rank() if use_ddp else 0
    world_size = dist.get_world_size() if use_ddp else 1
    output = Path(config.output_dir)
    if not resume and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} is not empty; choose a new run or --resume its checkpoint")
    if use_ddp:
        dist.barrier()
    seed_everything(config.seed)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    stats = saved["norm_stats"] if saved else json.loads(Path(config.norm_stats).read_text())
    if saved:
        if saved.get("world_size", 1) != world_size:
            raise ValueError("Resume with the original world size")
        for key in ("model", "data_index", "platform", "camera", "seed", "validation_fraction",
                    "batch_size", "frame_stride", "smoke", "model_type", "data_format",
                    "sample_manifest", "lora_rank", "lora_alpha", "backbone_dtype",
                    "pretrained_path", "tokenizer_path", "warmup_steps", "adam_betas"):
            previous = saved["config"].get(key, getattr(TrainConfig, key, None))
            if key == "model":
                previous = asdict(type(config.model)(**previous))
            if json.loads(json.dumps(previous)) != json.loads(json.dumps(asdict(config)[key])):
                raise ValueError(f"Cannot resume after changing {key}")
    adapter_only = bool(saved and saved.get("weights_kind") == "lora")
    model = create_model(config, initialize=not bool(saved) or adapter_only).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                  betas=tuple(config.adam_betas))
    start = 0
    if saved:
        load_weights(model, saved["model"], adapter_only=adapter_only)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
    if start >= config.num_steps:
        raise ValueError("num_steps must exceed the resumed step")
    initial_digest = parameter_digest(model) if config.lora_rank else None
    wrapped = (DistributedDataParallel(model, device_ids=[torch.device(config.device).index]
               if config.device.startswith("cuda") else None,
               find_unused_parameters=True, broadcast_buffers=False) if use_ddp else model)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "norm_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    if use_ddp:
        dist.barrier()
    config.norm_stats = str((output / "norm_stats.json").resolve())
    if rank == 0:
        config.save(output / "config.json")
    seed_everything(config.seed + rank)
    loader = create_data_loader(config)
    epoch, offset = divmod(start, len(loader))
    set_epoch(loader, config.seed, epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    if saved:
        restore_random_state(saved, config.device, rank)
        del saved
    checkpoint = output / "checkpoint.pt"
    log = (output / "metrics.jsonl").open("a") if rank == 0 else None
    try:
        for step in range(start, config.num_steps):
            begun = time.monotonic()
            learning_rate = config.learning_rate
            if config.warmup_steps and step < config.warmup_steps:
                initial = learning_rate / (config.warmup_steps + 1)
                learning_rate = initial + (learning_rate - initial) * step / config.warmup_steps
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                set_epoch(loader, config.seed, epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, config.device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16,
                                enabled=config.precision == "bfloat16"):
                losses = wrapped(batch["inputs"], batch["targets"])
                total = sum(config.loss_weights[key] * value for key, value in losses.items())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}, rank {rank}")
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            keys = list(losses)
            values = torch.stack([total.detach(), grad.detach(), *(losses[key].detach() for key in keys)]).float()
            identities = [dict(episode_id=m["episode_id"], task=m.get("task"),
                               action_row=m["action_row"]) for m in batch["metadata"]]
            if use_ddp:
                dist.all_reduce(values)
                values /= world_size
                collected = [None] * world_size
                dist.all_gather_object(collected, identities)
                identities = sum(collected, [])
            if rank == 0:
                record = dict(step=step + 1, loss=values[0].item(), grad_norm=values[1].item(),
                              **{key: value.item() for key, value in zip(keys, values[2:], strict=True)},
                              seconds=time.monotonic() - begun, smoke=config.smoke, world_size=world_size,
                              learning_rate=learning_rate,
                              global_batch_size=config.batch_size * world_size, resumed_from_step=start,
                              episodes=[m["episode_id"] for m in identities], samples=identities)
                log.write(json.dumps(record) + "\n")
                log.flush()
                if (step + 1) % config.log_every == 0 or step == start:
                    print(json.dumps(record), flush=True)
            if (step + 1) % config.save_every == 0 or step + 1 == config.num_steps:
                save_checkpoint(checkpoint, model=model, optimizer=optimizer, config=config,
                                step=step + 1, stats=stats)
                if rank == 0 and config.num_steps >= 1000:
                    # Hard links retain milestone adapters without copying optimizer state twice.
                    milestone = output / f"checkpoint_{step + 1:06d}.pt"
                    if not milestone.exists():
                        os.link(checkpoint, milestone)
        proof = dict(rank=rank, initial_trainable_sha256=initial_digest,
                     final_trainable_sha256=parameter_digest(model), world_size=world_size,
                     trainable_parameters=sum(p.numel() for p in parameters),
                     frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                     backend=dist.get_backend() if use_ddp else None,
                     peak_memory_bytes=torch.cuda.max_memory_allocated(config.device)
                     if config.device.startswith("cuda") else None)
        proofs = [proof]
        if use_ddp:
            proofs = [None] * world_size
            dist.all_gather_object(proofs, proof)
        if len({p["final_trainable_sha256"] for p in proofs}) != 1:
            raise RuntimeError("Trainable parameters diverged across ranks")
        if rank == 0:
            (output / "training_check.json").write_text(json.dumps(
                dict(step=config.num_steps, model_type=config.model_type, lora_rank=config.lora_rank,
                     ranks=proofs, all_ranks_equal=True), indent=2) + "\n")
    finally:
        if log is not None:
            log.close()
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Trusted checkpoint.pt produced by this project")
    args = parser.parse_args()
    try:
        result = train(TrainConfig.load(args.config), resume=args.resume)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Saved {result}")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
