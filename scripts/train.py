"""Train π0.5 or ESkill with one process or torchrun DDP."""
import argparse
from copy import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from openpi.models.model import to_device
from openpi.training.checkpoints import (check_resources, checkpoint_config, load_weights, resource_identities,
                                        restore_random_state, save_checkpoint, seed_everything)
from openpi.training.config import TrainConfig, WandbConfig, create_model
from openpi.training.data_loader import create_data_loader, episode_dataset
from openpi.training.validation import validation_losses
from openpi.training.metrics import (expert_gradient_norms, loss_metrics, motion_gradient_balance, phase_metrics,
                                    reduce_diagnostics, visual_lora_gradient_norms, wandb_metrics)
from openpi.augmentation import ImageAugmentation, GroundingAugmentation, augment_training_batch


def set_epoch(loader, seed, epoch):
    loader.batch_sampler.epoch = epoch
    sampler = loader.batch_sampler.sampler
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    else:
        sampler.generator.manual_seed(seed + epoch)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().numpy().tobytes())
    return digest.hexdigest()


def wandb_run_name(config, world_size):
    if config.model_type == "eskill":
        model = "pi05_" + config.model_type + ("_lora" if config.lora_rank else "")
    else:
        model = "pi05_lora" if config.lora_rank else "pi05_full"
    name = f"{model}_bs{config.batch_size * world_size}"
    if config.training_stage != 'joint':
        name += f'_{Path(config.data_index).parent.name or config.platform}_{config.training_stage}'
    return name


def init_wandb(config, *, rank, world_size=1, resume_config=None):
    if rank != 0:
        return None
    info = config.wandb_info
    if info is None:
        if config.smoke:
            return None
        info = config.wandb_info = WandbConfig.load()
    if info.mode == "disabled":
        return None
    import wandb

    previous = (resume_config or {}).get("wandb_info") or {}
    if (info.mode == "online" and info.id is None
            and previous.get("project") == info.project and previous.get("entity") == info.entity):
        info.id = previous.get("id")
        if info.id and info.name is None:
            info.name = previous.get("name")
    info.name = config.run_name or info.name or wandb_run_name(config, world_size)
    global_batch = config.batch_size * world_size
    method = "lora" if config.lora_rank else "full"
    run = wandb.init(
        project=info.project, entity=info.entity, name=info.name,
        tags=[config.model_type, method, f"bs{global_batch}", config.platform],
        mode=info.mode, id=info.id,
        resume="allow" if resume_config is not None and info.id and info.mode == "online" else None,
        config={**config.to_dict(), "world_size": world_size, "global_batch_size": global_batch},
        dir=config.output_dir,
        settings=wandb.Settings(api_key=(info.key or None) if info.mode == "online" else None,
                                disable_git=True, save_code=False),
    )
    info.id = run.id
    run.define_metric('chart/step')
    for namespace in ('train', 'aux_train', 'val', 'aux_val'):
        run.define_metric(namespace + '/*', step_metric='chart/step')
    for name in ('learning_rate', 'grad_norm', 'time_per_step'):
        run.define_metric('chart/' + name, step_metric='chart/step')
    return run


def calibrate_planner(model, loader, config):
    """Calibrate the native input on clean, deterministic samples from the training split."""
    clean = copy(loader.dataset)
    clean.augment = False
    clean.calibration_entities = True
    generator = torch.Generator().manual_seed(config.seed)
    rows = torch.randperm(len(clean), generator=generator)[:config.planner_calibration_samples].tolist()
    identities = []

    def batches():
        record_rows = not identities
        for start in range(0, len(rows), config.batch_size):
            batch = loader.collate_fn([clean[i] for i in rows[start:start + config.batch_size]])
            if record_rows:
                identities.extend({k: meta[k] for k in ('episode_id', 'frame')} for meta in batch['metadata'])
            yield to_device(batch, config.device)

    model.eval()
    try:
        with torch.no_grad(), torch.autocast(torch.device(config.device).type, dtype=torch.bfloat16,
                                             enabled=config.precision == 'bfloat16'):
            report = model.calibrate_motion_queries(batches)
    finally:
        model.train()
    return dict(report, samples=identities, initialization=getattr(model, 'planner_initialization', None))


def train(config, *, resume=None):
    config.validate_precision()
    planner = config.training_stage == 'motion_planner'
    if planner and not (config.model.planner_joint or config.smoke or resume or config.init_checkpoint):
        raise ValueError('Frozen Motion Planner training requires init_checkpoint from the KLAL run')
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
    saved = torch.load(resume, map_location="cpu", weights_only=False, mmap=True) if resume else None
    stats = saved["norm_stats"] if saved else json.loads(Path(config.norm_stats).read_text())
    if saved:
        previous_config = checkpoint_config(saved)
        if saved.get("world_size", 1) != world_size:
            raise ValueError("Resume with the original world size")
        if config.validation_fraction > 0 and 'split_path' not in previous_config:
            raise ValueError('This checkpoint predates task-stratified splits; start a new run')
        # Training semantics must match exactly. Resource files may live elsewhere (another
        # machine); check_resources compares their content instead of their paths.
        check_resources(saved, config)
        for key in ("model", "platform", "cameras", "action_padding", "norm_method", "loss_weights",
                    "seed", "validation_fraction", "batch_size", "frame_stride", "smoke", "model_type",
                    "lora_rank", "lora_alpha", "backbone_dtype", "warmup_steps", "adam_betas", "adam_fused", "weight_decay",
                    "skill_sampling", "skill_sampling_max_weight", "training_stage", "horizontal_flip_prob",
                    "image_augmentation", "grounding_augmentation", "planner_router_warmup_steps", "planner_calibration_samples",
                    "planner_init_step"):
            previous = previous_config.get(key, getattr(TrainConfig, key, None))
            if key == 'image_augmentation':
                previous = asdict(ImageAugmentation(**(previous or {})))
            if key == 'grounding_augmentation':
                if previous and previous.get('copy_prob', 0) and 'copy_max_objects' not in previous:
                    raise ValueError('Cannot resume an older Copy-Paste run with the new multi-object sampler')
                if previous and previous.get('patch_prob', 0) and 'target_margin_px' not in previous:
                    raise ValueError('Cannot resume after changing Patch Masking to protect the target; start a new run')
                previous = asdict(GroundingAugmentation(**(previous or {})))
            if key == "model":
                if config.model_type == 'eskill':
                    previous.setdefault('motion_min_horizon_frames', 0)
                previous = asdict(type(config.model)(**previous))
            if json.loads(json.dumps(previous)) != json.loads(json.dumps(asdict(config)[key])):
                raise ValueError(f"Cannot resume after changing {key}")
    if config.model_type == "eskill" and config.training_stage not in ('visual_primitive', 'motion_planner'):
        # Check command semantics before allocating the full model/VAE.
        episode_dataset(config)
    resources = resource_identities(config)
    adapter_only = bool(saved and saved.get("weights_kind") in ("lora", "visual_adapter", "planner_adapter"))
    model = create_model(config, initialize=not bool(saved) or adapter_only,
                         init_checkpoint=config.init_checkpoint if saved is None else None).train()
    loader = create_data_loader(config) if planner else None
    if planner:
        if saved:
            model.planner_calibration = saved['planner_calibration']
        else:
            reports = [calibrate_planner(model, loader, config) if rank == 0 else None]
            if use_ddp:
                # DDP initializes parameters from rank 0. Buffer sync is disabled, so copy the calibrated scale here.
                dist.broadcast(model.planner.query_time_scale, src=0)
                dist.broadcast_object_list(reports, src=0)
            model.planner_calibration = reports[0]
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay,
                                  betas=tuple(config.adam_betas), fused=config.adam_fused)
    start = 0
    if saved:
        load_weights(model, saved["model"], adapter_only=adapter_only)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
    if start >= config.num_steps:
        raise ValueError("num_steps must exceed the resumed step")
    wrapped = (DistributedDataParallel(model, device_ids=[torch.device(config.device).index]
               if config.device.startswith("cuda") else None,
               find_unused_parameters=True, broadcast_buffers=False,
               gradient_as_bucket_view=True) if use_ddp else model)
    initial_digest = parameter_digest(model) if config.lora_rank else None
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if planner:
            (output / "planner_calibration.json").write_text(json.dumps(model.planner_calibration, indent=2) + "\n")
        (output / "norm_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    if use_ddp:
        dist.barrier()
    config.norm_stats = str((output / "norm_stats.json").resolve())
    checkpoint = output / "checkpoint.pt"
    log, wandb_run = None, None
    exit_code = 1
    try:
        wandb_run = init_wandb(config, rank=rank, world_size=world_size,
                               resume_config=saved["config"] if saved else None)
        if rank == 0:
            config.save(output / "config.json")
            log = (output / "metrics.jsonl").open("a")
        seed_everything(config.seed + rank)
        if loader is None:
            loader = create_data_loader(config)
        sampling = getattr(loader.batch_sampler.sampler, 'summary', None)
        if rank == 0 and sampling is not None:
            (output / 'sampling.json').write_text(json.dumps(sampling, indent=2) + '\n')
        val_loader = (create_data_loader(config, 'val')
                      if config.validation_fraction > 0 and config.val_interval > 0 else None)
        epoch, offset = divmod(start, len(loader))
        set_epoch(loader, config.seed, epoch)
        loader.batch_sampler.start_batch = offset
        iterator = iter(loader)
        if saved:
            restore_random_state(saved, config.device, rank)
            del saved
        grounding = config.training_stage in ('grounding', 'visual_primitive', 'motion_planner')
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
                loader.batch_sampler.start_batch = 0
                iterator = iter(loader)
                batch = next(iterator)
            batch = to_device(batch, config.device)
            batch = augment_training_batch(batch, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(config.device).type, dtype=torch.bfloat16,
                                enabled=config.precision == "bfloat16"):
                router_only = planner and step < config.planner_router_warmup_steps
                forward_args = {'router_only': True} if router_only else {}
                if planner and config.model.planner_has_boundaries and (step + 1) % 500 == 0:
                    forward_args['planner_diagnostics'] = True
                losses = wrapped(batch["inputs"], batch["targets"], **forward_args)
                total = sum(config.loss_weights[key] * value for key, value in losses.items())
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}, rank {rank}")
            motion_gradients = None
            visual_gradients = None
            if config.training_stage == 'visual_primitive' and step == 0 and config.lora_rank:
                visual_gradients = visual_lora_gradient_norms(model, losses)
            if (config.motion_grad_interval and config.model_type == 'eskill' and 'action' in losses
                    and (step == 0 or (step + 1) % config.motion_grad_interval == 0)):
                motion_gradients = motion_gradient_balance(model, losses, config.loss_weights)
            total.backward()
            grad = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            visual = getattr(model, 'visual_supervision', None)
            logged = {**losses, **(visual.metrics if visual is not None else {})}
            keys = list(logged)
            values = torch.stack([total.detach(), grad.detach(), *(logged[key].detach() for key in keys)]).float()
            identities = [{k: m[k] for k in ('episode_id', 'task', 'action_row', 'frame', 'entity_id', 'source_task',
                                               'real_event_target', 'parent_trajectory_id', 'event_boundary', 'counterfactual')
                           if k in m} for m in batch['metadata']]
            if config.image_augmentation.enabled or grounding:
                for identity, metadata in zip(identities, batch['metadata'], strict=True):
                    identity['augmentation'] = (metadata['augmentation'] if grounding
                                                else metadata['augmentation']['counts'])
            if config.training_stage == 'stage1' or grounding:
                for identity, metadata in zip(identities, batch['metadata'], strict=True):
                    identity['mirrored'] = metadata.get('mirrored', False)
            if use_ddp:
                dist.all_reduce(values)
                values /= world_size
                collected = [None] * world_size
                dist.all_gather_object(collected, identities)
                identities = sum(collected, [])
            auxiliary = reduce_diagnostics(getattr(model, 'diagnostic_statistics', {}))
            payload = {}
            if rank == 0:
                record = dict(step=step + 1, loss=values[0].item(), grad_norm=values[1].item(),
                              **{key: value.item() for key, value in zip(keys, values[2:], strict=True)},
                              seconds=time.monotonic() - begun, smoke=config.smoke, world_size=world_size,
                              learning_rate=learning_rate,
                              global_batch_size=config.batch_size * world_size, resumed_from_step=start,
                              episodes=[m["episode_id"] for m in identities], samples=identities, aux=auxiliary)
                if grounding:
                    record['grounding_query_count'] = len(identities)
                    if config.cameras == ('image', 'wrist_image'):
                        totals = dict.fromkeys(('paired', 'copy_unverified', 'random_dropout', 'missing'), 0)
                        groups = {}
                        for row in identities:
                            aug = row['augmentation']
                            counts = groups.setdefault(aug['wrist_group'], {})
                            status = aug['wrist_status']
                            counts[status] = counts.get(status, 0) + 1
                            totals[status] += 1
                        totals['paired_text_swaps'] = groups.get('text_swap', {}).get('paired', 0)
                        record['wrist_samples'] = totals
                        record['wrist_by_augmentation'] = groups
                if planner:
                    record['router_only'] = router_only
                    record['expert_gradients'] = expert_gradient_norms(model)
                elif config.model_type == 'eskill' and not grounding:
                    record['expert_gradients'] = expert_gradient_norms(model)
                if config.image_augmentation.enabled:
                    statistics = [row['augmentation']['counts'] if grounding
                                  else row.pop('augmentation') for row in identities]
                    record['augmentation'] = {key: sum(row[key] for row in statistics) for key in statistics[0]}
                if motion_gradients is not None:
                    record['motion_gradients'] = dict(motion_gradients, loss_weights=config.loss_weights)
                if visual_gradients is not None:
                    record['unweighted_q_lora_grad_norm'] = visual_gradients
                log.write(json.dumps(record) + "\n")
                log.flush()
                if (step + 1) % config.log_every == 0 or step == start or step + 1 == config.num_steps:
                    print(json.dumps(record), flush=True)
                    payload = wandb_metrics(record)
            if val_loader is not None and ((step + 1) % config.val_interval == 0 or step + 1 == config.num_steps):
                val_losses = validation_losses(model, val_loader, config, include_aux=True)
                val_auxiliary = val_losses.pop('aux')
                if rank == 0:
                    metrics = {'val_' + key: value for key, value in loss_metrics(val_losses).items()}
                    with (output / 'validation.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(step=step + 1, **metrics, losses=val_losses, aux=val_auxiliary)) + '\n')
                    print(json.dumps(dict(step=step + 1, **metrics)), flush=True)
                    payload.update(phase_metrics('val', val_losses, val_auxiliary))
                    payload['chart/step'] = step + 1
            if rank == 0 and wandb_run is not None and payload:
                wandb_run.log(payload, step=step + 1)
            if (config.lora_rank and step == 0) or (step + 1) % config.save_every == 0 or step + 1 == config.num_steps:
                save_checkpoint(checkpoint, model=model, optimizer=optimizer, config=config,
                                step=step + 1, stats=stats, resources=resources)
                if rank == 0 and config.num_steps >= 1000:
                    # The latest pointer and its milestone share storage, including full-model checkpoints.
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
        exit_code = 0
    finally:
        if log is not None:
            log.close()
        if wandb_run is not None:
            wandb_run.finish(exit_code=exit_code)
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
