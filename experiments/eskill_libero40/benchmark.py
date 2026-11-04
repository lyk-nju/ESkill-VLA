"""Short real-weight timing runs; no W&B upload and no formal training state is reused."""
import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data._utils.collate import default_collate

from openpi.models.model import to_device
from openpi.models import pi0
from openpi.models.lora import LoRALinear
from openpi.training.config import TrainConfig, create_model
from openpi.training.data_loader import create_data_loader
from openpi.training.metrics import reduce_diagnostics
from scripts.train import seed_everything


def full_collate(samples):
    return dict(inputs=default_collate([s['inputs'] for s in samples]),
                targets=default_collate([s['targets'] for s in samples]),
                metadata=[s['metadata'] for s in samples])


def copied_attention(q, k, v, mask):
    repeats = q.shape[1] // k.shape[1]
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1),
        attn_mask=mask[:, None, None])
    return output.transpose(1, 2).flatten(2)


def cast_lora(self, value):
    linear = torch.nn.functional.linear
    base = linear(value, self.weight, self.bias)
    update = linear(linear(value.to(self.lora_A.dtype), self.lora_A), self.lora_B)
    return base + (update * self.scale).to(base.dtype)


class Timers:
    def __init__(self, device):
        self.device, self.events = device, {}

    @contextmanager
    def part(self, name):
        a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        yield
        b.record()
        self.events.setdefault(name, []).append((a,b))

    def wrap(self, obj, method, name):
        original = getattr(obj, method)
        def wrapped(*args, **kwargs):
            with self.part(name):
                return original(*args, **kwargs)
        setattr(obj, method, wrapped)

    def collect(self):
        torch.cuda.synchronize(self.device)
        result = {name:sum(a.elapsed_time(b) for a,b in pairs)/1000 for name,pairs in self.events.items()}
        self.events.clear()
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--modes',nargs='+',choices=['baseline','trim','optimized','trim_no_checkpoint'],
                        default=['baseline','trim','optimized'])
    parser.add_argument('--steps',type=int,default=8)
    args=parser.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=f'cuda:{local}';torch.cuda.set_device(local)
    if world>1:dist.init_process_group('nccl',device_id=torch.device(device))
    rank=dist.get_rank() if world>1 else 0
    config=TrainConfig.load(args.config);config.init_checkpoint=None;config.device=device
    seed_everything(config.seed)
    model=create_model(config).train()
    wrapped=DistributedDataParallel(model,device_ids=[local],find_unused_parameters=True,broadcast_buffers=False,
                                    gradient_as_bucket_view=True) if world>1 else model
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=config.learning_rate,betas=tuple(config.adam_betas),
                                weight_decay=config.weight_decay,fused=config.adam_fused)
    timer=Timers(device)
    for obj,method,name in (
        (model.paligemma_with_expert,'embed_images','vision'),(model,'_text_prefix','language'),
        (model.skill,'current','current_skill'),(model.skill,'future','future_skill'),
        (model.grounding,'current','current_grounding'),(model.grounding,'goal','goal_grounding'),
        (model.motion,'forward','motion'),(model,'velocity','action'),
        (model.visual_supervision,'encode_targets','vae'),
    ):timer.wrap(obj,method,name)
    summaries={}
    attention, lora_forward = pi0.attend, LoRALinear.forward
    for mode in args.modes:
        pi0.attend = attention if mode == 'optimized' else copied_attention
        LoRALinear.forward = lora_forward if mode == 'optimized' else cast_lora
        config.model=replace(config.model,gradient_checkpointing=mode!='trim_no_checkpoint')
        model.config=config.model
        loader=create_data_loader(config);loader.collate_fn=full_collate;iterator=iter(loader)
        rows=[];torch.cuda.reset_peak_memory_stats(device)
        try:
            for i in range(args.steps+3):
                row={};begin=time.perf_counter();t=time.perf_counter();batch=next(iterator)
                row['data_wait']=time.perf_counter()-t
                tokens=batch['inputs']['tokens'].shape[-1]
                if mode!='baseline':
                    tokens=int(batch['inputs']['token_mask'].sum(-1).max())
                    for key in ('tokens','token_mask'):batch['inputs'][key]=batch['inputs'][key][:,:tokens]
                row['text_tokens']=tokens
                with timer.part('transfer'):batch=to_device(batch,device)
                optimizer.zero_grad(set_to_none=True)
                with timer.part('forward'),torch.autocast('cuda',dtype=torch.bfloat16):
                    losses=wrapped(batch['inputs'],batch['targets'])
                    total=sum(config.loss_weights[k]*v for k,v in losses.items())
                with timer.part('backward'):total.backward()
                with timer.part('optimizer'):
                    grad=torch.nn.utils.clip_grad_norm_(params,config.grad_clip);optimizer.step()
                with timer.part('diagnostics'):
                    values=torch.stack([total.detach(),grad.detach(),*(v.detach() for v in losses.values())])
                    if world>1:
                        dist.all_reduce(values);values/=world
                        identities=[None]*world;dist.all_gather_object(identities,batch['metadata'])
                    aux=reduce_diagnostics(model.diagnostic_statistics)
                    row['loss']=float(values[0])
                    if rank==0:json.dumps(dict(values=values.cpu().tolist(),aux=aux))
                row.update(timer.collect());row['wall']=time.perf_counter()-begin
                if i>=3:rows.append(row)
            summary={k:statistics.mean(r[k] for r in rows) for k in rows[0]}
            summary['peak_gib']=torch.cuda.max_memory_allocated(device)/2**30
            summary['reserved_gib']=torch.cuda.max_memory_reserved(device)/2**30
            summary['steps']=args.steps
            summaries[mode]=summary
            if rank==0:print(json.dumps(dict(mode=mode,**summary)),flush=True)
        except torch.cuda.OutOfMemoryError:
            if world>1:raise
            summaries[mode]={'oom':True};print(mode,'OOM',flush=True)
            optimizer.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        del iterator,loader
    pi0.attend, LoRALinear.forward = attention, lora_forward
    if world>1:
        all_summaries=[None]*world;dist.all_gather_object(all_summaries,summaries)
    else:all_summaries=[summaries]
    if rank==0:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(world_size=world,per_rank=all_summaries),indent=2)+'\n')
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
