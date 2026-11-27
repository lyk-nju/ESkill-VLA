"""Capture selected native prefix attention rows with frozen official π0.5 weights."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

from inputs import load_rgb, query_inputs
from native_attention import capture_native_attention
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.transforms import TokenizePrompt, resize_image


METHODS = ("oracle", "source", "destination", "tail", "sentence", "locate")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, val):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(val, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def capture_batch(model, tokenizer, rows, device):
    """Full task and independent Locate text share one captured prefix batch."""
    examples, indexes, images, text = [], [], [], []
    max_len = tokenizer.max_len
    for i, row in enumerate(rows):
        rgb = load_rgb(row)
        im = resize_image(rgb, model.config.image_size).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1
        for variant, query in query_inputs(row, tokenizer).items():
            query_rows = np.zeros((len(METHODS), max_len), dtype=bool)
            for name, positions in query["queries"].items():
                query_rows[METHODS.index(name), np.asarray(positions, dtype=np.int64)] = True
            if not query_rows.any() and variant != "instruction":
                continue
            examples.append(i)
            indexes.append(query_rows.any(-1))
            images.append(im)
            text.append((query["tokens"], query["token_mask"], query_rows))
    inputs = dict(
        images=torch.from_numpy(np.stack(images))[:, None].to(device),
        image_mask=torch.ones(len(examples), 1, dtype=torch.bool, device=device),
        tokens=torch.from_numpy(np.stack([x[0] for x in text])).long().to(device),
        token_mask=torch.from_numpy(np.stack([x[1] for x in text])).bool().to(device),
    )
    query_mask = torch.from_numpy(np.asarray([x[2] for x in text])).to(device)
    result = capture_native_attention(model, inputs, query_mask)
    spatial = result.spatial.float().cpu().numpy()
    mass = result.visual_mass.float().cpu().numpy()
    valid = result.valid.cpu().numpy()
    if tuple(result.layers) != tuple(range(model.config.depth)):
        raise ValueError("Missing native layers")
    if spatial.shape[-1] != 256:
        raise ValueError(f"Unexpected native patch count: {spatial.shape[-1]}")
    out_maps = np.zeros((len(rows), *spatial.shape[1:]), np.float16)
    out_mass = np.zeros((len(rows), *mass.shape[1:]), np.float32)
    out_valid = np.zeros((len(rows), len(METHODS)), bool)
    for i, dst in enumerate(examples):
        use = valid[i]
        if not np.array_equal(use, indexes[i]) or np.any(out_valid[dst] & use):
            raise ValueError(f"Capture coverage mismatch at {rows[dst]['id']}")
        out_maps[dst, use] = spatial[i, use]
        out_mass[dst, use] = mass[i, use]
        out_valid[dst, use] = True
    if not np.isfinite(out_maps).all() or not np.isfinite(out_mass).all():
        raise ValueError("Nonfinite native attention")
    if np.any(out_mass < 0) or np.any(out_mass > 1.0001):
        raise ValueError("Invalid visual attention mass")
    if out_valid.any() and not np.allclose(out_maps[out_valid].sum(-1, dtype=np.float32), 1, atol=.002):
        raise ValueError("Conditional vision maps do not sum to one")
    return out_maps, out_mass, out_valid


def run_split(model, tokenizer, args, split, device, batch, *, output_root, limit=None):
    path = args.run / (split + "_rows.json")
    all_rows = read(path)
    if len({x["id"] for x in all_rows}) != len(all_rows):
        raise ValueError(f"Duplicate IDs in {path}")
    rows = all_rows[args.shard::args.shards]
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError(f"No rows for {split} shard {args.shard}")
    folder = output_root / split
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"shard_{args.shard:03d}.npz"
    info = dest.with_suffix(".json")
    if dest.exists():
        raise FileExistsError(dest)
    maps, mass, valid = [], [], []
    start = time.monotonic()
    for offset in range(0, len(rows), batch):
        results = capture_batch(model, tokenizer, rows[offset:offset + batch], device)
        for name, value in zip((maps, mass, valid), results, strict=True):
            name.append(value)
        done = min(offset + batch, len(rows))
        if offset % (batch * 10) == 0 or done == len(rows):
            print(f"{split} shard {args.shard}: {done}/{len(rows)}, {time.monotonic() - start:.1f}s", flush=True)
    tmp = dest.with_suffix(".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, sample_ids=np.asarray([row["id"] for row in rows]),
                            methods=np.asarray(METHODS), maps=np.concatenate(maps, 0),
                            mass=np.concatenate(mass, 0), valid=np.concatenate(valid, 0))
    tmp.replace(dest)
    write(info, dict(shard=args.shard, shards=args.shards, split=split, rows=len(rows),
                     python=sys.version.split()[0], torch=torch.__version__, device=torch.cuda.get_device_name(device),
                     hostname=socket.gethostname(), job_id=os.getenv("LSB_JOBID"), pid=os.getpid(),
                     cuda_visible_devices=os.getenv("CUDA_VISIBLE_DEVICES"),
                     source_rows_sha256=digest(path), elapsed_seconds=time.monotonic() - start,
                     seed=0, valid_per_method=dict(zip(METHODS, np.concatenate(valid, 0).sum(0).tolist(), strict=True))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--mode", choices=("smoke", "dev", "test", "both"), default="both")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--selection-timeout", type=int, default=1200)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error("Invalid shard index")
    config = read(args.run / "config.json")
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    print(f"loading original π0.5, shard {args.shard}/{args.shards}: {config['base_weights']}", flush=True)
    cfg = Pi0Config(action_horizon=10, gradient_checkpointing=False, discrete_state_input=False,
                    image_size=config["model_image_size"], patch_size=config["patch_size"],
                    heads=config["head_count"], depth=config["layer_count"], max_token_len=config["max_token_len"])
    model = Pi0(cfg)
    model.load_base_weights(config["base_weights"])
    model.to(device, dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = TokenizePrompt(config["tokenizer"], cfg.max_token_len)
    print(f"frozen weight parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    batch = args.batch or config["batch"]
    if args.mode == "smoke" or (args.mode == "both" and args.shard == 0):
        run_split(model, tokenizer, args, "dev", device, batch, output_root=args.run / "technical", limit=2)
        if args.mode == "smoke":
            return
    if args.mode in ("dev", "both"):
        run_split(model, tokenizer, args, "dev", device, batch, output_root=args.run / "captures")
    if args.mode in ("test", "both"):
        lock = args.run / "selection.json"
        deadline = time.monotonic() + args.selection_timeout
        while not lock.is_file():
            if (args.run / "metadata/selection_failed").exists():
                raise RuntimeError("Development head selection failed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Missing independently frozen development selection")
            time.sleep(2)
        selection_sha = digest(lock)
        print("formal selection SHA:", selection_sha, flush=True)
        run_split(model, tokenizer, args, "test", device, batch, output_root=args.run / "captures")
        if digest(lock) != selection_sha:
            raise RuntimeError("Head selection changed during formal capture")


if __name__ == "__main__":
    main()
