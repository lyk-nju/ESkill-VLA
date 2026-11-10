"""Compute statistics once per recorded training row, before horizon expansion."""
import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from openpi.training.config import TrainConfig, split_episodes
from openpi.training.data_loader import input_adapter


def statistics(values, valid, method):
    """Per-dimension moments and 1%/99% quantiles over valid, finite values; the method is stored."""
    values = np.asarray(values, np.float64)
    valid = np.broadcast_to(valid, values.shape) & np.isfinite(values)
    count = valid.sum(0)
    shape = values.shape[1:]
    mean, std, q01, q99 = np.zeros(shape), np.ones(shape), np.full(shape, -1.), np.ones(shape)
    for index in np.ndindex(shape):
        column = values[(slice(None), *index)][valid[(slice(None), *index)]]
        if len(column):
            mean[index], std[index] = column.mean(), max(column.std(), 1e-6)
            q01[index], q99[index] = np.quantile(column, [.01, .99])
    return dict(method=method, mean=mean.tolist(), std=std.tolist(), q01=q01.tolist(), q99=q99.tolist(),
                count=count.tolist())


def compute(config):
    """Training-split statistics from recorded rows, once per row, before action windows."""
    index_path = Path(config.data_index)
    selected = set(split_episodes(index_path, config.validation_fraction, config.seed, config.split_path)["train"])
    if not selected:
        raise ValueError("No training episodes in this split")
    adapter = input_adapter(config)
    if config.model_type == 'eskill':
        # Body input stays in physical units. Statistics require neither images
        # nor a live camera projection; state/actions retain the SFT normalization.
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.policies.robotwin_policy import RobotwinInputs
        adapter = {'libero': LiberoInputs, 'robotwin': RobotwinInputs}[config.platform](use_arm_state=False)
    columns = {key: ([], []) for key in ("state", "arm_state", "actions")}
    quality_warnings = {}
    for entry in json.loads(index_path.read_text())["episodes"]:
        if entry["episode_id"] not in selected:
            continue
        reader = EpisodeReader(index_path.parent / entry["path"])
        seq = reader.meta["state"]
        # Numeric fields only. No RGB decoding, Event targets, or repeated action windows.
        rows = [adapter(dict(state=row, state_valid=mask, images=[], prompt=""))
                for row, mask in zip(reader.array(seq["values"]), reader.array(seq["valid"]), strict=True)]
        for name in ("state", "arm_state"):
            if name in rows[0]:
                columns[name][0].append(np.stack([r[name] for r in rows]))
                columns[name][1].append(np.stack([r[name + "_valid"] for r in rows]))
        actions = reader.meta["actions"]
        columns["actions"][0].append(reader.array(actions["values"]))
        columns["actions"][1].append(reader.array(actions["valid"]))
        for warning in reader.meta.get("quality", {}).get("warnings", []):
            quality_warnings[warning] = quality_warnings.get(warning, 0) + 1
    result = {key: statistics(np.concatenate(values), np.concatenate(valid), config.norm_method)
              for key, (values, valid) in columns.items() if values}
    result["provenance"] = dict(
        data_index=str(index_path.resolve()), split="train", seed=config.seed,
        split_path=config.split_path,
        validation_fraction=config.validation_fraction, episode_ids=sorted(selected),
        frames=sum(len(v) for v in columns["actions"][0]),
        source_commands="unchanged", quality_warnings=quality_warnings,
        manual_review="unchanged; not an acceptance decision")
    if config.init_checkpoint:
        import torch
        from openpi.training.checkpoints import check_sft_split
        saved = torch.load(config.init_checkpoint, map_location="cpu", weights_only=False)
        check_sft_split(saved, config)
        # Keep the policy's input/output units across stages; only arm_state is new.
        previous = saved["norm_stats"]
        for key in ("state", "actions"):
            if previous[key].get("method") != config.norm_method:
                raise ValueError(f"SFT {key} statistics must explicitly use {config.norm_method}; convert old metadata first")
        result.update({key: previous[key] for key in ("state", "actions")})
        result["provenance"]["sft_checkpoint"] = str(Path(config.init_checkpoint).resolve())
        result["provenance"]["inherited_statistics"] = previous.get("provenance")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = TrainConfig.load(args.config)
    destination = Path(config.norm_stats)
    result = compute(config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A new split should have a new statistics file; do not replace a training run's assets.
    with destination.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Saved training-only normalization statistics to {destination}")


if __name__ == "__main__":
    main()
