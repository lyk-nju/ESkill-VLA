"""Compute statistics once per recorded training row, before horizon expansion."""
import argparse
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from openpi.policies.libero_policy import LiberoInputs
from openpi.policies.robotwin_policy import RobotwinInputs
from openpi.training.config import TrainConfig, split_episodes


class Moments:
    def __init__(self):
        self.count = self.mean = self.m2 = None

    def update(self, values, valid):
        values = np.asarray(values, np.float64)
        valid = np.broadcast_to(valid, values.shape) & np.isfinite(values)
        count = valid.sum(0)
        mean = np.where(valid, values, 0).sum(0) / np.maximum(count, 1)
        m2 = np.where(valid, (values - mean) ** 2, 0).sum(0)
        if self.count is None:
            self.count, self.mean, self.m2 = count, mean, m2
            return
        total = self.count + count
        delta = mean - self.mean
        self.m2 += m2 + delta ** 2 * self.count * count / np.maximum(total, 1)
        self.mean += delta * count / np.maximum(total, 1)
        self.count = total

    def result(self):
        std = np.sqrt(self.m2 / np.maximum(self.count, 1))
        return dict(mean=self.mean.tolist(),
                    std=np.where(self.count > 0, np.maximum(std, 1e-6), 1.).tolist(),
                    count=self.count.tolist())


def compute(config):
    index_path = Path(config.data_index)
    selected = set(split_episodes(index_path, config.validation_fraction, config.seed)["train"])
    adapter = {"libero": LiberoInputs, "robotwin": RobotwinInputs}[config.platform]()
    moments = {key: Moments() for key in ("state", "arm_state", "actions")}
    quality_warnings = {}
    for entry in json.loads(index_path.read_text())["episodes"]:
        if entry["episode_id"] not in selected:
            continue
        reader = EpisodeReader(index_path.parent / entry["path"])
        seq = reader.meta["state"]
        values, valid = reader.array(seq["values"]), reader.array(seq["valid"])
        # Numeric fields only. No RGB decoding, Event targets, or repeated action windows.
        rows = [adapter(dict(state=row, state_valid=mask, image=None, prompt=""))
                for row, mask in zip(values, valid, strict=True)]
        for name in ("state", "arm_state"):
            moments[name].update(np.stack([r[name] for r in rows]),
                                 np.stack([r[name + "_valid"] for r in rows]))
        actions = reader.meta["actions"]
        moments["actions"].update(reader.array(actions["values"]), reader.array(actions["valid"]))
        for warning in reader.meta.get("quality", {}).get("warnings", []):
            quality_warnings[warning] = quality_warnings.get(warning, 0) + 1
    if not selected:
        raise ValueError("No training episodes in this split")
    result = {key: value.result() for key, value in moments.items()}
    result["provenance"] = dict(
        data_index=str(index_path.resolve()), split="train", seed=config.seed,
        validation_fraction=config.validation_fraction, episode_ids=sorted(selected),
        source_commands="unchanged", quality_warnings=quality_warnings,
        manual_review="unchanged; not an acceptance decision")
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
