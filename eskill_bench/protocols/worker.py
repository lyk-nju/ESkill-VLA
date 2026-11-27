"""Worker process: import exactly one model generation per invocation."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from eskill_bench.protocols.assets import digest, read, write_new


def planner(job, root):
    import scripts.analysis

    scripts.analysis.__path__.append(str(root / "source/motion_v1_frozen/scripts/analysis"))
    from scripts.analysis.motion_planner_eval import episode_rows, pair_row, specification
    from scripts.analysis.motion_planner_metrics import factual_metrics, pair_metrics, timeline_metrics
    from scripts.analysis.motion_planner_report import write_csv
    from openpi.policies.motion_planner import create_motion_planner_policy
    from openpi.models.model import to_device
    from openpi.training.checkpoints import resource_identity

    manifest = read(job["manifest"])
    saved = torch.load(job["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    step = int(saved["step"])
    if job["formal_weights"]:
        index, split = (manifest["references"][k] for k in ("data_index", "split"))
        if (
            step != 20000
            or saved["resources"]["split_path"] != split["sha256"]
            or digest(split["path"]) != split["sha256"]
        ):
            raise ValueError("Unexpected Planner step or training split")
        if digest(index["path"]) != index["sha256"] or saved["resources"]["data_index"] != resource_identity(
            "data_index", index["path"]
        ):
            raise ValueError("Planner checkpoint does not match locked episode index and metadata")
    del saved

    policy = create_motion_planner_policy(
        job["checkpoint"],
        device=job["device"],
        tokenizer_path=job["tokenizer"],
        pretrained_path=job["base"],
    )
    spec = manifest["spec"]
    if specification(policy.model.config) != spec or policy.model.config.planner_joint != (
        job["variant"] == "joint"
    ):
        raise ValueError("Checkpoint has different routing or trajectory specification")
    if (getattr(policy.model.config, "planner_version", "trajectory") in ("boundary_first", "v2")) != (job["variant"] == "boundary"):
        raise ValueError("Checkpoint has different boundary slots")

    class Capture:
        def batch_features(self, observations):
            samples = [
                policy.transform.inputs(dict(images=[im], image_mask=np.ones(1, bool), prompt=t))
                for im, t in observations
            ]
            length = max(int(s["token_mask"].sum()) for s in samples)
            batch = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
            for k in ("tokens", "token_mask"):
                batch[k] = batch[k][:, :length]
            tensors = to_device({k: torch.from_numpy(v) for k, v in batch.items()}, policy.device)
            with (
                torch.inference_mode(),
                torch.autocast(
                    policy.device.type,
                    dtype=torch.bfloat16,
                    enabled=policy.precision == "bfloat16",
                ),
            ):
                self.features = policy.model.plan_features(tensors)
            return self.features

        def read_plan(self, *args, **kwargs):
            return policy.read_plan(*args, **kwargs)

    capture = Capture()
    rows, endpoints = [], []
    out = Path(job["out"])
    if job["factual"]:
        with (out / "factual.jsonl").open("x") as stream:
            for ep in manifest["episodes"]:
                for row in episode_rows(capture, ep, spec, 8):
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
        tables = factual_metrics(rows, manifest)
        counts = tables.pop("counts")
        for name, table in dict(tables, timeline=timeline_metrics(rows, manifest)).items():
            write_csv(out / (name + ".csv"), table)
        write_new(
            out / "coverage.json",
            dict(
                step=step, episodes=len(manifest["episodes"]), observations=len(rows), pairs=0, counts=counts
            ),
        )
        return
    with (out / "pairs.jsonl").open("x") as stream:
        for case in manifest["pairs"]:
            row = pair_row(capture, case, spec)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
            if job["variant"] == "boundary":
                endpoints.append(boundary_response(policy, capture.features, row, spec["size"]))
    if endpoints:
        write_csv(out / "boundary_pair_cases.csv", endpoints)
    details, groups = pair_metrics(rows, spec, manifest["pairs"])
    write_csv(out / "pair_cases.csv", details)
    write_csv(out / "pair_summary.csv", groups)
    write_new(out / "coverage.json", dict(step=step, pairs=len(rows), observations=0))


def boundary_response(policy, features, row, size):
    native = policy.read_plan(features)

    def coordinates(name):
        return native[name].detach().float().cpu().numpy()

    start, end = coordinates("start"), coordinates("end")
    boxes = np.asarray(row["boxes"])
    axis = (boxes[1, :2] + boxes[1, 2:] - boxes[0, :2] - boxes[0, 2:]) / 2
    result = dict(
        id=row["id"],
        kind=row["kind"],
        suite=row["suite"],
        start_a=start[0].tolist(),
        start_b=start[1].tolist(),
        end_a=end[0].tolist(),
        end_b=end[1].tolist(),
        start_drift_px=float(np.linalg.norm(start[1] - start[0]) * size),
    )
    if np.linalg.norm(axis) > 1e-6:
        direction = axis / np.linalg.norm(axis)
        result["end_toward_target_px"] = float((end[1] - end[0]) @ direction * size)
        if row["interventions"]:
            a = {k: v[:1] for k, v in features.items()}
            swapped = policy.read_plan(
                a,
                skill_id=native["skill_id"][:1],
                end_latent=native["boundary_latents"][1:2, 1:2],
            )
            delta = swapped["trace"][0].float().cpu().numpy() - np.asarray(row["a"]["trace"])
            result["end_swap_trace_toward_target_px"] = float(delta[-4:].mean(0) @ direction * size)
    return result


if __name__ == "__main__":
    job = read(sys.argv[1])
    planner(job, Path(job["bundle"]))
