"""Protocol-specific episode populations, resume keys, and native success summaries."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import runpy
import subprocess

from .assets import digest, read, write_new

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SAMPLE = "source/benchmarks/ESkill-VLA/scripts/evaluation"


def jobs_for(s, protocol, args):
    old = runpy.run_path(str(s.bundle.file(SAMPLE + "/sample_protocol.py")))
    if protocol in ("sample", "plus280"):
        if not args.jobs or not args.variants:
            raise ValueError("Supply original job-plan.json and selected-variants.json")
        if (
            digest(args.jobs) != "6651646fa57dcae7d0acba14cc2bbd4c606388c1784860000b1e6aa1109f5b85"
            or digest(args.variants) != "513d6d2de853ac00e6be559f3fe902bb6714f0d8ef2cc093c1ef41b9b6250f48"
        ):
            raise ValueError("Changed legacy job or variant selection")
        jobs = old["build_jobs"](read(args.jobs), read(args.variants)["variants"])
        keys = old["episode_keys"](jobs)
        if Counter(k[0] for k in keys) != {"libero": 200, "libero_plus": 280}:
            raise ValueError("Unexpected sampled episode inventory")
        return [j for j in jobs if protocol == "sample" or j["benchmark"] == "libero_plus"]
    if protocol == "libero200":
        return [
            dict(benchmark="libero", suite=suite, task_ids=[i], trials_per_task=5)
            for suite in SUITES
            for i in range(10)
        ]
    classification = read(s.paths["libero_plus"] / "libero/libero/benchmark/task_classification.json")
    counts = {suite: len(classification[suite]) for suite in SUITES}
    if sum(counts.values()) != 10030:
        raise ValueError("Wrong Plus variant inventory")
    return [
        dict(benchmark="libero_plus", suite=suite, task_ids=[i], trials_per_task=1)
        for suite, n in counts.items()
        for i in range(n)
    ]


def keyset(job):
    return {
        (job["benchmark"], job["suite"], t, i) for t in job["task_ids"] for i in range(job["trials_per_task"])
    }


def outcomes(root, jobs, protocol=None):
    wanted = set().union(*(keyset(j) for j in jobs))
    found = {}
    checked = set()
    if protocol and protocol["protocol"] == "plus-full":
        # The policy RNG is continuous over all trials. An interrupted and restarted stream is a new attempt.
        attempts = sorted((root / "full").glob("attempt_*"))
        paths = sorted(attempts[-1].glob("records/*.json")) if attempts else []
    else:
        paths = sorted(root.glob("batch_*/attempt_*/records/*.json"))
    for path in paths:
        attempt = path.parents[1]
        if protocol is not None and attempt not in checked:
            config = read(attempt / "config.json")
            metadata = config["server_metadata"]
            if Path(metadata["checkpoint"]).resolve() != Path(protocol["checkpoint"]).resolve():
                raise ValueError(f"Checkpoint differs in {attempt}")
            if config.get("purpose") == "integration_check" or metadata.get("technical_only"):
                raise ValueError("Technical rollout included in formal population")
            if protocol["protocol"] != "plus-full" and metadata.get("policy_rng_scope") != "episode":
                raise ValueError("Sampled protocol requires server-side per-episode RNG reset")
            if protocol["protocol"] == "plus-full" and metadata.get("policy_rng_scope") == "episode":
                raise ValueError("Full Plus requires a continuous policy RNG stream")
            checked.add(attempt)
        r = read(path)
        folder = path.parents[2].name
        job = None if folder == "full" else jobs[int(folder.split("_")[-1])]
        key = (
            "libero_plus" if job is None else job["benchmark"],
            r["suite"],
            r["task_id"],
            r["trial"],
        )
        if (
            key not in (wanted if job is None else keyset(job))
            or r["initial_state_id"] != r["trial"]
            or r["seed"] != 7
            or r.get("purpose") == "integration_check"
            or type(r["success"]) is not bool
        ):
            raise ValueError(f"Wrong rollout identity or scope: {path}")
        status = "error" if r["error"] is not None else "success" if r["success"] else "failure"
        if key in found and found[key] != "error":
            if status != "error" and found[key] != status:
                raise ValueError(f"Conflicting retry outcomes: {key}")
        else:
            found[key] = status
    result = {}
    for bench in sorted({k[0] for k in wanted}):
        total = sum(k[0] == bench for k in wanted)
        c = Counter(v for k, v in found.items() if k[0] == bench)
        missing = total - sum(c.values())
        recorded = sum(c.values())
        full = protocol and protocol["protocol"] == "plus-full"
        result[bench] = dict(
            planned=total,
            completed=c["success"] + c["failure"],
            success=c["success"],
            failure=c["failure"],
            error=c["error"],
            unfinished=missing,
            recorded_micro_rate=c["success"] / recorded if recorded else None,
            success_rate=c["success"] / total if not missing and (full or not c["error"]) else None,
        )
    return found, result


def call_module(s, python, module, args, paths, env=None):
    env = dict(
        os.environ,
        **(env or {}),
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run([str(python), "-m", module, *map(str, args)], cwd=paths[0], env=env, check=True)


def render_device(device):
    """Translate the worker's process-local CUDA ordinal to the EGL device ordinal."""
    if device == "cpu" and "MUJOCO_EGL_DEVICE_ID" in os.environ:
        return int(os.environ["MUJOCO_EGL_DEVICE_ID"])
    kind, _, ordinal = device.partition(":")
    if kind not in ("cpu", "cuda"):
        raise ValueError(f"Expected a CUDA or CPU policy device, got {device}")
    index = int(ordinal or "0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return index
    devices = visible.split(",")
    if not 0 <= index < len(devices) or not devices[index].strip().isdigit():
        raise ValueError("Set --render-gpu-device-id to the physical EGL index for this worker")
    return int(devices[index])


def rollout(s, command):
    proto, phase = command.protocol, command.phase
    composition = s.paths["composition"]
    if proto == "unseen-composition":
        spec = read(composition / "experiments/unseen_composition/spec.json")
        old = read(composition / "docs/history/experiments/skill_reuse/nine_combinations.json")
        goals = {t["task_id"]: t["goal_predicates"] for t in spec["tasks"]}
        scene_names = {
            "goal_native": "native_goal",
            "goal_replace_cheese_with_milk": "milk",
            "goal_replace_cheese_with_pudding": "chocolate_pudding",
        }
        new = [
            (
                scene_names[r["scene_id"]],
                r["task_id"],
                r["initial_indices"],
                goals[r["task_id"]],
            )
            for r in spec["evaluated_cases"]
        ]
        expected = [
            (
                r["scene"],
                r["task_id"].removeprefix(r["scene"] + "_scene__"),
                r["initial_indices"],
                r["goal_predicates"],
            )
            for r in old["tasks"]
        ]
        if sorted(new) != sorted(expected) or spec["eval_seeds"] != [old["eval_seed"]]:
            raise ValueError("Nine-combination cases differ from the original fixed study")
        if "--spec" in command.args or any(x.startswith("--spec=") for x in command.args):
            raise ValueError("Historical nine combinations uses the locked default spec")
        return call_module(
            s,
            s.paths["sim_python"],
            "libero_composition",
            [phase, *command.args],
            [composition],
        )
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--jobs", type=Path)
    parser.add_argument("--variants", type=Path)
    parser.add_argument("--checkpoint-tag", required=phase in ("plan", "serve"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--render-gpu-device-id", type=int, help="Physical EGL device index for the client")
    args = parser.parse_args(command.args)
    root = s.paths["output"] / args.run
    planfile = root / "protocol.json"
    if phase == "plan":
        jobs = jobs_for(s, proto, args)
        root.mkdir(parents=True, exist_ok=False)
        ckpt = Path(args.checkpoint_tag).resolve()
        write_new(
            planfile,
            dict(
                protocol=proto,
                jobs=jobs,
                checkpoint=str(ckpt),
                checkpoint_sha256=digest(ckpt) if ckpt.is_file() else None,
                sources={str(p): digest(p) for p in (args.jobs, args.variants) if p},
                seed=7,
            ),
        )
        print(planfile)
        return
    plan = read(planfile)
    if plan["protocol"] != proto or (
        args.checkpoint_tag and Path(args.checkpoint_tag).resolve() != Path(plan["checkpoint"])
    ):
        raise ValueError("Wrong protocol or checkpoint for this run")
    jobs = plan["jobs"]
    found, summary = outcomes(root, jobs, plan)
    if phase in ("status", "report"):
        print(json.dumps(summary, indent=2))
        return
    src = s.bundle.root / "source/benchmarks/ESkill-VLA"
    if phase in ("serve", "run") and (
        not plan.get("checkpoint_sha256") or digest(plan["checkpoint"]) != plan["checkpoint_sha256"]
    ):
        raise ValueError("Formal execution requires a content-identified checkpoint at plan time")
    if phase == "serve":
        server = Path(__file__).resolve().parents[2] / "scripts/serve_policy.py"
        subprocess.run(
            [
                str(s.paths["python"]), str(server), "--runtime", "local",
                "--policy-source", str(s.paths["policy_source"]),
                "--checkpoint", plan["checkpoint"], "--tokenizer-path", str(s.paths["tokenizer"]),
                "--device", s.device, "--openpi-root", str(s.paths["openpi"]),
                "--port", str(args.port), "--seed", "7", "--fm-steps", "10",
                *(["--reset-per-episode"] if proto != "plus-full" else []),
            ],
            check=True,
        )
        return
    if phase != "run" or not 0 <= args.worker < args.workers:
        raise ValueError("Expected run with a valid worker index")
    s.bundle.verify()
    egl = args.render_gpu_device_id if args.render_gpu_device_id is not None else render_device(s.device)
    if egl < 0:
        raise ValueError("Expected a non-negative EGL device index")
    render_env = dict(os.environ, MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID=str(egl), CUDA_VISIBLE_DEVICES=str(egl))
    vendor = s.paths["sim_configs"] / "nvidia_egl.json"
    if vendor.is_file():
        render_env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(vendor))

    if proto == "plus-full":
        if args.workers != 1:
            raise ValueError("Full Plus uses one ordered client and a continuous policy RNG stream")
        if summary["libero_plus"]["unfinished"] or summary["libero_plus"]["error"]:
            folder = root / "full"
            folder.mkdir(exist_ok=True)
            attempt = folder / f"attempt_{len(list(folder.glob('attempt_*'))):04}"
            subprocess.run(
                [
                    str(s.paths["plus_python"]),
                    str(src / "examples/libero_plus/main.py"),
                    "--render-gpu-device-id", str(egl),
                    "--task-suite-name",
                    "all",
                    "--libero-config",
                    str(s.paths["sim_configs"] / "libero_plus"),
                    "--num-trials-per-task",
                    "1",
                    "--port",
                    str(args.port),
                    "--seed",
                    "7",
                    "--checkpoint-tag",
                    plan["checkpoint"],
                    "--output",
                    str(attempt),
                ],
                env=render_env, check=True,
            )
        print(json.dumps(outcomes(root, jobs, plan)[1], indent=2))
        return
    for i, job in enumerate(jobs):
        if i % args.workers != args.worker or all(
            found.get(k) in ("success", "failure") for k in keyset(job)
        ):
            continue
        bench = job["benchmark"]
        py = s.paths["plus_python" if bench == "libero_plus" else "sim_python"]
        folder = root / f"batch_{i:05}"
        folder.mkdir(exist_ok=True)
        attempt = folder / f"attempt_{len(list(folder.glob('attempt_*'))):04}"
        argv = [
            "--render-gpu-device-id", str(egl),
            "--port",
            str(args.port),
            "--task-suite-name",
            job["suite"],
            "--libero-config",
            str(s.paths["sim_configs"] / bench),
            "--task-ids",
            ",".join(map(str, job["task_ids"])),
            "--num-trials-per-task",
            str(job["trials_per_task"]),
            "--seed",
            "7",
            "--replan-steps",
            "5",
            "--num-steps-wait",
            "10",
            "--checkpoint-tag",
            plan["checkpoint"],
            "--output",
            str(attempt),
            "--video",
            "all",
        ]
        env = dict(render_env, PYTHONPATH=str(src))
        subprocess.run(
            [
                str(py),
                str(s.bundle.file(SAMPLE + "/sample_client.py")),
                "--source",
                str(src),
                "--benchmark",
                bench,
                *argv,
            ],
            env=env,
            check=True,
        )
    print(json.dumps(outcomes(root, jobs, plan)[1], indent=2))
