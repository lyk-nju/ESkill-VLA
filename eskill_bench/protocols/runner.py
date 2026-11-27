"""Version-specific historical inference. Source snapshots and scoring execute in isolation."""

import os
from functools import partial
from pathlib import Path
import subprocess
import sys

from PIL import Image
from eskill_bench import ROOT, revision

from .assets import (
    KLAL,
    MOTION,
    NATIVE,
    VISUAL,
    digest,
    git_source,
    locked_paths,
    read,
    rgb,
    write_new,
)
from .motion import RUNS

STAGE = "source/stage1/experiments/stage1_visual_localization"
NATIVE_CODE = "run_code/native_grounding/run_001/source"
PROBE_CODE = "source/native_grounding/experiments/vlm_capability_diagnosis/representation"
SOURCES = dict(
    bbox="legacy_bbox_gac",
    bbox_gac="legacy_bbox_gac",
    legacy_vae="legacy_vae",
    vae_gac="legacy_bbox_gac",
    klal="klal_full_vlm_cf",
    frozen="motion_v1_frozen",
    joint="motion_v1_joint",
    boundary="motion_v2_boundary",
)


def execute(settings, entry, args, paths, *, device=None, log=None):
    interpreter = settings.paths.get("python", sys.executable)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONDONTWRITEBYTECODE="1",
    )
    if device and device.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]
    command = [str(interpreter), str(entry), *map(str, args)]
    if log is None:
        subprocess.run(command, env=env, check=True)
    else:
        with log.open("a") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def select_source(bundle, variant, identity, override=None):
    if identity:
        # Resolve actual source fingerprint, including signatures changed at the same architecture name.
        candidates = [
            bundle.root / "source" / name
            for name in dict.fromkeys(
                [
                    SOURCES[variant],
                    "bbox_vae_full_vlm",
                    "legacy_vae",
                    "visual_primitive",
                ]
            )
        ]
        for src in ([override] if override else []) + candidates:
            if all(
                (src / f).is_file() and digest(src / f) == sha for f, sha in identity["source_sha256"].items()
            ):
                return src
        raise ValueError(f"{variant} requires the source checkout matching {identity['source_sha256']}")
    return bundle.root / "source" / SOURCES[variant]


def inference(
    settings,
    variant,
    checkpoint,
    output,
    limit=None,
    dev_frame=False,
    technical_checkpoint=False,
    factual=False,
):
    from .audit import verify_locked

    b = settings.bundle
    b.verify()
    verify_locked(b)
    checkpoint = Path(checkpoint).resolve()
    out = Path(output).resolve()
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    motion = variant in ("frozen", "joint", "boundary")
    if factual and variant != "frozen":
        raise ValueError("The historical factual evaluation belongs to the frozen Planner")
    identity = None
    expected = None
    actual = digest(checkpoint)
    if motion:
        old = RUNS[("frozen", "joint", "boundary").index(variant)]
        expected = read(b.file("reference/motion/" + old + "/run.json"))["checkpoint_sha256"]
    elif variant != "klal":
        run = "run_002" if variant == "vae_gac" else "run_001"
        ident = read(b.file(f"reference/visual/stage1_visual_localization/{run}/results/summary.json"))
        identity = ident["models"][variant]["identity"]
        ten_k = read(b.file("reference/visual/stage1_visual_localization/run_003/results/summary.json"))
        if actual == ten_k["models"][variant]["identity"]["checkpoint_sha256"]:
            identity = ten_k["models"][variant]["identity"]
        expected = identity["checkpoint_sha256"]
    elif "klal_sha256" in settings.expected:
        expected = settings.expected["klal_sha256"]
    elif not technical_checkpoint:
        raise ValueError("Formal KLAL inference requires expected.klal_sha256")
    if technical_checkpoint and not (dev_frame or limit):
        raise ValueError("A technical checkpoint requires a bounded smoke run")
    if expected and actual != expected and not technical_checkpoint:
        raise ValueError(f"Wrong {variant} checkpoint: expected {expected}, got {actual}")
    src = select_source(b, variant, identity, settings.paths.get(variant + "_source"))
    if (
        identity
        and identity["readout"] == "vae_support_centroid"
        and not (src / "scripts/preprocess/libero/render.py").is_file()
    ):
        if not identity.get("revision"):
            raise ValueError(
                "Provide the complete matching VAE checkout, including its rendering dependencies"
            )
        full = git_source(settings.paths["output"], identity["revision"])
        if any(digest(full / f) != sha for f, sha in identity["source_sha256"].items()):
            raise ValueError("Full VAE source does not match the historical loader/decoder")
        src = full
    paths = [src / "src", src, Path(__file__).resolve().parents[2]]
    device = "cuda:0" if settings.device.startswith("cuda:") else settings.device
    if dev_frame and motion:
        raise ValueError("The native-development RGB smoke uses the KLAL rows interface")
    # All three legacy runs sealed the very same manifest bytes. Resolve RGB against the collecting run.
    key = MOTION if motion else KLAL if variant == "klal" else VISUAL
    manifest, mapping = locked_paths(settings, key)
    if dev_frame:
        key = NATIVE + "inputs/dev.json"
        dev, mapping = locked_paths(settings, key)
        row = dev[0]
        from src.openpi.policies.eskill_dataset import EpisodeReader
        import hashlib

        reader = EpisodeReader(row["episode_ref"])
        image = reader.rgb("image", row["frame"])
        if image is None or hashlib.sha256(image.tobytes()).hexdigest() != row["rgb_sha256"]:
            raise ValueError("Restored development frame RGB differs")
        region = reader.region(reader.meta["effector_regions"]["single"]["image"], row["frame"])
        manifest = dict(
            protocol="technical-native-dev",
            metric_space="source",
            rows=[
                dict(
                    id=row["id"],
                    split=row["split"],
                    parent=row["parent_id"],
                    scene=row["suite"],
                    task=row["prompt"],
                    target=row["target"],
                    boxes={row["target"]: row["current_bbox"]},
                    subject_box=region["bbox"].tolist(),
                    image=str(out / "dev.png"),
                )
            ],
        )
        if variant != "klal":
            manifest = dict(
                mode="technical",
                coverage=[],
                records=[
                    dict(
                        row,
                        image=str(out / "dev.npz"),
                        subject_bbox=region["bbox"].tolist(),
                    )
                ],
            )
    group = "pairs" if motion else "rows" if variant == "klal" else "records"
    if factual:
        group = "episodes"
    planned = len(manifest[group])
    if limit:
        candidates = manifest[group][:limit]
        if group == "records":
            # The legacy paired report reads both sides. Include any partner of a requested query.
            parents = {r["id"]: r for r in manifest[group]}
            wanted = {r["id"] for r in candidates}
            for r in list(candidates):
                while r.get("paired_id") and r["paired_id"] not in wanted:
                    wanted.add(r["paired_id"])
                    r = parents[r["paired_id"]]
            candidates = [r for r in manifest[group] if r["id"] in wanted]
        manifest[group] = candidates
    technical = bool(technical_checkpoint or dev_frame or (limit and limit < planned))
    if technical and "mode" in manifest:
        manifest["mode"] = "technical"
    out.mkdir(parents=True, exist_ok=False)
    if dev_frame:
        if variant == "klal":
            Image.fromarray(image).save(out / "dev.png")
        else:
            import numpy as np

            np.savez_compressed(out / "dev.npz", main=image)
    local = out / "manifest.json"
    write_new(local, manifest)
    meta = dict(
        variant=variant,
        protocol=key,
        source=str(src),
        original_sha256=b.files[key]["sha256"],
        local_sha256=digest(local),
        path_map=mapping,
        checkpoint=str(checkpoint),
        checkpoint_sha256=actual,
        benchmark_revision=revision(ROOT),
        planned=planned,
        requested=len(manifest[group]),
        population=group,
        technical=technical,
        device=settings.device,
    )
    write_new(out / "inputs.json", meta)
    base, tok = settings.paths["base"], settings.paths["tokenizer"]
    invoke = partial(execute, settings, log=out / "inference.log")
    try:
        if variant == "klal" and not dev_frame:
            # Converted KLAL rows name PNGs after the source RGB hash.
            for path in {r["image"] for r in manifest["rows"]}:
                rgb(path, expected=Path(path).stem)
        if motion:
            job = dict(
                bundle=str(b.root),
                variant=variant,
                checkpoint=str(checkpoint),
                device=device,
                tokenizer=str(tok),
                base=str(base),
                out=str(out),
                manifest=str(local),
                factual=factual,
                formal_weights=not technical_checkpoint,
            )
            write_new(out / "job.json", job)
            invoke(
                Path(__file__).with_name("worker.py"),
                [out / "job.json"],
                paths,
                device=settings.device,
            )
            write_new(
                out / "run.json",
                dict(
                    manifest=str(local),
                    manifest_sha256=digest(local),
                    source_manifest_sha256=meta["original_sha256"],
                    checkpoint=str(checkpoint),
                    checkpoint_sha256=actual,
                    **read(out / "coverage.json"),
                    scope="technical" if technical else "locked",
                    benchmark_revision=meta["benchmark_revision"],
                    device=settings.device,
                ),
            )
        elif variant == "klal":
            invoke(
                src / "scripts/analysis/visual_primitive_eval.py",
                [
                    "--manifest",
                    local,
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    out / "raw",
                    "--device",
                    device,
                    "--tokenizer-path",
                    tok,
                    "--pretrained-path",
                    base,
                    "--panels",
                    0,
                ],
                paths,
                device=settings.device,
            )
        else:
            write_new(out / "plan.json", {"seed": 7})
            args = [
                out,
                "--model",
                variant,
                "--source",
                src,
                "--checkpoint",
                checkpoint,
                "--device",
                device,
                "--tokenizer",
                tok,
            ]
            if identity["readout"] == "vae_support_centroid":
                args += ["--vae", settings.paths["vae"]]
            invoke(b.file(STAGE + "/infer.py"), args, paths, device=settings.device)
            invoke(b.file(STAGE + "/report.py"), [out, "--model", variant], paths)
    except Exception as exc:
        write_new(
            out / "status.json",
            dict(
                status="failed",
                failed_runs=1,
                scope="technical" if technical else "locked",
                population=group,
                requested=len(manifest[group]),
                error=str(exc),
                log="inference.log",
            ),
        )
        raise
    write_new(
        out / "status.json",
        dict(
            status="complete",
            scope="technical" if technical else "locked",
            population=group,
            completed=len(manifest[group]),
            failed=0,
        ),
    )
