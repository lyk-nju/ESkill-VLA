"""Native head/Probe development locks and simulation adapters."""

import subprocess
from pathlib import Path

from .assets import NATIVE, digest, git_source, locked_paths, read, write_new
from .audit import verify_locked
from .runner import NATIVE_CODE, PROBE_CODE, execute


def add_commands(commands):
    p = commands.add_parser("native")
    p.add_argument(
        "phase",
        choices=(
            "prepare",
            "smoke",
            "capture-dev",
            "select",
            "use-lock",
            "capture-test",
            "score",
            "route",
        ),
    )
    p.add_argument("run")
    p.add_argument("--recipe", type=Path)
    p.add_argument("--baseline", action="append", default=[])
    p = commands.add_parser("representation")
    p.add_argument("phase", choices=("prepare", "features", "fit", "test", "tables"))
    p.add_argument("run")
    p = commands.add_parser("libero90")
    p.add_argument("phase", choices=("score", "summarize", "verify"))
    p.add_argument("run")
    p = commands.add_parser("closed")
    p.add_argument(
        "protocol",
        choices=("sample", "libero200", "plus280", "plus-full", "nine", "craft"),
    )
    p.add_argument(
        "phase",
        choices=(
            "plan",
            "run",
            "report",
            "prepare",
            "audit",
            "render",
            "serve",
            "status",
        ),
    )
    p.add_argument("args", nargs="...")
    p = commands.add_parser("video")
    p.add_argument("output", type=Path)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--seconds", type=float, default=0.75)


def native_source(s):
    # Capture code imports matching Pi0 and prompt transforms. A training checkout is optional.
    commit = s.bundle.file(NATIVE + "metadata/base_code_sha.txt").read_text().strip()
    files = (
        "src/openpi/models/pi0.py",
        "src/openpi/models/pi0_config.py",
        "src/openpi/transforms.py",
    )
    repo = Path(__file__).resolve().parents[2]
    src = s.paths.get("native_source")
    if src is None:
        src = git_source(s.paths["output"], commit)
    for f in files:
        expected = subprocess.check_output(["git", "-C", str(repo), "show", commit + ":" + f])
        if (src / f).read_bytes() != expected:
            raise ValueError(f"Native model source differs from {commit}: {f}")
    return src


def prepare_native(s, run, recipe):
    b = s.bundle
    verify_locked(b)
    if not recipe:
        raise ValueError("Supply --recipe pointing to native_grounding/run_001/config.json")
    cfg = read(recipe)
    cfg.update(base_weights=str(s.paths["base"]), tokenizer=str(s.paths["tokenizer"]))
    run.mkdir(parents=True, exist_ok=False)
    write_new(run / "config.json", cfg)
    changes = {"recipe": dict(source=str(Path(recipe).resolve()), original_sha256=digest(recipe),
                              local_sha256=digest(run / "config.json"))}
    for split in ("dev", "test"):
        key = NATIVE + "inputs/" + split + ".json"
        doc, paths = locked_paths(s, key)
        file = run / (split + "_rows.json")
        write_new(file, doc)
        (run / "inputs").mkdir(exist_ok=True)
        (run / "inputs" / (split + ".json")).symlink_to("../" + file.name)
        changes[split] = dict(
            source=key,
            original_sha256=b.files[key]["sha256"],
            local_sha256=digest(file),
            paths=paths,
        )
    write_new(run / "mapping.json", changes)


def native(s, run, args):
    if args.phase == "prepare":
        return prepare_native(s, run, args.recipe)
    b = s.bundle
    b.verify()
    if args.phase != "smoke" and read(run / "config.json").get("scope") == "technical":
        raise ValueError("A technical Native run can only produce a development smoke capture")
    for split in ("dev", "test") if args.phase in ("capture-test", "score") else ("dev",):
        doc, _ = locked_paths(s, NATIVE + "inputs/" + split + ".json")
        if read(run / (split + "_rows.json")) != doc:
            raise ValueError(f"{split} differs from the relocated locked input")
    if args.phase == "use-lock":
        lock = read(b.file(NATIVE + "selection.json"))
        lock["dev_manifest_sha256"] = digest(run / "dev_rows.json")
        write_new(run / "selection.json", lock)
        return
    if args.phase in ("capture-test", "score"):
        old = read(b.file(NATIVE + "selection.json"))
        new = read(run / "selection.json")
        if (
            new["test_used"]
            or new["dev_manifest_sha256"] != digest(run / "dev_rows.json")
            or any(new[k] != old[k] for k in ("policies", "primary", "hit_threshold_px"))
        ):
            raise ValueError("Formal replay requires the archived development selection")
    op = {
        "smoke": ("capture.py", [run, "--mode", "smoke"]),
        "capture-dev": ("capture.py", [run, "--mode", "dev"]),
        "capture-test": ("capture.py", [run, "--mode", "test"]),
        "select": ("evaluate.py", ["select", run]),
        "score": (
            "evaluate.py",
            ["test", run, *[x for v in args.baseline for x in ("--baseline", v)]],
        ),
        "route": ("routing.py", [run, "--allow-after-e0"]),
    }
    script, argv = op[args.phase]
    paths = [b.root / NATIVE_CODE]
    if script == "capture.py":
        src = native_source(s)
        paths.extend([src / "src", src])
        if not s.device.startswith("cuda"):
            raise ValueError("Historical capture requires a CUDA device")
    execute(s, b.file(NATIVE_CODE + "/" + script), argv, paths, device=s.device)


def representation(s, run, phase):
    b = s.bundle
    b.verify()
    cfg = read(run / "config.json")
    if phase in ("fit", "test", "prepare", "features"):
        for split in ("dev",) if phase == "fit" else ("dev", "test"):
            locked, _ = locked_paths(s, NATIVE + "inputs/" + split + ".json")
            if read(Path(cfg["base"]) / "inputs" / (split + ".json")) != locked:
                raise ValueError("Representation parent data differ from E0")
            if phase in ("fit", "test"):
                parts = ("fit", "check") if split == "dev" else ("test",)
                for part in parts:
                    if phase == "fit" and part == "test":
                        continue
                    rows = [r for r in locked if r["stage"] == "Pick" and r["split"] == part]
                    if read(run / "protocol" / part / "rows.json") != rows:
                        raise ValueError(f"Wrong {part} rows for Probe")
    if phase == "test":
        selection = read(run / "selection.json")
        gold = read(
            b.file(
                "reference/visual/vlm_capability_diagnosis/frozen_representation_probe/run_001/selection.json"
            )
        )
        if selection["selection_uses_test"] or selection["winner"]["name"] != gold["winner"]["name"]:
            raise ValueError("Formal Probe differs from the archived Check selection")
    script = {
        "features": "features",
        "prepare": "prepare",
        "fit": "probe",
        "test": "probe",
        "tables": "report_tables",
    }[phase]
    paths = [b.root / NATIVE_CODE]
    if phase in ("features", "prepare"):
        src = native_source(s)
        paths.extend([src / "src", src])
    execute(
        s,
        b.file(PROBE_CODE + "/" + script + ".py"),
        [run, *([phase] if script == "probe" else [])],
        paths,
        device=s.device,
    )


def video(path, images, seconds):
    import imageio_ffmpeg
    from PIL import Image

    if seconds <= 0 or Path(path).suffix.lower() != ".mp4":
        raise ValueError("Expected positive duration and an MP4 destination")
    args = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-loglevel",
        "error",
        "-n",
        "-f",
        "image2pipe",
        "-c:v",
        "png",
        "-r",
        str(1 / seconds),
        "-i",
        "-",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2,fps=24",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with subprocess.Popen(args, stdin=subprocess.PIPE) as process:
        shape = None
        for name in images:
            with Image.open(name) as im:
                if shape and im.size != shape:
                    raise ValueError("Review frames have different dimensions")
                shape = im.size
                im.convert("RGB").save(process.stdin, format="PNG")
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")


def protocol_command(s, args):
    if args.command == "video":
        return video(args.output, args.images, args.seconds)
    if args.command == "closed":
        from .rollouts import closed

        return closed(s, args)
    run = s.paths["output"] / args.run
    if args.command == "libero90":
        b = s.bundle
        b.verify()
        directory = "source/native_grounding/experiments/vlm_capability_diagnosis/independent_pick"
        return execute(
            s,
            b.file(directory + "/" + args.phase + ".py"),
            [run],
            [b.root / PROBE_CODE, b.root / NATIVE_CODE],
        )
    if args.command == "native":
        return native(s, run, args)
    return representation(s, run, args.phase)
