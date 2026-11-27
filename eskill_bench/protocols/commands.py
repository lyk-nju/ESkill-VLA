"""Shared rollout and review video commands."""

import subprocess
from pathlib import Path


def add_commands(commands):
    rollout_phases = ("plan", "serve", "run", "status", "report")
    for name, phases in (
        ("libero", rollout_phases),
        ("libero-plus", rollout_phases),
        ("unseen-composition", ("plan", "prepare", "run", "status", "report", "audit", "render")),
    ):
        p = commands.add_parser(name)
        if name == "libero":
            p.add_argument("--with-plus", dest="protocol", action="store_const", const="sample",
                           default="libero200", help="Run the combined 200 LIBERO and 280 LIBERO-Plus sample")
        elif name == "libero-plus":
            p.add_argument("--full", dest="protocol", action="store_const", const="plus-full",
                           default="plus280", help="Run all 10,030 variants")
        else:
            p.set_defaults(protocol=name)
        p.add_argument("phase", choices=phases)
        p.add_argument("args", nargs="...")
    p = commands.add_parser("video")
    p.add_argument("output", type=Path)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--seconds", type=float, default=0.75)


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
    if hasattr(args, "protocol"):
        from .rollouts import rollout

        return rollout(s, args)
