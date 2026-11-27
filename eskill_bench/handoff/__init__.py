"""Run the November locked evaluation protocols and replay their regression anchors."""

import argparse
import json
from pathlib import Path

from .assets import Bundle, Settings, digest, locked_paths, unpack, write_new
from .audit import inventory, restore_dev, visual_regression
from .motion import reproduce
from .runner import SOURCES, inference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Host asset and environment mapping")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("unpack")
    p.add_argument("archive", type=Path)
    p.add_argument("destination", type=Path)
    p = commands.add_parser("replay")
    p.add_argument("bundle", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--suite", choices=("motion", "visual"), default="motion")
    commands.add_parser("audit")
    commands.add_parser("restore-dev").add_argument("run")
    p = commands.add_parser("relocate")
    p.add_argument("archive_key")
    p.add_argument("run")
    p = commands.add_parser("infer")
    p.add_argument("variant", choices=SOURCES)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("run")
    p.add_argument("--limit", type=int)
    p.add_argument("--native-dev-frame", action="store_true")
    p.add_argument(
        "--factual", action="store_true", help="Evaluate the frozen Planner's locked native episodes"
    )
    p.add_argument("--technical-checkpoint", action="store_true")
    from .protocols import add_commands, protocol_command

    add_commands(commands)
    args = parser.parse_args()
    if args.command == "unpack":
        unpack(args.archive, args.destination)
        return
    if args.command == "replay":
        bundle = Bundle(args.bundle)
        bundle.verify()
        result = reproduce(bundle) if args.suite == "motion" else visual_regression(bundle)
        if args.output:
            write_new(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.config is None:
        parser.error("--config is required for this operation")
    settings = Settings(args.config)
    if args.command == "audit":
        print(json.dumps(inventory(settings), indent=2))
    elif args.command == "restore-dev":
        print(json.dumps(restore_dev(settings, settings.paths["output"] / args.run)))
    elif args.command == "infer":
        inference(
            settings,
            args.variant,
            args.checkpoint,
            settings.paths["output"] / args.run,
            args.limit,
            args.native_dev_frame,
            args.technical_checkpoint,
            args.factual,
        )
    elif args.command == "relocate":
        dest = settings.paths["output"] / args.run
        doc, changes = locked_paths(settings, args.archive_key)
        dest.mkdir(parents=True, exist_ok=False)
        local = dest / "manifest.json"
        write_new(local, doc)
        write_new(
            dest / "mapping.json",
            dict(
                original=args.archive_key,
                original_sha256=digest(settings.bundle.file(args.archive_key)),
                copy_sha256=digest(local),
                paths=changes,
            ),
        )
    else:
        protocol_command(settings, args)
