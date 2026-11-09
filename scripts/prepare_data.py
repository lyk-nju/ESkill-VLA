"""Export existing labels: AtomicVLA or RoboTwin self-contained Episode bundles."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=["atomic-vla", "robotwin"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments for the source exporter")
    args = parser.parse_args()
    if args.source == "atomic-vla":
        from scripts.preprocess.libero.atomic_vla.export import main as export
        export(["export", *args.arguments])
    else:
        from scripts.preprocess.robotwin.episodes import main as export
        export(args.arguments)


if __name__ == "__main__":
    main()
