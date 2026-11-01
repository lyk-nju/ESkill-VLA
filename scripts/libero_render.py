"""Render a saved LIBERO episode bundle using the shared review layout."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    args = parser.parse_args()
    from scripts.preprocess.libero.view import render_saved
    render_saved(args.episode, args.video)


if __name__ == "__main__":
    main()
