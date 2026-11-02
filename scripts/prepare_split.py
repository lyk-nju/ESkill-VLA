"""Freeze a task-stratified parent-trajectory split shared by SFT and event supervision."""
import argparse
import json
from pathlib import Path

from openpi.training.config import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--val-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    split = make_split(args.index, args.val_fraction, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(split, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(f"Saved {len(split['tasks'])} tasks: {len(split['train'])} train / {len(split['val'])} val trajectories")


if __name__ == '__main__':
    main()
