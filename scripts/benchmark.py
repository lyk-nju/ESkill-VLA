"""Use the unified evaluation protocol with the current checkout's model."""
import os
from pathlib import Path
import sys


def unified_benchmark():
    model = Path(__file__).resolve().parents[1]
    default = model if (model / 'eskill_bench').is_dir() else model.parent / 'unified'
    root = Path(os.environ.get('ESKILL_BENCHMARK_ROOT', default)).expanduser().resolve()
    if not (root / 'eskill_bench/cli.py').is_file():
        raise FileNotFoundError(f'Unified benchmark not found at {root}. Set ESKILL_BENCHMARK_ROOT to its checkout.')
    # src.openpi imports benchmark helpers; openpi imports the model under evaluation.
    paths = [str(root), str(model / 'src'), str(model)]
    if sys.path[:3] != paths:
        sys.path[:0] = paths
    from eskill_bench import cli
    return cli


def benchmark(command):
    return unified_benchmark().implementation(command)


if __name__ == '__main__':
    unified_benchmark().main()
