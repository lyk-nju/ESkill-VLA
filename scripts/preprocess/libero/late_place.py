"""Compatibility entry point for the unified held-out Place manifest."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.benchmark import unified_benchmark

cli = unified_benchmark()

if __name__ == '__main__':
    cli.late_place(sys.argv[1:])
else:
    import runpy
    _source = cli.ROOT / 'scripts/preprocess/libero/late_place.py'
    _protocol = runpy.run_path(str(_source))
    export, late_frame, place_events = (_protocol[name] for name in ('export', 'late_frame', 'place_events'))
