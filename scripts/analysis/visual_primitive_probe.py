"""Compatibility entry point for the unified probe benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark import benchmark

module = benchmark('probe')
if __name__ == '__main__':
    module.main()
else:
    sys.modules[__name__] = module
