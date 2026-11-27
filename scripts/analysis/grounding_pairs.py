"""Compatibility entry point for the unified grounding_pairs benchmark."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.benchmark import benchmark

module = benchmark('grounding_pairs')
if __name__ == '__main__':
    module.main()
else:
    sys.modules[__name__] = module
