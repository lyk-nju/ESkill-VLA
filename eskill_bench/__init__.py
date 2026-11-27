"""Evaluation protocols owned by the unified checkout."""
from pathlib import Path
from src.openpi.revision import git_revision as revision

ROOT = Path(__file__).resolve().parents[1]


def revisions():
    import openpi
    model_root = Path(openpi.__file__).resolve().parents[2]
    return dict(benchmark=revision(ROOT), model=revision(model_root))
