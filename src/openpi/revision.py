"""Identify the checked-out source used for predictions or scoring."""
from pathlib import Path
import subprocess


def git_revision(root):
    root = Path(root)
    if (root / '.git').exists():
        cmd = ['git', '-C', str(root)]
        try:
            commit = subprocess.check_output([*cmd, 'rev-parse', 'HEAD'], text=True).strip()
            status = subprocess.check_output([*cmd, 'status', '--porcelain'], text=True)
            return dict(commit=commit, dirty=bool(status))
        except (OSError, subprocess.CalledProcessError):
            pass
    return dict(commit=None, dirty=None)
