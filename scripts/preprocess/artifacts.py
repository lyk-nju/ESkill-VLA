"""Small shared source/file helpers; no model, renderer or orchestration imports."""
import hashlib
import importlib
import json
from pathlib import Path

from scripts.preprocess import DATA_ROOT

ROOT = DATA_ROOT
SOURCE_DIRS = {"libero": "libero/guidedvla", "robotwin": "robotwin", "atomic-vla": "libero/atomic-vla"}


def processed_root(dataset):
    return ROOT / SOURCE_DIRS[dataset] / "processed"


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output exists; choose a new output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def episodes(dataset, manifest):
    if dataset not in SOURCE_DIRS:
        raise ValueError("Unknown source dataset")
    module = SOURCE_DIRS[dataset].replace("/", ".").replace("-", "_")
    source = importlib.import_module(f"scripts.preprocess.{module}.source")
    yield from source.iter_episodes(manifest)


def artifact(root, folder, key):
    path = Path(root).resolve() / folder / (key + ".json")
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Source episode key escapes output directory")
    return path


def check_source(episode, record):
    if record["source"] != episode.entry or record["frame_count"] != episode.count:
        raise ValueError("Annotation source identity/timeline differs from loaded episode")
