"""Content identities and path relocation for the sealed November handoff."""

import csv
import io
import subprocess
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np
from PIL import Image


ARCHIVE = "a7641006c8753a580b783c97751239955c4657923ef0fb79066cb10147fee9cd"
INDEX = "52d6394d49826ac5f986f63e4aa2ce3b36aa79d10f7640d651b30ea04859a7d2"
MOTION = "reference/motion/01/manifest.json"
VISUAL = "reference/visual/stage1_visual_localization/run_001/manifest.json"
NATIVE = "reference/visual/native_grounding/run_001/"
SPLIT = "reference/visual/current_subject_grounding/run_001/assets/split.json"
KLAL = "reference/visual/visual_primitive_full_vlm/run_001/evaluation_stage1/run_001/manifest.json"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x") as f:
        f.write(text)


def unpack(archive, destination):
    if digest(archive) != ARCHIVE:
        raise ValueError("Unexpected handoff archive SHA-256")
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=dest.parent) as tmp,
        tarfile.open(archive) as tar,
    ):
        # This exact digest identifies the reviewed bundle. Only accept regular files/directories.
        for m in tar:
            path = Path(m.name)
            if path.is_absolute() or ".." in path.parts or not (m.isfile() or m.isdir()):
                raise ValueError(m.name)
            tar.extract(m, tmp, set_attrs=False)
        tree = Path(tmp) / "eval_handoff_2026-11-27"
        Bundle(tree).verify()
        tree.rename(dest)


class Bundle:
    def __init__(self, root):
        self.root = Path(root).resolve()
        index = self.root / "FILES.tsv"
        if digest(index) != INDEX:
            raise ValueError("Unexpected FILES.tsv SHA-256")
        with index.open(newline="") as f:
            self.files = {r["archive_path"]: r for r in csv.DictReader(f, delimiter="\t")}

    def file(self, key):
        row = self.files[key]
        path = self.root / key
        if not path.is_file() or digest(path) != row["sha256"]:
            raise ValueError(f"Handoff file missing or changed: {key}")
        return path

    def verify(self):
        for key in self.files:
            self.file(key)
        return len(self.files)


class Settings:
    def __init__(self, file):
        cfg = read(file)
        self.base = Path(file).resolve().parent
        self.paths = {k: self.path(v) for k, v in cfg["paths"].items()}
        self.bundle = Bundle(self.paths["bundle"])
        self.device = cfg["device"]
        self.expected = cfg.get("expected", {})
        self.mapping = sorted(
            ((Path(k), self.path(v)) for k, v in cfg["path_map"].items()),
            key=lambda item: -len(item[0].parts),
        )

    def path(self, name):
        return Path(os.path.abspath(self.base / os.path.expanduser(name)))

    def relocate(self, path):
        path = Path(path)
        for old, new in self.mapping:
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path


def locked_paths(settings, key):
    """Rewrite only known path-bearing fields, preserving all sample labels and ordering."""
    bundle = settings.bundle
    original = read(bundle.file(key))
    parent = Path(bundle.files[key]["source_path"]).parent
    mappings = {}

    def walk(value, field=""):
        if isinstance(value, list):
            return [walk(v, field) for v in value]
        if isinstance(value, dict):
            if isinstance(value.get("image"), str) and isinstance(value.get("rgb_path"), str):
                image, resolved = Path(value["image"]), Path(value["rgb_path"])
                if (
                    not image.is_absolute()
                    and resolved.is_absolute()
                    and resolved.parts[-len(image.parts) :] == image.parts
                ):
                    value = dict(value, image=str(resolved))
            return {k: walk(v, k) for k, v in value.items()}
        if isinstance(value, str) and field in {
            "image",
            "rgb_path",
            "episode_ref",
            "path",
            "state_path",
            "mask_path",
            "source_index",
            "wrist_image",
            "current_mask",
            "subject_mask",
        }:
            src = parent / value
            dst = str(settings.relocate(src))
            mappings[str(src)] = dst
            return dst
        return value

    return walk(original), mappings


def rgb(path, key="main", expected=None):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            image = archive[key]
    else:
        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Invalid RGB: {path}")
    actual = hashlib.sha256(image.tobytes()).hexdigest()
    if expected and expected != actual:
        raise ValueError(f"Decoded RGB hash differs: {path}")
    return image


def audit_images(rows):
    unique = {}
    missing = []
    for r in rows:
        image = r.get("rgb_path") or r["image"]
        image_key = r.get("image_key", "main")
        expected = r.get("rgb_sha256")
        key = (image, image_key, expected)
        if key in unique:
            continue
        unique[key] = True
        if not Path(image).is_file():
            missing.append(image)
        else:
            rgb(image, image_key, expected)
    return dict(
        unique=len(unique),
        present=len(unique) - len(missing),
        missing=len(missing),
        first_missing=next(iter(missing), None),
    )


def git_source(output, commit):
    """Materialize the recorded checkout, including support modules omitted by the handoff."""
    repo = Path(__file__).resolve().parents[2]
    src = Path(output) / ("source_" + commit)
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=src.parent) as tmp:
            tree = Path(tmp) / "source"
            tree.mkdir()
            data = subprocess.check_output(["git", "-C", str(repo), "archive", commit, "src", "scripts"])
            with tarfile.open(fileobj=io.BytesIO(data)) as archive:
                for member in archive:
                    if member.isdir() or member.isfile():
                        archive.extract(member, tree, set_attrs=False)
            try:
                tree.rename(src)
            except OSError:
                if not src.is_dir():
                    raise
    return src
