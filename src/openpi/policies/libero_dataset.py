"""Action-only LIBERO windows from raw RLDS excerpts or saved ESkill episodes."""
from collections import OrderedDict
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image

from openpi.policies.eskill_dataset import EpisodeReader


class LiberoSFTDataset:
    """Memory-mapped original RGB/action records; no event supervision."""
    def __init__(self, index_path, *, action_horizon, episode_ids=None):
        self.root = Path(index_path).resolve().parent
        entries = json.loads(Path(index_path).read_text())["episodes"]
        self.entries = [e for e in entries if episode_ids is None or e["episode_id"] in episode_ids]
        self.horizon = action_horizon
        self.samples = [(i, row) for i, e in enumerate(self.entries) for row in range(e["frames"])]
        self.arrays = None

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, "arrays": None}

    def __getitem__(self, index):
        if self.arrays is None:
            self.arrays = {key: np.load(self.root / f"{key}.npy", mmap_mode="r")
                           for key in ("state", "actions", "image_offsets")}
            self.arrays["images"] = np.memmap(self.root / "images.bin", dtype=np.uint8, mode="r")
        number, row = self.samples[index]
        entry = self.entries[number]
        frame = entry["start"] + row
        images = []
        for offset, length in self.arrays["image_offsets"][frame]:
            with Image.open(BytesIO(self.arrays["images"][offset:offset + length].tobytes())) as im:
                images.append(np.asarray(im.convert("RGB")).copy())
        rows = np.minimum(np.arange(row, row + self.horizon), entry["frames"] - 1)
        actions = self.arrays["actions"][entry["start"] + rows].copy()
        state = self.arrays["state"][frame].copy()
        return dict(inputs=dict(image=images[0], wrist_image=images[1], state=state,
                                state_valid=np.ones_like(state, bool), prompt=entry["task"]),
                    targets=dict(actions=actions, action_valid=np.ones_like(actions, bool)),
                    metadata=dict(episode_id=entry["episode_id"], task=entry["task"], frame=row,
                                  action_row=row, action_frames=rows))


class LiberoDataset:
    def __init__(self, index_path, *, action_horizon, data_format="episodes", episode_ids=None,
                 sample_manifest=None, frame_stride=1, camera="image"):
        self.root = Path(index_path).resolve().parent
        entries = json.loads(Path(index_path).read_text())["episodes"]
        self.entries = [e for e in entries if episode_ids is None or e["episode_id"] in episode_ids]
        self.format, self.horizon, self.camera = data_format, action_horizon, camera
        if data_format not in ("episodes", "source_npz"):
            raise ValueError("Unsupported LIBERO data_format")
        if data_format == "source_npz" and camera != "image":
            raise ValueError("Raw RLDS excerpts contain only the main image camera")
        self.cache = OrderedDict()
        by_id = {entry["episode_id"]: i for i, entry in enumerate(self.entries)}
        if sample_manifest:
            records = json.loads(Path(sample_manifest).read_text())["samples"]
            self.samples = [(by_id[r["episode_id"]], r["action_row"]) for r in records if r["episode_id"] in by_id]
        else:
            counts = [json.loads((self.root / e["path"]).read_text())["actions"]["values"]["shape"][0]
                      if self.format == "episodes" else e["frames"] for e in self.entries]
            self.samples = [(i, row) for i, count in enumerate(counts)
                            for row in range(0, count, frame_stride)]

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, "cache": OrderedDict()}

    def episode(self, number):
        if number not in self.cache:
            path = self.root / self.entries[number]["path"]
            if self.format == "episodes":
                value = EpisodeReader(path)
            else:
                with np.load(path, allow_pickle=False) as archive:
                    value = {key: archive[key] for key in archive.files}
            self.cache[number] = value
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(number)
        return self.cache[number]

    def __getitem__(self, index):
        number, row = self.samples[index]
        entry, reader = self.entries[number], self.episode(number)
        if self.format == "episodes":
            action = reader.meta["actions"]
            frame_indices = reader.array(action["frame_indices"])
            image_row = int(reader.array(action["conditioning_image_indices"][self.camera])[row])
            if image_row < 0:
                raise ValueError("Action has no paired RGB")
            frame = int(reader.array(reader.meta["cameras"][self.camera]["frame_indices"])[image_row])
            image = reader.rgb(self.camera, frame)
            state_seq = reader.meta["state"]
            state_row = reader.row(state_seq, frame)
            if image is None or state_row is None:
                raise ValueError("Missing paired RGB or state")
            state, state_valid = reader.array(state_seq["values"])[state_row], reader.array(state_seq["valid"])[state_row]
            actions, valid = reader.array(action["values"]), reader.array(action["valid"])
            prompt = reader.meta["instructions"][0]
        else:
            frame = row
            a, b = reader["image_offsets"][row:row + 2]
            with Image.open(BytesIO(reader["image_bytes"][a:b].tobytes())) as im:
                image = np.asarray(im.convert("RGB")).copy()
            state = np.r_[reader["joints"][row], reader["state"][row]].astype(np.float32)
            state_valid = np.ones_like(state, bool)
            actions, frame_indices = reader["actions"], np.arange(len(reader["actions"]))
            valid = np.ones_like(actions, bool)
            prompt = entry["task"]
        n = min(self.horizon, len(actions) - row)
        chunk = np.zeros((self.horizon, actions.shape[-1]), np.float32)
        chunk_valid = np.zeros_like(chunk, bool)
        chunk[:n], chunk_valid[:n] = actions[row:row + n], valid[row:row + n]
        frames = np.full(self.horizon, -1, np.int64)
        frames[:n] = frame_indices[row:row + n]
        return dict(
            inputs=dict(image=image, state=state.copy(), state_valid=state_valid.copy(), prompt=prompt),
            targets=dict(actions=chunk, action_valid=chunk_valid),
            metadata=dict(episode_id=entry["episode_id"], task=prompt, frame=frame,
                          action_row=row, action_frames=frames))
