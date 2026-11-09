"""Read saved Episodes and build training samples.

Episodes are validated once at export time (scripts/preprocess/export.py); reading trusts that
structure and does not re-validate. Per-frame Event indices and source-pixel EEF projections are
stored; the Dataset does array slicing for action windows, future skills and Motion, and passes
raw bboxes to the training transforms for canvas construction.
"""
from collections import OrderedDict
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

SCHEMA_VERSION = "eskill.episode.v2-draft"
ACTION_PADDING = ("repeat_last", "mask")
ARM_SLOTS = {"single": 0, "left": 0, "right": 1}


def _load(path, key):
    """One array from an .npz; uncompressed members are memory-mapped."""
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(key + ".npy")
    if info.compress_type == zipfile.ZIP_STORED:
        with open(path, "rb") as f:
            f.seek(info.header_offset + 26)
            name_len, extra_len = map(int, np.frombuffer(f.read(4), "<u2"))
            f.seek(info.header_offset + 30 + name_len + extra_len)
            shape, _, dtype = np.lib.format._read_array_header(f, np.lib.format.read_magic(f))
            return np.memmap(path, dtype=dtype, mode="r", offset=f.tell(), shape=shape)
    with np.load(path) as archive:
        return archive[key]


class EpisodeReader:
    def __init__(self, path, *, metadata=None):
        self.path = Path(path).resolve()
        self.meta = metadata if metadata is not None else json.loads(self.path.read_text())
        self._raw = lru_cache(maxsize=64)(lambda uri, key: _load(self.path.parent / uri, key))

    def array(self, ref):
        values = self._raw(ref["uri"], ref["key"])
        if ref.get("encoding") == "packbits":
            n, h, w = ref["shape"]
            return np.unpackbits(values, axis=1, count=h * w, bitorder="big").reshape(n, h, w).astype(bool)
        return values

    def row(self, sequence, frame):
        indices = self.array(sequence["frame_indices"])
        r = int(np.searchsorted(indices, frame))
        return r if r < len(indices) and indices[r] == frame else None

    def rgb(self, camera, frame):
        stream = self.meta["cameras"][camera]
        r = self.row(stream, frame)
        if r is None or not self.array(stream["valid"])[r]:
            return None
        ref = stream["rgb"]
        if ref.get("encoding") != "jpeg_sequence":
            return self.array(ref)[r]
        offsets, raw = self._raw(ref["uri"], ref["offsets_key"]), self._raw(ref["uri"], ref["key"])
        return np.asarray(Image.open(BytesIO(raw[offsets[r]:offsets[r + 1]].tobytes())).convert("RGB"))

    # Convenience for preprocessing scripts and visualization; not used by the Dataset.
    def event_at(self, arm, frame):
        return next((e for e in self.meta["events"]
                     if e["arm"] == arm and e["start_frame"] <= frame < e["stop_frame"]), None)

    def region(self, sequence, frame, *, include_mask=False):
        row = None if sequence is None else self.row(sequence, frame)
        if row is None:
            return None
        box = mask = None
        if sequence["boxes_xyxy"] is not None and self.array(sequence["boxes_valid"])[row]:
            box = self.array(sequence["boxes_xyxy"])[row].copy()
        if include_mask:
            ref = sequence.get("masks")
            if ref is not None and self.array(sequence["masks_valid"])[row]:
                values = self._raw(ref["uri"], ref["key"])[row]
                if ref.get("encoding") == "packbits":
                    _, h, w = ref["shape"]
                    values = np.unpackbits(values, count=h * w, bitorder="big").reshape(h, w)
                mask = values.astype(bool)
        return None if box is None and mask is None else {"bbox": box, "mask": mask}

    def goal(self, event_id, camera, *, include_mask=False):
        event = next(e for e in self.meta["events"] if e["event_id"] == event_id)
        if not event["boundary_valid"] or event["goal_frame"] is None:
            return None
        frame = event["goal_frame"]
        return {"frame": frame, "rgb": self.rgb(camera, frame),
                "region": self.region(event["object_regions"].get(camera), frame, include_mask=include_mask)}


def load_episode(path):
    return EpisodeReader(path)


class EpisodeDataset:
    """Action-row windows over saved Episodes.

    `cameras` are input views in order; the first carries regions and Motion.
    `auxiliary=False` returns images, state and actions (π0.5).
    `auxiliary=True` adds skill labels, Goal-frame images, bboxes and Motion targets (ESkill).
    Canvas construction and image resizing happen in the training transforms, not here.
    """

    def __init__(self, index_path, *, action_horizon, cameras=("image",), auxiliary=True,
                 action_padding="mask", skill_names=(), motion_queries=8, motion_stride=4,
                 motion_tail_policy="hold_last", image_size=224,
                 episode_ids=None, frame_stride=1):
        self.horizon, self.cameras, self.camera = action_horizon, list(cameras), cameras[0]
        self.auxiliary, self.action_padding = auxiliary, action_padding
        self.skills = {name.casefold(): i for i, name in enumerate(skill_names)}
        self.motion_queries, self.motion_tail_policy = motion_queries, motion_tail_policy
        self.motion_offsets = np.arange(1, motion_queries + 1, dtype=np.int64) * motion_stride
        self.image_size = image_size
        index_path = Path(index_path).resolve()
        self.paths, self.metadata, self.samples = [], [], []
        for entry in json.loads(index_path.read_text())["episodes"]:
            if episode_ids is not None and entry["episode_id"] not in episode_ids:
                continue
            meta = json.loads((index_path.parent / entry["path"]).read_text())
            for c in cameras:
                if c not in meta["cameras"]:
                    raise ValueError(f'{meta["episode_id"]}: camera {c!r} not in episode')
            self.samples += [(len(self.paths), row)
                             for row in range(0, meta["actions"]["values"]["shape"][0], frame_stride)]
            self.paths.append(index_path.parent / entry["path"])
            self.metadata.append(meta)
        self._readers = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, "_readers": OrderedDict()}

    def reader(self, index):
        if index not in self._readers:
            self._readers[index] = EpisodeReader(self.paths[index], metadata=self.metadata[index])
            if len(self._readers) > 2:
                self._readers.popitem(last=False)
        self._readers.move_to_end(index)
        return self._readers[index]

    def __getitem__(self, index):
        episode, action_row = self.samples[index]
        reader = self.reader(episode)
        meta, actions = reader.meta, reader.meta["actions"]

        # Current observations: one image per configured camera.
        frames = []
        for camera in self.cameras:
            paired = int(reader.array(actions["conditioning_image_indices"][camera])[action_row])
            if paired < 0:
                raise ValueError(f'{meta["episode_id"]}: action row {action_row} has no paired image for {camera}')
            frames.append(int(reader.array(meta["cameras"][camera]["frame_indices"])[paired]))
        current = frames[0]
        images = [reader.rgb(c, f) for c, f in zip(self.cameras, frames)]
        state_row = reader.row(meta["state"], current)

        # Action window.
        action_indices = reader.array(actions["frame_indices"])
        n = min(self.horizon, len(action_indices) - action_row)
        values = reader.array(actions["values"])
        chunk = np.zeros((self.horizon, values.shape[1]), np.float32)
        valid = np.zeros_like(chunk, bool)
        chunk[:n], valid[:n] = values[action_row:action_row + n], reader.array(actions["valid"])[action_row:action_row + n]
        if self.action_padding == "repeat_last":
            chunk[n:], valid[n:] = chunk[n - 1], valid[n - 1]
        action_frames = np.full(self.horizon, -1, np.int64)
        action_frames[:n] = action_indices[action_row:action_row + n]

        sample = dict(
            inputs=dict(images=images, prompt=meta["instructions"][0],
                        state=reader.array(meta["state"]["values"])[state_row].copy(),
                        state_valid=reader.array(meta["state"]["valid"])[state_row].copy()),
            targets=dict(actions=chunk, action_valid=valid),
            metadata=dict(episode_id=meta["episode_id"], task=meta["instructions"][0], frame=current,
                          action_row=action_row, action_frames=action_frames))
        if self.auxiliary:
            self._auxiliary(sample, reader, current, images[0], action_frames[:n], valid[:n])
        return sample

    # ------------------------------------------------------------------
    # ESkill auxiliary targets
    # ------------------------------------------------------------------

    def _skill_index(self, event):
        if event is None or not event["boundary_valid"] or not event["skill_valid"] or event["skill"] is None:
            return -1
        return self.skills[event["skill"].casefold()]

    def _auxiliary(self, sample, reader, current, image, action_frames, action_valid):
        meta = reader.meta
        events = meta["events"]
        mq = self.motion_queries

        skills = np.full(2, -1, np.int64)
        future = np.full((2, self.horizon), -1, np.int64)
        trace = np.zeros((2, mq, 2), np.float32)
        trace_valid = np.zeros((2, mq), bool)
        observed, padded = np.zeros((2, mq), bool), np.zeros((2, mq), bool)
        source_frames = np.full((2, mq), -1, np.int64)
        event_ids, goal_frames = [None, None], [None, None]
        # role_sources[arm][role] = (rgb, bbox_or_None); transforms build canvases from these.
        role_sources = [[(None, None)] * 3 for _ in range(2)]

        for arm in meta["arm_ids"]:
            slot = ARM_SLOTS[arm]
            event_idx = reader.array(meta["event_index"][arm])
            ei = int(event_idx[current])
            event = events[ei] if ei >= 0 else None
            skills[slot] = self._skill_index(event)

            for j, frame in enumerate(action_frames):
                if action_valid[j].any():
                    fi = int(event_idx[frame])
                    future[slot, j] = self._skill_index(events[fi] if fi >= 0 else None)

            motion = self._motion(reader, arm, current)
            if motion is not None:
                trace[slot], trace_valid[slot] = motion["trace"], motion["trace_valid"]
                observed[slot], padded[slot] = motion["future_observed"], motion["future_padded"]
                source_frames[slot] = motion["future_source_frame_indices"]

            ee = meta["effector_regions"].get(arm, {}).get(self.camera)
            ee_region = reader.region(ee, current)
            role_sources[slot][0] = (image, ee_region.get("bbox") if ee_region else None)
            if event is not None:
                event_ids[slot], goal_frames[slot] = event["event_id"], event["goal_frame"]
                obj_region = reader.region(event["object_regions"].get(self.camera), current)
                role_sources[slot][1] = (image, obj_region.get("bbox") if obj_region else None)
                goal = reader.goal(event["event_id"], self.camera)
                if goal is not None:
                    goal_region = goal.get("region")
                    role_sources[slot][2] = (goal["rgb"], goal_region.get("bbox") if goal_region else None)

        sample["targets"].update(
            skill_labels=skills, skill_valid=skills >= 0, future_skill_labels=future, future_skill_valid=future >= 0,
            role_sources=role_sources,
            trace=trace, trace_valid=trace_valid, future_observed=observed, future_padded=padded)
        sample["metadata"].update(
            event_ids=event_ids, goal_frames=goal_frames, future_frame_indices=current + self.motion_offsets,
            future_source_frame_indices=source_frames)

    def _motion(self, reader, arm, current):
        """Select future EEF points from stored source-pixel projections; reproject to model canvas."""
        from openpi.transforms import future_targets, resize_points
        cameras = reader.meta.get("effector_pixels", {}).get(arm, {})
        pixels = cameras.get(self.camera)
        if pixels is None:
            return None
        indices = reader.array(pixels["frame_indices"])
        if not len(indices):
            return None
        uv = reader.array(pixels["values"])
        valid = reader.array(pixels["valid"])
        stream = reader.meta["cameras"][self.camera]
        shape = (stream["height"], stream["width"])
        uv, valid = resize_points(uv, valid, shape, self.image_size)
        return future_targets(indices, uv, valid, current, self.motion_offsets,
                              tail_policy=self.motion_tail_policy, last_frame=reader.meta["frame_count"] - 1)
