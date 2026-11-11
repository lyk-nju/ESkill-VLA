"""Read saved Episodes and build training samples.

Episodes are validated once at export time (scripts/preprocess/export.py); reading trusts that
structure and does not re-validate. Per-frame Event indices and source-pixel EEF projections are
stored; the Dataset builds action windows and Event-local Motion labels, and passes
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


def episode_timestamps(reader, timing=None):
    """Frame clock in physical seconds, never inferred from playback FPS or saved row count.

    An external timing record may contain timestamps_s, or a verified control_frequency_hz
    plus an explicit source_step_indices mapping. Evidence identifies the clock/mapping source.
    """
    meta, timing = reader.meta, timing or {}
    ref = timing.get("timestamps_s", meta.get("timestamps_s"))
    if ref is not None:
        times = reader.array(ref) if isinstance(ref, dict) else np.asarray(ref)
    elif timing.get("evidence") and "source_step_indices" in timing:
        hz = float(timing["control_frequency_hz"])
        if not np.isfinite(hz) or hz <= 0:
            raise ValueError("control_frequency_hz must be positive and verified")
        times = np.asarray(timing["source_step_indices"], np.float64) / hz
    else:
        raise ValueError(f'{meta["episode_id"]}: ESkill needs physical timestamps or an evidenced control clock '
                         'with source_step_indices; source_steps/nominal_fps are insufficient')
    times = np.asarray(times, np.float64)
    if (times.shape != (meta["frame_count"],) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)):
        raise ValueError(f'{meta["episode_id"]}: timestamps must be finite and strictly increasing per frame')
    return times


def gripper_semantics(meta, timing=None):
    """Explicit command channels/sign and command-after-observation alignment for future labels."""
    convention = (timing or {}).get("gripper") or meta["actions"].get("gripper_convention")
    if (not isinstance(convention, dict) or not convention.get("evidence")
            or convention.get("alignment") != "observation_then_action"
            or convention.get("close_sign") not in (-1, 1)
            or any(arm not in convention.get("channels", {}) for arm in meta["arm_ids"])):
        raise ValueError(f'{meta["episode_id"]}: ESkill needs evidenced gripper channels, close_sign and '
                         'alignment=observation_then_action; actual aperture is not a command label')
    return convention


def event_motion_targets(reader, current, convention, *, offsets_frames, camera="image", image_size=224,
                         min_horizon_frames=10):
    """Numeric supervision shared by Dataset and audit; no RGB decoding or simulator."""
    from openpi.transforms import resize_points
    meta, mq = reader.meta, len(offsets_frames)
    stream = meta["cameras"][camera]
    camera_frames = reader.array(stream["frame_indices"])
    current_row = reader.row(stream, current)
    requested = current_row + np.asarray(offsets_frames, dtype=np.int64)
    matched = requested < len(camera_frames)
    take = camera_frames[np.minimum(requested, len(camera_frames) - 1)]
    xy, grip = np.zeros((2, mq, 2), np.float32), np.zeros((2, mq), np.float32)
    xy_valid, grip_valid = np.zeros((2, mq), bool), np.zeros((2, mq), bool)
    projection_invalid, effector_missing, command_missing = (np.zeros((2, mq), bool) for _ in range(3))
    observed, event_tail = (np.zeros((2, mq), bool) for _ in range(2))
    cross_event = np.zeros((2, mq), bool)
    episode_tail = np.broadcast_to(~matched, (2, mq)).copy()
    source = np.full((2, mq), -1, np.int64)
    event_ids, goal_frames = [None, None], [None, None]
    stream = meta["cameras"][camera]
    shape = (stream["height"], stream["width"])
    action_frames = reader.array(meta["actions"]["frame_indices"])
    commands, command_valid = reader.array(meta["actions"]["values"]), reader.array(meta["actions"]["valid"])
    for arm in meta["arm_ids"]:
        slot = ARM_SLOTS[arm]
        pixels = meta.get("effector_pixels", {}).get(arm, {}).get(camera)
        if pixels is None:
            raise ValueError(f'{meta["episode_id"]}: ESkill needs calibrated future EEF pixels for {arm}')
        pix_indices = reader.array(pixels["frame_indices"])
        uv, uv_valid = resize_points(reader.array(pixels["values"]), reader.array(pixels["valid"]),
                                     shape, image_size)
        ei = int(reader.array(meta["event_index"][arm])[current])
        event = meta["events"][ei] if ei >= 0 else None
        if event is None or not event["boundary_valid"]:
            continue
        event_ids[slot], goal_frames[slot] = event["event_id"], event["goal_frame"]
        event_tail[slot] = ~matched | (take >= event["stop_frame"])
        # A short lookahead may follow the real trajectory beyond this Event, never beyond the Episode.
        in_window = matched & (take >= event["start_frame"]) & (
            ~event_tail[slot] | (np.asarray(offsets_frames) <= min_horizon_frames))
        cross_event[slot] = in_window & event_tail[slot]
        observed[slot] = in_window
        source[slot] = np.where(in_window, take, -1)
        pr = np.searchsorted(pix_indices, take)
        pi = np.minimum(pr, len(pix_indices) - 1)
        pixel_present = (pr < len(pix_indices)) & (pix_indices[pi] == take)
        projection_invalid[slot] = in_window & pixel_present & ~uv_valid[pi]
        effector_missing[slot] = in_window & ~pixel_present
        good_xy = in_window & pixel_present & uv_valid[pi]
        xy[slot] = np.where(good_xy[:, None], uv[pi], 0)
        xy_valid[slot] = good_xy
        channel = int(convention["channels"][arm])
        ar = np.searchsorted(action_frames, take)
        ai = np.minimum(ar, len(action_frames) - 1)
        # Multiple actions sharing one observation need an explicit action clock, not
        # an arbitrary first command. LIBERO's one-row pairing is unambiguous.
        next_row = np.minimum(ai + 1, len(action_frames) - 1)
        unique = (next_row == ai) | (action_frames[next_row] != take)
        value = commands[ai, channel]
        good_grip = (in_window & (ar < len(action_frames)) & (action_frames[ai] == take)
                     & unique & command_valid[ai, channel] & np.isfinite(value) & (value != 0))
        grip[slot] = np.where(good_grip, value * convention["close_sign"] > 0, 0)
        grip_valid[slot] = good_grip
        command_missing[slot] = in_window & ~good_grip
    padded = np.zeros((2, mq), bool)  # Terminal repeats are not observed or primary targets.
    targets = dict(trace=xy, trace_valid=xy_valid, gripper=grip, gripper_valid=grip_valid,
                   future_observed=observed, future_padded=padded, observed_mask=observed.copy(),
                   terminal_padded_mask=padded.copy(), future_event_tail=event_tail,
                   future_cross_event=cross_event,
                   future_episode_tail=episode_tail,
                   future_projection_invalid=projection_invalid, future_effector_missing=effector_missing,
                   future_command_missing=command_missing)
    metadata = dict(event_ids=event_ids, goal_frames=goal_frames,
                    future_camera_rows=requested, future_source_frame_indices=source)
    return targets, metadata


class EpisodeDataset:
    """Action-row windows over saved Episodes.

    `cameras` are input views in order; the first carries regions and Motion.
    `auxiliary=False` returns images, state and actions (π0.5).
    `auxiliary=True` adds skill labels, Goal-frame images, bboxes and Motion targets (ESkill).
    Canvas construction and image resizing happen in the training transforms, not here.
    """

    def __init__(self, index_path, *, action_horizon, cameras=("image",), auxiliary=True,
                 action_padding="mask", skill_names=(), motion_queries=8, image_size=224,
                 episode_ids=None, frame_stride=1,
                 motion_stride=5,
                 motion_min_horizon_frames=10,
                 active_skill_ids=tuple(range(6)), include_state=True):
        self.include_state = include_state
        self.horizon, self.cameras, self.camera = action_horizon, list(cameras), cameras[0]
        self.auxiliary, self.action_padding = auxiliary, action_padding
        self.skills = {name.casefold(): i for i, name in enumerate(skill_names)}
        self.motion_queries = motion_queries
        self.motion_offsets_frames = np.arange(1, motion_queries + 1, dtype=np.int64) * motion_stride
        self.motion_min_horizon_frames = motion_min_horizon_frames
        self.active_skill_ids = set(active_skill_ids)
        if not isinstance(motion_stride, int) or motion_stride < 1 or motion_queries < 1:
            raise ValueError("Motion queries and stride must be positive frame counts")
        self.image_size = image_size
        index_path = Path(index_path).resolve()
        self.paths, self.metadata, self.samples = [], [], []
        for entry in json.loads(index_path.read_text())["episodes"]:
            if episode_ids is not None and entry["episode_id"] not in episode_ids:
                continue
            meta = json.loads((index_path.parent / entry["path"]).read_text())
            if self.auxiliary:
                gripper_semantics(meta)  # Commands need semantics; frame queries do not require timestamps.
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
            inputs=dict(images=images, prompt=meta["instructions"][0]),
            targets=dict(actions=chunk, action_valid=valid),
            metadata=dict(episode_id=meta["episode_id"], task=meta["instructions"][0], frame=current,
                          action_row=action_row, action_frames=action_frames))
        if self.include_state:
            state_row = reader.row(meta["state"], current)
            sample["inputs"].update(state=reader.array(meta["state"]["values"])[state_row].copy(),
                                    state_valid=reader.array(meta["state"]["valid"])[state_row].copy())
        if self.auxiliary:
            self._auxiliary(sample, reader, current, images[0])
        return sample

    # ------------------------------------------------------------------
    # ESkill auxiliary targets
    # ------------------------------------------------------------------

    def _skill_index(self, event):
        if event is None or not event["boundary_valid"] or not event["skill_valid"] or event["skill"] is None:
            return -1
        return self.skills.get(event["skill"].casefold(), -1)

    def _auxiliary(self, sample, reader, current, image):
        meta = reader.meta
        targets, metadata = event_motion_targets(
            reader, current, gripper_semantics(meta),
            offsets_frames=self.motion_offsets_frames, camera=self.camera, image_size=self.image_size,
            min_horizon_frames=self.motion_min_horizon_frames)
        skills = np.full(2, -1, np.int64)
        sources = [[(None, None)] * 2 for _ in range(2)]
        for arm in meta["arm_ids"]:
            slot = ARM_SLOTS[arm]
            ei = int(reader.array(meta["event_index"][arm])[current])
            event = meta["events"][ei] if ei >= 0 else None
            skills[slot] = self._skill_index(event)
            if event is None or not event["boundary_valid"]:
                continue
            region = reader.region(event["object_regions"].get(self.camera), current)
            sources[slot][0] = (image, region.get("bbox") if region else None)
            goal = reader.goal(event["event_id"], self.camera)
            if goal is not None:
                sources[slot][1] = (goal["rgb"], goal["region"].get("bbox") if goal["region"] else None)
        sample["targets"].update(targets, skill_labels=skills,
                                skill_valid=np.isin(skills, list(self.active_skill_ids)), role_sources=sources)
        sample["metadata"].update(metadata)
