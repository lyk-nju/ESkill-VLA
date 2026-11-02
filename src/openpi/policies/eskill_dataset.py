"""Read saved episode fields and exact source frames; independent of preprocessing and training."""
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from collections import OrderedDict

import numpy as np

SCHEMA_VERSION = "eskill.episode.v2-draft"


class EpisodeReader:
    def __init__(self, path, *, metadata=None):
        self.path = Path(path).resolve()
        self.meta = metadata if metadata is not None else json.loads(self.path.read_text())
        if self.meta['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported episode schema; use an explicit source adapter')
        # The cache belongs to this episode, not to a global dataset iterator.
        self._raw = lru_cache(maxsize=64)(self._read_array)

    def _read_array(self, uri, key):
        path = Path(uri)
        if not path.is_absolute():
            path = self.path.parent/path
        with np.load(path, allow_pickle=False) as archive:
            return archive[key]

    def array(self, ref):
        """Decode a numeric/mask reference. RGB is read one frame at a time by rgb()."""
        values = self._raw(ref['uri'], ref['key'])
        encoding = ref.get('encoding', 'array')
        if encoding == 'packbits':
            n, h, w = ref['shape']
            if values.dtype != np.uint8 or values.shape != (n, (h*w+7)//8):
                raise ValueError('Packed mask storage shape/dtype mismatch')
            return np.unpackbits(values, axis=1, count=h*w, bitorder='big').reshape(n,h,w).astype(bool)
        if encoding != 'array':
            raise ValueError(f'Use rgb() for {encoding}')
        if list(values.shape) != list(ref['shape']) or str(values.dtype) != ref['dtype']:
            raise ValueError(f'Array shape/dtype mismatch: {ref["key"]}')
        return values

    def row(self, sequence, frame):
        """Exact index lookup only. Never copy a neighbouring observation."""
        indices = self.array(sequence['frame_indices'])
        row = int(np.searchsorted(indices, frame))
        return row if row < len(indices) and indices[row] == frame else None

    def rgb(self, camera, frame):
        stream = self.meta['cameras'][camera]
        row = self.row(stream, frame)
        if row is None or not self.array(stream['valid'])[row]:
            return None
        ref = stream['rgb']
        if ref.get('encoding', 'array') == 'jpeg_sequence':
            from PIL import Image
            offsets = self._raw(ref['uri'], ref['offsets_key'])
            raw = self._raw(ref['uri'], ref['key'])
            with Image.open(BytesIO(raw[offsets[row]:offsets[row+1]].tobytes())) as image:
                value = np.asarray(image.convert('RGB')).copy()
        else:
            value = self.array(ref)[row]
        if value.shape != (stream['height'],stream['width'],3) or value.dtype != np.uint8:
            raise ValueError('RGB must match the declared source image size')
        return value

    def event_at(self, arm, frame):
        matches = [e for e in self.meta['events'] if e['arm'] == arm
                   and e['start_frame'] <= frame < e['stop_frame']]
        if len(matches) > 1:
            raise ValueError('Overlapping events on the same arm')
        return matches[0] if matches else None

    def region(self, sequence, frame):
        """Return independent bbox/mask availability; an empty known mask is retained."""
        if sequence is None:
            return None
        row = self.row(sequence, frame)
        if row is None:
            return None
        box, mask = None, None
        if sequence['boxes_xyxy'] is not None and self.array(sequence['boxes_valid'])[row]:
            box = self.array(sequence['boxes_xyxy'])[row].copy()
        ref = sequence['masks']
        if ref is not None and self.array(sequence['masks_valid'])[row]:
            if ref.get('encoding') == 'packbits':
                _, h, w = ref['shape']
                bits = self._raw(ref['uri'],ref['key'])[row]
                mask = np.unpackbits(bits,count=h*w,bitorder='big').reshape(h,w).astype(bool)
            else:
                mask = self.array(ref)[row].copy()
        return {'bbox': box, 'mask': mask} if box is not None or mask is not None else None

    def goal(self, event_id, camera):
        event = next(e for e in self.meta['events'] if e['event_id'] == event_id)
        frame = event['goal_frame']
        if not event['boundary_valid'] or frame is None:
            return None
        return {'frame': frame, 'rgb': self.rgb(camera,frame),
                'region': self.region(event['object_regions'].get(camera),frame)}


def load_episode(path):
    return EpisodeReader(path)


class ESkillDataset:
    """Action-row windows over saved episodes; future data is isolated in targets."""

    def __init__(self, index_path, *, action_horizon, skill_names, motion_queries=8,
                 motion_stride=4, motion_tail_policy="hold_last", image_size=224,
                 canvas_size=256, camera="image", episode_ids=None, frame_stride=1):
        self.index_path = Path(index_path).resolve()
        self.horizon, self.motion_queries = action_horizon, motion_queries
        self.motion_offsets = np.arange(1, motion_queries + 1, dtype=np.int64) * motion_stride
        self.motion_tail_policy, self.image_size = motion_tail_policy, image_size
        self.canvas_size, self.camera = canvas_size, camera
        self.skills = {name.casefold(): i for i, name in enumerate(skill_names)}
        index = json.loads(self.index_path.read_text())
        selected = None if episode_ids is None else set(episode_ids)
        self.paths, self.metadata, self.samples = [], [], []
        for entry in index["episodes"]:
            if selected is not None and entry["episode_id"] not in selected:
                continue
            path = self.index_path.parent / entry["path"]
            meta = json.loads(path.read_text())
            count = meta["actions"]["values"]["shape"][0]
            episode = len(self.paths)
            self.paths.append(path)
            self.metadata.append(meta)
            self.samples.extend((episode, row) for row in range(0, count, frame_stride))
        self._readers = OrderedDict()

    def __len__(self):
        return len(self.samples)

    def __getstate__(self):
        return {**self.__dict__, "_readers": OrderedDict()}

    def reader(self, index):
        if index not in self._readers:
            self._readers[index] = EpisodeReader(self.paths[index], metadata=self.metadata[index])
            if len(self._readers) > 2:
                _, old = self._readers.popitem(last=False)
                old._raw.cache_clear()
        self._readers.move_to_end(index)
        return self._readers[index]

    def _skill(self, event):
        if event is None or not event["boundary_valid"] or not event["skill_valid"] or event["skill"] is None:
            return -1
        # Unknown vocabulary entries are configuration errors, not idle.
        return self.skills[event["skill"].casefold()]

    def _motion(self, reader, arm, current):
        """Project future positions into the current main view; Event boundaries are irrelevant."""
        from openpi.transforms import future_targets, project_points, resize_points

        stream = reader.meta["cameras"][self.camera]
        calibration = stream.get("calibration")
        positions = reader.meta["effector_positions"].get(arm)
        if (not calibration or positions is None or positions.get("unit") != "m"
                or positions.get("coordinate_frame") != "world"
                or positions.get("reference_point") in (None, "", "unknown")
                or calibration.get("world_frame") != "world"
                or calibration.get("extrinsics_convention") != "world_to_camera_cv"
                or any(calibration.get(k) is None for k in ("intrinsics", "extrinsics", "image_transform"))):
            return None
        indices = reader.array(positions["frame_indices"])
        if not len(indices):
            return None
        camera_row = reader.row(stream, current)
        def matrix(key):
            value = reader.array(calibration[key])
            return value if value.ndim == 2 else value[camera_row]
        shape = (stream["height"], stream["width"])
        uv, valid = project_points(reader.array(positions["values"]), matrix("intrinsics"),
                                   matrix("extrinsics"), matrix("image_transform"), shape)
        valid &= reader.array(positions["valid"]).all(-1)
        uv, valid = resize_points(uv, valid, shape, self.image_size)
        return future_targets(indices, uv, valid, current, self.motion_offsets,
                              tail_policy=self.motion_tail_policy, last_frame=reader.meta["frame_count"] - 1)

    def __getitem__(self, index):
        from openpi.transforms import role_canvas

        episode, action_row = self.samples[index]
        reader = self.reader(episode)
        meta = reader.meta
        actions = meta["actions"]
        action_indices = reader.array(actions["frame_indices"])
        pairing = reader.array(actions["conditioning_image_indices"][self.camera])
        image_row = int(pairing[action_row])
        if image_row < 0:
            raise ValueError(f'{meta["episode_id"]}: action has no paired observation')
        current = int(reader.array(meta["cameras"][self.camera]["frame_indices"])[image_row])
        image = reader.rgb(self.camera, current)
        if image is None:
            raise ValueError(f'{meta["episode_id"]}: missing current RGB at {current}')
        state_row = reader.row(meta["state"], current)
        if state_row is None:
            raise ValueError(f'{meta["episode_id"]}: missing state at observation {current}')
        state = reader.array(meta["state"]["values"])[state_row].copy()
        state_valid = reader.array(meta["state"]["valid"])[state_row].copy()

        n = min(self.horizon, len(action_indices) - action_row)
        values = reader.array(actions["values"])
        chunk = np.zeros((self.horizon, values.shape[1]), np.float32)
        action_valid = np.zeros_like(chunk, bool)
        chunk[:n] = values[action_row:action_row + n]
        action_valid[:n] = reader.array(actions["valid"])[action_row:action_row + n]
        future_frames = np.full(self.horizon, -1, np.int64)
        future_frames[:n] = action_indices[action_row:action_row + n]
        skills = np.full(2, -1, np.int64)
        future = np.full((2, self.horizon), -1, np.int64)
        size = self.canvas_size
        canvases = np.zeros((2, 3, size, size, 3), np.uint8)
        weights = np.zeros((2, 3, size, size), np.float32)
        role_valid = np.zeros((2, 3), bool)
        trace = np.zeros((2, self.motion_queries, 2), np.float32)
        trace_valid = np.zeros((2, self.motion_queries), bool)
        observed, padded = np.zeros_like(trace_valid), np.zeros_like(trace_valid)
        source_frames = np.full(trace_valid.shape, -1, np.int64)
        event_ids, goal_frames = [None, None], [None, None]
        for arm in meta["arm_ids"]:
            slot = {"single": 0, "left": 0, "right": 1}[arm]
            event = reader.event_at(arm, current)
            skills[slot] = self._skill(event)
            motion = self._motion(reader, arm, current)
            if motion is not None:
                trace[slot], trace_valid[slot] = motion["trace"], motion["trace_valid"]
                observed[slot], padded[slot] = motion["future_observed"], motion["future_padded"]
                source_frames[slot] = motion["future_source_frame_indices"]
            for j, frame in enumerate(future_frames[:n]):
                if action_valid[j].any():
                    future[slot, j] = self._skill(reader.event_at(arm, int(frame)))
            ee_sequence = meta["effector_regions"].get(arm, {}).get(self.camera)
            regions = [(image, reader.region(ee_sequence, current)), (None, None), (None, None)]
            if event is not None:
                event_ids[slot] = event["event_id"]
                goal_frames[slot] = event["goal_frame"]
                regions[1] = (image, reader.region(event["object_regions"].get(self.camera), current))
                if event["boundary_valid"] and event["goal_frame"] is not None:
                    goal = reader.goal(event["event_id"], self.camera)
                    regions[2] = (goal["rgb"], goal["region"])
            for role, (rgb, region) in enumerate(regions):
                canvases[slot, role], weights[slot, role], role_valid[slot, role] = role_canvas(
                    rgb, region, size)
        return dict(
            inputs=dict(image=image, state=state, state_valid=state_valid, prompt=meta["instructions"][0]),
            targets=dict(
                actions=chunk, action_valid=action_valid, skill_labels=skills,
                skill_valid=skills >= 0, future_skill_labels=future, future_skill_valid=future >= 0,
                role_images=canvases, role_weights=weights, role_valid=role_valid,
                trace=trace, trace_valid=trace_valid,
                future_observed=observed, future_padded=padded,
            ),
            metadata=dict(episode_id=meta["episode_id"], frame=current, action_row=action_row,
                          action_frames=future_frames, event_ids=event_ids, goal_frames=goal_frames,
                          future_frame_indices=current + self.motion_offsets,
                          future_source_frame_indices=source_frames),
        )
