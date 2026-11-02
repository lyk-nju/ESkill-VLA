"""Export and validate the current JSON + NPZ preprocessing format."""
import json
import os
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import SCHEMA_VERSION, EpisodeReader, load_episode


class EpisodeWriter:
    """Collect arrays while a source adapter constructs an EpisodeData dictionary."""

    def __init__(self):
        self.arrays = {}

    def add(self, key, values, *, pack_mask=False):
        if key in self.arrays:
            raise ValueError(f'Duplicate array key: {key}')
        values = np.asarray(values)
        if values.dtype.hasobject:
            raise ValueError('Object/pickle arrays are not part of the format')
        ref = dict(uri='arrays.npz', key=key, shape=list(values.shape), dtype=str(values.dtype))
        if pack_mask:
            if values.dtype != np.bool_ or values.ndim != 3:
                raise ValueError('Masks must be bool [N,H,W]')
            self.arrays[key] = np.packbits(values.reshape(len(values), -1), axis=1, bitorder='big')
            ref['encoding'] = 'packbits'
        else:
            self.arrays[key] = values
        return ref

    def save(self, directory, metadata):
        """One new episode directory. JSON is written only after structural validation."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/'arrays.npz', **self.arrays)
        reader = EpisodeReader(directory/'episode.json', metadata=metadata)
        validate_episode(reader)
        reader.path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return reader.path



def validate_episode(reader):
    """Index, shape and pixel-coordinate checks; not semantic/manual acceptance."""
    m = reader.meta
    count = m['frame_count']
    if count < 1 or not m['arm_ids'] or len(set(m['arm_ids'])) != len(m['arm_ids']):
        raise ValueError('Need a nonempty timeline and unique physical arms')
    if m['timeline'] not in {'source_steps','timestamps'}:
        raise ValueError('Unknown timeline convention')
    if m['timeline'] == 'timestamps' and m['timestamps_s'] is None:
        raise ValueError('Timestamp timeline requires actual timestamps')
    if m['timestamps_s'] is not None:
        times = reader.array(m['timestamps_s'])
        if times.shape != (count,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Timestamps must be finite, strictly increasing seconds')

    def indices(sequence, *, unique):
        value = reader.array(sequence['frame_indices'])
        if (value.ndim != 1 or value.dtype.kind not in 'iu' or
            np.any(value < 0) or np.any(value >= count) or
            np.any(value[1:] <= value[:-1] if unique else value[1:] < value[:-1])):
            raise ValueError('Invalid stream indices')
        return value

    def valid(ref, shape):
        value = reader.array(ref)
        if value.shape != shape or value.dtype != np.bool_:
            raise ValueError('Validity must be bool with the declared shape')
        return value

    for stream in m['cameras'].values():
        n = len(indices(stream,unique=True))
        valid(stream['valid'],(n,))
        ref = stream['rgb']
        shape = [n,stream['height'],stream['width'],3]
        if list(ref['shape']) != shape or ref['dtype'] != 'uint8':
            raise ValueError('RGB logical shape/dtype mismatch')
        if ref.get('encoding') == 'jpeg_sequence':
            offsets = reader._raw(ref['uri'],ref['offsets_key'])
            raw = reader._raw(ref['uri'],ref['key'])
            if (offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu' or offsets[0] != 0
                or np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(raw)
                or raw.dtype != np.uint8 or raw.ndim != 1):
                raise ValueError('JPEG offsets do not match the source bytes')
        else:
            reader.array(ref)
        calibration = stream.get('calibration')
        if calibration is not None:
            for key, width in [('intrinsics', 3), ('extrinsics', 4), ('image_transform', 3)]:
                ref = calibration.get(key)
                if ref is not None:
                    matrix = reader.array(ref)
                    if matrix.shape not in ((width, width), (n, width, width)) or not np.isfinite(matrix).all():
                        raise ValueError(f'Invalid camera calibration: {key}')

    sequences = [m.get('state'),m.get('actions'),*m['effector_positions'].values()]
    for sequence in sequences:
        if sequence is None:
            continue
        n = len(indices(sequence,unique=False))
        values = reader.array(sequence['values'])
        available = valid(sequence['valid'],values.shape)
        if values.ndim != 2 or len(values) != n or not np.isfinite(values[available]).all():
            raise ValueError('Invalid numeric sequence')
        if 'channels' in sequence and len(sequence['channels']) != values.shape[1]:
            raise ValueError('Numeric channels do not match values')
    if m.get('actions') is not None:
        action = m['actions']
        action_indices = indices(action,unique=False)
        for camera,ref in action['conditioning_image_indices'].items():
            rows = reader.array(ref)
            image_indices = indices(m['cameras'][camera],unique=True)
            if (rows.shape != action_indices.shape or rows.dtype.kind not in 'iu'
                or np.any(rows < -1) or np.any(rows >= len(image_indices))):
                raise ValueError('Invalid action-conditioning image rows')
            known = rows >= 0
            if np.any(image_indices[rows[known]] > action_indices[known]):
                raise ValueError('Future image used as an action condition')

    regions = [(camera,seq) for streams in m['effector_regions'].values()
               for camera,seq in streams.items()]
    seen, previous = set(), {}
    for event in sorted(m['events'],key=lambda e:e['start_frame']):
        start, stop, goal, arm = (event[k] for k in ['start_frame','stop_frame','goal_frame','arm'])
        if event['event_id'] in seen or arm not in m['arm_ids']:
            raise ValueError('Duplicate event ID or unknown arm')
        seen.add(event['event_id'])
        if not 0 <= start < stop <= count or start < previous.get(arm,0):
            raise ValueError('Invalid or overlapping same-arm intervals')
        if goal is not None and not start <= goal <= min(stop,count-1):
            raise ValueError('Goal must be an explicit recorded endpoint')
        previous[arm] = stop
        regions.extend(event['object_regions'].items())

    for camera,sequence in regions:
        n = len(indices(sequence,unique=True))
        stream = m['cameras'][camera]
        h,w = stream['height'],stream['width']
        if sequence['region_extent'] not in {'visible','full_in_frame','unknown'}:
            raise ValueError('Unknown region extent')
        for field,flag,shape in [('boxes_xyxy','boxes_valid',(n,4)),('masks','masks_valid',(n,h,w))]:
            ref,availability = sequence[field],sequence[flag]
            if (ref is None) != (availability is None):
                raise ValueError('Each region representation needs its own validity')
            if ref is None:
                continue
            values = reader.array(ref)
            active = valid(availability,(n,))
            if values.shape != shape:
                raise ValueError('Region shape must match its camera and frame indices')
            if field == 'masks':
                if values.dtype != np.bool_:
                    raise ValueError('Masks must decode to bool')
            else:
                b = values[active]
                if (not np.isfinite(b).all() or np.any(b[:,:2] < 0)
                    or np.any(b[:,2] > w) or np.any(b[:,3] > h)
                    or np.any(b[:,0] >= b[:,2]) or np.any(b[:,1] >= b[:,3])):
                    raise ValueError('Invalid source-pixel xyxy bbox')
    return {'frames': count, 'events': len(seen), 'structural_check': 'passed'}



def write_index(path, episode_paths):
    """Build a small batch index after the listed episodes have been saved."""
    path = Path(path).resolve()
    entries = []
    identities = set()
    for episode_path in episode_paths:
        reader = load_episode(episode_path)
        m,source = reader.meta,reader.meta['source']
        identity = (source['dataset'],source['revision'],source.get('parent_trajectory_id') or source['episode_id'])
        if identity in identities:
            raise ValueError('Duplicate source episode in batch index')
        identities.add(identity)
        entries.append(dict(episode_id=m['episode_id'],path=os.path.relpath(reader.path,path.parent),
                            frames=m['frame_count'],events=len(m['events'])))
    value = dict(schema_version=SCHEMA_VERSION,episodes=entries,
                 frame_count=sum(e['frames'] for e in entries),event_count=sum(e['events'] for e in entries))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value
