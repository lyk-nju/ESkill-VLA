"""Export and validate the current JSON + NPZ preprocessing format."""
import json
import os
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import SCHEMA_VERSION, EpisodeReader, load_episode


class EpisodeWriter:
    """Collect arrays while a source adapter constructs an EpisodeData dictionary.

    A saved Episode directory is self-contained: numeric fields, regions and calibration go to
    `arrays.npz`; encoded RGB frames go uncompressed to `images.npz` with their original bytes.
    """

    def __init__(self):
        self.arrays, self.images = {}, {}

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

    def add_encoded(self, key, frames, height, width):
        """Encoded frames (e.g. JPEG bytes) for one camera, stored without decoding or re-encoding."""
        if key in self.images or not frames:
            raise ValueError(f'Duplicate or empty encoded image stream: {key}')
        offsets = np.zeros(len(frames) + 1, np.int64)
        offsets[1:] = np.cumsum([len(frame) for frame in frames])
        self.images[key + '_bytes'] = np.frombuffer(b''.join(frames), np.uint8)
        self.images[key + '_offsets'] = offsets
        return dict(uri='images.npz', key=key + '_bytes', offsets_key=key + '_offsets',
                    encoding='jpeg_sequence', shape=[len(frames), height, width, 3], dtype='uint8')

    def save(self, directory, metadata, *, exist_ok=False):
        """One new episode directory. JSON is written only after structural validation."""
        add_derived(metadata, self)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=exist_ok)
        np.savez_compressed(directory/'arrays.npz', **self.arrays)
        if self.images:
            np.savez(directory/'images.npz', **self.images)
        reader = EpisodeReader(directory/'episode.json', metadata=metadata)
        validate_episode(reader)
        reader.path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return reader.path



def event_index(metadata, arm):
    """Per-frame position in metadata['events'] of this arm's Event; -1 outside every Event."""
    index = np.full(metadata['frame_count'], -1, np.int16)
    for i, event in enumerate(metadata['events']):
        if event['arm'] == arm:
            index[event['start_frame']:event['stop_frame']] = i
    return index


def effector_pixels(meta, arm, camera, array):
    """EEF positions projected into a statically calibrated camera, in source pixels.

    None without confirmed geometry: metric world positions of a named reference point and a
    static world_to_camera_cv calibration. `array(ref)` resolves a data reference.
    """
    from openpi.transforms import project_points
    stream = meta['cameras'][camera]
    calibration = stream.get('calibration')
    positions = meta['effector_positions'].get(arm)
    if (not calibration or not positions or positions.get('unit') != 'm'
            or positions.get('coordinate_frame') != 'world'
            or positions.get('reference_point') in (None, '', 'unknown')
            or calibration.get('world_frame') != 'world'
            or calibration.get('extrinsics_convention') != 'world_to_camera_cv'):
        return None
    matrices = [array(calibration[k]) for k in ('intrinsics', 'extrinsics', 'image_transform')]
    if any(m.ndim != 2 for m in matrices):
        return None  # Per-row (moving) cameras would need the current view for every future point.
    uv, valid = project_points(array(positions['values']), *matrices, (stream['height'], stream['width']))
    return array(positions['frame_indices']), uv, valid & array(positions['valid']).all(-1)


def add_derived(metadata, writer):
    """Fields derived from the saved labels, stored once: per-frame Event index and EEF pixels."""
    def array(ref):
        values = writer.arrays[ref['key']]
        if ref.get('encoding') == 'packbits':
            n, h, w = ref['shape']
            return np.unpackbits(values, axis=1, count=h*w, bitorder='big').reshape(n, h, w).astype(bool)
        return values
    metadata['event_index'] = {arm: writer.add(f'event_index_{arm}', event_index(metadata, arm))
                               for arm in metadata['arm_ids']}
    metadata['effector_pixels'] = {}
    for arm in metadata['effector_positions']:
        for camera in metadata['cameras']:
            result = effector_pixels(metadata, arm, camera, array)
            if result is None:
                continue
            frames, uv, valid = result
            metadata['effector_pixels'].setdefault(arm, {})[camera] = dict(
                frame_indices=writer.add(f'eef_pixel_frames_{arm}_{camera}', frames),
                values=writer.add(f'eef_uv_{arm}_{camera}', uv),
                valid=writer.add(f'eef_uv_valid_{arm}_{camera}', valid))


def derived_keys(metadata):
    keys = [ref['key'] for ref in metadata.get('event_index', {}).values()]
    for cameras in metadata.get('effector_pixels', {}).values():
        for pixels in cameras.values():
            keys += [pixels[k]['key'] for k in ('frame_indices', 'values', 'valid')]
    return keys


def writer_from_saved(reader, *, drop=()):
    """An EpisodeWriter holding a saved Episode's arrays, without derived fields, for re-saving."""
    import copy
    meta = copy.deepcopy(reader.meta)
    skip = set(derived_keys(meta)) | set(drop)
    meta.pop('event_index', None)
    meta.pop('effector_pixels', None)
    writer = EpisodeWriter()
    with np.load(reader.path.parent/'arrays.npz', allow_pickle=False) as archive:
        writer.arrays = {k: archive[k] for k in archive.files if k not in skip
                         and not k.startswith(('event_index_', 'eef_pixel_frames_', 'eef_uv_'))}
    return writer, meta


def validate_episode(reader):
    """Index, shape and pixel-coordinate checks; not semantic/manual acceptance."""
    m = reader.meta
    count = m['frame_count']

    def references(node):
        if isinstance(node, dict):
            if 'uri' in node and node.get('key') is not None:
                yield node
                return
            for value in node.values():
                yield from references(value)
        elif isinstance(node, list):
            for value in node:
                yield from references(value)
    # Source artifacts are provenance records (key=None); every data reference stays in the directory.
    if any(ref['uri'] not in ('arrays.npz', 'images.npz') for ref in references(
            {k: v for k, v in m.items() if k != 'source'})):
        raise ValueError('Data references must be arrays.npz or images.npz inside the Episode directory')
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
        if sequence['boxes_xyxy'] is None:
            raise ValueError('Training reads regions as bboxes; every region needs boxes_xyxy')

    if set(m['event_index']) != set(m['arm_ids']):
        raise ValueError('event_index must cover every arm')
    for arm, ref in m['event_index'].items():
        if not np.array_equal(reader.array(ref), event_index(m, arm)):
            raise ValueError(f'event_index for {arm} differs from the Event table')
    for arm, cameras in m['effector_pixels'].items():
        for camera, pixels in cameras.items():
            expected = effector_pixels(m, arm, camera, reader.array)
            if expected is None:
                raise ValueError(f'effector_pixels {arm}/{camera} without confirmed geometry')
            for got, want in zip((pixels['frame_indices'], pixels['values'], pixels['valid']), expected):
                if not np.array_equal(reader.array(got), want, equal_nan=True):
                    raise ValueError(f'effector_pixels {arm}/{camera} differ from the calibrated projection')
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
