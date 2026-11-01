"""Adapt existing LIBERO sources to display inputs; no model or label inference."""
from io import BytesIO
import importlib

import numpy as np
from PIL import Image

from .render import Episode, Event, Region


def decode(value):
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get('bytes')
    if value is None:
        return None
    with Image.open(BytesIO(value)) as image:
        return np.asarray(image.convert('RGB')).copy()


def region_lookup(regions, identity, frame, camera='image'):
    if regions is None or identity is None:
        return None
    item = regions.index.get((camera, identity, frame))
    if item is None:
        return None
    return Region(item.get('bbox'), regions.mask(item) if item.get('mask') else None)


def attach_regions(view, record, regions):
    """The active Event retains its own object callback for its future goal frame."""
    if record['frame_count'] != view.frame_count or record['source']['key'] != view.key:
        raise ValueError('Region record belongs to a different source episode')
    view.events = []
    for event in record.get('events', []):
        identity = event.get('object_id')
        view.events.append(Event(
            id=event['source_id'], start_frame=event.get('start_frame'), stop_frame=event.get('stop_frame'),
            goal_frame=event.get('goal_frame'), skill=event.get('skill'),
            object_name=event.get('object_name') or identity,
            object_region=(lambda frame, target=identity: region_lookup(regions, target, frame)) if identity else None))
    grippers = [t['id'] for t in record.get('targets', []) if t['role'] == 'gripper']
    if len(grippers) == 1:
        view.gripper_region = lambda frame: region_lookup(regions, grippers[0], frame)
    view.annotation_label = 'MODEL REGIONS / SOURCE EVENTS'
    view.review_status = record.get('manual_review', 'pending')
    if record.get('annotation_error'):
        view.note = 'TEXT ANNOTATION FAILED'
    return view


def from_episode(episode, record=None, regions=None, *, source=None, source_fps=None):
    """Adapter for existing source.Episode instances (GuidedVLA and AtomicVLA)."""
    state = episode.signals.get('state')
    xyz = episode.poses.get('single')
    view = Episode(
        key=episode.entry['key'], source=source or episode.entry.get('format'),
        frame_count=episode.count, rgb=lambda frame: episode.rgb('image', frame),
        task=episode.instructions[0] if episode.instructions else None,
        events=[Event(e['source_id'], e.get('start_frame'), e.get('stop_frame'), e.get('goal_frame'),
                      e.get('skill') or e.get('source_skill')) for e in episode.evidence.get('intervals', [])],
        gripper_state=state[:, 6:] if state is not None else None,
        actions=episode.signals.get('actions'), eef_xyz=xyz,
        xyz_note='state[:3] / coordinate and unit pending' if xyz is not None else None,
        source_fps=source_fps, playback_fps=episode.fps,
        annotation_label='SOURCE LABELS / REGIONS MISSING')
    return attach_regions(view, record, regions) if record is not None else view


def from_rlds(source, sample, data):
    """Raw release adapters. Atomic endpoint conversion stays in its source adapter."""
    n = sample['frames']
    def matrix(key):
        value = data.get(key)
        return np.asarray(value, np.float32).reshape(n, -1) if value is not None else None
    state, actions = matrix('steps/observation/state'), matrix('steps/action')
    images = data.get('steps/observation/image')
    view = Episode(
        key=sample['key'], source=source, frame_count=n,
        rgb=lambda frame: decode(images[frame]) if images is not None and 0 <= frame < len(images) else None,
        task=sample.get('task'), gripper_state=state[:, 6:] if state is not None else None,
        actions=actions, eef_xyz=state[:, :3] if state is not None else None,
        xyz_note=('state[:3] = EEF - target position / world axes; target pose absent'
                  if source == 'lilo-vla' else 'state[:3] / coordinate and unit pending'),
        annotation_label='SOURCE LABELS / REGIONS MISSING', playback_fps=10.0)
    if source == 'atomic-vla' and sample.get('annotation'):
        adapter = importlib.import_module('scripts.preprocess.libero.atomic_vla.source')
        events = adapter.source_intervals(sample['annotation'], n)
        for event in events:
            view.events.append(Event(event['source_id'], event['start_frame'], event['stop_frame'],
                                     event['goal_frame'], event['skill']))
        if any('boundary_correction' in event for event in events):
            view.annotation_label = 'SOURCE SKILLS / SHARED BOUNDARY -> NEXT EVENT'
        if any('skill_correction' in event for event in events):
            view.annotation_label = 'REVIEWED SKILL: PUSH / ORIGINAL SOURCE: PICK'
    elif source == 'lilo-vla':
        skill = sample.get('metadata', {}).get('skill_name')
        view.events = [Event('released_atomic_skill', 0, n, n-1, skill)]
    elif source == 'cyclevla':
        view.events = [Event('released_subtask', 0, n, n-1)]
        view.annotation_label = 'SOURCE SUBTASK TEXT / SKILL MISSING'
        view.note = 'PUBLISHED REPEATED TAIL RETAINED'
    return view


def from_saved(reader, *, arm='single', camera='image', playback_fps=10.0):
    """Render a saved EpisodeData through the shared layout, with no source scripts.

    This LIBERO layout needs streams already paired by source row. Other time axes
    require an explicit display adapter; this function never resamples them.
    """
    from openpi.policies.eskill_dataset import load_episode
    if not hasattr(reader, 'meta'):
        reader = load_episode(reader)
    m = reader.meta
    if arm not in m['arm_ids']:
        raise ValueError('Requested arm is not physically present')
    expected = np.arange(m['frame_count'])

    def aligned(sequence):
        if sequence is None:
            return None
        if not np.array_equal(reader.array(sequence['frame_indices']),expected):
            raise ValueError('LIBERO display requires an explicitly aligned source-step stream')
        values = reader.array(sequence['values']).copy()
        return np.where(reader.array(sequence['valid']),values,np.nan)

    if not np.array_equal(reader.array(m['cameras'][camera]['frame_indices']),expected):
        raise ValueError('LIBERO display requires one source image row per index')

    def region(sequence, frame):
        value = reader.region(sequence,frame)
        return Region(value['bbox'],value['mask']) if value is not None else None

    events = []
    for e in m['events']:
        if e['arm'] != arm:
            continue
        source = e['source_annotation']
        name = source.get('instance')
        part = (source.get('scope') or {}).get('part')
        if name and part and part != 'whole':
            name += ' / '+part
        events.append(Event(e['event_id'],e['start_frame'],e['stop_frame'],
                            e['goal_frame'] if e['boundary_valid'] else None,e['skill'] if e['skill_valid'] else None,
                            name,lambda frame,s=e['object_regions'].get(camera):region(s,frame)))
    state = aligned(m.get('state'))
    columns = [i for i,c in enumerate(m['state']['channels'])
               if c['arm'] == arm and c['name'].startswith('gripper_')] if state is not None else []
    effector = m['effector_regions'].get(arm,{}).get(camera)
    pose = m['effector_positions'].get(arm)
    extents = {seq['region_extent'] for e in m['events'] if e['arm']==arm
               for c,seq in e['object_regions'].items() if c==camera}
    if effector is not None:
        extents.add(effector['region_extent'])
    label = 'SAVED LABELS / '+('FULL-IN-FRAME' if extents=={'full_in_frame'} else ', '.join(sorted(extents)).upper())
    xyz_note = (f"{pose['coordinate_frame'] or 'FRAME UNKNOWN'} / {pose['unit'] or 'UNIT UNKNOWN'} / "
                f"{pose['reference_point'] or 'EEF POINT UNVERIFIED'}") if pose else None
    return Episode(key=m['episode_id'],frame_count=m['frame_count'],rgb=lambda frame:reader.rgb(camera,frame),
        source=m['source']['dataset'],task=m['instructions'][0] if m['instructions'] else None,events=events,
        gripper_state=state[:,columns] if columns else None,actions=aligned(m.get('actions')),
        eef_xyz=aligned(pose),xyz_note=xyz_note,gripper_region=lambda frame:region(effector,frame),
        playback_fps=playback_fps,source_fps=m['nominal_fps'],annotation_label=label,
        review_status=m['quality'].get('manual_review','pending'),
        note='STATE ALIGNMENT REVIEW PENDING' if 'state_alignment_exceeds_10mm' in m['quality'].get('warnings',[])
             else 'SAVED LABELS / USER REVIEW PENDING')


def render_saved(episode_path, video_path):
    """Render stored labels without recomputing regions or instance bindings."""
    import json
    from pathlib import Path

    from openpi.policies.eskill_dataset import load_episode
    from scripts.preprocess.artifacts import digest
    from scripts.preprocess.export import validate_episode
    from .render import Renderer, render_video

    reader = load_episode(episode_path)
    validate_episode(reader)
    view = from_saved(reader)
    video_path = Path(video_path)
    report = render_video(view, video_path)
    Image.fromarray(Renderer(view).frame(view.frame_count//2)).save(video_path.with_suffix('.jpg'))
    report.update(episode=str(reader.path), episode_sha256=digest(reader.path),
                  arrays_sha256=digest(reader.path.parent/'arrays.npz'), manual_review='pending')
    video_path.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    return report
