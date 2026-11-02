"""Saved-label audit must catch Event links as well as incorrect numeric values."""
import numpy as np
import pytest

from openpi.policies.eskill_dataset import SCHEMA_VERSION, load_episode
from scripts.preprocess.export import EpisodeWriter
from scripts.preprocess.libero.atomic_vla import batch


@pytest.fixture
def saved_batch_episode(tmp_path, monkeypatch):
    n, h, w = 3, 3, 5
    data = dict(joints=np.arange(n*7).reshape(n, 7).astype(float),
                state=np.arange(n*8).reshape(n, 8).astype(float), actions=np.zeros((n, 7)))
    monkeypatch.setattr(batch, 'source_data', lambda sample: data)
    writer = EpisodeWriter()
    indices = writer.add('indices', np.arange(n))
    valid = writer.add('valid', np.ones(n, bool))

    def numeric(name, values):
        return dict(values=writer.add(name, values), frame_indices=indices,
                    valid=writer.add(name+'_valid', np.ones_like(values, dtype=bool)))

    masks = np.zeros((2, n, h, w), bool)
    masks[0, :, :2, :2] = True
    masks[1, :2, :2, 2:4] = True
    boxes = np.array([[[0, 0, 2, 2]]*n, [[2, 0, 4, 2], [2, 0, 4, 2], [np.nan]*4]], float)
    bits = np.packbits(masks.reshape(2, n, -1), axis=2)
    archive = tmp_path/'native.npz'
    np.savez(archive, boxes=boxes, mask_bits=bits, mask_valid=np.ones((2, n), bool))
    regions = []
    for i in range(2):
        regions.append(dict(frame_indices=indices, boxes_xyxy=writer.add(f'boxes{i}', boxes[i]),
            boxes_valid=writer.add(f'box_valid{i}', np.isfinite(boxes[i]).all(axis=1)),
            masks=writer.add(f'masks{i}', masks[i], pack_mask=True), masks_valid=valid,
            region_extent='full_in_frame'))
    source = [dict(source_id='first', instance='cup_1'), dict(source_id='second', instance='cup_1')]
    events = [dict(event_id=e['source_id'], arm='single', start_frame=start, stop_frame=stop,
                   goal_frame=stop-1, boundary_valid=True, object_regions={'image':regions[1]},
                   source_annotation=e)
              for e, start, stop in [(source[0], 0, 1), (source[1], 1, 3)]]
    state = numeric('state', np.concatenate([data['joints'], data['state']], axis=1))
    state['channels'] = [{}]*15
    actions = numeric('actions', data['actions'])
    actions.update(channels=[{}]*7, conditioning_image_indices={'image':indices})
    metadata = dict(schema_version=SCHEMA_VERSION, frame_count=n, arm_ids=['single'],
        timeline='source_steps', timestamps_s=None,
        cameras={'image':dict(rgb=writer.add('rgb', np.zeros((n, h, w, 3), np.uint8)),
                             frame_indices=indices, valid=valid, height=h, width=w)},
        state=state, actions=actions, effector_positions={'single':numeric('eef', data['state'][:, :3])},
        effector_regions={'single':{'image':regions[0]}}, events=events,
        quality={'manual_review':'pending', 'warnings':[]})
    path = writer.save(tmp_path/'episode', metadata)
    labels = dict(events=source, state_alignment_exceeds_10mm=False)
    native = dict(archive=str(archive), bindings=[dict(source_id=e['source_id'], region_track=0) for e in source])
    return path, {'frames':n}, labels, native


def test_all_frame_audit_retains_known_empty_object_masks(saved_batch_episode):
    path, sample, labels, native = saved_batch_episode
    result = batch.check_saved(load_episode(path), sample, labels, native)
    assert result['mask_bbox_pairs'] == 6
    assert result['active_object_empty_frames'] == [2]
    assert result['nonempty_goal_boxes'] == 1


@pytest.mark.parametrize('case', ['action', 'bbox', 'event_region_link', 'review_status'])
def test_all_frame_audit_rejects_corruption(saved_batch_episode, case):
    path, sample, labels, native = saved_batch_episode
    reader = load_episode(path)
    if case == 'action':
        reader.array(reader.meta['actions']['values'])[0, 0] += 1
    elif case == 'bbox':
        reader.array(reader.meta['events'][0]['object_regions']['image']['boxes_xyxy'])[0, 0] += 1
    elif case == 'event_region_link':
        # Both Events share one target. Checking only the final occurrence misses this error.
        reader.meta['events'][0]['object_regions']['image'] = reader.meta['effector_regions']['single']['image']
    else:
        reader.meta['quality']['manual_review'] = 'accepted'
    with pytest.raises((AssertionError, ValueError)):
        batch.check_saved(reader, sample, labels, native)
