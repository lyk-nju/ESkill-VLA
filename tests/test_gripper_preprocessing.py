"""Source command supplementation and data statistics must not invent timing or labels."""
import json
from unittest import mock

import numpy as np
import pytest

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.libero.atomic_vla.gripper import REVISION, supplement
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.audit_training import audit
from test_eskill_data import fixture, dataset


def source_fixture(root):
    index = fixture(root, times=False)
    path = root / 'episode/episode.json'
    reader = EpisodeReader(path)
    original = root / 'source/0.npz'  # Source artifact names can be annotation IDs.
    original.parent.mkdir()
    np.savez(original, actions=reader.array(reader.meta['actions']['values']),
             state=reader.array(reader.meta['state']['values'])[:, -8:])
    meta = reader.meta
    meta['source'].update(dataset='AtomicVLA / openvla/modified_libero_rlds', revision=REVISION,
                          artifacts=[{'uri': str(original)}])
    meta['actions']['gripper_convention'] = None
    path.write_text(json.dumps(meta))
    return index, path, original


def test_supplement_preserves_arrays_checks_sources_and_is_idempotent(tmp_path):
    index, path, _ = source_fixture(tmp_path)
    before = (path.parent / 'arrays.npz').read_bytes()
    sums = tmp_path / 'SHA256SUMS'
    original_sums = (f'{digest(path)}  ./episode/episode.json\n'
                     f'{digest(path.parent / "arrays.npz")}  ./episode/arrays.npz\n')
    sums.write_text(original_sums)
    dry = supplement(index)
    assert dry['pending_episodes'] == 1 and not dry['written']
    assert json.loads(path.read_text())['actions']['gripper_convention'] is None
    assert sums.read_text() == original_sums
    report = supplement(index, write=True)
    assert report['updated_episodes'] == 1 and report['close'] == 4 and report['open'] == 5
    assert report['checksum_entries_updated'] == 1
    assert sums.read_text().splitlines() == [f'{digest(path)}  ./episode/episode.json',
        original_sums.splitlines()[1]]
    assert supplement(index, write=True)['updated_episodes'] == 0
    assert supplement(index, write=True)['checksum_entries_updated'] == 0
    # Repair a stale checksum after metadata was already written, without rewriting arrays.
    sums.write_text(original_sums)
    recovered = supplement(index, write=True)
    assert recovered['updated_episodes'] == 0 and recovered['checksum_entries_updated'] == 1
    assert (path.parent / 'arrays.npz').read_bytes() == before
    assert json.loads(path.read_text())['timestamps_s'] is None
    # Command readiness must not be reported as missing just because the clock is absent.
    result = audit(index)
    assert result['command_ready_episodes'] == 1 and result['physical_time_ready_episodes'] == 0
    assert result['effective_supervision']['samples'] == 9
    assert result['event_statistics']['event_length_frames']['median'] == 4.5
    assert result['by_skill']['Place']['action_rows'] == 5


def test_source_mismatch_prevents_metadata_write(tmp_path):
    index, path, original = source_fixture(tmp_path)
    meta_before = path.read_bytes()
    with np.load(original) as source:
        state, action = source['state'], source['actions']
    action[0, 6] *= -1
    np.savez(original, state=state, actions=action)
    with pytest.raises(AssertionError):
        supplement(index, write=True)
    assert path.read_bytes() == meta_before


def test_effective_coverage_equals_training_and_masks_remain_independent(tmp_path):
    def edit(meta, arrays):
        arrays['actions'][2, -1] = 0
        arrays['eef_valid'][1] = False
    index = fixture(tmp_path, edit=edit)
    with mock.patch.object(EpisodeReader, 'rgb', side_effect=AssertionError('audit decoded RGB')):
        result = audit(index, motion_stride=1)
    data = dataset(index)
    for name, key in [('trace', 'trace_valid'), ('gripper', 'gripper_valid')]:
        validity = np.stack([data[i]['targets'][key][0] for i in range(len(data))])
        stats = result['effective_supervision'][name]
        assert stats['valid_by_offset'] == validity.sum(0).tolist()
        assert stats['valid_points_per_sample'] == np.bincount(validity.sum(1), minlength=9).tolist()
    split = tmp_path / 'split.json'
    split.write_text(json.dumps(dict(train=['parent'], val=[])))
    assert audit(index, split_path=split, subset='train')['by_split']['train']['events'] == 2
    assert audit(index, split_path=split, subset='val')['episodes'] == 0


def test_motion_review_saves_the_same_targets_and_renders_without_inference(tmp_path):
    from scripts.preprocess.libero.review_motion import MotionRenderer, save_targets
    from scripts.preprocess.libero.render import Episode
    def edit(meta, arrays):
        meta['events'][0].update(stop_frame=8, goal_frame=7)
        meta['events'][1]['start_frame'] = 8
    index = fixture(tmp_path, times=False, edit=edit)
    reader = EpisodeReader(index.parent / 'episode/episode.json')
    path = tmp_path / 'targets.npz'
    save_targets(reader, path, np.arange(5, 41, 5))
    with np.load(path) as file:
        labels = {key: file[key] for key in file.files}
    data = dataset(index, motion_stride=5)
    for i in range(len(data)):
        for key in ('trace', 'trace_valid', 'gripper', 'gripper_valid'):
            np.testing.assert_array_equal(labels[key][i], data[i]['targets'][key])
    view = Episode('test', 9, rgb=lambda i: reader.rgb('image', i))
    assert MotionRenderer(view, labels).frame(0).shape == (1340, 1920, 3)
