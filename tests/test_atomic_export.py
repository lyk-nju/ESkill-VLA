"""AtomicVLA export and calibration write self-contained Episodes that training reads directly."""
from io import BytesIO
import json

import numpy as np
from PIL import Image
import pytest

from openpi.policies.eskill_dataset import EpisodeDataset, load_episode
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.atomic_vla import export, motion_2d


def jpeg(value):
    data = BytesIO()
    Image.fromarray(np.full((256, 256, 3), value, np.uint8)).save(data, format='JPEG', quality=95)
    return data.getvalue()


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    n = 4
    main, wrist = [jpeg(10 * t) for t in range(n)], [jpeg(100 + 10 * t) for t in range(n)]
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum([len(f) for f in main])
    cache = tmp_path / 'source.npz'
    np.savez(cache, image_bytes=np.frombuffer(b''.join(main), np.uint8), image_offsets=offsets,
             joints=np.zeros((n, 7), np.float32), actions=np.arange(n * 7, dtype=np.float32).reshape(n, 7),
             state=np.tile(np.arange(8, dtype=np.float32), (n, 1)))
    boxes = np.tile(np.array([0, 0, 8, 8], float), (2, n, 1))
    masks = np.zeros((2, n, 256, 256), bool)
    masks[:, :, :8, :8] = True
    archive = tmp_path / 'regions.npz'
    np.savez(archive, boxes=boxes, mask_bits=np.packbits(masks.reshape(2, n, -1), axis=2),
             mask_valid=np.ones((2, n), bool))
    for name in ('annotations.json', 'labels.json', 'scene.npz'):
        (tmp_path / name).write_text('{}')
    source = dict(source_id='event_000', start_frame=0, stop_frame=n, goal_frame=n - 1, skill='Pick',
                  instance='cup_1', scope=dict(part='whole'), status='rule_bound', description='pick cup',
                  geometry=dict(visual_geom_names=['cup']))
    sample = dict(key='ep', frames=n, task='pick cup', annotation_id=0, shard='shard', byte_offset=0,
                  source_cache=str(cache), source_cache_sha256=digest(cache))
    labels = dict(source_cache_sha256=sample['source_cache_sha256'], events=[source],
                  state_alignment_exceeds_10mm=False,
                  source_scene=dict(excerpt_sha256='scene', excerpt_path=str(tmp_path / 'scene.npz')))
    rules = dict(binder='b', catalog='c')
    region = tmp_path / 'region.json'
    region.write_text(json.dumps(dict(
        key='ep', frames=n, source_cache_sha256=sample['source_cache_sha256'], scene_match=dict(excerpt_sha256='scene'),
        region_extent='full_in_frame', binder_sha256='b', rules_sha256='c', archive=str(archive),
        archive_sha256=digest(archive), targets=[dict(instance='cup_1', part='whole', geom_names=['cup'])],
        bindings=[dict(source, region_track=0)], gripper_pose_source='joints', object_pose_source='scene',
        max_eef_error_m=0., scene_alignment='ok')))
    monkeypatch.setattr(export, 'wrist_frames', lambda sample: (wrist, main))
    manifest = dict(rules=rules, revision='r', annotation_path=str(tmp_path / 'annotations.json'),
                    annotation_sha256='a', annotation_revision='ar')
    return (sample, labels, region), dict(manifest=manifest, labels_path=tmp_path / 'labels.json'), main, wrist


@pytest.fixture
def exported(inputs, tmp_path):
    args, options, main, wrist = inputs
    return export.export_episode(*args, tmp_path / 'exported', **options), main, wrist


def test_exported_episode_holds_both_cameras_and_no_external_references(exported):
    path, main, wrist = exported
    reader = load_episode(path)
    assert sorted(p.name for p in path.parent.iterdir()) == ['arrays.npz', 'episode.json', 'images.npz']
    for camera, frames in (('image', main), ('wrist_image', wrist)):
        ref = reader.meta['cameras'][camera]['rgb']
        assert ref['uri'] == 'images.npz'
        assert reader._raw(ref['uri'], ref['key']).tobytes() == b''.join(frames)
    np.testing.assert_array_equal(reader.rgb('wrist_image', 2), np.asarray(Image.open(BytesIO(wrist[2]))))


def test_mismatched_rlds_main_camera_is_rejected(inputs, monkeypatch, tmp_path):
    args, options, main, wrist = inputs
    monkeypatch.setattr(export, 'wrist_frames', lambda sample: (wrist, wrist))
    with pytest.raises(ValueError, match='not paired'):
        export.export_episode(*args, tmp_path / 'second', **options)
    assert not (tmp_path / 'second').exists()


def test_calibrated_episode_is_self_contained_training_data(exported, tmp_path):
    path, main, wrist = exported
    reader = load_episode(path)
    n = reader.meta['frame_count']
    pose = np.eye(4)
    pose[2, 3] = 10.  # Camera 10 m in front of the world origin, so the EEF projects in view.
    # camera_parameters() returns JSON lists, as recorded in motion.json.
    calibration = dict(source_camera='image', intrinsic=[[10., 0, 127.5], [0, 10., 127.5], [0, 0, 1]],
                       world_to_camera=pose.tolist(), image_transform=np.eye(3).tolist())
    destination = tmp_path / 'training' / 'episodes' / '0000'
    destination.mkdir(parents=True)
    training = motion_2d.write_training_episode(reader, destination, calibration)
    # A new export can itself be reused; no removed eef_uv/eef_valid cache keys are needed.
    record = dict(episode_id=reader.meta['episode_id'], source_episode_sha256=digest(path),
                  arrays='arrays.npz', arrays_sha256=digest(destination/'arrays.npz'),
                  camera={**calibration, 'static': True}, checks={}, audit=str(destination/'audit.json'))
    (destination/'motion.json').write_text(json.dumps(record))
    _, uv, valid, _, _, _ = motion_2d.load_projection(destination/'motion.json', reader)
    saved = load_episode(training)
    pixels = saved.meta['effector_pixels']['single']['image']
    np.testing.assert_array_equal(uv, saved.array(pixels['values']))
    np.testing.assert_array_equal(valid, saved.array(pixels['valid']))
    path.parent.rename(tmp_path / 'source_removed')  # Training must not read the exported source.
    from scripts.preprocess.export import write_index
    write_index(tmp_path / 'training' / 'index.json', [training])
    data = EpisodeDataset(tmp_path / 'training' / 'index.json', action_horizon=2,
                          cameras=['image', 'wrist_image'], skill_names=['Pick'], image_size=224)
    sample = data[1]
    assert len(sample['inputs']['images']) == 2 and np.all(sample['inputs']['images'][1] > 90)
    assert sample['targets']['trace_valid'][0].any()
    meta = load_episode(training).meta
    assert meta['cameras']['image']['calibration']['intrinsics']['uri'] == 'arrays.npz'
    assert list(meta['effector_pixels']['single']) == ['image']  # Wrist view has no calibration.
