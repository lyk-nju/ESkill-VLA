"""Review contracts: missing values, bbox-only inputs, event identity and video timing."""
from copy import deepcopy
from types import SimpleNamespace

import av
import numpy as np

from scripts.preprocess.libero.render import Episode, Event, Region, Renderer, Trajectory3D, overlay, render_video
from scripts.preprocess.libero.view import attach_regions, from_rlds


def test_render_without_system_fonts_uses_pillow_fallback(monkeypatch):
    from PIL import ImageFont
    from scripts.preprocess.libero.render import font
    original = ImageFont.truetype
    def no_system_font(name, *args, **kwargs):
        if isinstance(name, str):
            raise OSError('No system fonts installed')
        return original(name, *args, **kwargs)
    font.cache_clear()
    monkeypatch.setattr(ImageFont, 'truetype', no_system_font)
    try:
        assert Renderer(Episode('missing fonts', 1)).frame(0).shape == (1340, 1920, 3)
        for kind in ('sans', 'bold', 'mono'):
            assert font(16, kind).getbbox('LIBERO')[2] > 0
    finally:
        font.cache_clear()


def test_missing_inputs_and_sparse_numeric_values_still_render():
    view = Episode('missing', 3)
    assert Renderer(view).frame(1).shape == (1340, 1920, 3)
    view.eef_xyz = np.asarray([[0, 1, 2], [np.nan, np.nan, np.nan], [2, 3, 4]])
    view.actions = np.full((3, 7), np.nan)
    view.gripper_state = np.full((3, 2), np.nan)
    assert Renderer(view).frame(1).shape == (1340, 1920, 3)


def test_bbox_only_and_mask_only_inputs_draw_independently_without_mutating_rgb():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    result = np.asarray(overlay(rgb, Region([1, 1, 4, 4]), '#58adf5'))
    np.testing.assert_array_equal(result[1, 1], [88, 173, 245])
    np.testing.assert_array_equal(result[4, 4], [0, 0, 0])  # xyxy upper edge is exclusive.
    assert not rgb.any()
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:4, 2:4] = True
    result = np.asarray(overlay(rgb, Region(mask=mask), '#79d9ad'))
    assert result[2, 2, 1] > 0 and not result[0, 0].any()


def test_goal_keeps_old_event_object_when_goal_is_at_next_event_start():
    calls = []
    def region(name, frame):
        calls.append((name, frame))
        return Region([1, 1, 3, 3])
    view = Episode('handover', 4, rgb=lambda _: np.zeros((8, 8, 3), dtype=np.uint8), events=[
        Event('old', 0, 2, 2, 'Place', 'cup', lambda t: region('cup', t)),
        Event('new', 2, 4, 3, 'Close', 'door', lambda t: region('door', t)),
    ])
    renderer = Renderer(view)
    renderer.frame(1)
    assert calls == [('cup', 1), ('cup', 2)]
    calls.clear()
    renderer.frame(2)
    assert calls == [('door', 2), ('door', 3)]
    view.events.append(Event('overlap', 0, 4, 3, 'Pick'))
    assert Renderer(view).active_event(1) is None  # Do not invent which overlapping label wins.


def test_raw_adapter_preserves_source_labels_and_does_not_invent_object_regions():
    sample = {'key': 'source', 'frames': 3, 'task': 'pick cup', 'annotation': {'segments': [
        {'start_frame': 0, 'end_frame': 3, 'primary_action_verb': 'pick',
         'chain_of_thought': 'This is step 1/1 of task, pick up cup, and the atomic skill is pick'}]}}
    original = deepcopy(sample)
    view = from_rlds('atomic-vla', sample, {})
    assert view.events[0].goal_frame == 2 and view.events[0].stop_frame == 3
    assert view.events[0].skill == 'Pick'
    assert view.events[0].object_name is None and view.events[0].object_region is None
    assert view.gripper_region is None and sample == original
    view = from_rlds('cyclevla', {'key': 'subtask', 'frames': 3, 'task': 'close gripper'}, {})
    assert view.events[0].skill is None  # A free-text instruction is not a supplied class label.


def test_regions_with_only_boxes_do_not_require_mask_archives():
    class Boxes:
        index = {('image', 'cup', 0): {'bbox': [1, 1, 3, 3]}}
        def mask(self, item):
            raise AssertionError('No mask field was supplied')
    record = {'source': {'key': 'case'}, 'frame_count': 1, 'manual_review': 'pending',
              'events': [{'source_id': 'event', 'start_frame': 0, 'stop_frame': 1,
                          'goal_frame': 0, 'skill': 'Pick', 'object_id': 'cup'}],
              'targets': []}
    view = attach_regions(Episode('case', 1), record, Boxes())
    assert view.events[0].object_region(0).bbox == [1, 1, 3, 3]
    assert view.review_status == 'pending'


def test_video_retains_source_order_color_and_display_timing(tmp_path):
    colors = [40, 100, 160]
    view = Episode('colors', 3, rgb=lambda t: np.full((12, 12, 3), [colors[t], 12, 5], dtype=np.uint8),
                   playback_fps=10)
    path = tmp_path/'review.mp4'
    result = render_video(view, path)
    decoded = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            assert abs(float(frame.pts*frame.time_base)-index/10) < 1e-6
            pixels = frame.to_ndarray(format='rgb24')[300:400, 200:300]
            decoded.append(pixels.mean(axis=(0, 1)))
    assert len(decoded) == result['frames'] == 3
    assert np.allclose([p[0] for p in decoded], colors, atol=6)
    assert all(p[0] > p[2]+25 for p in decoded)


def test_existing_grounding_entry_uses_common_libero_renderer(tmp_path):
    from scripts.preprocess.visual_grounding.render import render_episode
    identity = {'key': 'bridge', 'format': 'guidedvla.libero'}
    source = SimpleNamespace(entry=identity, count=2, fps=10, instructions=['pick cup'],
                             cameras={'image': None}, signals={}, poses={}, evidence={
                                 'intervals': [], 'fps_meaning': 'display_only'},
                             rgb=lambda camera, t: np.zeros((8, 8, 3), dtype=np.uint8))
    record = {'source': identity, 'frame_count': 2, 'targets': [], 'events': [], 'regions': []}
    report = render_episode(source, record, tmp_path, tmp_path/'video/bridge.mp4', ['image'])
    assert report['width'] == 1920 and report['height'] == 1340
    assert report['fps'] == 10 and report['source_fps'] is None
    assert report['video'] == 'video/bridge.mp4' and report['manual_review'] == 'pending'


def test_3d_axes_preserve_equal_units_and_missing_xyz_is_not_interpolated():
    xyz = np.asarray([[1.0, 2.0, 3.0], [np.nan, 2.1, 3.1], [1.2, 2.4, 3.8]])
    original = xyz.copy()
    plot = Trajectory3D(xyz)
    np.testing.assert_allclose(np.diff(plot.bounds, axis=1), np.repeat(np.diff(plot.bounds[:1], axis=1), 3, axis=0))
    assert plot.point(0) is not None and plot.point(1) is None and plot.point(2) is not None
    assert plot.point(3) is None
    np.testing.assert_array_equal(xyz, original)
    assert plot.frame(1, Event('event', 0, 2, 2, 'Place')).size == (720, 390)


def test_3d_missing_and_stationary_positions_remain_distinct():
    assert Trajectory3D(None).image is None
    assert Trajectory3D(np.full((3, 3), np.nan)).image is None
    plot = Trajectory3D(np.ones((3, 3)))
    assert plot.image is not None and plot.point(1) is not None
    np.testing.assert_allclose(plot.points[0], plot.points[2])
