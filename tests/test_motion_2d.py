"""Pixel geometry and fixed-time supervision, independent of MuJoCo and CUDA."""
import numpy as np
import pytest

from openpi.transforms import future_points, future_targets, project_points, resize_geometry, resize_image, resize_points


def test_projection_camera_pose_rotation_and_source_flip():
    # Camera at world (1, 2, 3), looking along world +X; camera right is world -Z.
    rotation = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, -rotation @ np.array([1, 2, 3])
    k = np.array([[100, 0, 49.5], [0, 100, 39.5], [0, 0, 1]])
    flip = np.array([[-1, 0, 99], [0, 1, 0], [0, 0, 1]])
    uv, valid = project_points([[3, 2, 3], [3, 2.2, 2.8], [0, 2, 3], [3, 2, -10]],
                               k, pose, flip, (80, 100))
    np.testing.assert_allclose(uv[:2], [[49.5, 39.5], [39.5, 49.5]], atol=1e-5)
    np.testing.assert_array_equal(valid, [True, True, False, False])
    assert uv[-1, 0] < 0  # Out-of-screen projection is not clamped to a false edge target.


def test_letterbox_integer_rounding_half_pixels_and_original_fov():
    assert resize_geometry((101, 203), 224) == (111, 224, 56, 0)
    image = resize_image(np.full((101, 203, 3), 255, np.uint8), 224)
    assert image[56:167].all() and not image[:56].any() and not image[167:].any()
    points = np.array([[0, 0], [202, 100], [101, -1]], np.float32)
    normalized, valid = resize_points(points, [True, True, True], (101, 203), 224)
    np.testing.assert_allclose(normalized[0], [.5/203, (56+.5*111/101)/224])
    np.testing.assert_allclose(normalized[1], [202.5/203, (56+100.5*111/101)/224])
    assert (normalized[-1] > 0).all() and not valid[-1]  # Padding is not camera FOV.


def test_future_samples_exact_indices_and_partial_tails():
    indices = np.array([0, 2, 4, 7])
    points = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], np.float32)
    target, valid = future_points(indices, points, [True]*4, [0, 2, 7], [2, 4, 7])
    np.testing.assert_array_equal(target[0], points[1:])
    np.testing.assert_array_equal(valid, [[True]*3, [True, False, False], [False]*3])
    assert not target[~valid].any()
    # Adjacent frames with different Event labels do not shorten this time window.
    target, valid = future_points(np.arange(5), np.tile([12, 34], (5, 1)), [True]*5, [1], [1, 2, 3])
    np.testing.assert_array_equal(target, [[[12, 34]]*3])
    assert valid.all()  # A stationary effector retains its absolute nonzero location.


def test_invalid_records_and_offsets_are_not_fabricated():
    points = np.array([[1, 2], [np.nan, np.nan], [3, 4]], np.float32)
    target, valid = future_points([0, 1, 2], points, [True, False, True], [0], [1, 2, 3])
    assert np.isfinite(target).all()
    np.testing.assert_array_equal(valid, [[False, True, False]])
    for offsets in ([0, 1], [2, 2], [2, 1], [1., 2.]):
        with pytest.raises(ValueError):
            future_points([0, 1, 2], points, [True]*3, [0], offsets)


def test_hold_last_uses_actual_episode_endpoint_and_keeps_fixed_offsets():
    points = np.array([[1, 2], [3, 4], [8, 9]], np.float32)
    target, valid = future_points([0, 4, 10], points, [True]*3, [0, 4, 10], [4, 8, 16],
                                 tail_policy='hold_last')
    # +16 uses the recorded endpoint, not the last sampled horizon (+4).
    np.testing.assert_array_equal(target[0], [[3, 4], [0, 0], [8, 9]])
    np.testing.assert_array_equal(valid[0], [True, False, True])  # Interior gap stays missing.
    np.testing.assert_array_equal(target[1], [[0, 0], [8, 9], [8, 9]])
    np.testing.assert_array_equal(target[2], [[8, 9]]*3)
    assert valid[2].all()


def test_hold_last_does_not_extend_an_invalid_endpoint_or_replace_it_with_an_earlier_point():
    for endpoint in ([8, 9], [np.nan, np.nan]):
        target, valid = future_points([0, 4, 10], [[1, 2], [3, 4], endpoint],
                                     [True, True, False], [0, 10], [4, 10, 40], tail_policy='hold_last')
        np.testing.assert_array_equal(valid, [[True, False, False], [False]*3])
        assert np.isfinite(target).all() and not target[~valid].any()


def test_hold_last_supports_a_single_record_without_changing_mask_policy():
    for policy, expected in [('mask', False), ('hold_last', True)]:
        target, valid = future_points([7], [[11, 19]], [True], [7], [8, 16, 20, 28, 36, 40],
                                     tail_policy=policy)
        assert (valid == expected).all()
        np.testing.assert_array_equal(target, [[[11, 19] if expected else [0, 0]]*6])


def test_known_episode_endpoint_is_not_the_last_available_record():
    result = future_targets([0, 4], [[1, 2], [3, 4]], [True, True], [0, 8], [4, 8, 12],
                            tail_policy='hold_last', last_frame=8)
    np.testing.assert_array_equal(result['trace_valid'], [[True, False, False], [False]*3])
    assert not result['future_padded'].any()
    np.testing.assert_array_equal(result['future_source_frame_indices'], [[4, -1, -1], [-1]*3])
