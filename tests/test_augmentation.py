"""Coordinate alignment, target preservation and augmentation configuration contracts."""
from dataclasses import replace
from unittest import mock
import torch
import pytest

from openpi.augmentation import ImageAugmentation, augment_eskill_batch, affine_eskill_batch
from openpi.training.config import TrainConfig


def sample():
    size = 256
    targets = dict(role_images=torch.full((1, 2, 2, 3, size, size), -1.),
                   role_weights=torch.zeros(1, 2, 2, size, size),
                   role_valid=torch.tensor([[[True, True], [False, False]]]),
                   trace=torch.tensor([[[[.5, .5], [.02, .9]], [[0., 0.], [0., 0.]]]]),
                   trace_valid=torch.tensor([[[True, True], [False, False]]]),
                   gripper=torch.ones(1, 2, 2), gripper_valid=torch.ones(1, 2, 2, dtype=torch.bool),
                   actions=torch.randn(1, 10, 32), skill_labels=torch.tensor([[0, -1]]))
    targets['role_weights'][0, 0, 0, :8, :8] = 1
    targets['role_weights'][0, 0, 1, 100:150, 100:150] = 1
    return dict(inputs=dict(images=torch.rand(1, 2, 3, 224, 224) * 2 - 1,
                            image_mask=torch.ones(1, 2, dtype=torch.bool),
                            arm_presence=torch.tensor([[True, False]])),
                targets=targets, metadata=[dict(frame=0)])


def test_disabled_augmentation_is_exact_and_consumes_no_rng():
    batch = sample()
    rng = torch.get_rng_state().clone()
    assert augment_eskill_batch(batch, ImageAugmentation()) is batch
    assert torch.equal(torch.get_rng_state(), rng)


def test_main_crop_aligns_224_and_256_geometry_and_excludes_outside_labels():
    batch = sample()
    # A linear pixel-center coordinate field exposes subpixel/size conversion errors.
    for size, value in [(224, batch['inputs']['images'][:, :1]),
                        (256, batch['targets']['role_images'])]:
        x = (torch.arange(size) + .5) / size
        value[:] = (2 * x - 1)[None, :].expand(size, -1)
    before = batch['inputs']['images'].clone()
    window = torch.tensor([[.125, .125, .5, .5]])
    result, _ = affine_eskill_batch(batch, window)
    for size, value in [(224, result['inputs']['images'][0, 0, 0]),
                        (256, result['targets']['role_images'][0, 0, 1, 0])]:
        x = .125 + .5 * (torch.arange(size) + .5) / size
        torch.testing.assert_close(value, (2 * x - 1)[None, :].expand(size, -1), atol=2e-6, rtol=0)
    torch.testing.assert_close(result['inputs']['images'][:, 1], before[:, 1], rtol=0, atol=0)
    torch.testing.assert_close(result['targets']['trace'][0, 0, 0], torch.tensor([.75, .75]))
    assert result['targets']['trace_valid'][0, 0].tolist() == [True, False]
    assert result['targets']['role_valid'][0, 0].tolist() == [False, True]
    for name in ('gripper', 'gripper_valid', 'actions', 'skill_labels'):
        assert result['targets'][name] is batch['targets'][name]
    assert torch.equal(batch['inputs']['images'], before)


def test_crop_preserves_black_background_and_identity_rows():
    batch = sample()
    result, _ = affine_eskill_batch(batch, torch.tensor([[.01, .02, .95, .95]]))
    assert (result['targets']['role_images'] == -1).all()
    identity, _ = affine_eskill_batch(batch, torch.tensor([[0., 0., 1., 1.]]))
    for name in batch['targets']:
        torch.testing.assert_close(identity['targets'][name], batch['targets'][name], rtol=0, atol=0)


def test_color_changes_both_views_but_keeps_all_supervision_and_restores_rng():
    batch = sample()
    cfg = ImageAugmentation(color_prob=1.)
    rng = torch.get_rng_state().clone()
    result = augment_eskill_batch(batch, cfg)
    torch.set_rng_state(rng)
    repeated = augment_eskill_batch(batch, cfg)
    assert torch.equal(result['inputs']['images'], repeated['inputs']['images'])
    assert result['metadata'] == repeated['metadata']
    for view in range(2):
        assert not torch.equal(result['inputs']['images'][:, view], batch['inputs']['images'][:, view])
    assert result['inputs']['images'].min() >= -1 and result['inputs']['images'].max() <= 1
    assert result['targets'] is batch['targets']
    assert result['metadata'][0]['augmentation']['counts']['colored_views'] == 2


def test_config_roundtrip_and_recipe_controls():
    base = TrainConfig('index', 'tokens', 'out', 'stats', image_augmentation={'crop_prob': .5})
    assert TrainConfig.from_dict(base.to_dict()).to_dict() == base.to_dict()
    with pytest.raises(ValueError, match='for ESkill'):
        replace(base, model_type='pi05')
    with pytest.raises(ValueError, match='crop_scale'):
        ImageAugmentation(crop_scale=[1., .95])
    with pytest.raises(ValueError, match='color_prob'):
        ImageAugmentation(color_prob=float('nan'))
    with pytest.raises(ValueError, match='rotation_degrees'):
        ImageAugmentation(rotation_degrees=-5)


@pytest.mark.parametrize('angle,quarter_turn', [(90., -1), (-90., 1)])
def test_rotation_pixel_and_point_directions_match_on_both_canvases(angle, quarter_turn):
    batch = sample()
    batch['inputs']['images'][:, 0].fill_(-1.)
    batch['targets']['role_images'].fill_(-1.)
    for size, value in [(224, batch['inputs']['images'][:, 0]),
                        (256, batch['targets']['role_images'].flatten(1, 3))]:
        value[..., size // 2, size * 3 // 4] = 1
    point = torch.tensor([(224 * .75 + .5) / 224, (224 * .5 + .5) / 224])
    batch['targets']['trace'][0, 0, 0] = point
    result, _ = affine_eskill_batch(batch, torch.tensor([[0., 0., 1., 1.]]), torch.tensor([angle]))
    for before, after in [(batch['inputs']['images'][:, 0], result['inputs']['images'][:, 0]),
                          (batch['targets']['role_images'], result['targets']['role_images'])]:
        torch.testing.assert_close(after, torch.rot90(before, quarter_turn, (-2, -1)), atol=5e-5, rtol=0)
    x, y = point
    expected = torch.stack((1 - y, x)) if angle > 0 else torch.stack((y, 1 - x))
    torch.testing.assert_close(result['targets']['trace'][0, 0, 0], expected, atol=1e-6, rtol=0)


def test_small_rotation_black_corners_invalid_points_and_unchanged_wrist():
    batch = sample()
    batch['inputs']['images'][:, 0].fill_(1.)
    batch['targets']['role_images'].fill_(1.)
    batch['targets']['trace'][0, 0, 1] = torch.tensor([.98, .98])
    result, _ = affine_eskill_batch(batch, torch.tensor([[0., 0., 1., 1.]]), torch.tensor([5.]))
    assert (result['inputs']['images'][0, 0, :, 0, 0] == -1).all()
    assert (result['targets']['role_images'][0, 0, 0, :, 0, 0] == -1).all()
    assert not result['targets']['trace_valid'][0, 0, 1]
    assert not result['targets']['role_valid'][0, 0, 0]
    torch.testing.assert_close(result['inputs']['images'][:, 1], batch['inputs']['images'][:, 1], rtol=0, atol=0)


def test_combined_crop_rotation_keeps_crop_fov_and_samples_each_stream_once():
    from torch.nn import functional as F
    batch = sample()
    batch['inputs']['images'][:, 0].fill_(1.)
    batch['targets']['trace'][0, 0, 0] = torch.tensor([.2, .5])
    with mock.patch.object(F, 'grid_sample', wraps=F.grid_sample) as sampler:
        result, _ = affine_eskill_batch(batch, torch.tensor([[.25, .25, .5, .5]]), torch.tensor([45.]))
    assert sampler.call_count == 3  # Main RGB, packed role RGB, packed role weights.
    assert not result['targets']['trace_valid'][0, 0, 0]  # Outside crop cannot re-enter after rotation.
    assert (result['inputs']['images'][0, 0, :, 16, 16] == -1).all()
    for name in ('actions', 'skill_labels', 'gripper', 'gripper_valid'):
        assert result['targets'][name] is batch['targets'][name]


def test_rotation_draw_is_reproducible_and_preserves_a2_crop_color_draws():
    batch = sample()
    a2 = ImageAugmentation(color_prob=1., crop_prob=1.)
    rng = torch.get_rng_state().clone()
    previous = augment_eskill_batch(batch, a2)
    torch.set_rng_state(rng)
    rotated = augment_eskill_batch(batch, replace(a2, rotation_prob=1.))
    torch.set_rng_state(rng)
    repeated = augment_eskill_batch(batch, replace(a2, rotation_prob=1.))
    assert torch.equal(rotated['inputs']['images'], repeated['inputs']['images'])
    before, after = previous['metadata'][0]['augmentation'], rotated['metadata'][0]['augmentation']
    assert before['crop'] == after['crop'] and before['color_factors'] == after['color_factors']
    assert abs(after['rotation_degrees']) <= 5
    assert after['counts']['rotated_samples'] == 1


def test_color_jitter_keeps_artificial_rotation_corners_black():
    batch = sample()
    batch['inputs']['images'].fill_(1.)
    torch.manual_seed(7)
    result = augment_eskill_batch(batch, ImageAugmentation(color_prob=1., rotation_prob=1.))
    angle = result['metadata'][0]['augmentation']['rotation_degrees']
    assert abs(angle) > 1
    assert (result['inputs']['images'][0, 0, :, 0, 0] == -1).all()
