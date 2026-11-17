"""Coordinate alignment, target preservation and augmentation configuration contracts."""
from dataclasses import replace
from unittest import mock
import torch
import pytest

from openpi.augmentation import ImageAugmentation, augment_eskill_batch, affine_eskill_batch
from openpi.training.config import TrainConfig


def grounding_batch():
    batch = sample()
    batch['inputs']['images'] = batch['inputs']['images'][:, :1]
    batch['inputs']['image_mask'] = batch['inputs']['image_mask'][:, :1]
    batch['inputs']['task_inputs'] = {'images':batch['inputs']['images'].clone()}
    batch['targets'] = {k:v for k,v in batch['targets'].items() if k.startswith('role_')}
    batch['metadata'] = [dict(mirror_eligible=True, spatial_augmentation_eligible=True,
                              augmentation=dict(copy_applied=True,target_swapped=True))]
    return batch


def test_grounding_geometry_aligns_both_roles_without_touching_original_task_view():
    from openpi.transforms import horizontal_mirror
    batch = grounding_batch()
    for size,value in [(224,batch['inputs']['images']), (256,batch['targets']['role_images'])]:
        value[:] = (2*(torch.arange(size)+.5)/size-1)[None,:].expand(size,-1)
    original_task = batch['inputs']['task_inputs']['images'].clone()
    mirrored = horizontal_mirror(batch,torch.tensor([True]))
    rotated,_ = affine_eskill_batch(mirrored,torch.tensor([[0.,0.,1.,1.]]),torch.tensor([90.]))
    for size,value in [(224,rotated['inputs']['images'][0,0,0]), (256,rotated['targets']['role_images'][0,0,1,0])]:
        expected = (1-2*(torch.arange(size)+.5)/size)[:,None].expand(-1,size)
        torch.testing.assert_close(value,expected,atol=5e-5,rtol=0)
    torch.testing.assert_close(rotated['inputs']['task_inputs']['images'],original_task,rtol=0,atol=0)
    assert 'trace' not in rotated['targets'] and 'actions' not in rotated['targets']


def test_combined_grounding_augmentation_is_replayable_and_keeps_online_metadata():
    from types import SimpleNamespace
    from openpi.augmentation import augment_training_batch
    batch = grounding_batch()
    config = SimpleNamespace(horizontal_flip_prob=1.,image_augmentation=ImageAugmentation(
        color_prob=1.,crop_prob=1.,rotation_prob=1.))
    state = torch.get_rng_state().clone()
    first = augment_training_batch(batch,config)
    torch.set_rng_state(state)
    second = augment_training_batch(batch,config)
    torch.testing.assert_close(first['inputs']['images'],second['inputs']['images'],rtol=0,atol=0)
    for key in first['targets']:
        torch.testing.assert_close(first['targets'][key],second['targets'][key],rtol=0,atol=0)
    info = first['metadata'][0]
    assert info['mirrored'] and info['augmentation']['copy_applied'] and info['augmentation']['target_swapped']
    assert info['augmentation']['counts']['rotated_samples'] == 1
    assert info['augmentation']['counts']['trace_after'] == 0
    assert first['inputs']['task_inputs'] is batch['inputs']['task_inputs']
    batch['metadata'][0].update(mirror_eligible=False,spatial_augmentation_eligible=False)
    config.image_augmentation = ImageAugmentation(crop_prob=1.,rotation_prob=1.)
    directional = augment_training_batch(batch,config)
    torch.testing.assert_close(directional['inputs']['images'],batch['inputs']['images'],rtol=0,atol=0)
    assert not directional['metadata'][0]['mirrored']
    assert directional['metadata'][0]['augmentation']['counts']['rotated_samples'] == 0


def test_grounding_color_changes_inputs_only_and_configuration_accepts_combination():
    from openpi.models.pi0_config import ESkillConfig
    from openpi.augmentation import augment_training_batch
    config = TrainConfig('index','tokenizer','out','stats',training_stage='grounding',
        model=ESkillConfig.tiny(visual_primitive_version='v3',grounding_subject=True),
        image_augmentation=ImageAugmentation(color_prob=1.))
    batch=grounding_batch()
    result=augment_training_batch(batch,config)
    assert not torch.equal(result['inputs']['images'],batch['inputs']['images'])
    assert result['targets'] is batch['targets']
    assert result['inputs']['task_inputs'] is batch['inputs']['task_inputs']
    restored=TrainConfig.from_dict(replace(config,horizontal_flip_prob=.5).to_dict())
    assert restored.horizontal_flip_prob == .5 and restored.image_augmentation.color_prob == 1.


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


def test_official_pi05_recipe_uses_integer_fixed_crop_and_color_ranges():
    from pathlib import Path
    from openpi.augmentation import augment_training_batch
    config = TrainConfig.load(Path(__file__).resolve().parents[1] / 'examples/libero/train_current_grounding.json')
    batch = grounding_batch()
    result = augment_training_batch(batch, config)
    info = result['metadata'][0]
    assert not info.get('mirrored', False)
    window = torch.tensor(info['augmentation']['crop'])
    torch.testing.assert_close(window[2:] * 224, torch.tensor([212., 212.]))
    torch.testing.assert_close(window[:2] * 224, (window[:2] * 224).round())
    assert (window[:2] >= 0).all() and (window[:2] * 224 <= 12).all()
    counts = info['augmentation']['counts']
    assert counts['crop_samples'] == counts['rotated_samples'] == counts['colored_views'] == 1
    factors = torch.tensor(info['augmentation']['color_factors'])
    assert ((factors >= torch.tensor([.7,.6,.5])) & (factors <= torch.tensor([1.3,1.4,1.5]))).all()
    assert result['inputs']['task_inputs'] is batch['inputs']['task_inputs']


def edge_grounding_batch(box, role=0, *, occluded=False):
    import numpy as np
    from openpi.transforms import role_canvas, resize_image
    image = np.full((256,256,3),128,np.uint8)
    x0,y0,x1,y1 = box
    if not occluded:
        image[y0:y1,x0:x1] = [255,0,0]
    canvas,weight,valid = role_canvas(image,{'bbox':box},256)
    batch = grounding_batch()
    batch['targets']['role_weights'][0,0].zero_()
    batch['targets']['role_weights'][0,0,:,100:150,100:150] = 1
    batch['inputs']['images'][0,0] = torch.tensor(resize_image(image,224).transpose(2,0,1))/127.5-1
    batch['targets']['role_images'][0,0,role] = torch.tensor(canvas.transpose(2,0,1))/127.5-1
    batch['targets']['role_weights'][0,0,role] = torch.tensor(weight)
    batch['targets']['role_valid'][0,0,role] = valid
    boxes = torch.zeros(1,2,2,4)
    boxes[0,0] = torch.tensor([100.,100.,150.,150.])/256
    boxes[0,0,role] = torch.tensor(box)/256
    batch['targets']['role_core_boxes'] = boxes
    return batch


@pytest.mark.parametrize('role',[0,1])
@pytest.mark.parametrize('angle',[-5.,0.,5.])
def test_crop_disables_target_when_only_background_margin_remains(role,angle):
    batch = edge_grounding_batch([0,100,12,120],role)
    result,_ = affine_eskill_batch(batch,torch.tensor([[12/224,0,212/224,212/224]]),torch.tensor([angle]))
    assert result['targets']['role_weights'][0,0,role].sum()>0  # Reproduces the old false positive.
    assert not result['targets']['role_valid'][0,0,role]
    assert result['targets']['role_valid'][0,0,1-role]
    assert not result['targets']['role_valid'][0,1].any()


def test_rotation_disables_corner_target_with_only_context_remaining():
    batch = edge_grounding_batch([0,0,4,10])
    result,_ = affine_eskill_batch(batch,torch.tensor([[0.,0.,1.,1.]]),torch.tensor([5.]))
    assert result['targets']['role_weights'][0,0,0].sum()>0
    assert not result['targets']['role_valid'][0,0,0]


@pytest.mark.parametrize('occluded',[False,True])
def test_partially_cropped_and_natively_occluded_targets_remain_valid(occluded):
    batch = edge_grounding_batch([0,100,20,120],occluded=occluded)
    result,_ = affine_eskill_batch(batch,torch.tensor([[12/224,0,212/224,212/224]]),torch.tensor([5.]))
    assert result['targets']['role_valid'][0,0,0]


def test_mirror_updates_core_boxes_before_crop():
    from openpi.transforms import horizontal_mirror
    batch = edge_grounding_batch([0,100,12,120])
    mirrored = horizontal_mirror(batch,torch.tensor([True]))
    torch.testing.assert_close(mirrored['targets']['role_core_boxes'][0,0,0],
                               torch.tensor([244.,100.,256.,120.])/256)
    # The right-edge target is now removed by cropping the right side.
    result,_ = affine_eskill_batch(mirrored,torch.tensor([[0.,0.,212/224,212/224]]))
    assert result['targets']['role_weights'][0,0,0].sum()>0
    assert not result['targets']['role_valid'][0,0,0]
    assert batch['targets']['role_valid'][0,0,0]  # No mutation of the unaugmented source.


def test_crop_updates_diagnostics_and_excludes_invalid_role_from_visual_loss():
    from openpi.models.pi0_config import ESkillConfig
    from openpi.training.config import create_model
    batch = edge_grounding_batch([0,100,12,120])
    with mock.patch('torch.randint',side_effect=[torch.tensor([12]),torch.tensor([0])]):
        result = augment_eskill_batch(batch,ImageAugmentation(crop_prob=1.,crop_scale=(.95,.95)))
    info = result['metadata'][0]['augmentation']
    assert not info['current_valid'] and info['subject_valid']
    assert info['counts']['role_before']==2 and info['counts']['role_after']==1
    cfg = ESkillConfig.tiny(canvas_size=256,role_queries=16,grounding_subject=True,
                            visual_primitive_version='v3',freeze_vlm=False)
    model = create_model(TrainConfig('none','none','none','none',model=cfg,training_stage='grounding',
                                    smoke=True,device='cpu',precision='float32'))
    anchors = torch.randn(1,2,2,cfg.role_queries,cfg.hidden_dim,requires_grad=True)
    losses = model.visual_supervision(anchors,result['targets'],role_names=('current','subject'))
    assert losses['current'].item()==0 and torch.isfinite(losses['subject'])
    sum(losses.values()).backward()
    assert anchors.grad[0,0,0].count_nonzero()==0
    assert anchors.grad[0,0,1].abs().sum()>0
