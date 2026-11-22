"""Stage V data semantics and query-independent augmentations on saved Episode fixtures."""
from dataclasses import replace
import json

import numpy as np
import torch

from openpi.augmentation import GroundingAugmentation, ImageAugmentation, affine_eskill_batch, augment_training_batch
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding
from openpi.transforms import horizontal_mirror
from test_current_grounding_training import saved_dataset, ToyTokenizer


def visual_dataset(tmp_path, *, goal=False, **aug):
    source = saved_dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    with np.load(tmp_path / 'arrays.npz') as archive:
        arrays = dict(archive)
    arrays['rgb'] = np.tile(np.arange(32, dtype=np.uint8)[None, None, :, None] * 6, (2, 32, 1, 3))
    for entity, annotation in meta['scene_entities'].items():
        mask = np.zeros((2, 32, 32), np.uint8)
        x, y, r, b = arrays[entity][0]
        mask[:, y:b, x:r] = 1
        key = entity + '_visible'
        arrays[key] = np.packbits(mask.reshape(2, -1), axis=1)
        annotation['regions']['image']['visible_masks']['key'] = key
    np.savez(tmp_path / 'arrays.npz', **arrays)
    if goal:
        # A second training episode witnesses bowl -> plate with the same placement relation.
        meta['scene_entities']['milk']['category'] = 'akita_black_bowl'
        meta['scene_entities']['basket']['category'] = 'flat_stove'
        meta['scene_entities']['pudding']['category'] = 'plate'
        meta['instructions'] = ['put the bowl on the stove']
        second = json.loads(json.dumps(meta))
        second['episode_id'] = 'second'
        second['instructions'] = ['put the bowl on the plate']
        second['events'][1]['interaction_target']['entity_id'] = 'pudding'
        (tmp_path / 'second.json').write_text(json.dumps(second))
        (tmp_path / 'index.json').write_text(json.dumps(dict(episodes=[
            dict(episode_id='test', path='episode.json'),dict(episode_id='second', path='second.json')])))
    path.write_text(json.dumps(meta))
    config = replace(source.config, training_stage='visual_v4',
                     model=replace(source.config.model, current_grounding='visual_v4', attention_layers=(0,)),
                     grounding_augmentation=GroundingAugmentation(**aug), lora_rank=8,
                     loss_weights={'gac': 1., 'current_align': .1, 'subject': 1.})
    return GroundingDataset(config.data_index, config, ToyTokenizer())


def test_current_and_subject_on_same_frame_after_rebinding(tmp_path):
    ds = visual_dataset(tmp_path)
    native = ds[1]
    assert ds[0]['metadata']['entity_id'] == 'milk'
    assert native['metadata']['entity_id'] == 'basket'
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    changed = ds[1]
    assert changed['metadata']['task'] == 'put the chocolate pudding in the basket'
    assert changed['metadata']['counterfactual'] == 'source'
    assert changed['metadata']['event_boundary']
    assert set(changed['inputs']) == {'images','image_mask','tokens','token_mask'}
    assert changed['targets']['current_mask'].sum() == 36
    assert changed['targets']['current_mask'][4,25] == 1
    assert changed['targets']['subject_mask'][13,13] == 1
    assert np.array_equal(native['targets']['subject_mask'], changed['targets']['subject_mask'])
    assert np.array_equal(native['inputs']['images'], changed['inputs']['images'])
    t = collate_grounding([native, changed])['targets']
    torch.testing.assert_close(t['entity_boxes'][torch.arange(2),t['positive_entity']], t['current_box'])


def test_goal_rebinding_and_phase2_geometry_are_shared(tmp_path):
    ds = visual_dataset(tmp_path, goal=True, patch_prob=1., patch_drop_prob=.12, patch_size=3)
    ds.config.horizontal_flip_prob = .5
    ds.config.image_augmentation = ImageAugmentation(crop_prob=1, color_prob=1, rotation_prob=1)
    native = ds[(1, 4, 0)]
    ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, target_swap_prob=1.)
    options = [ds[(1, 4, i)] for i in range(12)]
    changed = next(sample for sample in options if sample['metadata']['counterfactual']=='destination')
    assert changed['metadata']['task'] == 'put the bowl on the plate'
    augmented = augment_training_batch(collate_grounding([native, changed]), ds.config)
    rgb = augmented['inputs']['images']
    torch.testing.assert_close(rgb[0], rgb[1], rtol=0, atol=0)
    t = augmented['targets']
    torch.testing.assert_close(t['subject_mask'][0], t['subject_mask'][1])
    assert not torch.equal(t['current_mask'][0], t['current_mask'][1])
    assert native['metadata']['image_draws'] == changed['metadata']['image_draws']
    assert ds[(1, 5, 0)]['metadata']['image_draws'] != native['metadata']['image_draws']


def test_full_occlusion_skips_current_and_gac_independently(tmp_path):
    ds = visual_dataset(tmp_path)
    reader = ds.reader(0)
    reader._raw.cache_clear()
    with np.load(tmp_path / 'arrays.npz') as archive:
        arrays = dict(archive)
    arrays['basket_visible'][:] = 0
    np.savez(tmp_path / 'arrays.npz', **arrays)
    t = ds[1]['targets']
    assert t['current_box'].sum() > 0 and t['current_mask'].sum() == 0
    assert t['positive_entity'] == -1 and t['subject_mask'].sum() > 0


def test_independent_patch_drop_can_erase_target_and_flip_and_affine_follow_pixels(tmp_path):
    ds = visual_dataset(tmp_path, patch_prob=1, patch_drop_prob=1, patch_size=2, target_margin_px=0)
    t = ds[1]['targets']
    assert t['current_mask'].sum() == 0 and t['positive_entity'] == -1
    assert t['subject_mask'].sum() > 0
    ds.augment = False
    batch = collate_grounding([ds[1]])
    flipped = horizontal_mirror(batch, torch.tensor([True]))
    for role in ('current', 'subject'):
        torch.testing.assert_close(flipped['targets'][role + '_mask'], batch['targets'][role + '_mask'].flip(-1))
    rotated, _ = affine_eskill_batch(flipped, torch.tensor([[0.,0.,1.,1.]]), torch.tensor([90.]))
    for role in ('current', 'subject'):
        torch.testing.assert_close(rotated['targets'][role + '_mask'], flipped['targets'][role + '_mask'].rot90(-1,(-2,-1)))


def test_copy_paste_moves_visible_supervision_and_boxes(tmp_path, monkeypatch):
    ds = visual_dataset(tmp_path)
    def move(image, boxes, eligible, mask, robot, rng, cfg, **options):
        moved = np.roll(mask('milk'), 9, axis=0)
        rgb = image.copy()
        rgb[moved] = 255
        return rgb, dict(boxes, milk=boxes['milk'] + [0,9,0,9]), {'moved_offsets': {'milk': [0,9]}}
    monkeypatch.setattr('openpi.augmentation.copy_objects', move)
    row = ds[0]
    t = row['targets']
    np.testing.assert_array_equal(np.argwhere(t['current_mask'] > 0).min(0), [11,2])
    np.testing.assert_array_equal(t['current_box'] * 32, [2,11,10,19])
    np.testing.assert_array_equal(t['entity_boxes'][t['positive_entity']], t['current_box'])
    assert row['inputs']['images'][0,:,11,2].min() == 1.


def test_conflicting_boundary_skips_current_and_missing_subject_is_optional(tmp_path):
    original = visual_dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    del meta['effector_regions']
    meta['events'][1]['start_frame'] = 0
    path.write_text(json.dumps(meta))
    dataset = GroundingDataset(original.config.data_index, original.config, ToyTokenizer(), augment=False)
    row = dataset[0]
    assert row['metadata']['event_boundary']
    assert not row['targets']['current_mask'].any() and not row['targets']['subject_mask'].any()
    assert row['targets']['positive_entity'] == -1
    assert dataset[1]['targets']['current_mask'].any()
