from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grounding_fixtures import ToyTokenizer, visual_dataset
from openpi.augmentation import GroundingAugmentation, ImageAugmentation, augment_training_batch
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding
from openpi.training.config import TrainConfig
from scripts.preprocess.libero.counterfactual_states import state_windows


def dataset(path, *, plate=False, **kwargs):
    original = visual_dataset(path)
    if plate:
        meta = json.loads((path/'episode.json').read_text())
        meta['scene_entities']['pudding']['category'] = 'plate'
        (path/'episode.json').write_text(json.dumps(meta))
    entry = dict(source='milk', destination='basket', task='put the milk in the basket',
                 pre_grasp=[[0, 1]], holding=[[1, 2]], sources_touching_destination={})
    (path/'counterfactual_states.json').write_text(json.dumps(dict(version=2,
        index_sha256=hashlib.sha256((path/'index.json').read_bytes()).hexdigest(), episodes={'test': entry})))
    aug = GroundingAugmentation(target_swap_prob=1, state_consistent_swaps=True, **kwargs)
    cfg = replace(original.config, grounding_augmentation=aug)
    return GroundingDataset(cfg.data_index, cfg, ToyTokenizer())


def test_pick_swaps_source_and_holding_never_swaps_source(tmp_path):
    ds = dataset(tmp_path)
    pick, place = ds[0], ds[1]
    assert (pick['metadata']['task'], pick['metadata']['entity_id']) == ('put the chocolate pudding in the basket', 'pudding')
    assert pick['targets']['current_mask'][4, 25] and not pick['targets']['current_mask'][4, 4]
    assert (place['metadata']['task'], place['metadata']['entity_id']) == ('put the milk in the basket', 'basket')
    ds.state_windows['test']['pre_grasp'] = []
    assert ds[0]['metadata']['counterfactual'] == 'none'


def test_place_changes_preposition_and_destination_only(tmp_path, monkeypatch):
    ds = dataset(tmp_path, plate=True)
    sample = ds.sample(1, review=True)
    assert sample['review']['paired_queries'] == [('put the milk on the plate', 'pudding', 'destination')]
    assert sample['targets']['current_mask'][4, 25] and not sample['targets']['current_mask'][22, 22]
    validation = GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer(), augment=False)
    assert validation[1]['metadata']['task'] == 'put the milk in the basket'

    def obscure_source(image, *args, **kwargs):
        drop = np.zeros(image.shape[:2], bool)
        drop[2:10, 2:10] = True  # The plate is completely visible, but the payload is gone.
        return image, drop

    monkeypatch.setattr('openpi.augmentation.mask_grounding_patches', obscure_source)
    assert ds.sample(1, review=True)['review']['paired_queries'] == []


def test_holdout_canonical_triples_in_both_directions(tmp_path):
    ds = dataset(tmp_path, plate=True, held_out_transfers=(('milk', 'on', 'plate'), ('plate', 'in', 'basket')))
    assert all(ds.sample(i, review=True)['review']['paired_queries'] == [] for i in (0, 1))
    recipe = TrainConfig.load(Path(__file__).parents[1] / 'examples/libero/train_visual_primitive_state_swaps.json')
    assert len(set(recipe.grounding_augmentation.held_out_transfers)) == 9
    assert ('akita_black_bowl', 'on', 'plate') not in recipe.grounding_augmentation.held_out_transfers


def test_state_file_integrity_and_factual_fallback(tmp_path):
    ds = dataset(tmp_path)
    p = ds.config.counterfactual_states_path
    doc = json.loads(p.read_text())
    doc['episodes']['test']['holding'] = [[0, 1]]
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='invalid holding'):
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
    doc['episodes']['test'] = None
    p.write_text(json.dumps(doc))
    assert GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())[0]['metadata']['counterfactual'] == 'none'
    doc['index_sha256'] = 'other'
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='dataset index'):
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())


def test_native_windows_exclude_close_foreign_contact_release_and_invalid_rows():
    pick = dict(start_frame=0, stop_frame=6)
    place = dict(start_frame=6, stop_frame=13)
    valid = np.ones(13, bool)
    command = np.r_[-np.ones(3), np.ones(7), -np.ones(1), np.ones(2)]
    touch = np.zeros(13, bool)
    touch[[1, 5]] = True
    contact = np.zeros(13, bool)
    contact[5] = True
    pads = np.arange(13) >= 5
    dest = np.zeros(13, bool)
    assert state_windows(pick, place, valid, command, touch, contact, pads, dest) == ([[0, 1], [2, 3]], [[6, 10]])
    dest[7] = True
    assert state_windows(pick, place, valid, command, touch, contact, pads, dest)[1] == [[6, 7]]
    contact[:] = False
    assert not state_windows(pick, place, valid, command, touch, contact, pads, dest)[0]


@pytest.mark.parametrize('scale', [(.95, .95), (.75, .95)])
def test_host_view_matches_original_batch_geometry_and_color(tmp_path, scale):
    ds = dataset(tmp_path, plate=True)
    ds.config.image_augmentation = ImageAugmentation(crop_prob=1, crop_scale=scale, color_prob=1, rotation_prob=1)
    ds.config.horizontal_flip_prob = .5
    sample = ds[(1, 3, 10)]
    factual = dict(sample, metadata={k: v for k, v in sample['metadata'].items() if k != 'main_view'})
    new, old = [augment_training_batch(collate_grounding([s]), ds.config) for s in (sample, factual)]
    torch.testing.assert_close(new['inputs']['images'], old['inputs']['images'], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(new['targets']['current_mask'], old['targets']['current_mask'], atol=1e-5, rtol=1e-5)


def test_final_crop_eliminates_candidate_before_sampling(tmp_path, monkeypatch):
    ds = dataset(tmp_path)
    # Remove the rightmost half of the image, including Pudding. The gripper remains in view.
    monkeypatch.setattr('openpi.augmentation.sample_host_view', lambda *a: (0., 0., .56, 1., 0., True))
    sample = ds.sample(0, review=True)
    assert sample['metadata']['augmentation']['transfer_state'] == 'pre_grasp'
    assert not sample['review']['paired_queries'] and sample['metadata']['entity_id'] == 'milk'


def test_swap_keeps_the_same_host_rgb_and_rejects_patch_obscured_candidate(tmp_path, monkeypatch):
    ds = dataset(tmp_path)
    swapped = ds[(0, 2, 1)]
    ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, target_swap_prob=0)
    factual = ds[(0, 2, 2)]
    assert swapped['metadata']['entity_id'] != factual['metadata']['entity_id']
    assert swapped['metadata']['main_view'] == factual['metadata']['main_view']
    np.testing.assert_array_equal(swapped['inputs']['images'], factual['inputs']['images'])
    ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, target_swap_prob=1)

    def cover_half(image, *args, **kwargs):
        drop = np.zeros(image.shape[:2], bool)
        drop[2:5, 24:30] = True
        return image, drop

    monkeypatch.setattr('openpi.augmentation.mask_grounding_patches', cover_half)
    assert ds.sample(0, review=True)['review']['paired_queries'] == []


def test_source_already_touching_destination_is_excluded(tmp_path):
    ds = dataset(tmp_path)
    ds.state_windows['test']['sources_touching_destination'] = {'pudding': [[0, 1]]}
    assert not ds.sample(0, review=True)['review']['paired_queries']
