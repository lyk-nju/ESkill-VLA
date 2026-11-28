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
from scripts.preprocess.libero.counterfactual_states import contacts, state_windows


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


@pytest.mark.parametrize(('reference', 'receiver_movable'), [('ramekin', True), ('plate', False)])
def test_spatial_copy_releases_independent_place_receiver(tmp_path, monkeypatch, reference, receiver_movable):
    ds = dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    meta['instructions'] = [f'pick up the black bowl next to the {reference} and place it on the plate']
    entities = meta['scene_entities']
    entities['milk']['category'] = 'akita_black_bowl'
    entities['basket']['category'] = 'plate'
    entities['basket']['visible_fraction']['key'] = 'visible_fraction'
    entities['pudding']['category'] = 'glazed_rim_porcelain_ramekin'
    path.write_text(json.dumps(meta))
    ds.reader.cache_clear()
    ds.state_windows['test'] = None
    ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, copy_prob=1.)

    def copy(image, boxes, eligible, *args, **kwargs):
        assert ('basket' in eligible) == receiver_movable
        assert 'milk' not in eligible
        return image, boxes, {'moved_offsets': {}}

    monkeypatch.setattr('openpi.augmentation.copy_objects', copy)
    ds[1]


@pytest.mark.parametrize(('protected', 'free_window', 'released'), [
    (None, True, False), ({}, True, True), ({'milk': [[0, 1]]}, True, False), ({}, False, False),
])
def test_projected_pick_overlap_uses_verified_v3_window(tmp_path, monkeypatch, protected, free_window, released):
    ds = dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    meta['scene_entities']['milk']['visible_fraction']['key'] = 'visible_fraction'
    path.write_text(json.dumps(meta))
    ds.reader.cache_clear()
    if protected is not None:
        ds.state_windows['test']['protected'] = protected
    if not free_window:
        ds.state_windows['test']['pre_grasp'] = []
    ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, copy_prob=1.)

    def copy(image, boxes, eligible, *args, **kwargs):
        # The milk box overlaps the expanded gripper box in the toy scene.
        assert ('milk' in eligible) == released
        assert kwargs['release_priority_source'] == released
        return image, boxes, {'moved_offsets': {}}

    monkeypatch.setattr('openpi.augmentation.copy_objects', copy)
    ds[0]


def test_future_goal_preserves_the_pick_target_and_respects_holdout(tmp_path):
    ds = dataset(tmp_path, plate=True)
    ds.state_windows['test']['sources_touching_destination']['pudding'] = [[0, 1]]
    sample = ds.sample(0, review=True)
    assert sample['review']['paired_queries'] == [('put the milk on the plate', 'milk', 'future_goal')]
    assert sample['metadata']['counterfactual'] == 'future_goal'
    assert sample['targets']['current_mask'][4, 4] and not sample['targets']['current_mask'][4, 25]
    ds.held_out = {('milk', 'on', 'plate')}
    sample = ds.sample(0, review=True)
    assert sample['metadata']['task'] == 'put the milk in the basket'
    assert sample['metadata']['augmentation']['swap_rejections']['held_out'] == 1


def test_offscreen_unseen_instance_keeps_category_name_ambiguous(tmp_path):
    ds = dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    meta['scene_entities']['extra_pudding'] = dict(kind='object', category='chocolate_pudding', regions={})
    path.write_text(json.dumps(meta))
    ds = GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
    sample = ds.sample(0, review=True)
    assert sample['review']['paired_queries'] == []
    assert sample['metadata']['augmentation']['swap_rejections']['ambiguous_reference'] == 4


def test_combined_sample_uses_exact_pasted_mask(tmp_path, monkeypatch):
    ds = dataset(tmp_path)
    import openpi.policies.grounding_dataset as gd

    original_visible = gd.visible_pixels
    def visible(reader, sequence, row, shape):
        mask = original_visible(reader, sequence, row, shape)
        if mask is not None and sequence['visible_masks']['key'] == 'pudding_visible':
            mask[31, 0] = True
        return mask

    def copy(image, boxes, *args, **kwargs):
        new_boxes = dict(boxes, pudding=np.array([20, 10, 26, 16]))
        return image, new_boxes, dict(moved_offsets={'pudding': [-4, 8]})

    monkeypatch.setattr(gd, 'visible_pixels', visible)
    monkeypatch.setattr('openpi.augmentation.copy_objects', copy)
    sample = ds.sample(0, review=True)
    current = sample['targets']['current_mask']
    assert sample['metadata']['entity_id'] == 'pudding'
    assert current[12, 22] and current.sum() == 36


def test_crop_after_copy_can_restore_original_layout(tmp_path, monkeypatch):
    ds = dataset(tmp_path)
    def copy(image, boxes, *args, **kwargs):
        return np.zeros_like(image), dict(boxes, milk=np.array([24, 21, 32, 29])), dict(
            moved_offsets={'milk': [22, 19]}, copy_applied=True, copy_count=1)
    monkeypatch.setattr('openpi.augmentation.copy_objects', copy)
    monkeypatch.setattr('openpi.augmentation.sample_host_view', lambda *a: (0., 0., .56, 1., 0., True))
    sample = ds.sample(0, review=True)
    assert sample['metadata']['augmentation']['copy_reason'] == 'target_lost_after_view'
    assert sample['metadata']['augmentation']['moved_offsets'] == {}
    assert sample['metadata']['entity_id'] == 'milk'
    assert sample['targets']['current_mask'][4, 4]
    assert sample['review']['masked'].any()


def test_unmapped_external_contact_is_not_robot_self_contact():
    mujoco = pytest.importorskip('mujoco')
    xml = """
      <mujoco>
        <worldbody>
          <geom name="floor" type="plane" size="1 1 .1"/>
          <body name="robot0_base" pos="0 0 .03">
            <geom name="base" type="sphere" size=".025"/>
            <body name="hand">
              <joint type="slide" axis="0 0 1"/>
              <geom name="gripper0_finger1_pad_collision" pos="-.01 0 0" size=".02"/>
              <geom name="gripper0_finger2_pad_collision" pos=".01 0 0" size=".02"/>
            </body>
          </body>
          <body pos=".4 0 0"><geom name="milk_geom" size=".02"/></body>
          <body pos=".6 0 0"><geom name="basket_geom" size=".02"/></body>
        </worldbody>
        <contact>
          <pair geom1="base" geom2="gripper0_finger1_pad_collision"/>
        </contact>
      </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    base = model.geom('base').id
    pad = model.geom('gripper0_finger1_pad_collision').id
    assert any({c.geom1, c.geom2} == {base, pad} for c in data.contact)
    states = np.array([[0, 0, 0], [0, -.02, 0]])
    flags, _, _, _ = contacts(model, states, {'milk': {'kind': 'object'}, 'basket': {'kind': 'object'}},
                              'milk', 'basket')
    assert flags[0].tolist() == [False, True]
    assert not flags[1:].any()
