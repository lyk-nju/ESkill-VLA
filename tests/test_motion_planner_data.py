"""Main-view Current, skill and EEF supervision must describe the same augmented observation."""
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from openpi.augmentation import GroundingAugmentation, affine_eskill_batch, augment_training_batch
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding
from openpi.training.data_loader import skill_sampling_weights
from openpi.transforms import horizontal_mirror
from grounding_fixtures import ToyTokenizer, visual_dataset


def planner_dataset(tmp_path):
    baseline = visual_dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    ref = lambda key: dict(uri='arrays.npz', key=key)
    for i, e in enumerate(meta['events']):
        e.update(event_id=f'e{i}', goal_frame=i, boundary_valid=True, skill_valid=True)
    meta.update(arm_ids=['single'], event_index={'single':ref('event_index')},
                effector_pixels={'single':{'image':dict(frame_indices=ref('frames'), values=ref('eef'), valid=ref('valid'))}})
    meta['cameras']['image'].update(width=32, height=32)
    path.write_text(json.dumps(meta))
    with np.load(tmp_path / 'arrays.npz') as f:
        arrays = dict(f)
    np.savez(tmp_path / 'arrays.npz', **arrays, event_index=[0,1], eef=[[2,4],[20,23]])
    cfg = baseline.config
    cfg.training_stage = 'motion_planner'
    cfg.model = replace(cfg.model, current_grounding="motion_planner", motion_terminal_hold=True)
    return GroundingDataset(cfg.data_index, cfg, ToyTokenizer())


def test_factual_plan_uses_single_arm_and_saved_frame_clock(tmp_path):
    ds = planner_dataset(tmp_path)
    rows = [ds[0], ds[1]]
    b = collate_grounding(rows)
    t = b['targets']
    assert set(b['inputs']) == {'images','tokens','token_mask','image_mask'}
    assert t['skill_labels'].tolist() == [0, 1] and t['skill_valid'].all()
    assert 'gripper' not in t and 'subject_mask' not in t and 'actions' not in t
    assert 'entity_boxes' not in t
    ds.calibration_entities = True
    calibration = collate_grounding([ds[0], ds[1]])['targets']
    assert calibration['entity_valid'][torch.arange(2), calibration['positive_entity']].all()
    assert (calibration['positive_entity'] >= 0).all()
    assert (calibration['entity_valid'].sum(1) >= 2).all()
    assert t['trace'].shape == (2, 8, 2)
    assert t['trace_valid'].all() and t['trace_terminal_hold'].all()
    torch.testing.assert_close(t['trace'], torch.tensor([20.5,23.5]).expand_as(t['trace']) / 32)
    assert rows[0]['metadata']['trace_source_frame_indices'] == [1]*8
    _, summary = skill_sampling_weights(ds)
    assert summary['skills']['pick']['count'] == summary['skills']['place']['count'] == 1
    mirrored = horizontal_mirror(b, torch.tensor([True, False]))
    torch.testing.assert_close(mirrored['targets']['trace'][0,:,0], 1 - t['trace'][0,:,0])
    transformed, _ = affine_eskill_batch(b, torch.tensor([[0.,0.,1.,1.]]*2), torch.tensor([90.,0.]))
    torch.testing.assert_close(transformed['targets']['trace'][0,0], torch.tensor([1-23.5/32,20.5/32]))
    torch.testing.assert_close(transformed['targets']['trace'][1],t['trace'][1])
    cropped, _ = affine_eskill_batch(b, torch.tensor([[0.,0.,1.,1.],[.5,.5,.5,.5]]))
    assert cropped['targets']['skill_valid'].tolist() == [True,False]  # Second crop removed the gripper.
    assert cropped['targets']['trace_valid'][0].all() and not cropped['targets']['trace_valid'][1].any()
    augment_training_batch(b, ds.config)


@pytest.mark.parametrize('version', ['v1', 'v2_joint'])
def test_swapped_moved_and_occluded_samples_do_not_train_the_planner(tmp_path, monkeypatch, version):
    ds = planner_dataset(tmp_path)
    ds.config.model = replace(ds.config.model, planner_version=version)
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1.)
    alternative = ds[1]
    assert alternative['metadata']['counterfactual']=='source'
    def visual_only(sample):
        assert not sample['targets']['skill_valid'] and not sample['targets']['trace_valid'].any()
        if ds.config.model.planner_has_boundaries:
            assert not sample['targets']['boundary_valid'][1]
        assert not sample['metadata']['factual_plan']
    visual_only(alternative)
    if ds.config.model.planner_has_boundaries:
        assert alternative['targets']['boundary_valid'].tolist() == [True, False]
    assert alternative['targets']['current_mask'].any()
    ds.config.grounding_augmentation = GroundingAugmentation()
    monkeypatch.setattr('openpi.augmentation.copy_objects', lambda image, boxes, *a, **k: (
        image, boxes, {'moved_offsets': {'pudding': [1,0]}}))
    visual_only(ds[0])  # Even moving a distractor invalidates original planning labels.
    monkeypatch.undo()
    def patch(rgb, *args, **kwargs):
        drop = np.zeros(rgb.shape[:2], bool)
        drop[3,3] = True  # Intersects visible Milk but not the current Basket in Place.
        return rgb, drop
    monkeypatch.setattr('openpi.augmentation.mask_grounding_patches', patch)
    visual_only(ds[0])
    assert ds[1]['targets']['skill_valid'] and ds[1]['targets']['trace_valid'].all()
