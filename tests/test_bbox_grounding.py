"""Regression, same-frame entity contrast and label-free online localization."""
from dataclasses import replace
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.augmentation import GroundingAugmentation, affine_eskill_batch, transform_boxes
from openpi.models.grounding_bbox import box_loss, corners, entity_contrast, read_regions
from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding
from openpi.training.config import TrainConfig, create_model
from openpi.transforms import horizontal_mirror
from scripts.analysis.grounding_pairs import evaluate, score_pair
from test_current_grounding_training import ToyTokenizer, saved_dataset


def dataset_with_mode(original, mode, *, augment=True):
    model = replace(original.config.model, current_grounding=mode, subject_bbox=True)
    cfg = replace(original.config, model=model, loss_weights=None, smoke=True, device='cpu', precision='float32')
    return GroundingDataset(cfg.data_index, cfg, ToyTokenizer(), augment=augment)


def test_recipes_match_except_contrast():
    a, b = (TrainConfig.load('examples/libero/train_'+name+'.json').to_dict() for name in ('bbox', 'bbox_gac'))
    assert b['loss_weights'].pop('gac') == .2
    assert a['loss_weights'] == {'bbox': 1., 'subject': 1.}
    for cfg in (a, b):
        assert cfg['model']['subject_bbox'] and cfg['vae_path'] is None
        cfg['model'].pop('current_grounding')
        for key in ('run_name', 'output_dir'):
            cfg.pop(key)
    assert a == b


def test_regression_perfect_and_disjoint():
    box = torch.tensor([[.5, .5, .25, .5], [.4, .2, .1, .1]], requires_grad=True)
    gt = corners(box.detach())
    valid = torch.tensor([True, False])
    loss, diag = box_loss(box, gt, valid, 224)
    assert loss.item() == 0 and diag['grounding_bbox_iou'].tolist() == [1., 1.]
    assert diag['grounding_bbox_center_px'].tolist() == [0., 1.]
    loss, _ = box_loss(box, gt + .3, valid, 224)
    loss.backward()
    assert box.grad[0].abs().sum() > 0 and box.grad[1].abs().sum() == 0


def test_local_tokens_keep_absolute_coordinates_and_gradients():
    visual = torch.arange(16.).reshape(1, 16, 1)
    box = torch.tensor([[[0., 0., 1., 1.]]], requires_grad=True)
    region, points = read_regions(visual, box)
    torch.testing.assert_close(region.flatten(), visual.flatten())
    torch.testing.assert_close(points[0, 0, 0], torch.tensor([.125, .125]))
    region.sum().backward()
    assert torch.isfinite(box.grad).all() and box.grad.abs().sum() > 0


def test_crop_rotate_and_mirror_sync_contrast_and_regression(tmp_path):
    source = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    dataset = dataset_with_mode(source, 'bbox_gac')
    batch = collate_grounding([dataset[1]])
    assert batch['metadata'][0]['entity_id'] == 'pudding'
    assert 'entity_boxes' not in batch['inputs']
    for transformed in (batch, horizontal_mirror(batch, torch.ones(1, dtype=torch.bool)),
                        affine_eskill_batch(batch, torch.tensor([[0., 0., 1., 1.]]),torch.tensor([90.]))[0]):
        t = transformed['targets']
        i = t['positive_entity'].item()
        assert i >= 0 and t['entity_valid'][0, i] and t['role_valid'][0, 0, 0]
        torch.testing.assert_close(t['role_core_boxes'][0, 0, 0], t['entity_boxes'][0, i])
    boxes = torch.tensor([[[.1, .2, .3, .5], [.8, .3, .9, .4], [.2, .2, .4, .4]]])
    crop = torch.tensor([[.2, 0., .5, 1.]])
    moved, valid, full = transform_boxes(boxes, crop, torch.zeros(1), torch.ones(1))
    torch.testing.assert_close(moved[0, 0], torch.tensor([.5, 0., .8, .2]))
    assert valid.tolist() == [[True, False, True]] and full.tolist() == [[False, False, True]]


@pytest.mark.parametrize('mode', ['bbox','bbox_gac'])
def test_backward_checkpoint_and_two_tasks_online(tmp_path, mode):
    original = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds = dataset_with_mode(original, mode)
    batch = collate_grounding([ds[0], ds[1]])
    model = create_model(ds.config)
    assert model.visual_supervision is None
    losses = model(batch['inputs'], batch['targets'])
    assert set(losses) == set(ds.config.loss_keys)
    assert all(v == 2 for v in model.loss_counts.values())
    if mode == 'bbox_gac':
        params = (model.grounding.current_queries, model.grounding.current_projection.weight,
                  model.grounding.bbox_head[-1].weight, model.grounding.subject_queries)
        dq, dk, dbox, dsubject = torch.autograd.grad(losses['gac'], params, allow_unused=True, retain_graph=True)
        assert dq.norm() > 0 and dk.norm() > 0 and dbox is None and dsubject is None
    sum(losses.values()).backward()
    for weights in (model.grounding.current_queries, model.grounding.bbox_head[-1].weight,
                    model.grounding.subject_queries, model.grounding.subject_bbox_head[-1].weight):
        assert weights.grad.isfinite().all() and weights.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())
    path = tmp_path/'ckpt.pt'
    torch.save(dict(model=model.state_dict(), config=ds.config.to_dict()), path)
    with mock.patch('openpi.policies.current_grounding.TokenizePrompt', return_value=ToyTokenizer()):
        policy = create_current_grounding_policy(path)
    ds.augment = False
    sample = ds.sample(1, review=True)
    assert sample['review']['paired_query'] == ('put the chocolate pudding in the basket', 'pudding')
    rgb = sample['review']['original']
    a,b = [policy.infer([rgb], prompt) for prompt in (sample['metadata']['task'],sample['review']['paired_query'][0])]
    assert a['region_xy'].shape == (1, 16, 2) and a['grounding'].shape[1:3] == (2,16)
    assert (a['bbox_xyxy'] >= 0).all() and (a['bbox_xyxy'] <= 1).all()
    torch.testing.assert_close(a['subject_grounding'],b['subject_grounding'],rtol=0,atol=0)
    torch.testing.assert_close(a['subject_bbox_xyxy'], b['subject_bbox_xyxy'], rtol=0, atol=0)
    assert a['subject_region_xy'].shape == (1, 16, 2) and a['subject_bbox_xyxy'].shape == (1, 4)
    assert (a['subject_bbox_xyxy'] >= 0).all() and (a['subject_bbox_xyxy'] <= 1).all()
    assert evaluate(ds,policy,limit=2,seed=42)['counts']['scored'] == 2


def test_empty_and_cropped_contrast_skip_only_affected_rows():
    query = torch.randn(3, 8, requires_grad=True)
    visual = torch.randn(3, 16, 8)
    projection = torch.nn.Linear(8, 8)
    boxes = torch.tensor([[0., 0., .5, .5], [.5, .5, 1., 1.]]).expand(3,-1,-1)
    targets = dict(entity_boxes=boxes, entity_valid=torch.tensor([[1,1],[1,0],[1,1]]).bool(),
                   positive_entity=torch.tensor([0,1,-1]))
    loss, count, _ = entity_contrast(query,visual,projection,targets,torch.ones(3,dtype=torch.bool))
    loss.backward()
    assert count == 1 and query.grad[0].norm() > 0 and query.grad[1:].norm() == 0
    loss, count, _ = entity_contrast(query,visual,projection,targets,torch.zeros(3,dtype=torch.bool))
    assert count == loss == 0
    assert torch.autograd.grad(loss,query)[0].norm() == 0


def test_pair_scoring_rejects_ambiguous_and_measures_target_switch():
    gt = dict(milk=[0.,0.,.25,.25], basket=[.5,.6,.9,1.])
    assert score_pair(gt, list(gt), list(gt.values()), 16)['both_correct']
    assert not score_pair(gt, list(gt), [gt['milk']]*2, 16)['both_correct']
    assert score_pair(dict(gt, twin=gt['milk']), list(gt), list(gt.values()), 16) is None


@pytest.mark.parametrize('mode', ['bbox', 'bbox_gac'])
def test_training_entry_validates_saves_and_scores_pairs(tmp_path, mode, monkeypatch):
    import json
    import sys
    import sentencepiece
    from scripts.analysis import grounding_pairs
    from scripts.train import train

    original = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds = dataset_with_mode(original, mode)
    metadata = json.loads((tmp_path/'episode.json').read_text())
    metadata['episode_id'] = 'second'
    (tmp_path/'second.json').write_text(json.dumps(metadata))
    index = tmp_path/'index.json'
    entries = json.loads(index.read_text())
    entries['episodes'].append(dict(episode_id='second', path='second.json'))
    index.write_text(json.dumps(entries))
    corpus = tmp_path/'corpus'
    corpus.write_text('put the milk in the basket\nput the chocolate pudding in the basket\n')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path/'sp'),
                                            vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    stats = tmp_path/'stats.json'
    stats.write_text('{}')
    cfg = replace(ds.config, tokenizer_path=str(tmp_path/'sp.model'), output_dir=str(tmp_path/'run'),
                  norm_stats=str(stats), num_steps=1, batch_size=2, validation_fraction=.5, num_workers=0)
    checkpoint = train(cfg)
    validation = json.loads((checkpoint.parent/'validation.jsonl').read_text())
    assert validation['aux']['grounding_bbox_iou_count'] == 2
    assert validation['aux']['grounding_subject_bbox_iou_count'] == 2
    if mode == 'bbox_gac':
        assert validation['aux']['grounding_gac_recall_count'] == 2
    report = tmp_path/'pairs.json'
    monkeypatch.setattr(sys, 'argv', ['grounding_pairs', '--checkpoint', str(checkpoint),
                                      '--device', 'cpu', '--output', str(report)])
    grounding_pairs.main()
    assert json.loads(report.read_text())['counts']['scored'] == 2
