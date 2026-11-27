from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from safetensors.torch import save_file
import sentencepiece
import torch

from openpi.augmentation import GroundingAugmentation, ImageAugmentation, augment_training_batch
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.models.visual_primitive import patch_coverage, spatial_kl
from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding
from openpi.training.checkpoints import load_weights
from openpi.training.config import TrainConfig, WandbConfig
from scripts.analysis.visual_primitive_compare import compare
from scripts.analysis.visual_primitive_eval import evaluate, pick_place_timelines
from scripts.train import train
from grounding_fixtures import ToyTokenizer, visual_dataset


def paired_dataset(path):
    old = visual_dataset(path)
    meta = json.loads((path / 'episode.json').read_text())
    stream = meta['cameras']['image']
    meta['cameras']['wrist_image'] = dict(stream, rgb=dict(stream['rgb'], key='wrist'))
    with np.load(path / 'arrays.npz') as archive:
        arrays = dict(archive)
    arrays['wrist'] = 255 - arrays['rgb']
    arrays['wrist'][1] //= 2
    arrays['robot_mask'] = np.zeros_like(arrays['visible'])
    region = dict(meta['effector_regions']['single']['image'],
                  masks=dict(meta['scene_entities']['milk']['regions']['image']['visible_masks'], key='robot_mask'),
                  masks_valid=stream['valid'])
    meta['scene_entities']['robot::single'] = dict(kind='robot', regions={'image': region})
    np.savez(path / 'arrays.npz', **arrays)
    (path / 'episode.json').write_text(json.dumps(meta))
    model = replace(old.config.model, grounding_subject=False, finetune_vlm_layers=(1,), attention_layers=(0, 1))
    cfg = replace(old.config, model=model, cameras=('image', 'wrist_image'), lora_rank=0,
                  loss_weights={'current_align': .1}, grounding_augmentation=GroundingAugmentation(target_swap_prob=1))
    return GroundingDataset(cfg.data_index, cfg, ToyTokenizer())


def test_dual_view_native_kl_and_masked_pixel_invariance():
    torch.manual_seed(42)
    cfg = ESkillConfig.tiny(vocab_size=128, current_grounding='visual_primitive', finetune_vlm_layers=(1,))
    model = Pi0ESkill(cfg)
    inputs = dict(images=torch.randn(2, 2, 3, 32, 32), image_mask=torch.tensor([[True, True], [True, False]]),
                  tokens=torch.tensor([[1, 9, 2], [1, 2, 0]]), token_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]).bool())
    targets = dict(current_mask=torch.rand(2, 32, 32), current_box=torch.tensor([[0., .3, .5, .8]] * 2),
                   center_px_scale=torch.full((2, 2), 32.))
    pred = model.predict_grounding(inputs)
    with torch.no_grad():
        raw = model.encode_observation(inputs, attention_layers=(0, 1)).text_image_attention
        main, wrist = raw.split(16, -1)
        torch.testing.assert_close(pred['visual_attention_mass'], main.sum(-1))
        torch.testing.assert_close(pred['wrist_attention_mass'], wrist.sum(-1))
        assert wrist[0].sum() > 0 and wrist[1].count_nonzero() == 0
        expected = (main / main.sum(-1, keepdim=True).clamp_min(1e-30)).mean(2)
        expected = (pred['alpha'][:, None, :, None] * expected).sum(2)
        torch.testing.assert_close(pred['layer_maps'], expected)
        mono = model.predict_grounding(dict(inputs, images=inputs['images'][:, :1], image_mask=inputs['image_mask'][:, :1]))
        torch.testing.assert_close(pred['current_map'][1], mono['current_map'][1])
        changed = inputs['images'].clone()
        changed[:, 1] *= -1
        alternate = model.predict_grounding(dict(inputs, images=changed))
        torch.testing.assert_close(pred['current_map'][1], alternate['current_map'][1], rtol=0, atol=0)
        assert not torch.allclose(pred['layer_maps'][0], alternate['layer_maps'][0])
    losses = model(inputs, targets)
    assert set(losses) == {'current_align'}
    torch.testing.assert_close(losses['current_align'], spatial_kl(expected, patch_coverage(targets['current_mask'], 4))[0].mean())
    losses['current_align'].backward()
    grads = {key for key, p in model.named_parameters() if p.grad is not None}
    assert all('language_model.layers.1.' in key or key.startswith('grounding.alpha_scorer.') for key in grads)
    assert model.paligemma_with_expert.paligemma.language_model.layers[1].self_attn.q_proj.weight.grad.norm() > 0
    assert model.grounding.alpha_scorer[-1].weight.grad.norm() > 0
    assert not hasattr(model, 'planner') and not hasattr(model, 'action_out_proj') and 'subject_map' not in pred


def test_same_frame_text_swap_copy_and_phase_two(tmp_path):
    ds = paired_dataset(tmp_path)
    paired = ds[1]
    assert paired['metadata']['augmentation']['target_swapped'] and paired['inputs']['image_mask'].all()
    expected = ds.reader(0).rgb('wrist_image', 1).transpose(2, 0, 1).astype('float32') / 127.5 - 1
    np.testing.assert_allclose(paired['inputs']['images'][1], expected)
    assert set(paired['targets']) == {'current_mask', 'current_box', 'center_px_scale'}
    cfg = replace(ds.config, image_augmentation=ImageAugmentation(color_prob=1, crop_prob=1, rotation_prob=1),
                  horizontal_flip_prob=.5)
    mono_cfg = replace(cfg, cameras=('image',))
    mono = GroundingDataset(cfg.data_index, mono_cfg, ToyTokenizer())
    full = augment_training_batch(collate_grounding([paired]), cfg)
    single = augment_training_batch(collate_grounding([mono[1]]), mono_cfg)
    torch.testing.assert_close(full['inputs']['images'][:, 0], single['inputs']['images'][:, 0])
    torch.testing.assert_close(full['targets']['current_mask'], single['targets']['current_mask'])
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1, target_swap_prob=1, copy_max_shift=1,
                                                              copy_gripper_margin_px=0)
    for frame, group, target_moved in [(0, 'copy', False), (1, 'copy_and_text_swap', True)]:
        edited = ds[frame]
        aug = edited['metadata']['augmentation']
        assert aug['moved_offsets'] and aug['target_moved'] == target_moved
        assert (aug['wrist_group'], aug['wrist_status']) == (group, 'copy_unverified')
        assert edited['inputs']['image_mask'].tolist() == [True, False]


def test_random_wrist_dropout_is_shared_with_text_replacements(tmp_path):
    ds = paired_dataset(tmp_path)
    statuses = set()
    for epoch in range(32):
        ds.config.grounding_augmentation = GroundingAugmentation(wrist_dropout_prob=.2)
        factual = ds[(1, epoch, 0)]
        ds.config.grounding_augmentation = replace(ds.config.grounding_augmentation, target_swap_prob=1)
        swapped = ds[(1, epoch, 3)]
        a, b = (s['metadata']['augmentation'] for s in (factual, swapped))
        assert (a['wrist_group'], b['wrist_group']) == ('ordinary', 'text_swap')
        assert a['wrist_status'] == b['wrist_status']
        statuses.add(a['wrist_status'])
        expected_mask = [True, a['wrist_status'] == 'paired']
        assert factual['inputs']['image_mask'].tolist() == swapped['inputs']['image_mask'].tolist() == expected_mask
        np.testing.assert_array_equal(factual['inputs']['images'][1], swapped['inputs']['images'][1])
    assert statuses == {'paired', 'random_dropout'}

    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1, copy_max_shift=0,
                                                              copy_gripper_margin_px=0)
    failed = ds[0]['metadata']['augmentation']
    assert failed['copy_attempted'] and not failed['moved_offsets']
    assert (failed['wrist_group'], failed['wrist_status']) == ('ordinary', 'paired')
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1, wrist_dropout_prob=1)
    assert ds[1]['metadata']['augmentation']['wrist_status'] == 'random_dropout'
    validation = GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer(), augment=False)
    assert validation[1]['metadata']['augmentation']['wrist_status'] == 'paired'


def test_sparse_wrist_does_not_borrow_neighboring_frame(tmp_path):
    ds = paired_dataset(tmp_path)
    path = tmp_path / 'episode.json'
    meta = json.loads(path.read_text())
    stream = meta['cameras']['wrist_image']
    stream['frame_indices'] = dict(stream['frame_indices'], key='sparse_wrist_frames')
    with np.load(tmp_path / 'arrays.npz') as archive:
        arrays = dict(archive)
    np.savez(tmp_path / 'arrays.npz', **arrays, sparse_wrist_frames=[0, 2])
    path.write_text(json.dumps(meta))
    native = GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer(), augment=False)
    assert native[0]['inputs']['image_mask'].tolist() == [True, True]
    assert native[1]['inputs']['image_mask'].tolist() == [True, False]
    assert native[1]['metadata']['augmentation']['wrist_status'] == 'missing'


def test_current_only_train_restore_and_paired_eval(tmp_path):
    ds = paired_dataset(tmp_path)
    corpus = tmp_path / 'corpus'
    corpus.write_text('put the milk in the basket\nput the chocolate pudding in the basket\n')
    sentencepiece.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tmp_path / 'sp'),
                                            vocab_size=32, hard_vocab_limit=False, minloglevel=2)
    base = tmp_path / 'base.safetensors'
    save_file({k: v.clone() for k, v in Pi0(ds.config.model).state_dict().items()}, str(base))
    stats = tmp_path / 'stats.json'
    stats.write_text('{}')
    cfg = replace(ds.config, pretrained_path=str(base), norm_stats=str(stats), tokenizer_path=str(tmp_path / 'sp.model'),
                  output_dir=str(tmp_path / 'run'), device='cpu', precision='float32', smoke=False,
                  num_steps=1, num_workers=0, batch_size=2, validation_fraction=0,
                  wandb_info=WandbConfig(mode='disabled'))
    first = train(cfg)
    state = torch.load(first, weights_only=False)
    log = json.loads((Path(cfg.output_dir) / 'metrics.jsonl').read_text().splitlines()[0])
    assert log['wrist_samples'] == dict(paired=2, missing=0, copy_unverified=0, random_dropout=0, paired_text_swaps=1)
    assert log['wrist_by_augmentation'] == {'ordinary': {'paired': 1}, 'text_swap': {'paired': 1}}
    assert state['weights_kind'] == 'visual_adapter'
    key = next(k for k in state['model'] if 'layers.1.' in k)
    assert not any('layers.0.' in k for k in state['model'])
    with pytest.raises(ValueError, match='Incomplete'):
        load_weights(Pi0ESkill(cfg.model), {k: v for k, v in state['model'].items() if k != key}, adapter_only=True)
    ckpt = train(replace(cfg, num_steps=2), resume=first)
    policy = create_current_grounding_policy(ckpt)
    torch.testing.assert_close(policy.model.state_dict()[key], torch.load(ckpt, weights_only=False)['model'][key])
    assert policy.cameras == ('image', 'wrist_image')
    parent, frame = ds.reader(0).meta['episode_id'], 1
    image = ds.reader(0).rgb('image', frame)
    Image.fromarray(image).save(tmp_path / 'main.png')
    case = dict(id='one', parent=parent, frame=frame, split='val', scene='test', image='main.png',
                task='put the milk in the basket', boxes={'basket': [20, 20, 28, 28]}, target='basket')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(dict(protocol='dual-view-test', metric_space='source', rows=[case])))
    kwargs = dict(panels=0, episode_index=cfg.data_index)
    paired = evaluate(policy, manifest, tmp_path / 'paired', **kwargs)
    removed = evaluate(policy, manifest, tmp_path / 'removed', wrist_mode='removed', **kwargs)
    assert paired['rows'][0]['current']['wrist_attention_mass'] > 0
    assert removed['rows'][0]['current']['wrist_attention_mass'] == 0
    assert 'subject' not in paired['center_reference']
    assert 'removed' in compare({'paired': paired, 'removed': removed}, 'paired')
    Image.fromarray(255 - image).save(tmp_path / 'main.png')
    with pytest.raises(ValueError, match='differs from'):
        evaluate(policy, manifest, tmp_path / 'mismatched', **kwargs)
    assert evaluate(policy, manifest, tmp_path / 'conflict', wrist_mode='conflict', **kwargs)['wrist_mode'] == 'conflict'


def test_timeline_uses_observed_exclusive_hits_and_the_annotated_boundary():
    rows = [dict(timeline_id='pick-place', split='E', parent='episode', task='put milk in basket', frame=frame,
                 place_start_frame=10, skill=skill, target='milk' if skill == 'Pick' else 'basket',
                 entity_error_px={'milk': milk, 'basket': basket})
            for frame, skill, milk, basket in [(6, 'Pick', 2, 30), (8, 'Pick', 7, 4),
                                               (9, 'Pick', 23, 3), (12, 'Place', 19, 3)]]
    assert pick_place_timelines(rows)[0]['switch_delay_frames'] == -1
    assert pick_place_timelines(rows[1:])[0]['switch_delay_frames'] is None


def test_formal_multiview_recipe():
    cfg = TrainConfig.load(Path(__file__).resolve().parents[1] / 'examples/libero/train_visual_primitive_multiview.json')
    assert cfg.vlm_finetune.layers == cfg.model.attention_layers == tuple(range(10, 18))
    assert cfg.loss_keys == ('current_align',) and cfg.init_checkpoint is None and cfg.lora_rank == 0
    assert cfg.num_steps == 20000 and cfg.batch_size == 32
    assert cfg.grounding_augmentation.wrist_dropout_prob == .2
