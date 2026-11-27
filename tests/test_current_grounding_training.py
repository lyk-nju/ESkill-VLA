"""Counterfactual independence, full task conditioning, and inactive action/skill paths."""
import json
import pickle
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.augmentation import GroundingAugmentation, mask_grounding_patches, copy_objects, augmentation_invalidates
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.policies.grounding_dataset import (GroundingDataset, collate_grounding, entity_captions,
                                              spatial_copy_protection)
from openpi.training.config import TrainConfig, create_model


def inputs(config, b=2):
    return dict(images=torch.randn(b, 1, 3, config.image_size, config.image_size),
        image_mask=torch.ones(b, 1, dtype=torch.bool), tokens=torch.randint(1, 50, (b, 5)),
        token_mask=torch.ones(b, 5, dtype=torch.bool), arm_presence=torch.tensor([[True, False]]).expand(b, -1))


def model_config(**kwargs):
    return ESkillConfig.tiny(vocab_size=128,
                            **({'grounding_subject':True, 'freeze_vlm':True} | kwargs))


def test_patch_mask_without_target_protects_interaction():
    image = np.full((32,32,3),200,np.uint8)
    cfg=GroundingAugmentation(patch_prob=1,patch_drop_prob=1,patch_size=8)
    masked,drop=mask_grounding_patches(image,np.array([3,3,6,6]),np.random.default_rng(4),cfg,32)
    assert not drop[:8,:8].any() and drop[8:,8:].all()
    assert (masked[3:6,3:6]==200).all() and (masked[20:23,20:23]==128).all()
    assert (image==200).all()
    assert not augmentation_invalidates(np.zeros((32,32),bool),drop)
    assert augmentation_invalidates(np.pad(np.ones((2,2),bool),((20,10),(20,10))),drop)


def test_same_category_caption_is_not_instance_id_or_iteration_order():
    entities = {k: dict(kind='object', category='milk') for k in ['milk_9', 'milk_1']}
    boxes = {'milk_9': np.array([20, 0, 25, 5]), 'milk_1': np.array([1, 0, 6, 5])}
    captions = entity_captions(entities, boxes)
    assert captions['milk_1'] == 'leftmost milk in the image'
    assert captions['milk_9'] == 'rightmost milk in the image'
    assert entity_captions(entities, {k: np.array([1, 0, 6, 5]) for k in boxes}) == {}


def test_parts_and_regions_inherit_parent_disambiguation():
    entities, boxes = {}, {}
    for name,x in [('cabinet_9', 30), ('cabinet_1', 0)]:
        entities[name] = dict(kind='fixture', category='wooden_cabinet', instance=name)
        entities[name+'::top'] = dict(kind='part', category='wooden_cabinet', instance=name, part='top drawer')
        entities[name+'::region'] = dict(kind='region', category='wooden_cabinet', instance=name,
                                        site_name=name+'_back_contain_region')
        for key in (name, name+'::top', name+'::region'):
            boxes[key] = np.array([x, 0, x+10, 20])
    captions = entity_captions(entities, boxes)
    assert captions['cabinet_1::top'] == 'top drawer of the leftmost wooden cabinet in the image'
    assert captions['cabinet_9::top'] == 'top drawer of the rightmost wooden cabinet in the image'
    assert captions['cabinet_1::region'] == 'back compartment of the leftmost wooden cabinet in the image'
    boxes['cabinet_9'] = boxes['cabinet_1'].copy()
    assert entity_captions(entities, boxes) == {}  # Ambiguous parents cannot name their children uniquely.
    boxes.pop('cabinet_9')
    assert 'cabinet_9::top' not in entity_captions(entities, boxes)
    entities['table'] = dict(kind='background', category='living_room_table', instance='table')
    entities['table::right'] = dict(kind='region', category='living_room_table', instance='table',
                                    site_name='table_plate_right_region')
    boxes['table::right'] = np.array([10,20,15,25])
    assert entity_captions(entities, boxes)['table::right'] == 'plate right region of the living room table'
    entities['table_2'] = dict(entities['table'], instance='table_2')
    assert 'table::right' not in entity_captions(entities, boxes)


class ToyTokenizer:
    tokenizer = property(lambda self: self)

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def eos_id(self):
        return 2

    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)

    def __call__(self, text, state=None):
        tokens = np.array(self.encode(text)[:96], np.int64)
        return np.pad(tokens, (0, 96-len(tokens))), np.arange(96) < len(tokens)


def saved_dataset(tmp_path, *, aug=None):
    def ref(key, **kw):
        return dict(uri='arrays.npz', key=key, **kw)
    masks = np.zeros((2,32,32),bool)
    masks[:,2:10,2:10] = True
    arrays = dict(frames=np.arange(2), valid=np.ones(2,bool), rgb=np.full((2,32,32,3),180,np.uint8),
        milk=np.tile([2,2,10,10],(2,1)), basket=np.tile([20,20,28,28],(2,1)), pudding=np.tile([24,2,30,8],(2,1)),
        gripper=np.tile([12,12,16,16],(2,1)),
        visible=np.packbits(masks.reshape(2,-1),axis=1), visible_fraction=np.ones(2,np.float32),
        partially_visible=np.full(2,.85,np.float32),
        contact=np.full((2,4),np.nan), eligible=np.zeros(2,bool))
    np.savez(tmp_path/'arrays.npz',**arrays)
    entities = {key:dict(kind='object',category=key,part='whole',visible_fraction=ref('visible_fraction'),regions={'image':dict(frame_indices=ref('frames'),
        boxes_xyxy=ref(key),boxes_valid=ref('valid'),visible_masks=ref('visible',encoding='packbits',shape=[2,32,32]))})
        for key in ('milk','basket','pudding')}
    entities['pudding']['category'] = 'chocolate_pudding'
    entities['pudding']['copy_eligible'] = ref('valid')
    for key in ('milk', 'basket'):
        entities[key]['visible_fraction'] = ref('partially_visible')
    for entity in entities.values():
        entity['interaction_locked'] = ref('eligible')  # All False; separate from copy eligibility.
    meta = dict(episode_id='test',instructions=['put the milk in the basket'],scene_entities=entities,
        cameras={'image':dict(frame_indices=ref('frames'),valid=ref('valid'),rgb=ref('rgb'))},
        online_grounding=dict(interaction_boxes=ref('contact')),
        effector_regions={'single':{'image':dict(frame_indices=ref('frames'),boxes_xyxy=ref('gripper'),boxes_valid=ref('valid'))}},
        events=[dict(arm='single',skill='Pick' if i==0 else 'Place',start_frame=i,stop_frame=i+1,interaction_target=dict(entity_id=key),
                     grounding_template='pick up the {object}' if i==0 else None,
                     description='pick up the milk' if i==0 else 'place the milk in the basket')
                for i,key in enumerate(['milk','basket'])])
    (tmp_path/'episode.json').write_text(json.dumps(meta))
    (tmp_path/'index.json').write_text(json.dumps(dict(episodes=[dict(episode_id='test',path='episode.json')])))
    config = TrainConfig(str(tmp_path/'index.json'),'','','',model=model_config(freeze_vlm=True),
        training_stage='grounding',
        grounding_augmentation=aug or GroundingAugmentation())
    return GroundingDataset(config.data_index,config,ToyTokenizer())


@pytest.mark.parametrize('problem', ['absent', 'mask_ref', 'invalid', 'frames', None])
def test_copy_paste_requires_complete_robot_masks_at_initialization(tmp_path, problem):
    import copy
    ds = saved_dataset(tmp_path)
    meta = json.loads((tmp_path/'episode.json').read_text())
    if problem != 'absent':
        region = copy.deepcopy(meta['effector_regions']['single']['image'])
        region['masks'] = dict(meta['scene_entities']['milk']['regions']['image']['visible_masks'])
        region['masks_valid'] = dict(uri='arrays.npz', key='valid')
        if problem == 'mask_ref':
            region.pop('masks')
        elif problem == 'invalid':
            region['masks_valid']['key'] = 'eligible'
        elif problem == 'frames':
            region['frame_indices'] = dict(uri='arrays.npz', key='milk')
        meta['scene_entities']['robot::single'] = dict(kind='robot', regions={'image':region})
    (tmp_path/'episode.json').write_text(json.dumps(meta))
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=.5)
    if problem:
        with pytest.raises(ValueError, match='robot::single'):
            GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer(), augment=False)
    else:
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())


def test_single_counterfactual_changes_prompt_and_label_without_action_or_task_leakage(tmp_path):
    ds = saved_dataset(tmp_path)
    original = ds[1]
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    restored = pickle.loads(pickle.dumps(ds))
    swapped = restored[1]
    assert original['metadata']['task'] == 'put the milk in the basket'
    assert swapped['metadata']['task'] == 'put the chocolate pudding in the basket'
    assert np.array_equal(original['inputs']['images'],swapped['inputs']['images'])
    assert not np.array_equal(original['targets']['role_images'],swapped['targets']['role_images'])
    batch = collate_grounding([original,swapped])
    assert batch['inputs']['images'].shape[:2] == (2,1)
    assert 'task_inputs' not in batch['inputs']
    assert set(batch['targets']) == {'role_images','role_weights','role_valid','role_core_boxes'}
    ds.augment = False
    assert ds[0]['metadata']['entity_id'] == 'milk'  # Validation ignores configured training augmentation.


@pytest.mark.parametrize('fraction', [None, 0., .5, .799, .8, 1.])
def test_counterfactual_visibility_filter_preserves_real_event_labels(tmp_path, fraction):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    reader = ds.reader(0)
    if fraction is None:
        for entity in reader.meta['scene_entities'].values():
            entity.pop('visible_fraction')
    array = reader.array
    def visibility(ref):
        if ref['key'] == 'visible_fraction':
            return np.full(2, fraction, np.float32)
        return array(ref)
    with mock.patch.object(reader, 'array', side_effect=visibility):
        samples = [ds[(1, epoch, 0)] for epoch in range(4)]
    expected = 'pudding' if fraction is not None and fraction >= .8 else 'basket'
    for sample in samples:
        assert sample['metadata']['entity_id'] == expected
        assert sample['metadata']['augmentation']['target_swapped'] == (expected == 'pudding')
        assert sample['targets']['role_valid'][0, 0]  # An occluded real Event is still supervised.


def test_target_and_native_occlusion_kept_when_all_unprotected_patches_are_masked(tmp_path):
    ds = saved_dataset(tmp_path,aug=GroundingAugmentation(patch_prob=1,patch_drop_prob=1,patch_size=8))
    masked = ds[0]
    assert masked['targets']['role_valid'][0,0]
    assert not masked['metadata']['augmentation']['augmentation_invalidated']
    ds.config.grounding_augmentation = GroundingAugmentation()
    clean = ds[0]
    np.testing.assert_array_equal(clean['targets']['role_images'],masked['targets']['role_images'])
    # Full bbox and label survive a zero-visible native frame, even if every patch is dropped.
    reader = ds.reader(0)
    raw = reader._raw
    def hidden(uri,key):
        return np.zeros_like(raw(uri,key)) if key=='visible' else raw(uri,key)
    ds.config.grounding_augmentation = GroundingAugmentation(patch_prob=1,patch_drop_prob=1,patch_size=8)
    with mock.patch.object(reader,'_raw',side_effect=hidden):
        native = ds[0]
    assert native['targets']['role_valid'][0,0]
    assert native['metadata']['augmentation']['native_invisible']
    np.testing.assert_array_equal(native['targets']['role_images'],clean['targets']['role_images'])


def test_epoch_keys_replay_independently_of_worker_order_and_resume(tmp_path):
    from openpi.training.data_loader import ResumableBatchSampler
    ds = saved_dataset(tmp_path,aug=GroundingAugmentation(patch_prob=1))
    samples = [ds[(0,i,4)] for i in range(4)]
    assert any(not np.array_equal(samples[0]['inputs']['images'],s['inputs']['images']) for s in samples[1:])
    for i in reversed(range(4)):
        np.testing.assert_array_equal(samples[i]['inputs']['images'],ds[(0,i,4)]['inputs']['images'])
    batches = ResumableBatchSampler(list(range(10)),2,False)
    batches.sample_keys=True; batches.epoch=7
    full = list(batches); batches.start_batch=3
    assert list(batches) == full[3:]


def test_copy_translates_pixels_box_and_label_without_mutating_source_or_erasing_robot():
    from openpi.transforms import role_canvas
    image=np.full((40,40,3),90,np.uint8); image[4:12,4:12]=[220,30,10]
    # Simulate a native-mask/JPEG fringe outside the visible silhouette.
    image[3,4:12]=[120,80,70]
    original=image.copy()
    mask=np.zeros((40,40),bool);mask[4:12,4:12]=True
    robot=np.zeros_like(mask)
    boxes={'milk':np.array([4,4,12,12]),'other':np.array([30,30,36,36])}
    cfg=GroundingAugmentation(copy_prob=1,copy_max_shift=.5)
    rng=mock.Mock();rng.random.return_value=0; rng.integers.return_value=1; rng.permutation.return_value=[0]
    rng.choice.return_value=8*(40-8+1)+16
    result,updated,info=copy_objects(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
    json.dumps(info)  # The training logger writes offsets and placement counts to JSON.
    assert info['copy_applied'] and info['moved_offsets']=={'milk':[12,4]}
    np.testing.assert_array_equal(updated['milk'],[16,8,24,16])
    np.testing.assert_array_equal(result[8:16,16:24],original[4:12,4:12])
    assert (result[2:14,2:14]==90).all()
    np.testing.assert_array_equal(image,original)
    canvas,_,valid=role_canvas(result,{'bbox':updated['milk']},40)
    assert valid and (canvas[8:16,16:24]==[220,30,10]).all()
    robot[4:12,4:12]=True
    blocked,_,info=copy_objects(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
    assert blocked is image and info['copy_reason']=='source_robot_overlap'


def test_grounding_workers_and_mid_epoch_resume_replay_same_augmented_samples(tmp_path):
    from multiprocessing.connection import Listener
    from torch.utils.data import DataLoader
    from openpi.training.data_loader import ResumableBatchSampler
    try:
        with Listener(str(tmp_path/'ipc.sock'),family='AF_UNIX'):
            pass
    except PermissionError:
        pytest.skip('Spawn worker tensor transfer requires IPC permission')
    dataset=saved_dataset(tmp_path,aug=GroundingAugmentation(target_swap_prob=.5,patch_prob=1,patch_size=8))
    def load(workers,start=0):
        batches=ResumableBatchSampler([0,1,0,1],1,False)
        batches.sample_keys=True; batches.epoch=3; batches.start_batch=start
        loader=DataLoader(dataset,batch_sampler=batches,collate_fn=collate_grounding,num_workers=workers,
                          multiprocessing_context='spawn' if workers else None)
        return list(loader)
    reference=load(0)
    for actual,expected in [(load(2),reference),(load(2,2),reference[2:])]:
        for a,b in zip(actual,expected,strict=True):
            assert a['metadata']==b['metadata']
            torch.testing.assert_close(a['inputs']['images'],b['inputs']['images'],rtol=0,atol=0)
            torch.testing.assert_close(a['targets']['role_images'],b['targets']['role_images'],rtol=0,atol=0)
            assert torch.equal(a['targets']['role_valid'],b['targets']['role_valid'])


def test_patch_mask_protects_full_target_halo_and_contact_but_not_other_objects():
    image=np.full((64,64,3),200,np.uint8)
    cfg=GroundingAugmentation(patch_prob=1,patch_drop_prob=1,patch_size=8,target_margin_px=8)
    masked,drop=mask_grounding_patches(image,np.array([1,1,7,7]),np.random.default_rng(4),cfg,64,
                                     target_box=np.array([25,25,31,31]))
    assert not drop[17:39,17:39].any()  # Full box + 8 source pixels on every side.
    assert not drop[:8,:8].any()
    assert drop[48:56,48:56].all()  # An unrelated object's area can still be masked.
    np.testing.assert_array_equal(masked[17:39,17:39],image[17:39,17:39])
    _,edge=mask_grounding_patches(image,None,np.random.default_rng(4),cfg,64,
                                target_box=np.array([0,0,6,6]))
    assert not edge[:14,:14].any() and edge[56:,56:].all()


def test_subject_uses_one_unfused_image_encoding_and_is_independent_of_text():
    torch.manual_seed(41)
    model=Pi0ESkill(model_config(grounding_subject=True,freeze_vlm=True)).eval()
    data=inputs(model.config)
    with mock.patch.object(model.paligemma_with_expert,'embed_images',wraps=model.paligemma_with_expert.embed_images) as encode:
        first=model.predict_grounding(data)
        assert encode.call_count==1  # Subject adds no second vision/VLM pass.
    changed=dict(data,tokens=(data['tokens']+17)%128)
    second=model.predict_grounding(changed)
    torch.testing.assert_close(first['subject'],second['subject'],rtol=0,atol=0)
    assert not torch.allclose(first['current'],second['current'])
    first['subject'].square().mean().backward()
    assert model.grounding.subject_queries.grad.abs().sum()>0
    assert model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.grad is None
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())
    assert model.grounding.subject_projection.weight.grad.abs().sum() > 0
    assert model.grounding.current_projection.weight.grad is None
    assert model.grounding.text_projection.weight.grad is None


def test_subject_training_uses_visual_loss_and_does_not_train_gripper_commands():
    from openpi.training.metrics import loss_metrics
    config=TrainConfig('none','none','none','none',model=model_config(grounding_subject=True,freeze_vlm=True),
                       training_stage='grounding',smoke=True,device='cpu',precision='float32')
    model=create_model(config)
    data=inputs(config.model)
    targets=dict(role_images=torch.rand(2,2,2,3,32,32)*2-1,role_weights=torch.ones(2,2,2,32,32),
                 role_valid=torch.tensor([[[True,True],[False,False]],[[True,False],[False,False]]]))
    assert not hasattr(model, 'motion') and not hasattr(model, 'skill')
    assert not hasattr(model, 'arm_embedding') and not hasattr(model.grounding, 'future_decoder')
    with mock.patch.object(model,'flow_loss',side_effect=AssertionError('Action executed')):
        losses=model(data,targets)
        assert set(losses)=={'current','subject'}
        sum(losses.values()).backward()
    assert model.loss_counts['current']==2 and model.loss_counts['subject']==1
    assert model.grounding.subject_queries.grad.abs().sum()>0
    assert model.grounding.current_queries.grad.abs().sum()>0
    assert model.grounding.current_projection.weight.grad.abs().sum()>0
    assert model.grounding.subject_projection.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.paligemma_with_expert.parameters())
    assert not hasattr(model, 'action_in_proj')
    metrics=loss_metrics(dict(loss=0,**losses))
    assert 'subject_grounding_loss' in metrics and 'gripper_loss' not in metrics


def test_subject_canvas_and_protect_follow_gripper_independently_of_target(tmp_path):
    from dataclasses import replace
    ds=saved_dataset(tmp_path)
    ds.config.model=replace(ds.config.model,grounding_subject=True)
    initial=ds.sample(1,review=True)
    ds.config.grounding_augmentation=GroundingAugmentation(target_swap_prob=1,patch_prob=1,patch_drop_prob=1,patch_size=4)
    switched=ds.sample(1,review=True)
    assert initial['metadata']['entity_id']=='basket' and switched['metadata']['entity_id']=='pudding'
    np.testing.assert_array_equal(initial['targets']['role_images'][0,1],switched['targets']['role_images'][0,1])
    assert not np.array_equal(initial['targets']['role_images'][0,0],switched['targets']['role_images'][0,0])
    assert switched['targets']['role_valid'].tolist()==[[True,True],[False,False]]
    size = switched['review']['original'].shape[0]
    np.testing.assert_allclose(switched['targets']['role_core_boxes'][0,0],
                               switched['review']['boxes']['pudding']/size)
    np.testing.assert_allclose(switched['targets']['role_core_boxes'][0,1],
                               switched['review']['gripper_box']/size)
    for role, box in enumerate((switched['review']['boxes']['pudding'], switched['review']['gripper_box'])):
        x0, y0, x1, y1 = box.astype(int)
        inside = np.zeros((size, size), dtype=bool)
        inside[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(switched['targets']['role_weights'][0, role], inside)
        canvas = switched['targets']['role_images'][0, role].transpose(1, 2, 0)
        np.testing.assert_array_equal(canvas[~inside], -1.)
        np.testing.assert_allclose(canvas[inside], switched['review']['edited'][inside] / 127.5 - 1, atol=1e-7)
    assert 'role_core_boxes' not in switched['inputs']
    # Gripper is protected even with no interaction/contact box.
    assert not switched['review']['drop'][4:24,4:24].any()
    assert switched['review']['drop'][0,0]
    np.testing.assert_array_equal(switched['review']['masked'][4:24,4:24],switched['review']['edited'][4:24,4:24])


def test_online_policy_returns_subject_with_fixed_image_and_different_targets():
    from openpi.policies.current_grounding import CurrentGroundingPolicy
    model=Pi0ESkill(model_config(grounding_subject=True)).eval()
    with mock.patch('openpi.policies.current_grounding.TokenizePrompt',return_value=ToyTokenizer()):
        policy=CurrentGroundingPolicy(model,'unused')
    image=[np.full((32,32,3),150,np.uint8)]
    a=policy.infer(image,'put the milk in the basket')
    b=policy.infer(image,'put the pudding in the basket')
    torch.testing.assert_close(a['subject_grounding'],b['subject_grounding'],rtol=0,atol=0)
    assert a['subject_grounding'].shape==a['grounding'].shape
    assert not torch.allclose(a['grounding'],b['grounding'])


def test_current_reads_fused_features_and_subject_reads_unfused_patches():
    torch.manual_seed(9)
    model = Pi0ESkill(model_config()).eval()
    data = inputs(model.config)
    texts, current_visuals, subject_visuals = [], [], []
    hooks = [model.grounding.text_projection.register_forward_pre_hook(lambda m,a: texts.append(a[0].detach())),
             model.grounding.current_projection.register_forward_pre_hook(lambda m,a: current_visuals.append(a[0].detach())),
             model.grounding.subject_projection.register_forward_pre_hook(lambda m,a: subject_visuals.append(a[0].detach()))]
    with mock.patch.object(model, 'encode_observation', wraps=model.encode_observation) as encode, \
         mock.patch.object(model.paligemma_with_expert.paligemma.lm_head, 'forward', side_effect=AssertionError('LM head used')):
        first = model.predict_grounding(data)
        second = model.predict_grounding(dict(data, images=data['images'].flip(-1)))
        assert encode.call_count == 2
    with torch.no_grad():
        prefix = model.encode_observation(data)
    patches = prefix.visual_tokens.shape[1]
    torch.testing.assert_close(current_visuals[0], prefix.context[:, :patches], rtol=0, atol=0)
    torch.testing.assert_close(texts[0], prefix.context[:, patches:], rtol=0, atol=0)
    torch.testing.assert_close(subject_visuals[0], prefix.visual_tokens, rtol=0, atol=0)
    assert not torch.allclose(texts[0], texts[1])  # Current's text is also image-conditioned.
    assert not torch.allclose(current_visuals[0], subject_visuals[0])
    assert not torch.allclose(first['current'], second['current'])
    assert first['current'][:, 1].count_nonzero() == 0
    for hook in hooks:
        hook.remove()


def test_pick_keeps_original_prompt_and_target_even_with_counterfactual_enabled(tmp_path):
    ds = saved_dataset(tmp_path)
    pick, place = ds[0], ds[1]
    assert pick['metadata']['task'] == place['metadata']['task'] == 'put the milk in the basket'
    assert pick['metadata']['entity_id'] == 'milk' and place['metadata']['entity_id'] == 'basket'
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    augmented_pick, augmented_place = ds[0], ds[1]
    assert augmented_pick['metadata']['task'] == pick['metadata']['task']
    assert augmented_pick['metadata']['entity_id'] == 'milk'
    assert not augmented_pick['metadata']['augmentation']['target_swapped']
    assert augmented_pick['metadata']['augmentation']['swap_candidate_count'] == 0
    np.testing.assert_array_equal(augmented_pick['inputs']['tokens'], pick['inputs']['tokens'])
    np.testing.assert_array_equal(augmented_pick['targets']['role_core_boxes'], pick['targets']['role_core_boxes'])
    assert augmented_place['metadata']['task'] == 'put the chocolate pudding in the basket'
    assert augmented_place['metadata']['entity_id'] == 'pudding'
    ds.reader(0).meta['scene_entities']['pudding'].pop('copy_eligible')
    assert ds[1]['metadata']['entity_id'] == 'pudding'  # Querying does not require edit permission.


@pytest.mark.parametrize('fraction', [.85, .899, .90, .97])
def test_copy_uses_current_visibility_threshold_independent_of_counterfactual_and_old_flag(tmp_path, fraction):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    reader = ds.reader(0)
    reader.meta['scene_entities']['pudding']['copy_eligible']['key'] = 'eligible'  # Legacy 98% flag.
    array = reader.array

    def visibility(ref):
        if ref['key'] == 'visible_fraction':
            return np.full(2, fraction, np.float32)
        return array(ref)

    with mock.patch.object(reader, 'array', side_effect=visibility):
        sample = ds[1]
    assert sample['metadata']['entity_id'] == 'pudding'
    assert sample['metadata']['augmentation']['target_swapped']
    assert sample['metadata']['augmentation']['locatable_object_count'] == 3
    assert sample['metadata']['augmentation']['swap_candidate_count'] == 1
    assert sample['metadata']['augmentation']['copy_candidate_count'] == int(fraction >= .90)


def test_copy_gripper_margin_is_independent_of_saved_text_lock(tmp_path):
    ds = saved_dataset(tmp_path)
    reader = ds.reader(0)
    reader.meta['scene_entities']['pudding']['interaction_locked']['key'] = 'valid'
    raw = reader._raw

    def arrays_for_test(uri, key):
        return np.tile([16, 2, 22, 10], (2, 1)) if key == 'gripper' else raw(uri, key)

    def use_empty_robot_mask(image, boxes, eligible, visible, robot, rng, config, **kwargs):
        return copy_objects(image, boxes, eligible, visible, lambda: np.zeros(image.shape[:2], bool),
                            rng, config, **kwargs)

    with mock.patch.object(reader, '_raw', side_effect=arrays_for_test), \
         mock.patch('openpi.augmentation.copy_objects', side_effect=use_empty_robot_mask):
        for margin, candidates in ((4, 0), (2, 1)):
            ds.config.grounding_augmentation = GroundingAugmentation(
                copy_prob=1, target_swap_prob=1, copy_gripper_margin_px=margin)
            info = ds[1]['metadata']['augmentation']
            assert info['copy_candidate_count'] == candidates
            assert info['swap_candidate_count'] == 0


@pytest.mark.parametrize('task', ['put the milk in the basket', 'put the milk into the basket',
    'place the milk in the basket', 'place milk into basket.',
    'pick up the milk and place it in the basket', 'Pick up the milk and put it into the basket!'])
def test_transfer_paraphrases_replace_only_source_and_keep_punctuation(tmp_path, task):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.reader(0).meta['instructions'] = [task]
    assert ds[0]['metadata']['task'] == task
    sample = ds[1]
    assert sample['metadata']['task'] == task.replace('milk', 'chocolate pudding')
    assert sample['metadata']['entity_id'] == 'pudding'


@pytest.mark.parametrize(('task', 'source', 'destination', 'candidate', 'expected_task'), [
    ('put the bowl on the stove', 'akita_black_bowl', 'flat_stove', 'chocolate_pudding',
     'put the chocolate pudding on the stove'),
    ('put the bowl on the stove', 'akita_black_bowl', 'flat_stove', 'plate', 'put the plate on the stove'),
    ('put the wine bottle on the rack', 'wine_bottle', 'wine_rack', 'cream_cheese',
     'put the cream cheese on the rack'),
    ('put the bowl on the rack', 'akita_black_bowl', 'wine_rack', 'wine_bottle',
     'put the wine bottle on the rack'),
    ('put the bowl on top of the cabinet', 'akita_black_bowl', 'wooden_cabinet', 'cream_cheese',
     'put the cream cheese on top of the cabinet'),
    ('Place the bowl onto the plate.', 'akita_black_bowl', 'plate', 'cream_cheese',
     'Place the cream cheese onto the plate.'),
    ('put the bowl on the plate', 'akita_black_bowl', 'plate', 'wine_bottle', 'put the wine bottle on the plate'),
    ('put the cream cheese in the bowl', 'cream_cheese', 'akita_black_bowl', 'plate',
     'put the plate in the bowl'),
    ('place the milk onto the basket', 'milk', 'basket', 'chocolate_pudding',
     'place the chocolate pudding onto the basket'),
])
def test_transfer_grounds_replacement_regardless_of_placement_fit(tmp_path, task, source, destination, candidate, expected_task):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    meta = ds.reader(0).meta
    meta['instructions'] = [task]
    meta['scene_entities']['milk']['category'] = source
    meta['scene_entities']['basket']['category'] = destination
    meta['scene_entities']['pudding']['category'] = candidate
    sample = ds[1]['metadata']
    assert sample['augmentation']['transfer_resolved']
    assert sample['task'] == expected_task
    assert sample['entity_id'] == 'pudding'


def test_cabinet_alias_requires_unique_destination(tmp_path):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    meta = ds.reader(0).meta
    task = 'put the milk on top of the cabinet'
    meta['instructions'] = [task]
    meta['scene_entities']['basket'].update(kind='fixture', category='wooden_cabinet')
    meta['scene_entities']['pudding'].update(kind='fixture', category='white_cabinet')
    for frame in (0, 1):
        sample = ds[frame]['metadata']
        assert not sample['augmentation']['transfer_resolved']
        assert sample['task'] == task


@pytest.mark.parametrize(('candidate', 'opened_drawer', 'expected_swap'), [
    ('cream_cheese', 'top_drawer', True),
    ('plate', 'top_drawer', True),
    ('wine_bottle', 'top_drawer', True),
    ('cream_cheese', 'middle_drawer', False),
])
def test_open_then_place_changes_only_source_and_preserves_open_event(tmp_path, candidate, opened_drawer, expected_swap):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    reader = ds.reader(0)
    task = 'Open the top drawer and put the bowl inside!'
    reader.meta['instructions'] = [task]
    entities = reader.meta['scene_entities']
    entities['milk']['category'] = 'akita_black_bowl'
    entities['pudding']['category'] = candidate
    entities['top_drawer'] = dict(entities.pop('basket'), kind='part', category='wooden_cabinet',
                                  instance='cabinet', part='top drawer')
    entities['middle_drawer'] = dict(entities['top_drawer'], part='middle drawer')
    entities['cabinet'] = dict(kind='fixture', category='wooden_cabinet', regions={})
    pick, place = reader.meta['events']
    place['interaction_target']['entity_id'] = 'top_drawer'
    opened = dict(pick, skill='Open', interaction_target=dict(entity_id=opened_drawer))
    reader.meta['events'] = [opened, pick, place]
    for event, frame in ((opened, 0), (pick, 0), (place, 1)):
        with mock.patch.object(reader, 'event_at', return_value=event):
            sample = ds[frame]
        info = sample['metadata']
        swapped = expected_swap and event['skill'] == 'Place'
        assert info['augmentation']['target_swapped'] == swapped
        assert info['task'] == (task.replace('bowl', candidate.replace('_', ' ')) if swapped else task)
        assert info['entity_id'] == ('pudding' if swapped else event['interaction_target']['entity_id'])


@pytest.mark.parametrize('task', ['turn on the stove', 'do the milk transfer'])
def test_copy_paste_attempt_is_independent_of_transfer_parser(tmp_path, task):
    ds = saved_dataset(tmp_path)
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1)
    ds.reader(0).meta['instructions'] = [task]
    def copy(image, boxes, candidates, visible, robot, rng, config, **kwargs):
        assert config.copy_prob == 1 and candidates == ['pudding']
        return image, boxes, dict(copy_attempted=True, copy_applied=False, moved_offsets={}, copy_reason='entity_overlap')
    with mock.patch('openpi.augmentation.copy_objects', side_effect=copy) as paste:
        sample = ds[0]
    assert paste.call_count == 1
    assert sample['metadata']['augmentation']['copy_attempted']
    assert not sample['metadata']['augmentation']['transfer_resolved']


def test_spatial_instruction_locks_references_and_allows_copy_for_other_objects(tmp_path):
    ds = saved_dataset(tmp_path)
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1)
    meta = ds.reader(0).meta
    meta['instructions'] = ['pick up the milk next to the basket and place it in the basket']
    for entity in meta['scene_entities'].values():
        entity['visible_fraction'] = dict(uri='arrays.npz', key='visible_fraction')

    def deterministic_copy(image, boxes, candidates, visible, robot, rng, config, **kwargs):
        assert candidates == ['pudding']
        fixed = mock.Mock()
        fixed.random.return_value = 0.
        fixed.integers.return_value = 1
        fixed.permutation.return_value = [0]
        fixed.choice.side_effect = lambda positions: positions[0]
        mask = np.zeros(image.shape[:2], bool)
        mask[2:8, 24:30] = True
        return copy_objects(image, boxes, candidates, lambda _: mask, lambda: np.zeros_like(mask),
                            fixed, config, **kwargs)

    with mock.patch('openpi.augmentation.copy_objects', side_effect=deterministic_copy):
        sample = ds.sample(0, review=True)
    audit = sample['metadata']['augmentation']
    assert set(audit['moved_offsets']) == {'pudding'}
    assert audit['copy_task_protected'] == ['basket', 'milk']
    assert sample['metadata']['entity_id'] == 'milk'
    np.testing.assert_array_equal(sample['review']['boxes']['milk'], sample['review']['original_boxes']['milk'])


def test_copy_can_move_object_in_empty_corner_of_old_interaction_union(tmp_path):
    ds = saved_dataset(tmp_path)
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1, copy_gripper_margin_px=4)
    reader = ds.reader(0)
    raw = reader._raw
    legacy_union = np.array([2, 2, 27, 27])
    overrides = {'basket': [18, 18, 27, 27], 'pudding': [24, 9, 28, 13], 'contact': legacy_union}

    def arrays_for_test(uri, key):
        return np.tile(overrides[key], (2, 1)) if key in overrides else raw(uri, key)

    def deterministic_copy(image, boxes, candidates, visible, robot, rng, config, **kwargs):
        assert candidates == ['pudding']
        protected = kwargs['protected_boxes']
        assert any(np.array_equal(box, overrides['basket']) for box in protected)
        assert not any(np.array_equal(box, legacy_union) for box in protected)
        fixed = mock.Mock()
        fixed.random.return_value = 0.
        fixed.integers.return_value = 1
        fixed.permutation.return_value = [0]
        fixed.choice.side_effect = lambda positions: positions[0]
        mask = np.zeros(image.shape[:2], bool)
        mask[9:13, 24:28] = True
        return copy_objects(image, boxes, candidates, lambda _: mask, lambda: np.zeros_like(mask),
                            fixed, config, **kwargs)

    with mock.patch.object(reader, '_raw', side_effect=arrays_for_test), \
         mock.patch('openpi.augmentation.copy_objects', side_effect=deterministic_copy):
        sample = ds.sample(0, review=True)
    assert set(sample['metadata']['augmentation']['moved_offsets']) == {'pudding'}
    assert sample['metadata']['mirror_eligible']  # "in the basket" stays true after reflection.
    assert sample['metadata']['spatial_augmentation_eligible']
    np.testing.assert_array_equal(sample['review']['interaction_box'], legacy_union)


def test_spatial_protection_includes_same_category_instances_and_named_regions():
    entities = {
        'bowl_1': dict(kind='object', category='akita_black_bowl'),
        'bowl_2': dict(kind='object', category='akita_black_bowl'),
        'rack': dict(kind='fixture', category='wine_rack'),
        'cookies': dict(kind='object', category='cookies'),
        'plate': dict(kind='object', category='plate'),
        'target_area': dict(kind='region', category='spatial_region', instance='rack'),
    }
    events = [dict(source_annotation=dict(instance='bowl_1'), interaction_target=dict(entity_id='target_area'))]
    task = 'pick up the leftmost bowl next to the cookie box and place it to the right of the rack.'
    assert spatial_copy_protection(task, entities, events, None) == {
        'bowl_1', 'bowl_2', 'cookies', 'rack', 'target_area',
    }


@pytest.mark.parametrize(('reference', 'duplicate_plate', 'unprotected'), [
    ('the stove', False, True), ('the plate', False, False), ('the stove', True, False),
])
def test_spatial_place_releases_only_a_unique_destination_independent_of_pick(reference, duplicate_plate, unprotected):
    task = f'pick up the black bowl next to {reference} and place it on the plate'
    entities = {
        'bowl': dict(kind='object', category='akita_black_bowl'),
        'plate': dict(kind='object', category='plate'),
        'stove': dict(kind='fixture', category='flat_stove'),
    }
    if duplicate_plate:
        entities['other_plate'] = dict(kind='object', category='plate')
    pick = dict(skill='Pick', interaction_target=dict(entity_id='bowl'))
    place = dict(skill='Place', interaction_target=dict(entity_id='plate'))
    assert 'plate' in spatial_copy_protection(task, entities, [pick, place], None, event=pick)
    protected = spatial_copy_protection(task, entities, [pick, place], None, event=place)
    assert ('plate' not in protected) == unprotected
    assert 'bowl' in protected


@pytest.mark.parametrize(('second_destination', 'first_mug_is_reference'), [
    ('on the right plate', False), ('to the right of the white mug', True),
])
def test_long_task_pick_can_move_unique_mug_without_changing_destination_references(
    second_destination, first_mug_is_reference,
):
    task = f'put the white mug on the left plate and put the yellow and white mug {second_destination}'
    entities = {
        'white_mug': dict(kind='object', category='porcelain_mug'),
        'yellow_mug': dict(kind='object', category='white_yellow_mug'),
        'plate_1': dict(kind='object', category='plate'),
        'plate_2': dict(kind='object', category='plate'),
    }
    pick = dict(skill='Pick', interaction_target=dict(entity_id='white_mug'))
    place = dict(skill='Place', interaction_target=dict(entity_id='plate_1'))
    protected = spatial_copy_protection(task, entities, [pick, place], None, event=pick)
    assert ('white_mug' in protected) == first_mug_is_reference
    assert {'plate_1', 'plate_2', 'yellow_mug'} <= protected
    assert 'plate_1' in spatial_copy_protection(task, entities, [pick, place], None, event=place)


@pytest.mark.parametrize('problem', ['missing', 'missing_array', 'shape', 'dtype', None])
def test_counterfactual_requires_interaction_locks_at_initialization(tmp_path, problem):
    ds = saved_dataset(tmp_path)
    meta = ds.reader(0).meta
    entity = meta['scene_entities']['pudding']
    if problem == 'missing':
        entity.pop('interaction_locked')
    elif problem:
        entity['interaction_locked']['key'] = {'missing_array':'unknown', 'shape':'milk', 'dtype':'frames'}[problem]
    (tmp_path/'episode.json').write_text(json.dumps(meta))
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    if problem:
        with pytest.raises(ValueError, match='interaction_locked.*pudding.*interaction_locks'):
            GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
        # Validation and runs without counterfactual swaps do not need the flags.
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer(), augment=False)
        ds.config.grounding_augmentation = GroundingAugmentation()
        GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
    else:
        valid = GroundingDataset(ds.config.data_index, ds.config, ToyTokenizer())
        assert valid[1]['metadata']['entity_id'] == 'pudding'


@pytest.mark.parametrize('known', [True, False])
def test_sample_guard_keeps_locked_or_unknown_objects_out_of_new_pick_targets(tmp_path, known):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    entity = ds.reader(0).meta['scene_entities']['pudding']
    if known:
        entity['interaction_locked']['key'] = 'valid'  # All True, even with high visibility.
    else:
        entity.pop('interaction_locked')
    sample = ds[1]
    assert sample['metadata']['entity_id'] == 'basket'
    info = sample['metadata']['augmentation']
    assert info['locatable_object_count'] == 3 and info['swap_candidate_count'] == 0
    assert info['missing_interaction_lock_count'] == int(not known)


@pytest.mark.parametrize('target', ['region::basket_inside', 'basket', 'milk'])
def test_copy_keeps_supervised_region_parent_still_but_allows_unused_region_parent(tmp_path, target):
    import copy
    ds = saved_dataset(tmp_path)
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1)
    reader = ds.reader(0)
    entities = reader.meta['scene_entities']
    child = copy.deepcopy(entities['basket'])
    child.update(kind='region', category='spatial_region', instance='basket', site_name='basket_inside')
    child.pop('copy_eligible', None)
    entities['region::basket_inside'] = child
    entities['basket']['visible_fraction'] = dict(uri='arrays.npz', key='visible_fraction')
    reader.meta['events'][1]['interaction_target']['entity_id'] = target
    def guarded_copy(image, boxes, candidates, visible, robot, rng, config, *, priority_target=None, **kwargs):
        assert ('basket' in candidates) == (target != 'region::basket_inside')
        assert 'pudding' in candidates  # Other independent objects stay editable.
        return copy_objects(image, boxes, candidates, visible, lambda:np.zeros((32,32),bool), rng, config,
                           priority_target=priority_target)
    with mock.patch('openpi.augmentation.copy_objects', side_effect=guarded_copy):
        sample = ds.sample(1, review=True)
    if target == 'region::basket_inside':
        np.testing.assert_array_equal(sample['review']['boxes']['basket'], [20,20,28,28])
        np.testing.assert_array_equal(sample['review']['boxes'][target], [20,20,28,28])
        assert sample['metadata']['entity_id'] == target
        assert sample['targets']['role_valid'][0,0]
        assert sample['metadata']['augmentation']['moved_offsets'].get('basket') is None


def test_counterfactual_uses_second_moved_objects_new_box_mask_and_canvas(tmp_path):
    from openpi.transforms import role_canvas
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1,copy_prob=1,
                                                              copy_target_prob=1,copy_max_shift=.5)
    reader = ds.reader(0)
    basket = reader.meta['scene_entities']['basket']
    basket['visible_fraction'] = dict(uri='arrays.npz', key='visible_fraction')
    basket['regions']['image']['visible_masks']['key'] = 'basket_visible'
    basket_visible = np.zeros((32,32), bool)
    basket_visible[20:28,20:28] = True
    basket_packed = np.packbits(np.stack((basket_visible,basket_visible)).reshape(2,-1),axis=1)
    entity = reader.meta['scene_entities']['pudding']
    entity['regions']['image']['visible_masks']['key'] = 'pudding_visible'
    visible = np.zeros((32,32), bool)
    visible[2:8,24:30] = True
    packed = np.packbits(np.stack((visible,visible)).reshape(2,-1),axis=1)
    raw = reader._raw
    image = np.full((32,32,3),180,np.uint8)
    image[visible] = [220,30,10]
    moved_gripper = np.tile([0,28,4,32], (2,1))

    def arrays_for_test(uri, key):
        if key == 'pudding_visible':
            return packed
        if key == 'basket_visible':
            return basket_packed
        if key == 'gripper':
            return moved_gripper
        return raw(uri, key)

    def deterministic_copy(rgb, boxes, eligible, visible_mask, robot_mask, rng, config, **kwargs):
        fixed = mock.Mock()
        fixed.random.return_value = 0
        fixed.integers.return_value = 2
        fixed.permutation.return_value = [0]
        fixed.choice.side_effect = [12*(32-8+1)+4, 2*(32-6+1)+10]
        return copy_objects(rgb,boxes,eligible,visible_mask,lambda:np.zeros((32,32),bool),fixed,config,**kwargs)
    with mock.patch.object(reader,'rgb',return_value=image), \
         mock.patch.object(reader,'_raw',side_effect=arrays_for_test), \
         mock.patch('openpi.augmentation.copy_objects',side_effect=deterministic_copy):
        sample = ds.sample(1,review=True)
    info,review = sample['metadata']['augmentation'],sample['review']
    assert info['target_swapped'] and info['copy_applied'] and info['target_moved']
    assert list(info['moved_offsets']) == ['basket', 'pudding']
    np.testing.assert_array_equal(review['boxes']['pudding'],[10,2,16,8])
    np.testing.assert_array_equal(review['visible'],np.roll(visible,-14,axis=1))
    assert (review['edited'][2:8,10:16]==[220,30,10]).all()
    canvas,_,_ = role_canvas(review['edited'],{'bbox':review['boxes']['pudding']},ds.config.model.canvas_size)
    np.testing.assert_allclose(sample['targets']['role_images'][0,0],canvas.transpose(2,0,1)/127.5-1,atol=1e-7)


@pytest.mark.parametrize('task', ['put both the milk and the chocolate pudding in the basket',
    'pick up the milk next to the chocolate pudding and place it in the basket'])
def test_complex_tasks_keep_original_text_and_target(tmp_path, task):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.reader(0).meta['instructions'] = [task]
    sample = ds[1]
    assert sample['metadata']['task'] == task and sample['metadata']['entity_id'] == 'basket'
    assert not sample['metadata']['augmentation']['target_swapped']


def test_grounding_config_requires_frozen_backbone():
    with pytest.raises(ValueError, match='freeze_vlm=true'):
        TrainConfig('none','none','none','none',model=model_config(freeze_vlm=False),training_stage='grounding')


def test_online_policy_uses_full_task_once():
    from openpi.policies.current_grounding import CurrentGroundingPolicy
    model = Pi0ESkill(model_config()).eval()
    tokenizer = ToyTokenizer()
    with mock.patch('openpi.policies.current_grounding.TokenizePrompt', return_value=tokenizer):
        policy = CurrentGroundingPolicy(model, 'unused')
    task = 'put the milk in the basket'
    with mock.patch.object(model, 'predict_grounding', wraps=model.predict_grounding) as predict:
        result = policy.infer([np.full((32,32,3),180,np.uint8)], task)
    assert predict.call_count == 1 and result['task'] == task
    data = predict.call_args.args[0]
    assert tokenizer.decode(data['tokens'][0][data['token_mask'][0]].tolist()) == task
    assert set(data) == {'images','image_mask','tokens','token_mask','arm_presence'}


def test_checkpoint_roundtrip_and_trainable_scope(tmp_path):
    from openpi.policies.current_grounding import create_current_grounding_policy
    config = TrainConfig('none','none','none','none',model=model_config(),training_stage='grounding',
                         smoke=True,device='cpu',precision='float32')
    model = create_model(config).eval()
    assert all(name.startswith(('grounding.', 'visual_supervision.head.'))
               for name, p in model.named_parameters() if p.requires_grad)
    task = 'put the milk in the basket'
    from openpi.policies.current_grounding import CurrentGroundingPolicy
    with mock.patch('openpi.policies.current_grounding.TokenizePrompt',return_value=ToyTokenizer()):
        original = CurrentGroundingPolicy(model, 'unused')
        path = tmp_path/'checkpoint.pt'
        torch.save(dict(config=config.to_dict(), model=model.state_dict()), path)
        restored = create_current_grounding_policy(path)
    rgb = [np.full((32,32,3),200,np.uint8)]
    a,b = original.infer(rgb,task),restored.infer(rgb,task)
    for key in ('grounding','subject_grounding'):
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)


def test_task_padding_does_not_change_predictions():
    model = Pi0ESkill(model_config()).eval()
    data = inputs(model.config)
    padded = dict(data, tokens=torch.nn.functional.pad(data['tokens'],(0,4),value=99),
                  token_mask=torch.nn.functional.pad(data['token_mask'],(0,4),value=False))
    a,b = model.predict_grounding(data),model.predict_grounding(padded)
    for key in a:
        torch.testing.assert_close(a[key],b[key],atol=1e-6,rtol=1e-5)


def test_saved_interaction_lock_migration_preserves_existing_arrays_and_is_idempotent(tmp_path):
    import zipfile
    from scripts.preprocess.libero.interaction_locks import backfill
    from openpi.policies.eskill_dataset import EpisodeReader
    saved_dataset(tmp_path)
    meta_path = tmp_path/'episode.json'
    meta = json.loads(meta_path.read_text())
    meta['cameras']['image'].update(height=32,width=32)
    for entity in meta['scene_entities'].values():
        entity.pop('interaction_locked')
    meta_path.write_text(json.dumps(meta))
    with zipfile.ZipFile(tmp_path/'arrays.npz') as archive:
        original = {key:archive.read(key) for key in archive.namelist()}
    first = backfill(tmp_path/'index.json',tmp_path/'first')
    reader = EpisodeReader(meta_path)
    for key, expected in [('milk',True),('basket',False),('pudding',False)]:
        assert reader.array(reader.meta['scene_entities'][key]['interaction_locked']).tolist()==[expected,expected]
    second = backfill(tmp_path/'index.json',tmp_path/'second')
    assert first['updated']==1 and second['updated']==0
    with zipfile.ZipFile(tmp_path/'arrays.npz') as archive:
        for key,value in original.items():
            assert archive.read(key)==value
        assert len(archive.namelist())==len(set(archive.namelist()))


@pytest.mark.parametrize(('frame', 'current'), [(0, 'milk'), (1, 'basket')])
def test_dataset_prioritizes_current_before_text_replacement(tmp_path, frame, current):
    ds = saved_dataset(tmp_path)
    reader = ds.reader(0)
    for key in ('milk', 'basket'):
        reader.meta['scene_entities'][key]['visible_fraction'] = dict(uri='arrays.npz', key='visible_fraction')
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1,copy_target_prob=1,
                                                            target_swap_prob=0,copy_max_shift=.5)
    raw = reader._raw

    def arrays_for_test(uri, key):
        return np.tile([0, 28, 4, 32], (2, 1)) if key == 'gripper' else raw(uri, key)

    def fixed_copy(rgb,boxes,eligible,visible,robot,rng,config,**kwargs):
        assert kwargs['priority_target'] == current
        fixed = mock.Mock()
        fixed.random.side_effect = [0.,0.]
        fixed.integers.return_value = 1
        fixed.permutation.side_effect = np.arange
        fixed.choice.side_effect = lambda positions: positions[0]

        def visible_box(key):
            x0, y0, x1, y1 = boxes[key].astype(int)
            mask = np.zeros(rgb.shape[:2], bool)
            mask[y0:y1, x0:x1] = True
            return mask

        return copy_objects(rgb, boxes, eligible, visible_box, lambda: np.zeros(rgb.shape[:2], bool),
                            fixed, config, **kwargs)
    with mock.patch.object(reader, '_raw', side_effect=arrays_for_test), \
         mock.patch('openpi.augmentation.copy_objects', side_effect=fixed_copy):
        sample = ds.sample(frame,review=True)
    info = sample['metadata']['augmentation']
    assert info['copy_priority_target_eligible'] and info['copy_priority_target_moved']
    assert info['target_moved'] and not info['target_swapped']
    assert sample['metadata']['task']=='put the milk in the basket'
    assert sample['metadata']['entity_id'] == current
    assert set(info['moved_offsets']) == {current}


def test_resume_rejects_old_single_object_copy_configuration(tmp_path):
    from scripts.train import train
    config = TrainConfig('none','none',str(tmp_path/'run'),'none',model=model_config(),
        training_stage='grounding',grounding_augmentation=GroundingAugmentation(copy_prob=.5),
        smoke=True,device='cpu',precision='float32',validation_fraction=0)
    saved_config = config.to_dict()
    saved_config['grounding_augmentation'].pop('copy_max_objects')
    path = tmp_path/'old.pt'
    torch.save(dict(config=saved_config,norm_stats={},world_size=1),path)
    with mock.patch('scripts.train.check_resources'), \
         pytest.raises(ValueError,match='new multi-object sampler'):
        train(config,resume=path)


def test_refresh_copy_flags_allows_clear_container_and_preserves_saved_arrays(tmp_path):
    import zipfile
    from scripts.preprocess.libero.interaction_locks import backfill
    from openpi.policies.eskill_dataset import EpisodeReader
    saved_dataset(tmp_path)
    path = tmp_path/'episode.json'
    meta = json.loads(path.read_text())
    meta['cameras']['image'].update(height=32,width=32)
    for key,entity in meta['scene_entities'].items():
        entity['copy_eligible'] = dict(uri='arrays.npz',key='eligible')
        entity.pop('interaction_locked')  # Recomputed by the migration.
        entity['regions']['image']['masks'] = dict(entity['regions']['image']['visible_masks'])
    path.write_text(json.dumps(meta))
    with zipfile.ZipFile(tmp_path/'arrays.npz') as archive:
        before={name:archive.read(name) for name in archive.namelist()}
    first=backfill(tmp_path/'index.json',tmp_path/'first',refresh_copy_eligibility=True)
    reader=EpisodeReader(path)
    basket=reader.meta['scene_entities']['basket']
    assert reader.array(basket['copy_eligible']).tolist()==[True,True]
    milk=reader.meta['scene_entities']['milk']
    assert reader.array(milk['copy_eligible']).tolist()==[False,False]  # Gripper proximity remains protected.
    second=backfill(tmp_path/'index.json',tmp_path/'second',refresh_copy_eligibility=True)
    assert first['newly_copy_eligible_frames']==4 and second['updated']==0
    with zipfile.ZipFile(tmp_path/'arrays.npz') as archive:
        assert all(archive.read(name)==value for name,value in before.items())
        assert len(archive.namelist())==len(set(archive.namelist()))


def test_copy_eligibility_keeps_visibility_contact_and_object_kind_requirements():
    from scripts.preprocess.libero.atomic_vla.scene_regions import copy_eligibility
    locked=np.array([False,True,False,False,False])
    visible=np.array([1.,1.,.899,.90,1.])
    full=np.array([4,4,4,4,0])
    assert copy_eligibility('object',locked,visible,full).tolist()==[True,False,False,True,False]
    assert not copy_eligibility('fixture',locked,visible,full).any()
    assert not copy_eligibility('part',locked,visible,full).any()
