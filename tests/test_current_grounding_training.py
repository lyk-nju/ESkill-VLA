"""Counterfactual independence, full task conditioning, and inactive action/skill paths."""
import json
import pickle
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.augmentation import GroundingAugmentation, mask_grounding_patches, copy_object, augmentation_invalidates
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.policies.grounding_dataset import (GroundingDataset, collate_grounding, entity_captions,
                                              copy_safe_for_grounding)
from openpi.training.config import TrainConfig, create_model


def inputs(config, b=2):
    return dict(images=torch.randn(b, 1, 3, config.image_size, config.image_size),
        image_mask=torch.ones(b, 1, dtype=torch.bool), tokens=torch.randint(1, 50, (b, 5)),
        token_mask=torch.ones(b, 5, dtype=torch.bool), arm_presence=torch.tensor([[True, False]]).expand(b, -1))


def model_config(**kwargs):
    return ESkillConfig.tiny(visual_primitive_version='v3', vocab_size=128,
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
        contact=np.full((2,4),np.nan), eligible=np.zeros(2,bool))
    np.savez(tmp_path/'arrays.npz',**arrays)
    entities = {key:dict(kind='object',category=key,visible_fraction=ref('visible_fraction'),regions={'image':dict(frame_indices=ref('frames'),
        boxes_xyxy=ref(key),boxes_valid=ref('valid'),visible_masks=ref('visible',encoding='packbits',shape=[2,32,32]))})
        for key in ('milk','basket','pudding')}
    entities['pudding']['copy_eligible'] = ref('valid')
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
    original = ds[0]
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    restored = pickle.loads(pickle.dumps(ds))
    swapped = restored[0]
    assert original['metadata']['task'] == 'put the milk in the basket'
    assert swapped['metadata']['task'] == 'put the pudding in the basket'
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
        samples = [ds[(0, epoch, 0)] for epoch in range(4)]
    expected = 'pudding' if fraction is not None and fraction >= .8 else 'milk'
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
    rng=mock.Mock();rng.random.return_value=0; rng.integers.return_value=0
    rng.choice.return_value=8*(40-8+1)+16
    result,updated,info=copy_object(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
    json.dumps(info)  # The training logger writes offsets and placement counts to JSON.
    assert info['copy_applied'] and info['offset']==[12,4]
    np.testing.assert_array_equal(updated['milk'],[16,8,24,16])
    np.testing.assert_array_equal(result[8:16,16:24],original[4:12,4:12])
    assert (result[2:14,2:14]==90).all()
    np.testing.assert_array_equal(image,original)
    canvas,_,valid=role_canvas(result,{'bbox':updated['milk']},40)
    assert valid and (canvas[8:16,16:24]==[220,30,10]).all()
    robot[4:12,4:12]=True
    blocked,_,info=copy_object(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
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
    assert all(p.grad is None for p in model.action_in_proj.parameters())
    metrics=loss_metrics(dict(loss=0,**losses))
    assert 'subject_grounding_loss' in metrics and 'gripper_loss' not in metrics


def test_subject_canvas_and_protect_follow_gripper_independently_of_target(tmp_path):
    from dataclasses import replace
    ds=saved_dataset(tmp_path)
    ds.config.model=replace(ds.config.model,grounding_subject=True)
    initial=ds.sample(0,review=True)
    ds.config.grounding_augmentation=GroundingAugmentation(target_swap_prob=1,patch_prob=1,patch_drop_prob=1,patch_size=4)
    switched=ds.sample(0,review=True)
    assert initial['metadata']['entity_id']=='milk' and switched['metadata']['entity_id']=='pudding'
    np.testing.assert_array_equal(initial['targets']['role_images'][0,1],switched['targets']['role_images'][0,1])
    assert not np.array_equal(initial['targets']['role_images'][0,0],switched['targets']['role_images'][0,0])
    assert switched['targets']['role_valid'].tolist()==[[True,True],[False,False]]
    size = switched['review']['original'].shape[0]
    np.testing.assert_allclose(switched['targets']['role_core_boxes'][0,0],
                               switched['review']['boxes']['pudding']/size)
    np.testing.assert_allclose(switched['targets']['role_core_boxes'][0,1],
                               switched['review']['gripper_box']/size)
    assert 'role_core_boxes' not in switched['inputs']
    # Empty-hand gripper is protected even with no interaction/contact box.
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


def test_full_task_event_switch_and_place_counterfactual(tmp_path):
    ds = saved_dataset(tmp_path)
    pick, place = ds[0], ds[1]
    assert pick['metadata']['task'] == place['metadata']['task'] == 'put the milk in the basket'
    assert pick['metadata']['entity_id'] == 'milk' and place['metadata']['entity_id'] == 'basket'
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    for frame in (0, 1):
        sample = ds[frame]
        assert sample['metadata']['task'] == 'put the pudding in the basket'
        assert sample['metadata']['entity_id'] == 'pudding'
    ds.reader(0).meta['scene_entities']['pudding'].pop('copy_eligible')
    assert ds[1]['metadata']['entity_id'] == 'pudding'  # Querying does not require edit permission.


@pytest.mark.parametrize('editable', [True, False, None])
def test_visible_counterfactual_does_not_require_copy_eligibility(tmp_path, editable):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    entity = ds.reader(0).meta['scene_entities']['pudding']
    if editable is None:
        entity.pop('copy_eligible')
    elif not editable:
        entity['copy_eligible']['key'] = 'eligible'  # Both saved values are False.
    sample = ds[1]
    assert sample['metadata']['entity_id'] == 'pudding'
    assert sample['metadata']['augmentation']['target_swapped']
    assert sample['metadata']['augmentation']['locatable_object_count'] == 3
    assert sample['metadata']['augmentation']['swap_candidate_count'] == 1
    assert sample['metadata']['augmentation']['copy_candidate_count'] == int(editable is True)


@pytest.mark.parametrize('task', ['put the milk in the basket', 'put the milk into the basket',
    'place the milk in the basket', 'place milk onto basket.',
    'pick up the milk and place it in the basket', 'Pick up the milk and put it into the basket!'])
def test_transfer_paraphrases_replace_only_source_and_keep_punctuation(tmp_path, task):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.reader(0).meta['instructions'] = [task]
    for frame in (0, 1):
        sample = ds[frame]
        assert sample['metadata']['task'] == task.replace('milk', 'pudding')
        assert sample['metadata']['entity_id'] == 'pudding'


@pytest.mark.parametrize('task', ['turn on the stove', 'do the milk transfer'])
def test_copy_paste_attempt_is_independent_of_transfer_parser(tmp_path, task):
    ds = saved_dataset(tmp_path)
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1)
    ds.reader(0).meta['instructions'] = [task]
    def copy(image, boxes, candidates, visible, robot, rng, config, *, task_target=None):
        assert config.copy_prob == 1 and candidates == ['pudding']
        return image, boxes, dict(copy_attempted=True, copy_applied=False, moved_entity='', copy_reason='entity_overlap')
    with mock.patch('openpi.augmentation.copy_object', side_effect=copy) as paste:
        sample = ds[0]
    assert paste.call_count == 1
    assert sample['metadata']['augmentation']['copy_attempted']
    assert not sample['metadata']['augmentation']['transfer_resolved']


@pytest.mark.parametrize('task', ['pick up the leftmost bowl', 'pick up the bowl beside the plate',
    'pick up the black bowl on the cookies and place it on the plate', 'pick up the nearest milk'])
def test_spatial_copy_guard_is_independent_of_transfer_parser(task):
    assert not copy_safe_for_grounding(task)
    assert copy_safe_for_grounding('put the milk into the basket')
    assert copy_safe_for_grounding('pick up the milk')
    assert copy_safe_for_grounding('open the drawer')


@pytest.mark.parametrize('known', [True, False])
def test_interaction_locked_or_unknown_object_is_locatable_but_not_a_new_pick_target(tmp_path, known):
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


def test_copied_counterfactual_target_updates_pixels_box_visible_mask_and_vae_canvas(tmp_path):
    from openpi.transforms import role_canvas
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1,copy_prob=1,copy_max_shift=.5)
    reader = ds.reader(0)
    entity = reader.meta['scene_entities']['pudding']
    entity['regions']['image']['visible_masks']['key'] = 'pudding_visible'
    visible = np.zeros((32,32), bool)
    visible[2:8,24:30] = True
    packed = np.packbits(np.stack((visible,visible)).reshape(2,-1),axis=1)
    raw = reader._raw
    image = np.full((32,32,3),180,np.uint8)
    image[visible] = [220,30,10]
    def deterministic_copy(rgb, boxes, eligible, visible_mask, robot_mask, rng, config, *, task_target=None):
        fixed = mock.Mock()
        fixed.random.return_value = 0
        fixed.integers.return_value = 0
        fixed.choice.return_value = 2*(32-6+1)+10
        return copy_object(rgb,boxes,eligible,visible_mask,lambda:np.zeros((32,32),bool),fixed,config,task_target=task_target)
    with mock.patch.object(reader,'rgb',return_value=image), \
         mock.patch.object(reader,'_raw',side_effect=lambda uri,key: packed if key=='pudding_visible' else raw(uri,key)), \
         mock.patch('openpi.augmentation.copy_object',side_effect=deterministic_copy):
        sample = ds.sample(0,review=True)
    info,review = sample['metadata']['augmentation'],sample['review']
    assert info['target_swapped'] and info['copy_applied'] and info['target_moved']
    np.testing.assert_array_equal(review['boxes']['pudding'],[10,2,16,8])
    np.testing.assert_array_equal(review['visible'],np.roll(visible,-14,axis=1))
    assert (review['edited'][2:8,10:16]==[220,30,10]).all()
    canvas,_,_ = role_canvas(review['edited'],{'bbox':review['boxes']['pudding']},ds.config.model.canvas_size)
    np.testing.assert_allclose(sample['targets']['role_images'][0,0],canvas.transpose(2,0,1)/127.5-1,atol=1e-7)


@pytest.mark.parametrize('task', ['put both the milk and the pudding in the basket',
    'pick up the milk next to the pudding and place it in the basket'])
def test_complex_tasks_keep_original_text_and_target(tmp_path, task):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    ds.reader(0).meta['instructions'] = [task]
    sample = ds[0]
    assert sample['metadata']['task'] == task and sample['metadata']['entity_id'] == 'milk'
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


def test_dataset_prefers_original_event_target_and_keeps_its_grounding_label(tmp_path):
    ds = saved_dataset(tmp_path)
    reader = ds.reader(0)
    reader.meta['scene_entities']['milk']['copy_eligible'] = dict(uri='arrays.npz',key='valid')
    ds.config.grounding_augmentation = GroundingAugmentation(copy_prob=1,copy_target_prob=1,
                                                            target_swap_prob=0,copy_max_shift=.5)
    def fixed_copy(rgb,boxes,eligible,visible,robot,rng,config,*,task_target=None):
        assert task_target=='milk'
        fixed = mock.Mock()
        fixed.random.side_effect = [0.,0.]
        fixed.choice.return_value = 2*(32-8+1)+12
        return copy_object(rgb,boxes,eligible,visible,lambda:np.zeros((32,32),bool),fixed,config,
                           task_target=task_target)
    with mock.patch('openpi.augmentation.copy_object',side_effect=fixed_copy):
        sample = ds.sample(0,review=True)
    info = sample['metadata']['augmentation']
    assert info['copy_task_target_available'] and info['copied_task_target']
    assert info['target_moved'] and not info['target_swapped']
    assert sample['metadata']['task']=='put the milk in the basket'
    assert sample['metadata']['entity_id']=='milk'
    np.testing.assert_array_equal(sample['review']['boxes']['milk'],[12,2,20,10])


def test_resume_rejects_old_copy_configuration_without_target_probability(tmp_path):
    from scripts.train import train
    config = TrainConfig('none','none',str(tmp_path/'run'),'none',model=model_config(),
        training_stage='grounding',grounding_augmentation=GroundingAugmentation(copy_prob=.5),
        smoke=True,device='cpu',precision='float32',validation_fraction=0)
    saved_config = config.to_dict()
    saved_config['grounding_augmentation'].pop('copy_target_prob')
    path = tmp_path/'old.pt'
    torch.save(dict(config=saved_config,norm_stats={},world_size=1),path)
    with mock.patch('scripts.train.check_resources'), \
         pytest.raises(ValueError,match='task-target-biased Copy-Paste'):
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
    visible=np.array([1.,1.,.97,.98,1.])
    full=np.array([4,4,4,4,0])
    assert copy_eligibility('object',locked,visible,full).tolist()==[True,False,False,True,False]
    assert not copy_eligibility('fixture',locked,visible,full).any()
    assert not copy_eligibility('part',locked,visible,full).any()
