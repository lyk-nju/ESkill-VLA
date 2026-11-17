"""Counterfactual independence, causal target text, and inactive action/skill paths."""
import json
import pickle
from unittest import mock

import numpy as np
import pytest
import torch

from openpi.augmentation import GroundingAugmentation, mask_grounding_patches, copy_object, augmentation_invalidates
from openpi.models.pi0_config import ESkillConfig
from openpi.models.pi0_eskill import Pi0ESkill
from openpi.models.target_text import completion_logits, generate_target_text
from openpi.policies.grounding_dataset import GroundingDataset, collate_grounding, entity_captions
from openpi.training.config import TrainConfig, create_model


def inputs(config, b=2):
    return dict(images=torch.randn(b, 1, 3, config.image_size, config.image_size),
        image_mask=torch.ones(b, 1, dtype=torch.bool), tokens=torch.randint(1, 50, (b, 5)),
        token_mask=torch.ones(b, 5, dtype=torch.bool), arm_presence=torch.tensor([[True, False]]).expand(b, -1))


def model_config(**kwargs):
    return ESkillConfig.tiny(visual_primitive_version='v3', vocab_size=128, **kwargs)


def test_current_is_identical_to_existing_v3_current_without_future_execution():
    torch.set_num_threads(2)
    torch.manual_seed(5)
    config = model_config()
    model = Pi0ESkill(config).eval()
    data = inputs(config)
    prefix = model.encode_observation(data)
    context = model.context_projection(prefix.context)
    kwargs = dict(text_context=context[:, -5:], text_mask=data['token_mask'])
    old = model.grounding(context, prefix.mask, model.arm_embedding.weight, **kwargs)[:, :, 0]
    with mock.patch.object(model.grounding.future_decoder, 'forward', side_effect=AssertionError('Future executed')):
        current = model.predict_current(data)
    torch.testing.assert_close(current, old)
    changed = dict(data, tokens=data['tokens'].flip(-1))
    assert not torch.allclose(current, model.predict_current(changed))


def test_target_text_teacher_forcing_is_causal_and_uses_no_future_tokens():
    torch.manual_seed(6)
    model = Pi0ESkill(model_config()).eval()
    prefix = model.encode_observation(inputs(model.config))
    tokens = torch.tensor([[11, 12, 13], [21, 22, 23]])
    valid = torch.ones_like(tokens, dtype=torch.bool)
    a = completion_logits(model, prefix, tokens, valid)
    modified = tokens.clone(); modified[:, 1:] += 30
    b = completion_logits(model, prefix, modified, valid)
    torch.testing.assert_close(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:])
    first = completion_logits(model, prefix, tokens[:, :0], valid[:, :0])
    torch.testing.assert_close(a[:, :1], first)


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


def test_grounding_and_text_losses_never_run_motion_action_or_skill():
    torch.set_num_threads(2)
    config = TrainConfig(data_index='none', tokenizer_path='none', output_dir='none', norm_stats='none',
        model=model_config(freeze_vlm=False), training_stage='grounding', grounding_target_text=True,
        device='cpu', precision='float32', smoke=True)
    model = create_model(config)
    data = inputs(config.model)
    data['task_inputs'] = inputs(config.model, 1)
    targets = dict(role_images=torch.rand(2, 2, 1, 3, 32, 32)*2-1,
        role_weights=torch.ones(2, 2, 1, 32, 32), role_valid=torch.tensor([[[True], [False]]]).expand(2, -1, -1),
        target_text_tokens=torch.tensor([[11, 12, 2]]), target_text_valid=torch.ones(1, 3, dtype=torch.bool))
    with mock.patch.object(model.motion, 'forward', side_effect=AssertionError('Motion executed')), \
         mock.patch.object(model.skill, 'forward', side_effect=AssertionError('Skill executed')), \
         mock.patch.object(model, 'flow_loss', side_effect=AssertionError('Action executed')), \
         mock.patch.object(model.grounding.future_decoder, 'forward', side_effect=AssertionError('Future executed')):
        losses = model(data, targets)
        assert set(losses) == {'current', 'target_text'}
        sum(losses.values()).backward()
    for module in [model.motion, model.skill, model.action_in_proj, model.grounding.future_decoder]:
        assert all(p.grad is None for p in module.parameters())
    assert model.grounding.queries.grad[0].abs().sum() > 0
    assert model.grounding.queries.grad[1].abs().sum() == 0
    assert model.paligemma_with_expert.paligemma.language_model.embed_tokens.weight.grad.abs().sum() > 0


def test_frozen_vlm_text_learning_cannot_silently_be_enabled():
    with pytest.raises(ValueError, match='frozen VLM'):
        TrainConfig(data_index='none', tokenizer_path='none', output_dir='none', norm_stats='none',
                    model=model_config(), training_stage='grounding', grounding_target_text=True)


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
        milk=np.tile([2,2,10,10],(2,1)), basket=np.tile([20,20,28,28],(2,1)),
        gripper=np.tile([12,12,16,16],(2,1)),
        visible=np.packbits(masks.reshape(2,-1),axis=1), visible_fraction=np.ones(2,np.float32),
        contact=np.full((2,4),np.nan), eligible=np.zeros(2,bool))
    np.savez(tmp_path/'arrays.npz',**arrays)
    entities = {key:dict(kind='object',category=key,visible_fraction=ref('visible_fraction'),regions={'image':dict(frame_indices=ref('frames'),
        boxes_xyxy=ref(key),boxes_valid=ref('valid'),visible_masks=ref('visible',encoding='packbits',shape=[2,32,32]))})
        for key in ('milk','basket')}
    meta = dict(episode_id='test',instructions=['put milk into basket'],scene_entities=entities,
        cameras={'image':dict(frame_indices=ref('frames'),valid=ref('valid'),rgb=ref('rgb'))},
        online_grounding=dict(interaction_boxes=ref('contact')),
        effector_regions={'single':{'image':dict(frame_indices=ref('frames'),boxes_xyxy=ref('gripper'),boxes_valid=ref('valid'))}},
        events=[dict(arm='single',start_frame=i,stop_frame=i+1,interaction_target=dict(entity_id=key),
                     grounding_template='pick up the {object}' if i==0 else None,
                     description='pick up the milk' if i==0 else 'place the milk in the basket')
                for i,key in enumerate(['milk','basket'])])
    (tmp_path/'episode.json').write_text(json.dumps(meta))
    (tmp_path/'index.json').write_text(json.dumps(dict(episodes=[dict(episode_id='test',path='episode.json')])))
    config = TrainConfig(str(tmp_path/'index.json'),'','','',model=model_config(freeze_vlm=False),
        training_stage='grounding',grounding_target_text=True,grounding_max_target_tokens=96,
        grounding_augmentation=aug or GroundingAugmentation())
    return GroundingDataset(config.data_index,config,ToyTokenizer())


def test_single_counterfactual_changes_prompt_and_label_without_action_or_task_leakage(tmp_path):
    ds = saved_dataset(tmp_path)
    original = ds[0]
    ds.config.grounding_augmentation = GroundingAugmentation(target_swap_prob=1)
    restored = pickle.loads(pickle.dumps(ds))
    swapped = restored[0]
    assert original['metadata']['target_text'] == 'pick up the milk'
    assert swapped['metadata']['target_text'] == 'pick up the basket'
    assert np.array_equal(original['inputs']['images'],swapped['inputs']['images'])
    assert not np.array_equal(original['targets']['role_images'],swapped['targets']['role_images'])
    assert np.array_equal(original['targets']['target_text_tokens'],swapped['targets']['target_text_tokens'])
    batch = collate_grounding([original,swapped])
    assert batch['inputs']['images'].shape[:2] == (2,1)
    assert batch['inputs']['task_inputs']['images'].shape[0] == 2
    assert set(batch['targets']) == {'role_images','role_weights','role_valid','target_text_tokens','target_text_valid'}
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
    expected = 'basket' if fraction is not None and fraction >= .8 else 'milk'
    for sample in samples:
        assert sample['metadata']['entity_id'] == expected
        assert sample['metadata']['augmentation']['target_swapped'] == (expected == 'basket')
        assert sample['targets']['role_valid'][0, 0]  # An occluded real Event is still supervised.
        assert sample['targets']['target_text_valid'].any()


def test_counterfactual_chooses_visible_alternative_without_requiring_copy_eligibility(tmp_path):
    ds = saved_dataset(tmp_path, aug=GroundingAugmentation(target_swap_prob=1))
    reader = ds.reader(0)
    entities = reader.meta['scene_entities']
    entities['pudding'] = dict(entities['basket'], category='pudding')
    entities['basket'] = dict(entities['basket'], visible_fraction=dict(uri='arrays.npz', key='hidden_fraction'))
    array = reader.array
    def visibility(ref):
        return np.zeros(2, np.float32) if ref['key'] == 'hidden_fraction' else array(ref)
    with mock.patch.object(reader, 'array', side_effect=visibility):
        for epoch in range(8):
            sample = ds[(0, epoch, 0)]
            assert sample['metadata']['entity_id'] == 'pudding'
            assert sample['metadata']['target_text'] == 'pick up the pudding'
            assert sample['metadata']['augmentation']['target_visible_fraction'] == 1.


def test_target_and_native_occlusion_kept_when_all_unprotected_patches_are_masked(tmp_path):
    ds = saved_dataset(tmp_path,aug=GroundingAugmentation(patch_prob=1,patch_drop_prob=1,patch_size=8))
    masked = ds[0]
    assert masked['targets']['role_valid'][0,0]
    assert not masked['metadata']['augmentation']['augmentation_invalidated']
    assert masked['targets']['target_text_valid'].any()
    ds.config.grounding_augmentation = GroundingAugmentation()
    clean = ds[0]
    np.testing.assert_array_equal(clean['targets']['role_images'],masked['targets']['role_images'])
    np.testing.assert_array_equal(clean['inputs']['task_inputs']['images'],masked['inputs']['task_inputs']['images'])
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


def test_relational_or_incompatible_templates_fall_back_to_locate():
    from openpi.policies.grounding_dataset import goal_text
    entities={'milk':dict(kind='object'),'drawer':dict(kind='part')}
    event=dict(interaction_target=dict(entity_id='drawer'),description='put milk in drawer',grounding_template=None)
    assert goal_text(event,'milk','milk',entities)=='Locate milk.'
    assert goal_text(event,'drawer','drawer',entities,edited=True)=='Locate drawer.'
    event['grounding_template']='open the {object}'
    assert goal_text(event,'milk','milk',entities)=='Locate milk.'


def test_generation_stops_at_eos_without_teacher_targets():
    model = Pi0ESkill(model_config()).eval()
    data = inputs(model.config, 1)
    def logits(model, prefix, tokens, valid):
        value = torch.zeros(1, len(tokens[0])+1, 128)
        value[:, -1, 2 if tokens.shape[-1] else 11] = 10
        return value
    with mock.patch('openpi.models.target_text.completion_logits', side_effect=logits):
        result = generate_target_text(model, data, eos_id=2, max_tokens=4)
    assert result['tokens'].tolist() == [[11, 2]]
    assert result['terminated'].all() and result['valid'].all()


def test_online_policy_generates_then_locates_and_does_not_fallback_to_labels():
    from openpi.policies.current_grounding import CurrentGroundingPolicy
    model = Pi0ESkill(model_config()).eval()
    tokenizer = ToyTokenizer()
    with mock.patch('openpi.policies.current_grounding.TokenizePrompt', return_value=tokenizer):
        policy = CurrentGroundingPolicy(model, 'unused')
    rgb = [np.full((32, 32, 3), 200, np.uint8)]
    generated = dict(tokens=torch.tensor([tokenizer.encode('milk')+[2]]),
                     valid=torch.ones(1, 5, dtype=torch.bool), terminated=torch.tensor([True]))
    with mock.patch('openpi.policies.current_grounding.generate_target_text', return_value=generated) as generate, \
         mock.patch.object(model, 'predict_grounding', wraps=model.predict_grounding) as locate:
        result = policy.infer(rgb, 'put milk into basket')
        assert result['target_text'] == 'milk' and result['target_text_source'] == 'predicted'
        assert result['target_presence_confidence'] is None
        assert result['grounding'].shape == (1, 2, 4, model.config.hidden_dim)
        assert set(generate.call_args.args[1]) == {'images', 'image_mask', 'tokens', 'token_mask', 'arm_presence'}
        target_input = locate.call_args.args[0]
        prompt = tokenizer.decode(target_input['tokens'][0][target_input['token_mask'][0]].tolist())
        assert prompt == 'milk'
    generated['terminated'][:] = False
    with mock.patch('openpi.policies.current_grounding.generate_target_text', return_value=generated), \
         mock.patch.object(model, 'predict_grounding', side_effect=AssertionError('No fallback on incomplete generation')):
        assert policy.infer(rgb, 'put milk into basket')['grounding'] is None


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
    rng=mock.Mock();rng.random.return_value=0; rng.integers.side_effect=[0,12,4]
    result,updated,info=copy_object(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
    assert info['copy_applied'] and info['offset']==[12,4]
    np.testing.assert_array_equal(updated['milk'],[16,8,24,16])
    np.testing.assert_array_equal(result[8:16,16:24],original[4:12,4:12])
    assert (result[2:14,2:14]==90).all()
    np.testing.assert_array_equal(image,original)
    canvas,_,valid=role_canvas(result,{'bbox':updated['milk']},40)
    assert valid and (canvas[8:16,16:24]==[220,30,10]).all()
    robot[4:12,4:12]=True
    rng.integers.side_effect=[0,12,4]
    blocked,_,info=copy_object(image,boxes,['milk'],lambda k:mask,lambda:robot,rng,cfg)
    assert blocked is image and info['copy_reason']=='robot_overlap'


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
    model=Pi0ESkill(model_config(grounding_subject=True,freeze_vlm=False)).eval()
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
    assert any(p.grad is not None and p.grad.abs().sum()>0
               for p in model.paligemma_with_expert.paligemma.vision_tower.parameters())


def test_subject_training_uses_visual_loss_and_does_not_train_gripper_commands():
    from openpi.training.metrics import loss_metrics
    config=TrainConfig('none','none','none','none',model=model_config(grounding_subject=True,freeze_vlm=False),
                       training_stage='grounding',smoke=True,device='cpu',precision='float32')
    model=create_model(config)
    data=inputs(config.model)
    targets=dict(role_images=torch.rand(2,2,2,3,32,32)*2-1,role_weights=torch.ones(2,2,2,32,32),
                 role_valid=torch.tensor([[[True,True],[False,False]],[[True,False],[False,False]]]))
    with mock.patch.object(model.motion,'forward',side_effect=AssertionError('Motion executed')), \
         mock.patch.object(model.grounding.future_decoder,'forward',side_effect=AssertionError('Future executed')), \
         mock.patch.object(model,'flow_loss',side_effect=AssertionError('Action executed')):
        losses=model(data,targets)
        assert set(losses)=={'current','subject'}
        sum(losses.values()).backward()
    assert model.loss_counts['current']==2 and model.loss_counts['subject']==1
    assert model.grounding.subject_queries.grad.abs().sum()>0
    assert model.grounding.queries.grad[0].abs().sum()>0
    assert all(p.grad is None for p in model.motion.gripper_head.parameters())
    metrics=loss_metrics(dict(loss=0,**losses))
    assert 'subject_grounding_loss' in metrics and 'gripper_loss' not in metrics


def test_subject_canvas_and_protect_follow_gripper_independently_of_target(tmp_path):
    from dataclasses import replace
    ds=saved_dataset(tmp_path)
    ds.config.model=replace(ds.config.model,grounding_subject=True)
    initial=ds.sample(0,review=True)
    ds.config.grounding_augmentation=GroundingAugmentation(target_swap_prob=1,patch_prob=1,patch_drop_prob=1,patch_size=4)
    switched=ds.sample(0,review=True)
    assert initial['metadata']['entity_id']=='milk' and switched['metadata']['entity_id']=='basket'
    np.testing.assert_array_equal(initial['targets']['role_images'][0,1],switched['targets']['role_images'][0,1])
    assert not np.array_equal(initial['targets']['role_images'][0,0],switched['targets']['role_images'][0,0])
    assert switched['targets']['role_valid'].tolist()==[[True,True],[False,False]]
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
    a=policy.infer(image,'task',target_text='pick milk')
    b=policy.infer(image,'task',target_text='pick pudding')
    torch.testing.assert_close(a['subject_grounding'],b['subject_grounding'],rtol=0,atol=0)
    assert a['subject_grounding'].shape==a['grounding'].shape
    assert not torch.allclose(a['grounding'],b['grounding'])
