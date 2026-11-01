"""Storage semantics: explicit goals, packed masks, asynchronous actions and review provenance."""
import json

import numpy as np
import pytest

from openpi.policies.eskill_dataset import SCHEMA_VERSION, load_episode
from scripts.preprocess.export import EpisodeWriter, write_index, validate_episode


def bundle(tmp_path):
    writer = EpisodeWriter()
    index = writer.add('index',np.arange(3,dtype=np.int64))
    valid = writer.add('valid',np.ones(3,dtype=bool))
    rgb = np.arange(3*3*5*3,dtype=np.uint8).reshape(3,3,5,3)
    mask = np.zeros((3,3,5),dtype=bool)
    mask[:2,:2,1:3] = True
    seq = dict(frame_indices=index,boxes_xyxy=writer.add('boxes',np.array([[1,0,3,2],[1,0,3,2],[np.nan]*4])),
        boxes_valid=writer.add('box_valid',np.array([True,True,False])),masks=writer.add('masks',mask,pack_mask=True),
        masks_valid=valid,visibility=None,region_extent='full_in_frame',evidence={})
    def event(identity,arm,start,stop,goal,object_regions):
        return dict(event_id=identity,arm=arm,start_frame=start,stop_frame=stop,goal_frame=goal,
            boundary_valid=True,description='push bowl',skill='Push',skill_valid=True,
            object_present=True if object_regions else None,object_regions=object_regions,
            cooperative_group_id=None,evidence={'skill':{'method':'human','review_status':'accepted'}},
            source_annotation={'source_skill':'Pick','instance':'bowl_1'})
    actions = np.ones((5,2),np.float32)
    action_indices = writer.add('action_indices',np.array([0,0,1,1,2],np.int64))
    meta = dict(schema_version=SCHEMA_VERSION,episode_id='test',
        source={'dataset':'fixture','revision':'test','episode_id':'1','parent_trajectory_id':'parent_1'},
        instructions=['push bowl'],embodiment='test',arm_ids=['left','right'],source_arm_mapping={},
        frame_count=3,timeline='source_steps',timestamps_s=None,nominal_fps=None,alignment='source paired',
        cameras={'main':dict(rgb=writer.add('rgb',rgb),frame_indices=index,valid=valid,
                            height=3,width=5,role='external',calibration=None)},state=None,
        actions=dict(values=writer.add('action',actions),frame_indices=action_indices,
            valid=writer.add('action_valid',np.ones_like(actions,dtype=bool)),
            channels=[{},{}],conditioning_image_indices={
                'main':writer.add('condition_images',np.array([0,0,1,1,2],np.int64))}),
        effector_positions={},effector_regions={'left':{'main':seq}},
        events=[event('first','left',0,1,1,{'main':seq}),event('second','left',1,3,2,{}),
                event('parallel','right',0,3,2,{})],cooperative_goal_groups=[],
        quality={'manual_review':'pending','warnings':['state_alignment_exceeds_10mm']})
    path = writer.save(tmp_path/'episode',meta)
    return path,rgb,mask


def test_roundtrip_retains_source_values_masks_and_provenance(tmp_path):
    path,rgb,mask = bundle(tmp_path)
    reader = load_episode(path)
    assert validate_episode(reader) == {'frames':3,'events':3,'structural_check':'passed'}
    seq = reader.meta['events'][0]['object_regions']['main']
    np.testing.assert_array_equal(reader.array(seq['masks']),mask)
    np.testing.assert_array_equal(reader.rgb('main',1),rgb[1])
    assert reader.meta['timestamps_s'] is None and reader.meta['nominal_fps'] is None
    assert reader.array(reader.meta['actions']['values']).shape == (5,2)
    assert reader.meta['events'][0]['source_annotation']['source_skill'] == 'Pick'
    assert reader.meta['quality']['manual_review'] == 'pending'
    assert reader.meta['quality']['warnings'] == ['state_alignment_exceeds_10mm']
    with np.load(path.parent/'arrays.npz') as arrays:
        assert arrays['masks'].shape == (3,2)  # 15 pixels per row, not byte-aligned.


def test_goal_uses_own_object_at_explicit_endpoint_not_next_event(tmp_path):
    path,_,mask = bundle(tmp_path)
    reader = load_episode(path)
    assert reader.event_at('left',1)['event_id'] == 'second'
    assert reader.event_at('right',1)['event_id'] == 'parallel'
    goal = reader.goal('first','main')
    assert goal['frame'] == 1
    np.testing.assert_array_equal(goal['region']['mask'],mask[1])
    assert reader.goal('second','main')['region'] is None
    assert reader.rgb('main',3) is None


def test_empty_full_mask_is_known_region_not_missing(tmp_path):
    path,_,_ = bundle(tmp_path)
    reader = load_episode(path)
    sequence = reader.meta['events'][0]['object_regions']['main']
    region = reader.region(sequence,2)
    assert region['bbox'] is None and not region['mask'].any()
    assert reader.region(sequence,3) is None


@pytest.mark.parametrize('change', ['overlap','goal_outside','future_image','timestamp_missing'])
def test_rejects_temporal_misinterpretation(tmp_path,change):
    path,_,_ = bundle(tmp_path)
    reader = load_episode(path)
    if change == 'overlap':
        reader.meta['events'][1]['start_frame'] = 0
    elif change == 'goal_outside':
        reader.meta['events'][1]['goal_frame'] = 3
    elif change == 'future_image':
        ref = reader.meta['actions']['conditioning_image_indices']['main']
        reader.array(ref)[0] = 2
    else:
        reader.meta['timeline'] = 'timestamps'
    with pytest.raises(ValueError):
        validate_episode(reader)


def test_index_is_relative_and_duplicate_source_is_rejected(tmp_path):
    path,_,_ = bundle(tmp_path)
    value = write_index(tmp_path/'index.json',[path])
    assert value['episodes'][0]['path'] == 'episode/episode.json'
    assert value['frame_count'] == 3 and value['event_count'] == 3
    assert json.loads((tmp_path/'index.json').read_text()) == value
    with pytest.raises(ValueError,match='Duplicate source'):
        write_index(tmp_path/'bad.json',[path,path])
    second = json.loads(path.read_text())
    second['source']['episode_id'] = 'second_camera_copy'
    second_path = path.with_name('other_view.json')
    second_path.write_text(json.dumps(second))
    with pytest.raises(ValueError,match='Duplicate source'):
        write_index(tmp_path/'bad_views.json',[path,second_path])


def test_saved_view_respects_validity_and_requires_explicit_row_alignment(tmp_path):
    from scripts.preprocess.libero.view import from_saved
    path,_,_ = bundle(tmp_path)
    reader = load_episode(path)
    with pytest.raises(ValueError,match='aligned'):
        from_saved(reader,arm='left',camera='main')
    reader.meta['actions'] = None
    reader.meta['events'][0].update(boundary_valid=False,skill_valid=False)
    reader.meta['events'][0]['object_regions']['main']['region_extent'] = 'visible'
    reader.meta['effector_regions']['left']['main']['region_extent'] = 'visible'
    view = from_saved(reader,arm='left',camera='main')
    assert view.events[0].goal_frame is None and view.events[0].skill is None
    assert view.annotation_label == 'SAVED LABELS / VISIBLE'
    assert view.source_fps is None and view.review_status == 'pending'
