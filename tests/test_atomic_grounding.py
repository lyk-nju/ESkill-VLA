"""Contracts for source event endpoints and manipulation-object identity."""
from copy import deepcopy
import importlib
from types import SimpleNamespace

import pytest

source = importlib.import_module('scripts.preprocess.libero.atomic_vla.source')
grounding = importlib.import_module('scripts.preprocess.libero.atomic_vla.grounding')


def test_atomic_terminal_stop_maps_to_last_image_without_rewriting_source():
    annotation = {'segments': [
        {'start_frame': 0, 'end_frame': 1, 'primary_action_verb': 'pick',
         'chain_of_thought': 'This is step 1/2 of task, pick up the mug, and the atomic skill is pick'},
        {'start_frame': 2, 'end_frame': 5, 'primary_action_verb': 'place',
         'chain_of_thought': 'This is step 2/2 of task, place the mug on the plate, and the atomic skill is place'},
    ]}
    before = deepcopy(annotation)
    events = source.source_intervals(annotation, 5)
    assert [(e['start_frame'], e['stop_frame'], e['goal_frame']) for e in events] == [(0, 2, 1), (2, 5, 4)]
    assert events[-1]['source_end_frame'] == 5
    assert annotation == before
    annotation['segments'][-1]['end_frame'] = 6
    with pytest.raises(ValueError):
        source.source_intervals(annotation, 5)


def test_place_keeps_carried_object_but_close_targets_door():
    intervals = [
        {'source_id': 'event_000', 'source_skill': 'pick', 'description': 'pick up the yellow and white mug'},
        {'source_id': 'event_001', 'source_skill': 'place', 'description': 'place the mug into the microwave'},
        {'source_id': 'event_002', 'source_skill': 'close', 'description': 'close the microwave door'},
    ]
    before = deepcopy(intervals)
    episode = SimpleNamespace(entry={'key': 'case'}, count=12, instructions=['put mug in microwave and close it'],
                              evidence={'intervals': intervals, 'endpoint_rule': 'test'})
    record = grounding.object_records(episode)
    first, second, third = record['events']
    assert first['object_id'] == second['object_id'] == 'yellow_and_white_mug'
    assert third['object_id'] == 'microwave_door'
    assert record['manual_review'] == 'pending'
    assert intervals == before


def test_shared_frame_moves_to_next_event_and_goal_uses_updated_endpoint():
    annotation = {'segments': [
        {'start_frame': 0, 'end_frame': 2, 'primary_action_verb': 'pick', 'chain_of_thought': ''},
        {'start_frame': 2, 'end_frame': 5, 'primary_action_verb': 'place', 'chain_of_thought': ''},
    ]}
    before = deepcopy(annotation)
    events = source.source_intervals(annotation, 5)
    assert [(e['start_frame'],e['stop_frame'],e['goal_frame']) for e in events] == [(0,2,1),(2,5,4)]
    assert [[e['source_id'] for e in events if e['start_frame']<=f<e['stop_frame']] for f in range(5)] == [
        ['event_000'],['event_000'],['event_001'],['event_001'],['event_001']]
    assert events[0]['source_end_frame'] == 2
    assert events[0]['boundary_correction']['source_goal_frame'] == 2
    assert annotation == before
    raw = source.source_intervals(annotation,5,resolve_shared_boundaries=False)
    assert raw[0]['stop_frame'] == 3


def test_wider_overlap_is_not_truncated_by_shared_boundary_rule():
    annotation = {'segments': [
        {'start_frame':0,'end_frame':3,'primary_action_verb':'pick','chain_of_thought':''},
        {'start_frame':2,'end_frame':5,'primary_action_verb':'place','chain_of_thought':''},
    ]}
    with pytest.raises(ValueError,match='single shared boundary'):
        source.source_intervals(annotation,5)


def test_user_reviewed_push_changes_skill_and_preserves_source_label():
    annotation = {'segments': [
        {'start_frame':0,'end_frame':5,'primary_action_verb':'pick',
         'chain_of_thought':'This is step 1/1 of task, Push the plate to the front of the stove, and the atomic skill is pick'},
    ]}
    before = deepcopy(annotation)
    event, = source.source_intervals(annotation,5)
    assert event['skill']=='Push' and event['source_skill']=='pick'
    assert event['skill_correction']['skill_review']=='user_confirmed'
    assert annotation==before
    original, = source.source_intervals(annotation,5,apply_reviewed_skills=False)
    assert original['skill']=='Pick' and 'skill_correction' not in original


def test_push_in_task_context_does_not_relabel_a_pick_event():
    annotation = {'segments': [
        {'start_frame':0,'end_frame':5,'primary_action_verb':'pick',
         'chain_of_thought':'The Task Chain: [push the plate, pick up the mug].This is step 2/2 of task, pick up the mug, and the atomic skill is pick'},
    ]}
    event, = source.source_intervals(annotation,5)
    assert event['skill']=='Pick' and 'skill_correction' not in event


def test_reviewed_push_does_not_create_a_carried_object_for_following_place():
    episode = SimpleNamespace(entry={'key':'push_then_place'},count=10,instructions=['push the plate then place the mug'],
        evidence={'endpoint_rule':'test','intervals':[
            {'source_id':'push','source_skill':'pick','skill':'Push','description':'push the plate'},
            {'source_id':'place','source_skill':'place','skill':'Place','description':'place the mug'},
        ]})
    record=grounding.object_records(episode)
    first,second=record['events']
    assert first['object_id']=='plate' and first['skill']=='Push'
    assert first['object_id']!=second['object_id']
