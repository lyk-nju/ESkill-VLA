"""Use the deployed planner and the training event contract in offline diagnostics."""
import numpy as np
import torch

from openpi.policies.eskill_dataset import EpisodeReader, planning_event
from openpi.policies.motion_planner import MotionPlannerPolicy
from scripts.analysis.motion_planner import entity_coverage, episode_rows, plan_pair, receiver_event
from test_motion_planner_data import planner_dataset
from test_motion_planner_model import build


def test_continuous_and_paired_predictions(tmp_path):
    planner_dataset(tmp_path)
    reader = EpisodeReader(tmp_path / 'episode.json')
    model, _, _ = build()

    def tokenize(task, state):
        return np.array([1, 5 if 'milk' in task else 6, 0]), np.array([True, True, False])

    policy = MotionPlannerPolicy(model, tokenize, 'float32')
    rows = list(episode_rows(policy, reader, [0, 1], {'1': True}))
    assert [r['annotated_skill'] for r in rows] == ['Pick', 'Place']
    assert [r['event_id'] for r in rows] == ['e0', 'e1']
    assert [r['simulator_grasp'] for r in rows] == [None, True]
    assert rows[0]['receiver'] == rows[1]['current_entity'] == rows[1]['receiver']
    assert len(rows[1]['hold_error_px']) == 8
    regions = [entity_coverage(reader, key, 0, 32, 4) for key in ('milk', 'pudding')]
    assert all(r is not None for r in regions)
    pair = plan_pair(policy, reader.rgb('image', 0), ['pick up milk', 'pick up pudding'],
                     regions=regions, observed=np.zeros(8, bool), other_skill=2)
    conditions = pair['conditions']
    assert len(conditions) == 9
    assert conditions['task_a']['first_readout_delta_rms'] == 0.
    assert all(c['toward_b_px_on_observed_slots'] is None for c in conditions.values())
    assert conditions['change_expert']['skill_id'] == 2
    assert conditions['remove_specialists']['motion_delta_rms'] > 0.
    baseline = policy.features(reader.rgb('image', 0), 'pick up milk', attention_diagnostics=True)
    assert baseline['native_attention'].shape[-2] == int(baseline['text_mask'].sum())
    torch.testing.assert_close(baseline['current_map'], policy.infer(reader.rgb('image', 0), 'pick up milk')['current_map'])


def test_receiver_does_not_jump_across_unrelated_or_invalid_events(tmp_path):
    planner_dataset(tmp_path)
    reader = EpisodeReader(tmp_path / 'episode.json')
    pick = planning_event(reader, 0)
    assert receiver_event(reader, dict(pick, skill='Open')) is None
    place = planning_event(reader, 1)
    place['boundary_valid'] = False
    assert receiver_event(reader, pick) is None
