"""Per-task quotas, indivisible parents, cross-format reuse and training-only statistics."""
import json

import numpy as np
import pytest

from openpi.training.config import TrainConfig, make_split, split_episodes
from scripts.compute_norm_stats import compute


def index_file(tmp_path, counts=(20, 30)):
    entries = [dict(episode_id=f't{task}_p{parent}', task=f'task {task}',
                    source=dict(parent_trajectory_id=f't{task}_p{parent}'), start=0, frames=2)
               for task, count in enumerate(counts) for parent in range(count)]
    path = tmp_path / 'index.json'
    path.write_text(json.dumps(dict(episodes=entries)))
    return path, entries


def test_each_task_has_its_own_quota_and_order_does_not_change_membership(tmp_path):
    path, entries = index_file(tmp_path)
    first = make_split(path)
    assert [(task['train'], task['val']) for task in first['tasks']] == [(18, 2), (27, 3)]
    assert not set(first['train']) & set(first['val'])
    path.write_text(json.dumps(dict(episodes=entries[::-1])))
    second = make_split(path)
    assert (first['train'], first['val']) == (second['train'], second['val'])
    assert make_split(path, seed=43)['val'] != first['val']


def test_views_count_as_one_parent_and_single_trajectory_stays_in_training(tmp_path):
    path, entries = index_file(tmp_path, (10, 1))
    for entry in entries[:10]:
        entries.append({**entry, 'episode_id': entry['episode_id'] + '_other_view'})
    path.write_text(json.dumps(dict(episodes=entries)))
    manifest = make_split(path)
    assert manifest['tasks'] == [dict(task='task 0', total=10, train=9, val=1),
                                 dict(task='task 1', total=1, train=1, val=0)]
    result = split_episodes(path)
    for entry in entries[:10]:
        assert (entry['episode_id'] in result['val']) == (entry['episode_id'] + '_other_view' in result['val'])


def test_frozen_split_preserves_sft_assignments_for_event_format_and_subsets(tmp_path):
    path, entries = index_file(tmp_path)
    manifest = make_split(path)
    frozen = tmp_path / 'split.json'
    frozen.write_text(json.dumps(manifest))
    selected = entries[:8] + entries[-8:]
    event_entries = []
    for i, entry in enumerate(selected):
        episode = tmp_path / f'meta{i}.json'
        episode.write_text(json.dumps(dict(episode_id=f'view_{i}', source=entry['source'],
                                           instructions=[entry['task'].replace(' ', '_') + '  '])))
        event_entries.append(dict(episode_id=f'view_{i}', path=episode.name))
    subset = tmp_path / 'events.json'
    subset.write_text(json.dumps(dict(episodes=event_entries)))
    result = split_episodes(subset, split_path=frozen)
    for i, entry in enumerate(selected):
        assert (f'view_{i}' in result['val']) == (entry['episode_id'] in manifest['val'])
    assert split_episodes(path, seed=43, split_path=frozen) == split_episodes(path, seed=42, split_path=frozen)
    with pytest.raises(ValueError, match='fraction'):
        split_episodes(path, fraction=.2, split_path=frozen)


def test_conflicting_tasks_for_one_parent_are_rejected(tmp_path):
    path, entries = index_file(tmp_path, (2, 2))
    entries[-1]['source'] = entries[0]['source']
    path.write_text(json.dumps(dict(episodes=entries)))
    with pytest.raises(ValueError, match='conflicting task'):
        make_split(path)


def test_quantiles_exclude_all_validation_rows(tmp_path):
    from scripts.preprocess.export import write_index
    from test_training_data import saved_episode
    paths = [saved_episode(tmp_path, f'ep{i}') for i in range(10)]
    index = tmp_path / 'index.json'
    write_index(index, paths)
    manifest = make_split(index, .2)
    frozen = tmp_path / 'split.json'
    frozen.write_text(json.dumps(manifest))
    for path in paths:
        if path.parent.name in manifest['val']:
            with np.load(path.parent / 'arrays.npz') as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays['state'][:] = arrays['actions'][:] = 999.
            np.savez_compressed(path.parent / 'arrays.npz', **arrays)
    cfg = TrainConfig(str(index), 'unused', 'unused', 'unused', model_type='pi05',
                      validation_fraction=.2, split_path=str(frozen))
    stats = compute(cfg)
    assert manifest['val'] and stats['provenance']['frames'] == 6 * len(manifest['train'])
    assert set(stats['provenance']['episode_ids']) == set(manifest['train'])
    assert stats['state']['method'] == stats['actions']['method'] == 'quantile'
    assert max(stats['state']['q99']) < 999 and max(stats['actions']['q99']) < 999
    assert 'arm_state' not in stats
