"""Exercise the documented CLI with a model implementation absent from unified."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from PIL import Image
import pytest

from scripts.benchmark import benchmark


def test_model_checkout_supplies_predictions_and_unified_scores(tmp_path):
    unified = Path(__file__).resolve().parents[1]
    model = tmp_path / 'other-model'
    scripts = model / 'scripts'
    scripts.mkdir(parents=True)
    shutil.copy(unified / 'scripts/benchmark.py', scripts)
    policies = model / 'src/openpi/policies'
    policies.mkdir(parents=True)
    (policies.parent / '__init__.py').touch()
    (policies / 'current_grounding.py').write_text("""
from types import SimpleNamespace
import torch

def create_current_grounding_policy(checkpoint, **kwargs):
    assert checkpoint == 'branch-only'
    cfg = SimpleNamespace(image_size=32, patch_size=8, attention_layers=[], subject_bbox=True)
    def predict(views, task):
        assert len(views) == 1 and task == 'pick milk'
        box = torch.tensor([[0., 0., .5, .5]])
        return dict(bbox_xyxy=box, subject_bbox_xyxy=box)
    return SimpleNamespace(model=SimpleNamespace(config=cfg, training_stage='grounding'),
                           infer=predict, precision='float32')
""")
    Image.new('RGB', (32, 32)).save(tmp_path / 'main.png')
    manifest = tmp_path / 'cases.json'
    manifest.write_text(json.dumps(dict(protocol='fixture', metric_space='source', rows=[
        dict(id='case-b', split='Check', parent='ep', scene='table', image='main.png', task='pick milk',
             target='milk', boxes={'milk': [0, 0, 16, 16]}, subject_box=[0, 0, 16, 16])])))
    output = tmp_path / 'result'
    # The benchmark source and the model checkout deliberately have unrelated paths.
    env = dict(os.environ, PYTHONPATH=str(unified / 'src'), ESKILL_BENCHMARK_ROOT=str(unified))
    subprocess.run([sys.executable, str(scripts / 'benchmark.py'), 'visual', '--manifest', str(manifest),
                    '--checkpoint', 'branch-only', '--output', str(output), '--panels', '0'],
                   cwd=tmp_path, env=env, check=True)
    report = json.loads((output / 'metrics.json').read_text())
    assert report['rows'][0]['id'] == 'case-b'
    assert report['rows'][0]['current']['center_error_px'] == 0
    assert report['readout'] == 'regressed_bbox'
    commit = subprocess.check_output(['git', '-C', str(unified), 'rev-parse', 'HEAD'], text=True).strip()
    assert report['revisions']['benchmark']['commit'] == commit
    assert report['revisions']['model']['commit'] is None


def test_paired_reports_identify_incompatible_benchmark_commits():
    compare = benchmark('compare').compare
    base = dict(revisions={'benchmark': {'commit': 'a'}}, metric_space='source', input_sha256='locked')
    other = dict(base, revisions={'benchmark': {'commit': 'b'}})
    with pytest.raises(ValueError, match='same benchmark commit'):
        compare({'a': base, 'b': other}, 'a')


def test_native_planner_labels_are_fixed_across_training_recipes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    planner = benchmark('planner')
    index = tmp_path / 'index.json'
    index.write_text(json.dumps({'episodes': [{'episode_id': 'a', 'path': 'episode.json'}]}))
    manifest = tmp_path / 'frames.json'
    frames = [{'episode_id': 'a', 'frame': 4}]
    manifest.write_text(json.dumps({'rows': frames}))
    monkeypatch.setattr(planner, 'EpisodeReader', lambda path: object())

    def rows(policy, reader, frames, grasp, *, protocol):
        assert protocol['terminal_hold'] is True and protocol['min_horizon_frames'] == 10
        yield dict(frame=frames[0], event_id='pick', annotated_skill=None,
                   observed_error_px=[], hold_error_px=[], observed_error_by_slot_px=[None] * 8)

    monkeypatch.setattr(planner, 'episode_rows', rows)
    cfg = SimpleNamespace(motion_offsets_frames=list(range(5, 41, 5)), planner_has_boundaries=False,
                          image_size=224, planner_version='test', motion_terminal_hold=False, motion_min_horizon_frames=0)
    policy = SimpleNamespace(model=SimpleNamespace(config=cfg))
    report = planner.native_report(policy, index, manifest)
    assert report['rows'][0]['episode_id'] == 'a' and report['frames'] == 1
    manifest.write_text(json.dumps({'rows': frames, 'motion_protocol': dict(report['motion_protocol'], offsets_frames=[5])}))
    with pytest.raises(ValueError, match='offsets'):
        planner.native_report(policy, index, manifest)
