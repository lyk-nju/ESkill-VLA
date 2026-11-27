import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from eskill_bench.protocols.assets import Bundle, locked_paths, rgb
from eskill_bench.protocols.audit import verify_locked, visual_regression
from eskill_bench.protocols.motion import reproduce
from eskill_bench.protocols.commands import video
from eskill_bench.protocols.rollouts import outcomes
from eskill_bench.protocols.worker import boundary_response


@pytest.mark.parametrize(("command", "protocol"), [
    (["unseen-composition"], "unseen-composition"),
    (["libero"], "libero200"),
    (["libero", "--with-plus"], "sample"),
    (["libero-plus"], "plus280"),
    (["libero-plus", "--full"], "plus-full"),
])
def test_named_rollout_dispatch(monkeypatch, command, protocol):
    from eskill_bench import cli, protocols
    from eskill_bench.protocols import rollouts

    monkeypatch.setattr(sys, "argv", ["benchmark", "--config", "local.json", *command, "report", "run01"])
    monkeypatch.setattr(protocols, "Settings", lambda path: path)
    calls = []
    monkeypatch.setattr(rollouts, "rollout", lambda settings, args: calls.append((args.protocol, args.phase, args.args)))
    cli.main()
    assert calls == [(protocol, "report", ["run01"])]


def test_relocation_only_changes_asset_fields(tmp_path):
    doc = {
        "records": [
            dict(
                id="/old/id",
                image="rgb/case.npz",
                frame=10,
                current_bbox=[1, 2, 3, 4],
                prompt="/old/rgb/case.npz",
                rgb_sha256="abc",
            )
        ]
    }
    source = tmp_path / "sealed.json"
    source.write_text(json.dumps(doc))
    b = SimpleNamespace(file=lambda key: source, files={"m": {"source_path": "/old/run/manifest.json"}})
    s = SimpleNamespace(bundle=b, relocate=lambda p: tmp_path / p.relative_to("/old"))
    new, paths = locked_paths(s, "m")
    row = new["records"][0]
    assert len(paths) == 1 and row.pop("image") == str(tmp_path / "run/rgb/case.npz")
    assert row == {k: v for k, v in doc["records"][0].items() if k != "image"}
    assert json.loads(source.read_text()) == doc

    row = doc["records"][0]
    row["rgb_path"] = "/old/other/rgb/case.npz"
    source.write_text(json.dumps(doc))
    relocated, _ = locked_paths(s, "m")
    assert (
        relocated["records"][0]["image"]
        == relocated["records"][0]["rgb_path"]
        == str(tmp_path / "other/rgb/case.npz")
    )


def test_pixel_checksum_is_over_decoded_rgb(tmp_path):
    import hashlib

    a = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    expected = hashlib.sha256(a.tobytes()).hexdigest()
    np.savez_compressed(tmp_path / "a.npz", main=a)
    np.testing.assert_equal(rgb(tmp_path / "a.npz", expected=expected), a)
    with pytest.raises(ValueError, match="hash"):
        rgb(tmp_path / "a.npz", expected="bad")


def test_boundary_swap_only_end_slot_and_fixed_route():
    import torch

    class Policy:
        def read_plan(self, features, **kwargs):
            if not kwargs:
                return dict(
                    start=torch.zeros(2, 2),
                    end=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                    boundary_latents=torch.arange(4).reshape(2, 2, 1),
                    skill_id=torch.tensor([4, 5]),
                )
            assert kwargs["end_latent"].tolist() == [[[3]]] and kwargs["skill_id"].tolist() == [4]
            assert list(features["x"]) == [9]
            return dict(trace=torch.ones(1, 8, 2))

    row = dict(
        id="x",
        kind="switch_pick",
        suite="goal",
        boxes=[[0, 0, 0.1, 0.1], [0.9, 0, 1, 0.1]],
        a={"trace": [[0, 0]] * 8},
        interventions={"prior_only": {}},
    )
    result = boundary_response(Policy(), {"x": torch.tensor([9, 8])}, row, 224)
    assert result["end_swap_trace_toward_target_px"] == 224
    assert result["end_toward_target_px"] == 224


def test_rollout_retries_and_incomplete_denominators(tmp_path):
    jobs = [dict(benchmark="libero", suite="goal", task_ids=[1], trials_per_task=2)]

    def record(attempt, trial, success, error=None):
        p = tmp_path / f"batch_00000/attempt_{attempt:04}/records/{trial}.json"
        p.parent.mkdir(exist_ok=True, parents=True)
        p.write_text(
            json.dumps(
                dict(
                    suite="goal",
                    task_id=1,
                    trial=trial,
                    initial_state_id=trial,
                    seed=7,
                    success=success,
                    error=error,
                )
            )
        )

    record(0, 0, False, "reset failed")
    assert outcomes(tmp_path, jobs)[1]["libero"]["success_rate"] is None
    record(1, 0, True)
    record(1, 1, False)
    assert outcomes(tmp_path, jobs)[1]["libero"]["success_rate"] == 0.5
    record(2, 0, False)
    with pytest.raises(ValueError, match="Conflicting"):
        outcomes(tmp_path, jobs)


def test_full_plus_uses_single_continuous_attempt(tmp_path):
    jobs = [dict(benchmark="libero_plus", suite="goal", task_ids=[0, 1], trials_per_task=1)]
    plan = dict(protocol="plus-full", checkpoint=str(tmp_path / "model.pt"))
    for i in range(2):
        attempt = tmp_path / "full" / f"attempt_{i:04}"
        records = attempt / "records"
        records.mkdir(parents=True)
        (attempt / "config.json").write_text(
            json.dumps(dict(server_metadata=dict(checkpoint=plan["checkpoint"])))
        )
        row = dict(suite="goal", task_id=i, trial=0, initial_state_id=0, seed=7, success=True, error=None)
        (records / "row.json").write_text(json.dumps(row))
    _, result = outcomes(tmp_path, jobs, plan)
    assert result["libero_plus"]["completed"] == result["libero_plus"]["unfinished"] == 1
    assert result["libero_plus"]["success_rate"] is None


def test_static_video_default_duration_and_codec(tmp_path):
    import imageio_ffmpeg
    import subprocess

    files = []
    for i in range(2):
        p = tmp_path / f"{i}.png"
        Image.new("RGB", (31, 29), (i * 255, 0, 0)).save(p)
        files.append(p)
    out = tmp_path / "test.mp4"
    video(out, files, 0.75)
    info = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(out), "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr
    assert "h264" in info and "yuv420p" in info and "00:00:01.50" in info


def test_sealed_november_regression():
    path = os.environ.get("ESKILL_EVAL_BUNDLE")
    if not path:
        pytest.skip("Set ESKILL_EVAL_BUNDLE to the sealed extraction")
    b = Bundle(path)
    assert verify_locked(b)["visual_unique_rgb"] == 351
    assert reproduce(b)["totals"]["both_region_prefer"] == [140, 118, 134]
    visual = visual_regression(b)
    assert visual["stage1_task_totals"]["10000"]["vae_gac"]["current_hits"] == 234
    assert visual["e0_formal_by_unit"] == dict(planned=1307, valid=1307, hits=509)


@pytest.mark.parametrize('protocol', ['libero200', 'plus280', 'plus-full'])
def test_rollout_worker_uses_local_tokenizer_and_render_device(tmp_path, monkeypatch, protocol):
    from eskill_bench.protocols import rollouts

    root = tmp_path / 'run'
    root.mkdir()
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'checkpoint')
    bench = 'libero' if protocol == 'libero200' else 'libero_plus'
    plan = dict(protocol=protocol, checkpoint=str(checkpoint), checkpoint_sha256=rollouts.digest(checkpoint),
                jobs=[dict(benchmark=bench, suite='libero_goal', task_ids=[0], trials_per_task=1)])
    (root / 'protocol.json').write_text(json.dumps(plan))
    paths = {k: tmp_path / k for k in ('composition', 'tokenizer', 'openpi', 'policy_source', 'python',
                                         'sim_python', 'plus_python', 'sim_configs')}
    bundle = SimpleNamespace(root=tmp_path, file=lambda key: tmp_path / key, verify=lambda: None)
    settings = SimpleNamespace(paths=dict(paths, output=tmp_path), bundle=bundle, device='cuda:1')
    calls = []
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '2,3')
    monkeypatch.setattr(rollouts.subprocess, 'run', lambda cmd, **kwargs: calls.append((cmd, kwargs)))
    rollouts.rollout(settings, SimpleNamespace(protocol=protocol, phase='serve',
                                              args=['run', '--checkpoint-tag', str(checkpoint)]))
    serve, _ = calls.pop()
    assert serve[serve.index('--tokenizer-path') + 1] == str(paths['tokenizer'])
    assert serve[serve.index('--device') + 1] == 'cuda:1'
    assert ('--reset-per-episode' in serve) == (protocol != 'plus-full')
    rollouts.rollout(settings, SimpleNamespace(protocol=protocol, phase='run', args=['run']))
    client, kwargs = calls.pop()
    assert client[client.index('--render-gpu-device-id') + 1] == '3'
    assert kwargs['env']['MUJOCO_EGL_DEVICE_ID'] == kwargs['env']['CUDA_VISIBLE_DEVICES'] == '3'


def test_sample_policy_resets_only_on_marked_episode_request():
    from scripts.serve_policy import EpisodeSeededPolicy

    seeds = []
    policy = EpisodeSeededPolicy(SimpleNamespace(infer=dict), seeds.append)
    first = {'evaluation/episode_seed': 7, 'prompt': 'pick'}
    assert policy.infer(first) == policy.infer({'prompt': 'pick'}) == {'prompt': 'pick'}
    assert policy.infer(first) == {'prompt': 'pick'}
    assert seeds == [7, 7]
    with pytest.raises(ValueError):
        policy.infer({'evaluation/episode_seed': 8})
