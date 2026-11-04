"""Config-only W&B setup, rank ownership, resume, credential redaction and lifecycle."""
import json
import sys
from unittest import mock

import pytest
torch = pytest.importorskip("torch")

from openpi.models.pi0_config import ESkillConfig
from openpi.training.config import TrainConfig, WandbConfig
from scripts.train import init_wandb, train, wandb_metrics
from openpi.training.metrics import phase_metrics


SECRET = "test-only-key-that-must-not-be-serialized"


def test_dashboard_groups_skill_losses_and_keeps_only_requested_fields():
    record = dict(loss=42., action=3., grounding=5., skill=4., future=6., trace=7.,
                  learning_rate=1e-5, grad_norm=2., seconds=.5, step=10,
                  grounding_ee=1., grounding_obj=2., grounding_goal=2., world_size=4,
                  global_batch_size=8, resumed_from_step=0, smoke=False)
    before = dict(record)
    result = wandb_metrics(record)
    assert {key for key in result if key.startswith('train/')} == {
        'train/loss', 'train/action_loss', 'train/grounding_loss', 'train/skill_loss', 'train/traj_loss'}
    assert {key for key in result if key.startswith('chart/')} == {
        'chart/learning_rate', 'chart/grad_norm', 'chart/time_per_step', 'chart/step'}
    assert all(key.startswith(('train/', 'aux_train/', 'chart/')) for key in result)
    assert result['train/skill_loss'] == 10. and result['train/loss'] == 42.
    assert result['train/action_loss'] == 3. and result['train/grounding_loss'] == 5. and result['train/traj_loss'] == 7.
    assert result['aux_train/grounding_ee'] == 1.
    assert 'aux_train/current_skill_loss' not in result and 'aux_train/future_skill_loss' not in result
    assert result['chart/time_per_step'] == .5
    assert record == before


def test_full_diagnostics_are_filtered_to_six_dashboard_summaries_without_mutation():
    from openpi.models.diagnostics import prediction_statistics
    from openpi.training.metrics import reduce_diagnostics
    from test_diagnostics import example
    cfg, inputs, prediction, targets, trace = example()
    auxiliary = reduce_diagnostics(prediction_statistics(prediction, trace, inputs, targets, cfg))
    for role in ('ee', 'obj', 'goal'):
        auxiliary.update({f'grounding_{role}': 1., f'grounding_{role}_count': 2,
                          f'grounding_{role}_fg': 3., f'grounding_{role}_bg': 4.})
    original = dict(auxiliary)
    losses = dict(loss=10., action=1., grounding=3., skill=1., future=1., trace=1.)
    expected = {'grounding_ee', 'grounding_obj', 'grounding_goal', 'current_skill_accuracy',
                'future_skill_accuracy', 'motion_l2_px'}
    for phase in ('train', 'val'):
        metrics = phase_metrics(phase, losses, auxiliary)
        assert len(metrics) == 11
        assert {key.split('/')[1] for key in metrics if key.startswith('aux_' + phase + '/')} == expected
        assert len([key for key in metrics if key.startswith(phase + '/')]) == 5
        assert metrics['aux_' + phase + '/motion_l2_px'] == pytest.approx(35.)
        assert not any('_pred_' in key or '_offset_' in key or key.endswith('_count') for key in metrics)
    assert auxiliary == original  # The original detailed local record survives.
    auxiliary['motion_count'] = 0
    assert 'aux_train/motion_l2_px' not in phase_metrics('train', losses, auxiliary)


@pytest.fixture(autouse=True)
def shared_config(tmp_path, monkeypatch):
    path = tmp_path / "wandb.json"
    path.write_text(json.dumps(dict(key=SECRET, entity="shared-team", project="shared-project", mode="online")))
    monkeypatch.setattr("openpi.training.config.WANDB_CONFIG_PATH", path)
    return path


@pytest.mark.parametrize("suffix", [".json", ".yaml", ".yml"])
def test_config_loads_wandb_and_redacts_only_saved_credentials(tmp_path, suffix):
    raw = dict(data_index="index", tokenizer_path="tokenizer", output_dir="output", norm_stats="stats",
               model={"action_horizon": 10},
               wandb_info=dict(key=SECRET, entity="test-team", project="test-project"))
    path = tmp_path / ("input" + suffix)
    if suffix == ".json":
        path.write_text(json.dumps(raw))
    else:
        yaml = pytest.importorskip("yaml")
        path.write_text(yaml.safe_dump(raw))
    config = TrainConfig.load(path)
    assert config.model.action_horizon == 10
    assert config.wandb_info.key == SECRET and config.wandb_info.mode == "online"
    assert SECRET not in repr(config)
    output = tmp_path / "config.json"
    config.save(output)
    assert SECRET not in output.read_text()
    assert TrainConfig.load(output).wandb_info.key == ""
    assert config.wandb_info.key == SECRET


@pytest.fixture
def sdk(monkeypatch):
    sdk = mock.MagicMock()
    sdk.init.side_effect = lambda **kw: mock.Mock(id=kw.get("id") or "new-run")
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    return sdk


@pytest.mark.parametrize("rank,info", [(0, None), (0, {"mode": "disabled"}), (1, {"mode": "online"})])
def test_disabled_or_other_rank_does_not_initialize_wandb(sdk, rank, info):
    config = TrainConfig("index", "tokenizer", "output", "stats", smoke=True, wandb_info=info)
    assert init_wandb(config, rank=rank) is None
    sdk.init.assert_not_called()


@pytest.mark.parametrize("model_type,lora_rank,name", [
    ("pi05", 16, "pi05_lora_bs128"), ("pi05", 0, "pi05_full_bs128"),
    ("eskill", 0, "pi05_eskill_bs128"), ("eskill", 16, "pi05_eskill_lora_bs128"),
])
def test_shared_config_and_automatic_global_batch_name(sdk, shared_config, model_type, lora_rank, name):
    before = shared_config.read_text()
    config = TrainConfig("index", "tokenizer", "output", "stats", model_type=model_type,
                         lora_rank=lora_rank, batch_size=32)
    run = init_wandb(config, rank=0, world_size=4)
    kwargs = sdk.init.call_args.kwargs
    assert (kwargs["entity"], kwargs["project"], kwargs["name"]) == ("shared-team", "shared-project", name)
    assert kwargs["config"]["global_batch_size"] == 128 and kwargs["config"]["world_size"] == 4
    assert kwargs["config"]["wandb_info"]["key"] == ""
    assert "bs128" in kwargs["tags"] and model_type in kwargs["tags"]
    assert sdk.Settings.call_args.kwargs["api_key"] == SECRET
    run.define_metric.assert_any_call('chart/step')
    for namespace in ('train', 'aux_train', 'val', 'aux_val'):
        run.define_metric.assert_any_call(namespace + '/*', step_metric='chart/step')
    assert shared_config.read_text() == before  # Never write run IDs/names to the shared file.


def test_shared_config_run_name_override_and_resume(sdk):
    config = TrainConfig("index", "tokenizer", "output", "stats", model_type="pi05", lora_rank=16,
                         batch_size=32, run_name="pi05_lora_bs128_libero40_seed42")
    init_wandb(config, rank=0, world_size=4)
    assert sdk.init.call_args.kwargs["name"] == "pi05_lora_bs128_libero40_seed42"
    resumed = TrainConfig("index", "tokenizer", "output", "stats", model_type="pi05", lora_rank=16, batch_size=32)
    init_wandb(resumed, rank=0, world_size=4, resume_config=config.to_dict())
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["resume"] == "allow" and kwargs["id"] == "new-run"
    assert kwargs["name"] == "pi05_lora_bs128_libero40_seed42"


def test_nonzero_rank_does_not_read_shared_credentials(sdk, shared_config):
    shared_config.unlink()
    config = TrainConfig("index", "tokenizer", "output", "stats")
    assert init_wandb(config, rank=1, world_size=4) is None
    sdk.init.assert_not_called()


@pytest.mark.parametrize("mode,entity,expected_id", [
    ("online", "team", "previous-run"), ("online", "other-team", None), ("offline", "team", None),
])
def test_resume_scopes_run_identity_and_keeps_key_out_of_uploaded_config(sdk, mode, entity, expected_id):
    config = TrainConfig("index", "tokenizer", "output", "stats",
                         wandb_info=dict(key=SECRET, mode=mode, entity=entity, project="project"))
    previous = {"wandb_info": dict(id="previous-run", entity="team", project="project")}
    run = init_wandb(config, rank=0, resume_config=previous)
    kwargs = sdk.init.call_args.kwargs
    assert kwargs["id"] == expected_id
    assert kwargs["resume"] == ("allow" if expected_id else None)
    assert SECRET not in json.dumps(kwargs["config"])
    settings = sdk.Settings.call_args.kwargs
    assert settings["api_key"] == (SECRET if mode == "online" else None)
    assert settings["disable_git"] and not settings["save_code"]
    assert config.wandb_info.id == run.id


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, inputs, targets):
        return {"action": ((self.weight * inputs["state"] - targets["actions"]) ** 2).mean()}


@pytest.fixture
def training_config(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setattr("scripts.train.create_model", lambda *a, **kw: TinyModel())
    sample = dict(inputs={"state": torch.ones(1, 2)}, targets={"actions": torch.zeros(1, 2)},
                  metadata=[dict(episode_id="test", action_row=0)])
    from openpi.training.data_loader import ResumableBatchSampler
    sampler = torch.utils.data.RandomSampler([sample], generator=torch.Generator())
    loader = torch.utils.data.DataLoader(
        [sample], batch_sampler=ResumableBatchSampler(sampler, 1, drop_last=False),
        collate_fn=lambda rows: rows[0], generator=torch.Generator())
    monkeypatch.setattr("scripts.train.create_data_loader", lambda config: loader)
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    return TrainConfig("index", "tokenizer", str(tmp_path / "train"), str(stats),
                       model=ESkillConfig.tiny(), device="cpu", precision="float32", num_workers=0,
                       num_steps=3, log_every=2, save_every=3, smoke=True, loss_weights={"action": 1.},
                       validation_fraction=0,
                       wandb_info=WandbConfig(key=SECRET, project="test-project", entity="test-team"))


def test_training_logs_sanitizes_checkpoint_and_resumes_same_run(training_config, sdk):
    runs = []
    def start(**kwargs):
        run = mock.Mock(id=kwargs.get("id") or "new-run")
        runs.append(run)
        return run
    sdk.init.side_effect = start
    checkpoint = train(training_config)
    assert [call.kwargs["step"] for call in runs[0].log.call_args_list] == [1, 2, 3]
    metrics = runs[0].log.call_args.args[0]
    assert set(metrics) == {'train/loss', 'train/action_loss', 'chart/grad_norm', 'chart/learning_rate',
                            'chart/time_per_step', 'chart/step'}
    runs[0].finish.assert_called_once_with(exit_code=0)
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["config"]["wandb_info"]["key"] == ""
    assert saved["config"]["wandb_info"]["id"] == "new-run"
    assert SECRET.encode() not in checkpoint.read_bytes()
    assert SECRET not in (checkpoint.parent / "config.json").read_text()

    training_config.num_steps = 4
    training_config.wandb_info.id = None  # Restore identity from the checkpoint, not this Python object.
    train(training_config, resume=checkpoint)
    assert sdk.init.call_args.kwargs["resume"] == "allow"
    assert sdk.init.call_args.kwargs["id"] == "new-run"
    runs[1].finish.assert_called_once_with(exit_code=0)
    assert [call.kwargs["step"] for call in runs[1].log.call_args_list] == [4]


def test_training_failure_finishes_wandb(training_config, sdk, monkeypatch):
    run = mock.Mock(id="failed-run")
    sdk.init.side_effect = None
    sdk.init.return_value = run
    monkeypatch.setattr("scripts.train.create_data_loader", mock.Mock(side_effect=RuntimeError("loader failed")))
    with pytest.raises(RuntimeError, match="loader failed"):
        train(training_config)
    run.finish.assert_called_once_with(exit_code=1)


def test_train_and_validation_use_one_wandb_event_per_step(training_config, sdk, monkeypatch):
    import scripts.train as module
    loader = module.create_data_loader(training_config)
    sample = next(iter(loader))
    sample['targets']['action_valid'] = torch.ones_like(sample['targets']['actions'], dtype=torch.bool)
    monkeypatch.setattr(module, 'create_data_loader', lambda config, split='train': loader if split == 'train' else [sample])
    training_config.model_type = 'pi05'
    training_config.validation_fraction = .5
    training_config.val_interval = 2
    run = mock.Mock(id='val-run')
    sdk.init.side_effect = None
    sdk.init.return_value = run
    train(training_config)
    assert [call.kwargs['step'] for call in run.log.call_args_list] == [1, 2, 3]
    assert 'val/loss' not in run.log.call_args_list[0].args[0]
    assert {'val/loss', 'val/action_loss', 'chart/step'} <= run.log.call_args_list[1].args[0].keys()
    assert {'val/loss', 'val/action_loss', 'chart/step'} <= run.log.call_args_list[2].args[0].keys()


def test_real_wandb_offline_training(training_config, tmp_path, monkeypatch, shared_config):
    wandb = pytest.importorskip("wandb")
    for name in ("WANDB_CONFIG_DIR", "WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    monkeypatch.setenv("WANDB_INIT_TIMEOUT", "20")
    settings = json.loads(shared_config.read_text())
    settings["mode"] = "offline"
    shared_config.write_text(json.dumps(settings))
    training_config.wandb_info = None
    training_config.smoke = False
    checkpoint = train(training_config)
    assert wandb.run is None
    archives = list((checkpoint.parent / "wandb").glob("offline-run-*/run-*.wandb"))
    assert len(archives) == 1 and archives[0].stat().st_size > 0
    assert SECRET.encode() not in archives[0].read_bytes()
    saved = torch.load(checkpoint, weights_only=False)
    assert saved["config"]["wandb_info"]["id"] and saved["config"]["wandb_info"]["key"] == ""
