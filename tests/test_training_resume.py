"""The real train/save/resume/infer entry points on a small stored episode."""
import json
from multiprocessing.connection import Listener

import numpy as np
import pytest
torch = pytest.importorskip("torch")
sentencepiece = pytest.importorskip("sentencepiece")
pytest.importorskip("diffusers")

from openpi.models.pi0_config import ESkillConfig, SKILLS
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import TrainConfig
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from scripts.train import train
from test_training_data import saved_episode


def test_resume_skips_indices_without_loading_consumed_samples():
    from torch.utils.data import DataLoader, DistributedSampler
    from openpi.training.data_loader import ResumableBatchSampler

    class Dataset:
        def __init__(self):
            self.reads = []

        def __len__(self):
            return 23

        def __getitem__(self, index):
            self.reads.append(index)
            return index

    for rank in (0, 1):
        dataset = Dataset()
        sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42)
        sampler.set_epoch(3)
        batches = ResumableBatchSampler(sampler, batch_size=5, drop_last=False)
        expected = list(batches)
        batches.start_batch = 2
        loader = DataLoader(dataset, batch_sampler=batches)
        assert len(loader) == 1
        assert [row.tolist() for row in loader] == expected[2:]
        assert dataset.reads == expected[2]  # No reads for skipped batches, including DDP padding.
        batches.start_batch = 0
        assert list(batches) == expected


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("stop", [2, 3], ids=["within_epoch", "epoch_boundary"])
@pytest.mark.parametrize("full", [False, True], ids=["frozen_vlm", "full_amp"])
def test_resume_matches_uninterrupted_training_and_policy_roundtrip(tmp_path, workers, stop, full):
    if workers:
        try:
            with Listener(str(tmp_path / "ipc.sock"), family="AF_UNIX"):
                pass
        except PermissionError:
            pytest.skip("Multi-worker tensor transfer requires local IPC socket permission")
    torch.set_num_threads(2)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("\n".join(text for _, text in SKILLS) + "\nTask: pick cup then place it State: 0 1 2 3 4 5 6 7 8 9 Action:\n")
    sentencepiece.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(tmp_path / "tokenizer"), vocab_size=64,
        hard_vocab_limit=False, minloglevel=2)
    episode = saved_episode(tmp_path)
    index = tmp_path / "index.json"
    write_index(index, [episode, saved_episode(tmp_path, identity='two')])
    config = TrainConfig(
        data_index=str(index), tokenizer_path=str(tmp_path / "tokenizer.model"),
        output_dir=str(tmp_path / "resumed"), norm_stats=str(tmp_path / "stats.json"),
        model=ESkillConfig.tiny(vocab_size=128, freeze_vlm=not full, gradient_checkpointing=full),
        batch_size=2, num_workers=workers, skill_sampling=bool(workers),
        num_steps=stop, log_every=100, save_every=3, device="cpu",
        precision="bfloat16" if full else "float32",
        smoke=True, validation_fraction=.5, val_interval=2, val_max_batches=2)
    stats = compute(config)
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    path = train(config)
    first = torch.load(path, weights_only=False)
    assert first["step"] == stop
    validation = [json.loads(row) for row in (path.parent / 'validation.jsonl').read_text().splitlines()]
    assert [row['step'] for row in validation] == sorted({2, stop})
    assert all('val_current_object_loss' in row and 'val_future_object_loss' in row and 'val_skill_loss' in row for row in validation)
    config.num_steps = 7  # Cross epoch boundaries after resuming partway through an epoch.
    train(config, resume=path)
    resumed = torch.load(path, weights_only=False)
    config.output_dir = str(tmp_path / "continuous")
    continuous_path = train(config)
    continuous = torch.load(continuous_path, weights_only=False)
    resumed_records = [json.loads(row) for row in (path.parent / 'metrics.jsonl').read_text().splitlines()]
    continuous_records = [json.loads(row) for row in (continuous_path.parent / 'metrics.jsonl').read_text().splitlines()]
    assert [r['samples'] for r in resumed_records] == [r['samples'] for r in continuous_records]
    for key in continuous["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    policy = create_trained_policy(path)
    assert policy.precision == config.precision
    other = create_trained_policy(continuous_path)
    observation = dict(images=[np.zeros((32, 32, 3), np.uint8)],
                       state=np.arange(8, dtype=np.float32) / 10, prompt="pick cup")
    noise = np.zeros((config.model.action_horizon, config.model.action_dim), np.float32)
    result = policy.infer(observation, noise=noise, steps=2)
    np.testing.assert_array_equal(result["actions"], other.infer(observation, noise=noise, steps=2)["actions"])
    assert result["actions"].shape == (config.model.action_horizon, 7)
    assert np.isfinite(result["actions"]).all()
    assert result["current_skills"][1] == -1
    assert policy.model.visual_supervision is None
