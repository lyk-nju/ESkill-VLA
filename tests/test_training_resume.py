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


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_matches_uninterrupted_training_and_policy_roundtrip(tmp_path, workers):
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
    write_index(index, [episode])
    config = TrainConfig(
        data_index=str(index), tokenizer_path=str(tmp_path / "tokenizer.model"),
        output_dir=str(tmp_path / "resumed"), norm_stats=str(tmp_path / "stats.json"),
        model=ESkillConfig.tiny(vocab_size=128), batch_size=2, num_workers=workers,
        num_steps=3, log_every=100, save_every=3, device="cpu", precision="float32",
        smoke=True, validation_fraction=0)
    stats = compute(config)
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    path = train(config)
    first = torch.load(path, weights_only=False)
    assert first["step"] == 3
    config.num_steps = 4
    train(config, resume=path)
    resumed = torch.load(path, weights_only=False)
    config.output_dir = str(tmp_path / "continuous")
    continuous_path = train(config)
    continuous = torch.load(continuous_path, weights_only=False)
    for key in continuous["model"]:
        torch.testing.assert_close(resumed["model"][key], continuous["model"][key], rtol=0, atol=0)
    policy = create_trained_policy(path)
    other = create_trained_policy(continuous_path)
    observation = dict(image=np.zeros((32, 32, 3), np.uint8),
                       state=np.arange(15, dtype=np.float32) / 10, prompt="pick cup")
    noise = np.zeros((config.model.action_horizon, config.model.action_dim), np.float32)
    result = policy.infer(observation, noise=noise, steps=2)
    np.testing.assert_array_equal(result["actions"], other.infer(observation, noise=noise, steps=2)["actions"])
    assert result["actions"].shape == (8, 7)
    assert np.isfinite(result["actions"]).all()
    assert (result["future_skills"][1] == -1).all()
    assert policy.model.visual_supervision is None
