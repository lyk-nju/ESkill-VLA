"""Attention capture leaves sampling unchanged; the viewer renders every shared source frame."""
import json

import av
import numpy as np
import pytest
torch = pytest.importorskip("torch")
sentencepiece = pytest.importorskip("sentencepiece")

import openpi.models.pi0 as pi0
from openpi.models.pi0_config import ESkillConfig, Pi0Config, SKILLS
from openpi.training.config import TrainConfig, WandbConfig, create_model
from scripts.analysis.attention_viewer import AttentionCapture, box_mass, render
from scripts.attention_viewer import main
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from test_training_data import saved_episode


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp("viewer")
    (root / "corpus.txt").write_text("\n".join(t for _, t in SKILLS) + "\npick cup then place it")
    sentencepiece.SentencePieceTrainer.train(input=str(root / "corpus.txt"), model_prefix=str(root / "tokenizer"),
                                             vocab_size=64, hard_vocab_limit=False, minloglevel=2)
    write_index(root / "index.json", [saved_episode(root)])
    paths = {}
    for model_type in ("pi05", "eskill"):
        config = TrainConfig(str(root / "index.json"), str(root / "tokenizer.model"), str(root / model_type),
                             str(root / "stats.json"), model=(ESkillConfig if model_type == 'eskill' else Pi0Config).tiny(vocab_size=128),
                             model_type=model_type, validation_fraction=0, precision="float32", device="cpu",
                             smoke=True, wandb_info=WandbConfig(mode="disabled"))
        stats = compute(config)
        torch.manual_seed(0)
        model = create_model(config, training=False)
        paths[model_type] = root / f"{model_type}.pt"
        torch.save(dict(model=model.state_dict(), config=config.to_dict(), norm_stats=stats,
                        weights_kind="full"), paths[model_type])
    return root, paths


def test_capture_matches_sampling_and_restores_attend(checkpoints):
    from openpi.policies.policy_config import create_trained_policy
    _, paths = checkpoints
    policy = create_trained_policy(paths["eskill"])
    observation = dict(images=[np.random.default_rng(0).integers(0, 255, (32, 32, 3), np.uint8)],
                       state=np.arange(8, dtype=np.float32) / 10, prompt="pick cup",
                       effector_start=np.array([[.5, .5], [0., 0.]], np.float32),
                       effector_start_valid=np.array([True, False]))
    noise = np.random.default_rng(1).standard_normal((policy.model.config.action_horizon, 32)).astype(np.float32)
    expected = policy.infer(observation, noise=noise, steps=2)["actions"]
    original = pi0.attend
    capture = AttentionCapture(policy.model, [0, 1])
    capture.image_keys = 16
    with capture.active():
        actions = policy.infer(observation, noise=noise, steps=2)["actions"]
        maps = capture.result()
    assert pi0.attend is original and "velocity" not in vars(policy.model)
    np.testing.assert_array_equal(actions, expected)
    assert maps.shape == (16,) and 0 < maps.sum() <= 1 + 1e-6


def test_box_mass_uses_model_input_pixels():
    attention = np.zeros((4, 4))
    attention[3, 3] = 1
    assert box_mass(attention, [(24, 24, 32, 32)], 32) > .5
    assert box_mass(attention, [(0, 0, 8, 8)], 32) == 0
    assert box_mass(attention, [], 32) is None


def test_viewer_renders_one_row_per_model(checkpoints, tmp_path):
    root, paths = checkpoints
    video = tmp_path / "attention.mp4"
    main(["--model", f"sft={paths['pi05']}", "--model", f"eskill={paths['eskill']}", "--episode", "one",
          "--video", str(video), "--device", "cpu", "--layers", "1", "--steps", "2"])
    with av.open(str(video)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    # Actions 0..5 observe frames 0, 2 and 3.
    assert len(frames) == 3
    assert frames[0].shape[0] > 2 * 320
    with pytest.raises(FileExistsError):
        render([("sft", paths["pi05"])], "one", video, device="cpu")
    with pytest.raises(ValueError, match="not in"):
        render([("sft", paths["pi05"])], "missing", tmp_path / "missing.mp4", device="cpu")
