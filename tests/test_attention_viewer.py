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
from scripts.analysis.attention_viewer import AttentionCapture, box_mass, percent, render
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
    for model_type in ("pi05",):
        config = TrainConfig(str(root / "index.json"), str(root / "tokenizer.model"), str(root / model_type),
                             str(root / "stats.json"), model=(ESkillConfig.tiny(current_grounding='action_execution', vocab_size=128) if model_type == 'eskill' else Pi0Config.tiny(vocab_size=128)),
                             model_type=model_type, validation_fraction=0, precision="float32", device="cpu",
                             smoke=True, wandb_info=WandbConfig(mode="disabled"))
        stats = compute(config)
        torch.manual_seed(0)
        model = create_model(config, training=False)
        paths[model_type] = root / f"{model_type}.pt"
        torch.save(dict(model=model.state_dict(), config=config.to_dict(), norm_stats=stats,
                        weights_kind="full"), paths[model_type])
    return root, paths


def test_plain_pi05_has_zero_motion_mass_and_capture_restores_on_error(checkpoints):
    from openpi.policies.policy_config import create_trained_policy
    _, paths = checkpoints
    policy = create_trained_policy(paths['pi05'])
    capture = AttentionCapture(policy.model, [0, 1])
    observation = dict(images=[np.zeros((32, 32, 3), np.uint8)], state=np.zeros(8), prompt='pick cup')
    with pytest.raises(RuntimeError, match='test exit'):
        with capture.active():
            policy.infer(observation, noise=np.zeros((8, 32), np.float32), steps=1)
            assert capture.key_mass()['motion'] == 0
            assert capture.motion_attention() is None
            assert sum(capture.key_mass().values()) == pytest.approx(1., abs=1e-6)
            raise RuntimeError('test exit')
    assert all(name not in vars(policy.model) for name in ('velocity', 'predict_condition', 'encode_observation'))


def test_box_mass_uses_model_input_pixels():
    attention = np.zeros((4, 4))
    attention[3, 3] = 1
    assert box_mass(attention, [(24, 24, 32, 32)], 32) > .5
    assert box_mass(attention, [(0, 0, 8, 8)], 32) == 0
    assert box_mass(attention, [], 32) is None
    assert percent(.0003869) == '0.039%'
    assert percent(.26) == '26.0%'


def test_viewer_renders_one_row_per_model(checkpoints, tmp_path):
    root, paths = checkpoints
    video = tmp_path / "attention.mp4"
    main(["--model", f"sft={paths['pi05']}", "--model", f"second={paths['pi05']}", "--episode", "one",
          "--video", str(video), "--device", "cpu", "--layers", "1", "--steps", "2"])
    with av.open(str(video)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    # Actions 0..5 observe frames 0, 2 and 3.
    assert len(frames) == 3
    assert frames[0].shape[0] > 2 * 320
    report = json.loads(video.with_suffix('.json').read_text())
    assert report['models']['sft']['0']['key_mass']['motion'] == 0
    assert sum(report['models']['second']['0']['key_mass'].values()) == pytest.approx(1., abs=1e-6)
    with pytest.raises(FileExistsError):
        render([("sft", paths["pi05"])], "one", video, device="cpu")
    with pytest.raises(ValueError, match="not in"):
        render([("sft", paths["pi05"])], "missing", tmp_path / "missing.mp4", device="cpu")
