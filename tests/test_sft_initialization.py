"""SFT -> new auxiliary stage -> independent resume and inference."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")
sentencepiece = pytest.importorskip("sentencepiece")
from diffusers import AutoencoderKL
from safetensors.torch import save_file

from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import ESkillConfig, SKILLS
from openpi.policies.policy_config import create_trained_policy
from openpi.training.checkpoints import load_sft_weights
from openpi.training.config import TrainConfig, WandbConfig, create_model, make_split
from scripts.compute_norm_stats import compute
from scripts.preprocess.export import write_index
from scripts.train import train
from test_training_data import saved_episode


@pytest.mark.parametrize("rank", [0, 2])
@pytest.mark.parametrize("use_arm_state", [True, False])
def test_sft_weights_transfer_but_training_state_starts_fresh(tmp_path, rank, use_arm_state):
    torch.set_num_threads(2)
    cfg = ESkillConfig.tiny(vocab_size=128, use_arm_state=use_arm_state)
    base = Pi0(cfg)
    save_file({k: v.clone() for k, v in base.state_dict().items()}, str(tmp_path / "base.safetensors"))
    vae = AutoencoderKL(down_block_types=("DownEncoderBlock2D",) * 2,
                        up_block_types=("UpDecoderBlock2D",) * 2, block_out_channels=(16, 32),
                        layers_per_block=1, norm_num_groups=8, latent_channels=4)
    vae.save_pretrained(tmp_path / "vae")
    (tmp_path / "corpus.txt").write_text("\n".join(t for _, t in SKILLS) + "\npick cup then place it")
    sentencepiece.SentencePieceTrainer.train(input=str(tmp_path / "corpus.txt"),
        model_prefix=str(tmp_path / "tokenizer"), vocab_size=64, hard_vocab_limit=False, minloglevel=2)
    write_index(tmp_path / "index.json", [saved_episode(tmp_path), saved_episode(tmp_path, identity='two')])
    split_path = tmp_path / 'split.json'
    split_path.write_text(json.dumps(make_split(tmp_path / 'index.json', .5)))
    source = TrainConfig(str(tmp_path / "index.json"), str(tmp_path / "tokenizer.model"),
        str(tmp_path / "sft"), str(tmp_path / "stats.json"), model=cfg,
        pretrained_path=str(tmp_path / "base.safetensors"), lora_rank=rank, lora_alpha=2,
        model_type="pi05", batch_size=2, num_workers=0, num_steps=2, save_every=2,
        validation_fraction=.5, split_path=str(split_path), val_interval=1, val_max_batches=1,
        precision="float32", device="cpu", wandb_info=WandbConfig(mode="disabled"))
    (tmp_path / "stats.json").write_text(json.dumps(compute(source)))
    sft_path = train(source)
    saved = torch.load(sft_path, weights_only=False)
    auxiliary = replace(source, model_type="eskill", output_dir=str(tmp_path / "auxiliary"),
                        vae_path=str(tmp_path / "vae"), init_checkpoint=str(sft_path))
    inherited = compute(auxiliary)
    assert inherited["actions"] == saved["norm_stats"]["actions"]
    assert inherited["state"] == saved["norm_stats"]["state"]
    assert inherited["arm_state"]["method"] == "standard"
    initialized = create_model(auxiliary, init_checkpoint=sft_path)
    assert (initialized.state_projection is not None) == use_arm_state
    for key, value in saved["model"].items():
        torch.testing.assert_close(initialized.state_dict()[key], value, rtol=0, atol=0)
    assert torch.count_nonzero(initialized.step_projection.weight) == 0
    assert all(not p.requires_grad for p in initialized.visual_supervision.vae.parameters())
    with pytest.raises(ValueError, match='same fixed split_path'):
        load_sft_weights(initialized, sft_path, replace(auxiliary, split_path=None))
    legacy = dict(saved, config={**saved['config'], 'validation_fraction': 0, 'split_path': None})
    legacy_path = tmp_path / 'legacy_all_data.pt'
    torch.save(legacy, legacy_path)
    with pytest.raises(ValueError, match='same fixed split_path'):
        compute(replace(auxiliary, init_checkpoint=str(legacy_path)))
    with pytest.raises(ValueError, match="lora_alpha"):
        load_sft_weights(initialized, sft_path, replace(auxiliary, lora_alpha=4))
    damaged = dict(saved, model=dict(saved["model"]))
    damaged["model"].pop(next(iter(damaged["model"])))
    torch.save(damaged, tmp_path / "incomplete.pt")
    with pytest.raises(ValueError, match="Incomplete SFT"):
        load_sft_weights(initialized, tmp_path / "incomplete.pt", auxiliary)
    stats = json.loads(Path(source.norm_stats).read_text())
    stats["actions"]["mean"][0] += 1
    (tmp_path / "different_stats.json").write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="normalization: actions"):
        load_sft_weights(initialized, sft_path, replace(auxiliary, norm_stats=str(tmp_path / "different_stats.json")))

    path = train(auxiliary)
    records = [json.loads(line) for line in (path.parent / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [1, 2]
    assert all(r["resumed_from_step"] == 0 for r in records)
    assert all(np.isfinite(r[k]) for r in records for k in ("action", "skill", "future", "grounding", "trace"))
    trained = torch.load(path, weights_only=False)
    assert torch.count_nonzero(trained["model"]["step_projection.weight"]) > 0
    assert all(int(s["step"]) == 2 for s in trained["optimizer"]["state"].values())
    # Auxiliary checkpoints stand alone: resume/inference no longer read the SFT checkpoint.
    sft_path.rename(sft_path.with_suffix(".unavailable"))
    auxiliary.num_steps = 3
    train(auxiliary, resume=path)
    resumed = torch.load(path, weights_only=False)
    assert resumed["step"] == 3
    assert all(int(s["step"]) == 3 for s in resumed["optimizer"]["state"].values())
    policy = create_trained_policy(path)
    state = np.arange(15, dtype=np.float32) / 10
    result = policy.infer(dict(image=np.zeros((32, 32, 3), np.uint8),
                              state=state if use_arm_state else state[-8:], prompt="pick cup"), steps=2)
    assert result["actions"].shape == (cfg.action_horizon, 7)
    assert np.isfinite(result["actions"]).all()
    assert policy.model.visual_supervision is None
