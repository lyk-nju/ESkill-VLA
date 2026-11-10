"""Original π0.5 path: adapter-only updates and independent reload."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from safetensors.torch import save_file

from openpi.models.lora import apply_lora
from openpi.models.pi0 import Pi0
from openpi.models.pi0_config import Pi0Config
from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.training.checkpoints import load_weights, save_checkpoint
from openpi.training.config import TrainConfig, create_model
from test_eskill_model import batch
from test_training_data import dataset


@pytest.mark.parametrize('kv_heads', [1, 2, 4])
def test_attention_kv_expansion_preserves_outputs_and_gradients(kv_heads):
    from openpi.models.gemma import attend
    from torch.nn import functional as F

    torch.manual_seed(17)
    q = torch.randn(2, 4, 5, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, kv_heads, 9, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    mask = torch.rand(2, 9) > .3
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(4 // kv_heads, 1), v.repeat_interleave(4 // kv_heads, 1),
        attn_mask=mask[:, None, None]).transpose(1, 2).flatten(2)
    actual = attend(q, k, v, mask)
    upstream = torch.randn_like(actual)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (q, k, v), upstream),
                    torch.autograd.grad(expected, (q, k, v), upstream)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('autocast', [False, True])
def test_lora_cast_elision_preserves_outputs_and_gradients(autocast):
    from openpi.models.lora import LoRALinear
    from torch.nn import functional as F

    torch.manual_seed(19)
    layer = LoRALinear(torch.nn.Linear(16, 8).bfloat16(), rank=4, alpha=4)
    torch.nn.init.normal_(layer.lora_B)
    value = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        expected = F.linear(value, layer.weight, layer.bias) + (
            F.linear(F.linear(value.to(layer.lora_A.dtype), layer.lora_A), layer.lora_B)
            * layer.scale).to(value.dtype)
        actual = layer(value)
    upstream = torch.randn_like(actual)
    parameters = (value, layer.lora_A, layer.lora_B)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for a, b in zip(torch.autograd.grad(actual, parameters, upstream),
                    torch.autograd.grad(expected, parameters, upstream)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_lora_is_initially_noop_and_only_adapter_weights_update(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = Pi0Config.tiny(vocab_size=128)
    model = Pi0(config)
    inputs, targets = batch(config)
    noise = torch.randn_like(targets["actions"])
    original = model.sample_actions(inputs, noise=noise, steps=2)[0]
    base = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = tmp_path / "base.safetensors"
    save_file(base, str(path))
    assert apply_lora(model, rank=2, alpha=2) == 28
    torch.testing.assert_close(original, model.sample_actions(inputs, noise=noise, steps=2)[0], rtol=0, atol=0)
    assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    loss = model(inputs, targets)["action"]
    loss.backward()
    optimizer.step()
    assert any(p.abs().sum() > 0 for name, p in model.named_parameters() if name.endswith("lora_B"))
    for name, value in base.items():
        torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
    settings = TrainConfig("unused", "unused", str(tmp_path), "unused",
                           pretrained_path=str(path), model=config, model_type="pi05",
                           lora_rank=2, lora_alpha=2, device="cpu")
    save_checkpoint(tmp_path / "checkpoint.pt", model=model, optimizer=optimizer,
                    config=settings, step=1, stats={})
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert saved["weights_kind"] == "lora"
    assert all("lora_" in name for name in saved["model"])
    restored = create_model(settings, training=False, initialize=True)
    load_weights(restored, saved["model"], adapter_only=True)
    torch.testing.assert_close(model.sample_actions(inputs, noise=noise, steps=2)[0],
                               restored.sample_actions(inputs, noise=noise, steps=2)[0])
    assert not hasattr(restored, "skill") and not hasattr(restored, "grounding")
    missing = dict(saved["model"])
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing adapter"):
        load_weights(restored, missing, adapter_only=True)


def test_action_only_dataset_preserves_asynchronous_pairing_and_padding(tmp_path):
    _, index = dataset(tmp_path)
    data = EpisodeDataset(index, action_horizon=4, auxiliary=False, action_padding="mask")
    assert len(data) == 6
    item = data[2]
    assert item["metadata"]["frame"] == 0
    assert item["metadata"]["action_frames"].tolist() == [1, 2, 2, 3]
    assert set(item["targets"]) == {"actions", "action_valid"}
    last = data[5]
    assert last["targets"]["action_valid"][0].all()
    assert not last["targets"]["action_valid"][1:].any()
