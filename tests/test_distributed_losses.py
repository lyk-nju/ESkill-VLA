"""Two-process DDP must match a single global batch, including empty ranks."""
from copy import deepcopy
from datetime import timedelta
from unittest import mock

import pytest
torch = pytest.importorskip("torch")
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from openpi.models.model import masked_mean


class MaskedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 9))

    def forward(self, values, valid):
        return masked_mean(self.weight * values, valid)


def masked_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        model = DDP(MaskedLoss())
        values = torch.tensor([[9.] * 9, [1.] * 9])
        # Unequal counts, equal counts, either rank empty, and globally empty.
        for counts in ((1, 9), (9, 9), (0, 9), (9, 0), (0, 0)):
            valid = torch.arange(9)[None] < torch.tensor(counts)[:, None]
            local_valid = valid & (torch.arange(2)[:, None] == rank)
            model.zero_grad(set_to_none=True)
            loss = model(values, local_valid)
            loss.backward()
            logged = loss.detach().clone()
            dist.all_reduce(logged)
            logged /= 2

            # Independent global-batch reference, without the distributed helper.
            weight = torch.ones(2, 9, requires_grad=True)
            expected = torch.where(valid, weight * values, 0).sum() / valid.sum().clamp_min(1)
            expected.backward()
            torch.testing.assert_close(logged, expected.detach())
            torch.testing.assert_close(model.module.weight.grad, weight.grad)
    finally:
        dist.destroy_process_group()


class VisualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKL
        from openpi.models.grounding import VisualSupervision
        from openpi.models.pi0_config import ESkillConfig

        config = ESkillConfig.tiny(canvas_size=8)
        vae = AutoencoderKL(
            down_block_types=("DownEncoderBlock2D",), up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,), norm_num_groups=4, layers_per_block=1, latent_channels=4)
        self.anchors = nn.Parameter(torch.randn(2, 3, config.role_queries, config.hidden_dim))
        self.supervision = VisualSupervision(config, vae)

    def forward(self, targets):
        anchors = self.anchors[None].expand(len(targets["role_valid"]), -1, -1, -1, -1)
        return self.supervision(anchors, targets)


def visual_worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        model = DDP(VisualLoss(), find_unused_parameters=True, broadcast_buffers=False)
        targets = dict(role_images=torch.randn(4, 2, 3, 3, 8, 8),
                       role_weights=torch.rand(4, 2, 3, 8, 8))
        uneven = torch.tensor([
            [[1, 1, 0], [0, 1, 0]], [[0, 1, 0], [0, 1, 0]],
            [[1, 1, 1], [1, 0, 1]], [[1, 0, 1], [1, 0, 0]],
        ], dtype=torch.bool)
        empty_first = uneven.clone()
        empty_first[:2] = False
        empty_second = uneven.clone()
        empty_second[2:] = False
        for valid in (uneven, empty_first, empty_second, torch.zeros_like(uneven)):
            targets["role_valid"] = valid
            local = {key: value[rank * 2:(rank + 1) * 2] for key, value in targets.items()}
            model.zero_grad(set_to_none=True)
            loss = model(local)
            loss.backward()
            logged = loss.detach().clone()
            dist.all_reduce(logged)
            logged /= 2

            reference = deepcopy(model.module)
            reference.zero_grad(set_to_none=True)
            with mock.patch("openpi.models.model.dist.is_initialized", return_value=False):
                expected = reference(targets)
            expected.backward()
            torch.testing.assert_close(logged, expected.detach(), rtol=1e-5, atol=1e-6)
            for name, parameter in model.module.named_parameters():
                if not parameter.requires_grad:
                    continue
                actual = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
                target = reference.get_parameter(name).grad
                target = target if target is not None else torch.zeros_like(parameter)
                torch.testing.assert_close(actual, target, rtol=1e-4, atol=1e-6, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("worker", [masked_worker, visual_worker], ids=["masked_losses", "visual_roles"])
def test_ddp_matches_global_batch(tmp_path, worker):
    if worker is visual_worker:
        pytest.importorskip("diffusers")
        pytest.importorskip("transformers")
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
