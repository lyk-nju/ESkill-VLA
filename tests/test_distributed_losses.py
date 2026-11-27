"""Two-process DDP must match a single global batch, including empty ranks."""
from datetime import timedelta

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


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Requires Gloo")
@pytest.mark.parametrize("worker", [masked_worker], ids=["masked_losses"])
def test_ddp_matches_global_batch(tmp_path, worker):
    mp.spawn(worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
