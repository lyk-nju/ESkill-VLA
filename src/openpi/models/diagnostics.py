"""Detached diagnostic sums/counts; never used as model conditions or losses."""
import torch


@torch.no_grad()
def mean_statistics(values, valid):
    return torch.stack((torch.where(valid, values.detach(), 0).sum(dtype=torch.float64),
                        valid.sum(dtype=torch.float64)))
