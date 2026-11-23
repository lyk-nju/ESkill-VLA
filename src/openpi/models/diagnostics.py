"""Detached diagnostic sums/counts; never used as model conditions or losses."""
from math import isqrt

import torch
from torch.nn import functional as F


HIT_RADIUS_PX = 10.


@torch.no_grad()
def mean_statistics(values, valid):
    return torch.stack((torch.where(valid, values.detach(), 0).sum(dtype=torch.float64),
                        valid.sum(dtype=torch.float64)))


@torch.no_grad()
def spatial_centers(weights):
    """Return normalized xy for the highest-mass 3×3 window and the single peak."""
    side = isqrt(weights.shape[-1])
    grid = weights.float().reshape(-1, 1, side, side)
    # Zero padding lets a window centered at any edge compete. Argmax breaks ties in raster order.
    window_mass = F.avg_pool2d(grid, 3, stride=1, padding=1, divisor_override=1)
    window = window_mass.flatten(1).argmax(-1)
    index = weights.argmax(-1)
    argmax = (torch.stack((index % side, index // side), -1).float() + .5) / side

    offsets = torch.arange(-1, 2, device=weights.device)
    x = (window % side)[:, None] + offsets
    y = (window // side)[:, None] + offsets
    indices = y[:, :, None] * side + x[:, None, :]
    inside = ((x[:, None, :] >= 0) & (x[:, None, :] < side)
              & (y[:, :, None] >= 0) & (y[:, :, None] < side))
    patch_weights = grid.flatten(1).gather(1, indices.flatten(1).clamp(0, side * side - 1)).view_as(indices)
    patch_weights = patch_weights * inside
    mass = patch_weights.sum((1, 2))
    centroid = torch.stack(((patch_weights.sum(1) * (x + .5)).sum(1),
                            (patch_weights.sum(2) * (y + .5)).sum(1)), dim=-1)
    peak3 = centroid / (mass.clamp_min(torch.finfo(grid.dtype).tiny)[:, None] * side)
    return peak3.reshape_as(argmax), argmax
