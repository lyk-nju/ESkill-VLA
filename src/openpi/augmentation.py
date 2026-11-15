"""Training-only ESkill appearance and shared image-coordinate transforms."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ImageAugmentation:
    color_prob: float = 0.
    brightness: float = .15
    contrast: float = .15
    saturation: float = .1
    crop_prob: float = 0.
    # Retained side length, not area; a normalized square on the model canvas.
    crop_scale: tuple[float, float] = (.95, 1.)
    rotation_prob: float = 0.
    rotation_degrees: float = 5.

    def __post_init__(self):
        object.__setattr__(self, 'crop_scale', tuple(self.crop_scale))
        for name in ('color_prob', 'crop_prob', 'rotation_prob', 'brightness', 'contrast', 'saturation'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'image_augmentation.{name} must lie in [0, 1]')
        if (len(self.crop_scale) != 2 or not all(math.isfinite(x) for x in self.crop_scale)
                or not 0 < self.crop_scale[0] <= self.crop_scale[1] <= 1):
            raise ValueError('image_augmentation.crop_scale needs 0 < min <= max <= 1')
        if not math.isfinite(self.rotation_degrees) or not 0 <= self.rotation_degrees <= 180:
            raise ValueError('image_augmentation.rotation_degrees must lie in [0, 180]')

    @property
    def enabled(self):
        return bool(self.color_prob or self.crop_prob or self.rotation_prob)


def affine_eskill_batch(batch, windows, angles=None):
    """Crop then rotate about (.5,.5), sampling once; positive degrees are clockwise in image xy."""
    import torch
    from torch.nn import functional as F

    inputs, targets = dict(batch['inputs']), dict(batch['targets'])
    if angles is None:
        angles = windows.new_zeros(len(windows))
    radians = angles * (math.pi / 180)
    cosine, sine = radians.cos(), radians.sin()
    rotated = angles != 0
    active = (windows != windows.new_tensor([0., 0., 1., 1.])).any(-1) | rotated

    def warp(value, background):
        height, width = value.shape[-2:]
        y, x = torch.meshgrid((torch.arange(height, device=value.device) + .5) / height,
                              (torch.arange(width, device=value.device) + .5) / width, indexing='ij')
        output = torch.stack((x, y), -1)[None].expand(len(windows), -1, -1, -1)
        centered = output - .5
        c, s = cosine[:, None, None], sine[:, None, None]
        # Image sampling is inverse rotation, while trajectory points below rotate forward.
        inverse = torch.stack((c * centered[..., 0] + s * centered[..., 1],
                               -s * centered[..., 0] + c * centered[..., 1]), -1) + .5
        uncropped = torch.where(rotated[:, None, None, None], inverse, output)
        grid = uncropped * windows[:, None, None, 2:] + windows[:, None, None, :2]
        sampled = F.grid_sample(value - background, 2 * grid - 1, mode='bilinear',
                                padding_mode='zeros', align_corners=False) + background
        # Preserve the cropped field of view: rotation must not reveal pixels outside it.
        support = (uncropped >= 0).all(-1) & (uncropped < 1).all(-1)
        sampled = torch.where(support[:, None], sampled, background)
        return torch.where(active[:, None, None, None], sampled, value), support

    images = inputs['images'].clone()
    images[:, 0], image_support = warp(images[:, 0], -1.)
    inputs['images'] = images  # Wrist geometry is untouched.
    roles = targets['role_images']
    targets['role_images'] = warp(roles.flatten(1, 3), -1.)[0].reshape_as(roles)
    weights = targets['role_weights']
    targets['role_weights'] = warp(weights.flatten(1, 2), 0.)[0].reshape_as(weights)
    targets['role_valid'] = targets['role_valid'] & (
        ~active[:, None, None] | targets['role_weights'].gt(0).any((-1, -2)))
    points = targets['trace']
    moved = (points - windows[:, None, None, :2]) / windows[:, None, None, 2:]
    in_crop = (moved >= 0).all(-1) & (moved < 1).all(-1)
    centered = moved - .5
    c, s = cosine[:, None, None], sine[:, None, None]
    forward = torch.stack((c * centered[..., 0] - s * centered[..., 1],
                           s * centered[..., 0] + c * centered[..., 1]), -1) + .5
    moved = torch.where(rotated[:, None, None, None], forward, moved)
    moved = torch.where(active[:, None, None, None], moved, points)
    in_frame = in_crop & torch.isfinite(moved).all(-1) & (moved >= 0).all(-1) & (moved < 1).all(-1)
    valid = targets['trace_valid'] & (~active[:, None, None] | in_frame)
    targets['trace'] = torch.where(active[:, None, None, None], torch.where(valid[..., None], moved, 0.), points)
    targets['trace_valid'] = valid  # Off-screen points are excluded, never clamped to an edge.
    return dict(batch, inputs=inputs, targets=targets), image_support


def augment_eskill_batch(batch, config):
    """Use torch's device RNG so ordinary checkpoint RNG restoration is sufficient."""
    if not config.enabled:
        return batch
    import torch

    images = batch['inputs']['images']
    count, views = images.shape[:2]
    device = images.device
    windows = images.new_tensor([0., 0., 1., 1.]).repeat(count, 1)
    selected = torch.zeros(count, dtype=torch.bool, device=device)
    if config.crop_prob:
        selected = torch.rand(count, device=device) < config.crop_prob
        lo, hi = config.crop_scale
        side = lo + (hi - lo) * torch.rand(count, device=device)
        origin = torch.rand(count, 2, device=device) * (1 - side[:, None])
        sampled = torch.cat((origin, side[:, None].expand(-1, 2)), -1)
        windows = torch.where(selected[:, None], sampled, windows)
    factors = images.new_ones(count, views, 3)
    colored = torch.zeros(count, views, dtype=torch.bool, device=device)
    if config.color_prob:
        colored = (torch.rand(count, views, device=device) < config.color_prob) & batch['inputs']['image_mask']
        amplitude = images.new_tensor([config.brightness, config.contrast, config.saturation])
        sampled = 1 + (2 * torch.rand(count, views, 3, device=device) - 1) * amplitude
        factors = torch.where(colored[..., None], sampled, factors)
    # Draw rotation last: A2 and A3 share crop/color draws for the same seed.
    angles = images.new_zeros(count)
    if config.rotation_prob:
        rotate = torch.rand(count, device=device) < config.rotation_prob
        sampled = (2 * torch.rand(count, device=device) - 1) * config.rotation_degrees
        angles = torch.where(rotate, sampled, angles)
    if config.crop_prob or config.rotation_prob:
        augmented, image_support = affine_eskill_batch(batch, windows, angles)
    else:
        augmented, image_support = batch, None
    inputs = dict(augmented['inputs'])
    if config.color_prob:
        rgb = (inputs['images'] + 1) / 2
        gray_weights = rgb.new_tensor([.2989, .587, .114]).view(1, 1, 3, 1, 1)
        value = (rgb * factors[..., 0, None, None, None]).clamp(0, 1)
        mean = (value * gray_weights).sum(2, keepdim=True).mean((-1, -2), keepdim=True)
        value = (mean + factors[..., 1, None, None, None] * (value - mean)).clamp(0, 1)
        gray = (value * gray_weights).sum(2, keepdim=True)
        value = (gray + factors[..., 2, None, None, None] * (value - gray)).clamp(0, 1)
        inputs['images'] = torch.where(colored[..., None, None, None], 2 * value - 1, inputs['images'])
    if image_support is not None:
        # Contrast jitter must not brighten artificial rotation corners into gray.
        inputs['images'][:, 0] = torch.where(image_support[:, None], inputs['images'][:, 0], -1.)

    presence = inputs['arm_presence']
    def counts(targets):
        padded = targets.get('terminal_padded_mask', targets.get('future_padded', torch.zeros_like(targets['trace_valid'])))
        held = targets.get('trace_terminal_hold', torch.zeros_like(padded))
        trace = targets['trace_valid'] & presence[:, :, None] & (~padded | held)
        role = targets['role_valid'] & presence[:, :, None]
        return trace.sum((1, 2)).tolist(), role.sum((1, 2)).tolist()
    before_trace, before_role = counts(batch['targets'])
    after_trace, after_role = counts(augmented['targets'])
    def area(targets):
        weights = targets['role_weights']
        visible = targets['role_valid'] & presence[:, :, None]
        return (weights * visible[..., None, None]).sum((1, 2, 3, 4)) / (weights.shape[-1] * weights.shape[-2])
    before_area = area(batch['targets']).tolist()
    # Remove the magnification from crop+resize when measuring retained foreground area.
    after_area = (area(augmented['targets']) * windows[:, 2:].prod(-1)).tolist()
    rotations = (angles != 0).tolist()
    windows, factors, angles = windows.tolist(), factors.tolist(), angles.tolist()
    crops, colors = selected.tolist(), colored.sum(1).tolist()
    metadata = [dict(row, augmentation=dict(
        crop=windows[i], rotation_degrees=angles[i], color_factors=factors[i],
        counts=dict(crop_samples=int(crops[i]), rotated_samples=int(rotations[i]), colored_views=colors[i], trace_before=before_trace[i],
                    trace_after=after_trace[i], role_before=before_role[i], role_after=after_role[i],
                    role_area_before=before_area[i], role_area_after=after_area[i])))
        for i, row in enumerate(batch['metadata'])]
    return dict(augmented, inputs=inputs, metadata=metadata)
