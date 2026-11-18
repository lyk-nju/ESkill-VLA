"""Training-only ESkill appearance and shared image-coordinate transforms."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GroundingAugmentation:
    """Augmentation phase 1: counterfactual query, object copy-paste and protected patch masking."""
    target_swap_prob: float = 0.
    target_min_visible_fraction: float = .8
    copy_prob: float = 0.
    copy_target_prob: float = .5  # Include the source object named by the task when eligible.
    copy_max_objects: int = 3
    copy_max_shift: float = .3  # Fraction of source image dimensions.
    patch_prob: float = 0.
    patch_drop_prob: float = .25
    patch_size: int = 28  # Model-input pixels; scaled to the source RGB.
    target_margin_px: int = 8  # Halo around the full target bbox, in source-image pixels.

    def __post_init__(self):
        for name in ('target_swap_prob', 'target_min_visible_fraction', 'copy_prob', 'copy_target_prob', 'copy_max_shift', 'patch_prob', 'patch_drop_prob'):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        if self.patch_size < 1:
            raise ValueError('patch_size must be positive')
        if type(self.copy_max_objects) is not int or not 1 <= self.copy_max_objects <= 3:
            raise ValueError('copy_max_objects must be an integer from 1 to 3')
        if self.target_margin_px < 0:
            raise ValueError('target_margin_px must be nonnegative')

    @property
    def enabled(self):
        return bool(self.target_swap_prob or self.copy_prob or self.patch_prob)


def copy_destination_mask(source, obstacles, robot, max_shift):
    """Valid integer top-left positions for a fixed-size bbox, computed before sampling."""
    import numpy as np
    h, w = robot.shape
    x0, y0, x1, y1 = source
    bh, bw = y1 - y0, x1 - x0
    occupied = robot.astype(bool, copy=True)
    for box in obstacles:
        a, b = np.floor(box[:2]).astype(int).clip([0, 0], [w, h])
        c, d = np.ceil(box[2:]).astype(int).clip([0, 0], [w, h])
        occupied[b:d, a:c] = True
    # Rectangle sums find every empty placement in O(HW), without trying random offsets.
    integral = np.pad(occupied.astype(np.int32), ((1, 0), (1, 0)))
    integral = integral.cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32)
    counts = integral[bh:, bw:] - integral[:-bh, bw:] - integral[bh:, :-bw] + integral[:-bh, :-bw]
    yy, xx = np.ogrid[:counts.shape[0], :counts.shape[1]]
    dx, dy = np.abs(xx - x0), np.abs(yy - y0)
    return ((counts == 0) & (dx <= round(w * max_shift)) & (dy <= round(h * max_shift))
            & ((dx >= 4) | (dy >= 4)))


def copy_objects(image, boxes, eligible, visible_mask, robot_mask, rng, config, *,
                 task_source=None, protected_boxes=()):
    """Move up to the sampled budget, planning each move against the latest boxes."""
    import numpy as np
    moved_offsets = {}
    info = dict(copy_attempted=False, copy_applied=False, moved_offsets=moved_offsets, copy_reason='not_sampled',
                copy_budget=0, copy_count=0, valid_destination_count=0, copy_task_source_eligible=False,
                copy_task_source_selected=False, copied_task_source=False)
    if rng.random() >= config.copy_prob:
        return image, boxes, info
    info.update(copy_attempted=True, copy_reason='no_editable_object')

    def overlaps(box, obstacles):
        return np.any((box[0] < obstacles[:, 2]) & (obstacles[:, 0] < box[2]) &
                      (box[1] < obstacles[:, 3]) & (obstacles[:, 1] < box[3]))

    protected = np.asarray([box for box in protected_boxes if box is not None and np.isfinite(box).all()],
                           dtype=float).reshape(-1, 4)
    if len(protected):
        eligible = [key for key in eligible if not overlaps(boxes[key], protected)]
    if not eligible:
        return image, boxes, info

    source_eligible = task_source in eligible
    budget = int(rng.integers(1, min(config.copy_max_objects, len(eligible)) + 1))
    info.update(copy_budget=budget, copy_task_source_eligible=source_eligible)

    source_selected = source_eligible and rng.random() < config.copy_target_prob
    info['copy_task_source_selected'] = source_selected
    others = [key for key in eligible if key != task_source]
    order = ([task_source] if source_selected else []) + [others[i] for i in rng.permutation(len(others))]
    info['copy_reason'] = 'task_source_not_sampled'
    h, w = image.shape[:2]
    robot = None
    edited, updated = image, dict(boxes)
    for key in order:
        src = np.asarray(updated[key], dtype=int)
        other_boxes = np.asarray([v for k, v in updated.items() if k != key], dtype=float).reshape(-1, 4)
        obstacles = np.concatenate((other_boxes, protected))
        # Include the RGB/JPEG fringe in the original erasure protection.
        erase = (src + np.array([-2,-2,2,2])).clip([0,0,0,0],[w,h,w,h])
        if overlaps(erase, obstacles):
            info['copy_reason'] = 'source_entity_overlap'
            continue
        if robot is None:
            robot = robot_mask()
        ex0,ey0,ex1,ey1 = erase
        if robot[ey0:ey1,ex0:ex1].any():
            info['copy_reason'] = 'source_robot_overlap'
            continue
        mask = visible_mask(key)
        if mask is None or not mask.any():
            info['copy_reason'] = 'missing_visible_pixels'
            continue
        valid = copy_destination_mask(src, obstacles, robot, config.copy_max_shift)
        positions = np.flatnonzero(valid)
        if not len(positions):
            info['copy_reason'] = 'no_valid_destination'
            continue

        x0, y0, x1, y1 = src
        b, a = divmod(int(rng.choice(positions)), valid.shape[1])
        c, d = a + x1 - x0, b + y1 - y0
        border = np.concatenate([edited[max(0,ey0-1),ex0:ex1], edited[min(h-1,ey1),ex0:ex1],
                                 edited[ey0:ey1,max(0,ex0-1)], edited[ey0:ey1,min(w-1,ex1)]])
        if edited is image:
            edited = image.copy()
        edited[ey0:ey1,ex0:ex1] = np.median(border, axis=0).astype(np.uint8)
        crop_mask = mask[y0:y1,x0:x1]
        edited[b:d,a:c][crop_mask] = image[y0:y1,x0:x1][crop_mask]
        updated[key] = np.array([a,b,c,d], np.float32)
        moved_offsets[key] = [int(a - x0), int(b - y0)]
        info['valid_destination_count'] += len(positions)
        if len(moved_offsets) == budget:
            break

    if moved_offsets:
        info.update(copy_applied=True, copy_reason='applied', copy_count=len(moved_offsets),
                    copied_task_source=task_source in moved_offsets)
    return edited, updated, info


def mask_grounding_patches(image, interaction_box, rng, config, image_size, *, target_box=None, gripper_box=None):
    """Protect full target/gripper boxes plus halos, and the local interaction area."""
    import numpy as np
    h, w = image.shape[:2]
    drop = np.zeros((h,w), bool)
    if rng.random() >= config.patch_prob:
        return image, drop
    size = max(1, round(config.patch_size * max(h,w) / image_size))
    cells = rng.random((math.ceil(h/size), math.ceil(w/size))) < config.patch_drop_prob
    protected = [interaction_box]
    margin = config.target_margin_px
    for box in (target_box, gripper_box):
        if box is not None:
            protected.append(np.asarray(box) + [-margin,-margin,margin,margin])
    for box in protected:
        if box is None or not np.isfinite(box).all():
            continue
        x0,y0,x1,y1 = np.clip(box, [0,0,0,0], [w,h,w,h])
        cells[max(0,int(y0)//size):math.ceil(y1/size), max(0,int(x0)//size):math.ceil(x1/size)] = False
    drop = cells.repeat(size,0).repeat(size,1)[:h,:w]
    edited = image.copy()
    edited[drop] = 128
    return edited, drop


def augmentation_invalidates(visible, drop):
    """Native invisibility never invalidates the existing full Grounding label."""
    return bool(visible is not None and visible.any() and not (visible & ~drop).any())


@dataclass(frozen=True)
class ImageAugmentation:
    """Augmentation phase 2: conventional geometry/color; horizontal flip is configured separately."""
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


def transform_boxes(boxes, windows, cosine, sine):
    """Clip to crop, rotate, intersect with screen, and bound the surviving polygon."""
    import torch

    flat = boxes.reshape(len(boxes), -1, 4)
    origin, size = windows[:, None].split(2, -1)
    lo = (flat[..., :2] - origin) / size
    hi = (flat[..., 2:] - origin) / size
    within_crop = ((lo >= 0) & (hi <= 1)).all(-1)
    lo, hi = lo.clamp(0, 1), hi.clamp(0, 1)
    unit = boxes.new_tensor([[0, 0], [1, 0], [1, 1], [0, 1]])
    rect = lo[..., None, :] + unit * (hi - lo)[..., None, :]
    c, s = cosine[:, None, None], sine[:, None, None]

    def rotate(points, sin):
        x, y = (points - .5).unbind(-1)
        return torch.stack((c * x - sin * y, sin * x + c * y), -1) + .5

    vertices = rotate(rect, s)
    inside = ((vertices >= 0) & (vertices <= 1)).all(-1)
    # Vertices of the intersection are rectangle corners, screen corners or edge crossings.
    screen = unit.expand_as(vertices)
    inverse = rotate(screen, -s)
    screen_inside = ((inverse >= lo[..., None, :]) & (inverse <= hi[..., None, :])).all(-1)
    points, valid = [vertices, screen], [inside, screen_inside]
    delta = vertices.roll(-1, dims=-2) - vertices
    for axis in (0, 1):
        step = delta[..., axis]
        moving = step.abs() > 1e-8
        t = (boxes.new_tensor([0, 1]) - vertices[..., axis, None]) / torch.where(moving, step, 1.)[..., None]
        crossings = vertices[..., None, :] + t[..., None] * delta[..., None, :]
        other = crossings[..., 1 - axis]
        valid.append((moving[..., None] & (t >= 0) & (t <= 1) & (other >= 0) & (other <= 1)).flatten(-2))
        points.append(crossings.flatten(-3, -2))
    points = torch.cat(points, -2)
    valid = torch.cat(valid, -1)[..., None]
    lower = points.masked_fill(~valid, 1.).amin(-2)
    upper = points.masked_fill(~valid, 0.).amax(-2)
    survives = (hi > lo).all(-1) & (upper > lower).all(-1)
    result = torch.cat((lower, upper), -1).clamp(0, 1)
    result = torch.where(survives[..., None], result, 0.).reshape_as(boxes)
    return result, survives.reshape(boxes.shape[:-1]), (within_crop & inside.all(-1)).reshape(boxes.shape[:-1])


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
        return torch.where(active[:, None, None, None], sampled, value), support, grid

    images = inputs['images'].clone()
    images[:, 0], image_support, image_grid = warp(images[:, 0], -1.)
    inputs['images'] = images  # Wrist geometry is untouched.
    roles = targets['role_images']
    targets['role_images'] = warp(roles.flatten(1, 3), -1.)[0].reshape_as(roles)
    weights = targets['role_weights']
    targets['role_weights'] = warp(weights.flatten(1, 2), 0.)[0].reshape_as(weights)
    targets['role_valid'] = targets['role_valid'] & (
        ~active[:, None, None] | targets['role_weights'].gt(0).any((-1, -2)))
    if 'role_core_boxes' in targets:
        # Reuse the RGB inverse sampling grid: a background halo alone cannot
        # keep a target valid. Full bboxes preserve native-occlusion semantics.
        boxes = targets['role_core_boxes'][:, :, :, None, None]
        points = image_grid[:, None, None]
        retained = ((points >= boxes[..., :2]) & (points < boxes[..., 2:])).all(-1)
        retained = (retained & image_support[:, None, None]).any((-1, -2))
        targets['role_valid'] = targets['role_valid'] & (~active[:, None, None] | retained)
        transformed, survives, _ = transform_boxes(targets['role_core_boxes'], windows, cosine, sine)
        targets['role_core_boxes'] = torch.where(active[:, None, None, None], transformed, targets['role_core_boxes'])
        targets['role_valid'] &= ~active[:, None, None] | survives
    if 'entity_boxes' in targets:
        boxes, survives, full = transform_boxes(targets['entity_boxes'], windows, cosine, sine)
        targets['entity_boxes'] = torch.where(active[:, None, None], boxes, targets['entity_boxes'])
        # Visibility was measured before geometry. Use intact ROIs for the appearance contrast.
        targets['entity_valid'] = targets['entity_valid'] & (~active[:, None] | (survives & full))
    if 'trace' in targets:
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
    eligible = torch.tensor([row.get('spatial_augmentation_eligible', True) for row in batch['metadata']],
                            dtype=torch.bool, device=device)
    if config.crop_prob:
        selected = (torch.rand(count, device=device) < config.crop_prob) & eligible
        lo, hi = config.crop_scale
        if lo == hi:
            # Official pi05 uses an integer crop size and an integer random origin.
            height, width = images.shape[-2:]
            crop_h, crop_w = max(1, int(height * lo)), max(1, int(width * lo))
            x = torch.randint(width - crop_w + 1, (count,), device=device) / width
            y = torch.randint(height - crop_h + 1, (count,), device=device) / height
            extent = images.new_tensor([crop_w / width, crop_h / height]).expand(count, -1)
            sampled = torch.cat((torch.stack((x, y), -1), extent), -1)
        else:
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
        rotate = (torch.rand(count, device=device) < config.rotation_prob) & eligible
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
        trace_count = [0] * count
        if 'trace_valid' in targets:
            padded = targets.get('terminal_padded_mask', targets.get('future_padded', torch.zeros_like(targets['trace_valid'])))
            held = targets.get('trace_terminal_hold', torch.zeros_like(padded))
            trace = targets['trace_valid'] & presence[:, :, None] & (~padded | held)
            trace_count = trace.sum((1, 2)).tolist()
        role = targets['role_valid'] & presence[:, :, None]
        return trace_count, role.sum((1, 2)).tolist()
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
    metadata = [dict(row, augmentation=dict(row.get('augmentation', {}),
        crop=windows[i], rotation_degrees=angles[i], color_factors=factors[i],
        counts=dict(crop_samples=int(crops[i]), rotated_samples=int(rotations[i]), colored_views=colors[i], trace_before=before_trace[i],
                    trace_after=after_trace[i], role_before=before_role[i], role_after=after_role[i],
                    role_area_before=before_area[i], role_area_after=after_area[i])))
        for i, row in enumerate(batch['metadata'])]
    if 'role_core_boxes' in augmented['targets']:
        for row, valid in zip(metadata, augmented['targets']['role_valid'][:, 0].tolist(), strict=True):
            row['augmentation']['current_valid'] = valid[0]
            if len(valid) > 1:
                row['augmentation']['subject_valid'] = valid[1]
    return dict(augmented, inputs=inputs, metadata=metadata)


def augment_training_batch(batch, config):
    """Apply phase 2 after Dataset phase 1: mirror, shared geometry, then input-only color.

    These augmentation phases are independent of the model training stage.
    """
    import torch
    from openpi.transforms import horizontal_mirror
    if config.horizontal_flip_prob:
        device = batch['inputs']['images'].device
        eligible = torch.tensor([row['mirror_eligible'] for row in batch['metadata']], device=device)
        selected = (torch.rand(len(eligible), device=device) < config.horizontal_flip_prob) & eligible
        batch = horizontal_mirror(batch, selected)
    return augment_eskill_batch(batch, config.image_augmentation)
