"""Deterministic sample transforms. Future observations stay in targets."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def resize_geometry(shape, size):
    """Actual integer resize and padding, shared by images and pixel-center points."""
    h, w = shape[:2]
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    return nh, nw, (size - nh) // 2, (size - nw) // 2


def resize_image(image, size, *, nearest=False):
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    nh, nw, y, x = resize_geometry(image.shape, size)
    value = np.asarray(Image.fromarray(image).resize((nw, nh), resample))
    result = np.zeros((size, size, *image.shape[2:]), dtype=image.dtype)
    result[y:y + nh, x:x + nw] = value
    return result


def project_points(points, intrinsic, world_to_camera, image_transform, image_shape):
    """World XYZ -> source RGB pixel centers, with positive camera Z forward.

    image_transform maps the calibrated camera pixels to the saved RGB pixels.
    Pixel (0, 0) is the top-left pixel center; image edges are at -0.5 and W/H-0.5.
    Occlusion is deliberately absent from validity. Invalid points are not clamped.
    """
    points = np.asarray(points, np.float64)
    transform = np.asarray(world_to_camera, np.float64)
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    pixels = camera @ np.asarray(intrinsic, np.float64).T
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = pixels / pixels[..., 2:3]
        pixels = pixels @ np.asarray(image_transform, np.float64).T
        uv = pixels[..., :2] / pixels[..., 2:3]
    h, w = image_shape[:2]
    valid = (np.isfinite(uv).all(-1) & np.isfinite(camera).all(-1) & (camera[..., 2] > 0)
             & (uv >= -.5).all(-1) & (uv < np.array([w, h]) - .5).all(-1))
    return uv.astype(np.float32), valid


def resize_points(points, valid, image_shape, size):
    """Source pixel centers -> normalized model-canvas points, matching resize_image."""
    h, w = image_shape[:2]
    nh, nw, y, x = resize_geometry(image_shape, size)
    points = np.asarray(points, np.float32)
    in_frame = (np.isfinite(points).all(-1) & (points >= -.5).all(-1)
                & (points < np.array([w, h]) - .5).all(-1))
    normalized = ((points + .5) * np.array([nw / w, nh / h]) + [x, y]) / size
    return normalized.astype(np.float32), np.asarray(valid, bool) & in_frame


def future_targets(frame_indices, points, valid, anchors, offsets, *, tail_policy='mask', last_frame=None):
    """Fixed source offsets; optionally extend the final recorded position past the tail.

    hold_last supplies a terminal padding target, not an observed future position.
    Interior gaps and invalid final records remain invalid under either policy.
    """
    indices = np.asarray(frame_indices)
    offsets = np.asarray(offsets)
    if (not len(indices) or np.any(np.diff(indices) <= 0) or offsets.ndim != 1
            or not len(offsets) or offsets[0] <= 0 or np.any(np.diff(offsets) <= 0)
            or not np.issubdtype(offsets.dtype, np.integer)):
        raise ValueError('Need increasing source indices and positive increasing integer offsets')
    if tail_policy not in ('mask', 'hold_last'):
        raise ValueError('Unknown future-point tail policy')
    requested = np.asarray(anchors)[..., None] + offsets
    endpoint = indices[-1] if last_frame is None else last_frame
    source = np.minimum(requested, endpoint) if tail_policy == 'hold_last' else requested
    rows = np.searchsorted(indices, source)
    take = np.minimum(rows, len(indices) - 1)
    matched = (rows < len(indices)) & (indices[take] == source) & (source <= endpoint)
    available = matched & np.asarray(valid, bool)[take]
    values = np.asarray(points, np.float32)[take]
    available &= np.isfinite(values).all(-1)
    return dict(trace=np.where(available[..., None], values, 0), trace_valid=available,
                future_observed=available & (requested <= endpoint),
                future_padded=available & (requested > endpoint),
                future_frame_indices=requested.astype(np.int64),
                future_source_frame_indices=np.where(available, source, -1).astype(np.int64))


def future_points(frame_indices, points, valid, anchors, offsets, *, tail_policy='mask', last_frame=None):
    """Coordinate-only view of future_targets, used by existing projection checks."""
    targets = future_targets(frame_indices, points, valid, anchors, offsets,
                             tail_policy=tail_policy, last_frame=last_frame)
    return targets['trace'], targets['trace_valid']


def role_canvas(image, region, size, margin=4):
    """Keep the original layout; bbox extent is already full_in_frame."""
    canvas = np.zeros((size, size, 3), np.uint8)
    weight = np.zeros((size, size), np.float32)
    if image is None or region is None:
        return canvas, weight, False
    box = region.get("bbox")
    if box is None and region.get("mask") is not None:
        y, x = np.where(region["mask"])
        if len(x):
            box = [x.min(), y.min(), x.max() + 1, y.max() + 1]
    if box is None:
        return canvas, weight, False
    h, w = image.shape[:2]
    x0, y0, x1, y1 = np.asarray(box, float)
    x0, y0 = max(0, int(np.floor(x0)) - margin), max(0, int(np.floor(y0)) - margin)
    x1, y1 = min(w, int(np.ceil(x1)) + margin), min(h, int(np.ceil(y1)) + margin)
    if x1 <= x0 or y1 <= y0:
        return canvas, weight, False
    cropped = np.zeros_like(image)
    cropped[y0:y1, x0:x1] = image[y0:y1, x0:x1]
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return resize_image(cropped, size), resize_image(mask, size).astype(np.float32) / 255, True


def spatial_trace(points, count):
    """Arc-length samples of the complete remaining path, in source XYZ units."""
    points = np.asarray(points, np.float32)
    if len(points) == 0 or not np.isfinite(points).all():
        return np.zeros((count, 3), np.float32), np.zeros(count, bool)
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if distance[-1] <= 1e-8:
        return np.zeros((count, 3), np.float32), np.ones(count, bool)
    unique = np.r_[True, np.diff(distance) > 1e-8]
    positions = np.linspace(0, distance[-1], count)
    sampled = np.stack([np.interp(positions, distance[unique], points[unique, i]) for i in range(3)], -1)
    return (sampled - points[0]).astype(np.float32), np.ones(count, bool)


class Normalize:
    def __init__(self, stats):
        self.stats = json.loads(Path(stats).read_text()) if isinstance(stats, (str, Path)) else stats

    def apply(self, key, value, valid=None, *, inverse=False):
        item = self.stats[key]
        if item.get("method", self.stats.get("method")) == "quantile":
            q01, q99 = np.asarray(item["q01"], np.float32), np.asarray(item["q99"], np.float32)
            output = ((value + 1) / 2 * (q99 - q01 + 1e-6) + q01 if inverse
                      else (value - q01) / (q99 - q01 + 1e-6) * 2 - 1)
        else:
            mean, std = np.asarray(item["mean"], np.float32), np.asarray(item["std"], np.float32)
            output = value * std + mean if inverse else (value - mean) / np.maximum(std, 1e-6)
        return np.where(valid, output, 0).astype(np.float32) if valid is not None else output.astype(np.float32)


class TokenizePrompt:
    def __init__(self, path, max_len):
        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))
        self.max_len = max_len

    def __call__(self, prompt, state=None):
        text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            bins = np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1
            text = f"Task: {text}, State: {' '.join(map(str, bins))};\nAction: "
        tokens = self.tokenizer.encode(text, add_bos=True)
        if state is None:
            tokens += self.tokenizer.encode("\n")
        count = min(len(tokens), self.max_len)
        result = np.zeros(self.max_len, np.int64)
        result[:count] = tokens[:count]
        return result, np.arange(self.max_len) < count


class ModelTransforms:
    def __init__(self, model_config, stats, tokenizer):
        self.config, self.normalize, self.tokenizer = model_config, Normalize(stats), tokenizer

    def inputs(self, data):
        data = dict(data)
        state = self.normalize.apply("state", data.pop("state"), data.pop("state_valid"))
        tokens, token_mask = self.tokenizer(data.pop("prompt"), state if self.config.discrete_state_input else None)
        body = {}
        if "arm_state" in data:
            body = dict(arm_state=self.normalize.apply("arm_state", data.pop("arm_state"), data.pop("arm_state_valid")),
                        arm_presence=data["arm_presence"])
        images = np.stack([resize_image(im, self.config.image_size).transpose(2, 0, 1) for im in data.pop("images")])
        return dict(
            images=images.astype(np.float32) / 127.5 - 1,
            image_mask=data["image_mask"], tokens=tokens, token_mask=token_mask,
            state=np.pad(state, (0, self.config.action_dim - len(state))),
            **body,
        )

    def __call__(self, sample):
        targets = dict(sample["targets"])
        action = self.normalize.apply("actions", targets["actions"], targets["action_valid"])
        pad = self.config.action_dim - action.shape[-1]
        targets["actions"] = np.pad(action, ((0, 0), (0, pad)))
        targets["action_valid"] = np.pad(targets["action_valid"], ((0, 0), (0, pad)))
        if "role_images" in targets:
            targets["role_images"] = targets["role_images"].transpose(0, 1, 4, 2, 3).astype(np.float32) / 127.5 - 1
        return dict(inputs=self.inputs(sample["inputs"]), targets=targets, metadata=sample["metadata"])
