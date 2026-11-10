"""Action-query attention over image patches for trained checkpoints, rendered as one video.

Each checkpoint reads the episode with the views and transforms of its own training config,
so π0.5 SFT and ESkill models can be compared on the same source frames. Attention is recorded by
temporarily wrapping `openpi.models.pi0.attend` inside this process; model code is not modified and
the sampled actions are unchanged.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

import openpi.models.pi0 as pi0
from openpi.policies.eskill_dataset import ARM_SLOTS, EpisodeDataset, EpisodeReader
from openpi.policies.policy_config import create_trained_policy, load_train_config
from openpi.transforms import resize_geometry, resize_image, resize_points
from scripts.preprocess.libero.render import BG, BORDER, GREEN, MUTED, PANEL, WHITE, font


PANEL_SIZE, LABEL_WIDTH, HEADER, GAP = 320, 180, 64, 12
ALPHA = .5


def default_layers(depth):
    """Third quartile of the action expert, where GuidedVLA reports the most object-focused heads."""
    return list(range(depth // 2, max(depth // 2 + 1, 3 * depth // 4)))


class AttentionCapture:
    """Action-query softmax over image keys, last denoising step, mean over heads and action steps."""

    def __init__(self, model, layers):
        self.model, self.layers = model, set(layers)
        self.image_keys = None
        self._layer = None
        self._maps = []

    @contextmanager
    def active(self):
        original_attend, original_velocity = pi0.attend, self.model.velocity

        def attend(q, k, v, mask):
            if self._layer is not None:
                if self._layer in self.layers:
                    self._record(q, k, mask)
                self._layer += 1
            return original_attend(q, k, v, mask)

        def velocity(*args, **kwargs):
            # Every call is one flow step; later steps overwrite earlier ones.
            self._layer, self._maps = 0, []
            try:
                return original_velocity(*args, **kwargs)
            finally:
                self._layer = None

        pi0.attend, self.model.velocity = attend, velocity
        try:
            yield self
        finally:
            pi0.attend = original_attend
            del self.model.velocity

    def _record(self, q, k, mask):
        k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = scores.softmax(-1).mean((1, 2))  # [B, K]
        self._maps.append(probs[:, :self.image_keys])

    def result(self):
        if len(self._maps) != len(self.layers):
            raise RuntimeError(f"Captured {len(self._maps)} of {len(self.layers)} expert layers")
        return torch.stack(self._maps).mean(0)[0].cpu().numpy()  # [views * patches]


def load_config(checkpoint):
    return load_train_config(torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True))


@dataclass
class FrameView:
    images: list          # Model-input RGB per view, resized to image_size.
    attention: list       # Per-view [grid, grid] maps, each renormalized to sum 1.
    image_share: float    # Fraction of action-query attention on image keys.


@dataclass
class ModelRun:
    name: str
    image_size: int
    frames: dict = field(default_factory=dict)


def run_model(name, checkpoint, episode, *, device, layers=None, stride=1, seed=0, steps=10,
              tokenizer_path=None):
    config = load_config(checkpoint)
    # Inputs only: the same episode and views the checkpoint was trained on, without Event targets.
    dataset = EpisodeDataset(config.data_index, action_horizon=config.model.action_horizon,
                             cameras=config.cameras, auxiliary=False, episode_ids={episode})
    if len(dataset) == 0:
        raise ValueError(f"{name}: episode {episode} is not in {config.data_index}")
    policy = create_trained_policy(checkpoint, device=device, tokenizer_path=tokenizer_path)
    model, m = policy.model, policy.model.config
    layers = default_layers(m.depth) if layers is None else layers
    if not all(0 <= layer < m.depth for layer in layers):
        raise ValueError(f"{name}: layers must be within [0, {m.depth})")
    grid = m.image_size // m.patch_size
    capture = AttentionCapture(model, layers)
    run = ModelRun(name, m.image_size)
    with capture.active():
        for index in range(len(dataset)):
            sample = dataset[index]
            frame = int(sample["metadata"]["frame"])
            if frame % stride or frame in run.frames:
                continue
            observation = sample["inputs"]
            if config.model_type == 'eskill':
                # Current geometry only: attention inspection does not need Event or future labels.
                reader = dataset.reader(dataset.samples[index][0])
                start, valid = np.zeros((2, 2), np.float32), np.zeros(2, bool)
                for arm in reader.meta['arm_ids']:
                    pixels = reader.meta['effector_pixels'][arm][config.cameras[0]]
                    row = reader.row(pixels, frame)
                    if row is not None:
                        point, available = resize_points(reader.array(pixels['values'])[row:row + 1],
                            reader.array(pixels['valid'])[row:row + 1], observation['images'][0].shape, m.image_size)
                        start[ARM_SLOTS[arm]], valid[ARM_SLOTS[arm]] = point[0], available[0]
                observation.update(effector_start=start, effector_start_valid=valid)
            images = [resize_image(im, m.image_size) for im in observation["images"]]
            capture.image_keys = len(images) * grid * grid
            generator = torch.Generator().manual_seed(seed + frame)
            noise = torch.randn(m.action_horizon, m.action_dim, generator=generator).numpy()
            policy.infer(observation, noise=noise, steps=steps)
            values = capture.result()
            share = float(values.sum())
            maps = [v / max(v.sum(), 1e-12) for v in values.reshape(len(images), grid, grid)]
            run.frames[frame] = FrameView(images, maps, share)
    del policy, model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return run, config


class Annotations:
    """Event skill and current-object box per source frame, from a saved episodes index."""

    def __init__(self, index, episode, camera):
        import json
        entries = json.loads(Path(index).read_text())["episodes"]
        entry = next((e for e in entries if e["episode_id"] == episode), None)
        if entry is None or "path" not in entry:
            raise ValueError(f"Episode {episode} has no saved annotations in {index}")
        self.reader, self.camera = EpisodeReader(Path(index).parent / entry["path"]), camera
        stream = self.reader.meta["cameras"][camera]
        self.shape = (stream["height"], stream["width"])
        self.task = self.reader.meta["instructions"][0]

    def events(self, frame):
        return [(arm, self.reader.event_at(arm, frame)) for arm in self.reader.meta["arm_ids"]]

    def object_boxes(self, frame, size):
        """Source xyxy boxes mapped to the model's padded `size` input."""
        nh, nw, y, x = resize_geometry(self.shape, size)
        sy, sx = nh / self.shape[0], nw / self.shape[1]
        boxes = []
        for _, event in self.events(frame):
            if event is None:
                continue
            region = self.reader.region(event["object_regions"].get(self.camera), frame)
            if region is not None and region["bbox"] is not None:
                x0, y0, x1, y1 = region["bbox"]
                boxes.append((x0 * sx + x, y0 * sy + y, x1 * sx + x, y1 * sy + y))
        return boxes


def upsample(attention, size):
    value = np.asarray(Image.fromarray(attention.astype(np.float32)).resize(
        (size, size), Image.Resampling.BILINEAR))
    return value / max(value.sum(), 1e-12)


def box_mass(attention, boxes, size):
    """Share of one view's image attention inside the union of boxes (pixels of the model input)."""
    if not boxes:
        return None
    dense, inside = upsample(attention, size), np.zeros((size, size), bool)
    for x0, y0, x1, y1 in boxes:
        inside[max(0, math.floor(y0)):min(size, math.ceil(y1)),
               max(0, math.floor(x0)):min(size, math.ceil(x1))] = True
    return float(dense[inside].sum())


def overlay(image, attention):
    from matplotlib import colormaps
    dense = upsample(attention, PANEL_SIZE)
    heat = colormaps["jet"](dense / max(dense.max(), 1e-12))[..., :3] * 255
    base = np.asarray(Image.fromarray(image).resize((PANEL_SIZE, PANEL_SIZE), Image.Resampling.BILINEAR))
    return Image.fromarray(((1 - ALPHA) * base + ALPHA * heat).astype(np.uint8))


class Renderer:
    def __init__(self, runs, annotations, episode, *, views):
        self.runs, self.annotations, self.episode, self.views = runs, annotations, episode, views
        self.width = LABEL_WIDTH + views * (PANEL_SIZE + GAP) + GAP
        self.height = HEADER + len(runs) * (PANEL_SIZE + GAP) + GAP
        self.width += self.width % 2
        self.height += self.height % 2

    def frame(self, frame):
        canvas = Image.new("RGB", (self.width, self.height), BG)
        draw = ImageDraw.Draw(canvas)
        task = self.annotations.task if self.annotations else ""
        draw.text((GAP, 10), f"{self.episode}   frame {frame}   {task}", fill=WHITE, font=font(22))
        if self.annotations:
            skills = "   ".join(f'{arm}: {event["skill"] if event else "—"}'
                                for arm, event in self.annotations.events(frame))
            draw.text((GAP, 38), skills, fill=MUTED, font=font(18))
        for row, run in enumerate(self.runs):
            top = HEADER + row * (PANEL_SIZE + GAP)
            view = run.frames[frame]
            boxes = self.annotations.object_boxes(frame, run.image_size) if self.annotations else []
            mass = box_mass(view.attention[0], boxes, run.image_size)
            draw.rectangle((GAP, top, LABEL_WIDTH - GAP, top + PANEL_SIZE), fill=PANEL, outline=BORDER)
            draw.text((GAP * 2, top + 14), run.name, fill=WHITE, font=font(20, "bold"))
            if mass is not None:
                draw.text((GAP * 2, top + 52), "obj box", fill=MUTED, font=font(16))
                draw.text((GAP * 2, top + 74), f"{mass:.0%}", fill=GREEN, font=font(34, "bold"))
            for col, (image, attention) in enumerate(zip(view.images, view.attention)):
                left = LABEL_WIDTH + col * (PANEL_SIZE + GAP)
                panel = overlay(image, attention)
                if col == 0 and boxes:
                    scale = PANEL_SIZE / run.image_size
                    pen = ImageDraw.Draw(panel)
                    for box in boxes:
                        pen.rectangle(tuple(v * scale for v in box), outline=GREEN, width=3)
                canvas.paste(panel, (left, top))
        return np.asarray(canvas)


def write_video(renderer, frames, destination, fps):
    """Encode, then decode in full to check frame count and timing, as the review renderers do."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.mp4")
    try:
        with av.open(str(temporary), "w") as container:
            stream = container.add_stream("libx264", rate=Fraction(str(fps)))
            stream.width, stream.height, stream.pix_fmt = renderer.width, renderer.height, "yuv420p"
            stream.options = {"crf": "20", "preset": "fast", "threads": "2"}
            for frame in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(renderer.frame(frame), format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(temporary)) as container:
            count = sum(1 for _ in container.decode(video=0))
        if count != len(frames):
            raise ValueError(f"Rendered {count} of {len(frames)} frames")
        temporary.rename(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def render(models, episode, destination, *, annotations=None, camera=None, device="cuda", layers=None,
           stride=1, seed=0, steps=10, fps=10., tokenizer_path=None):
    """`models` is a list of (name, checkpoint); one row per model in the output video."""
    if not models:
        raise ValueError("Give at least one model")
    runs, configs = [], []
    for name, checkpoint in models:
        run, config = run_model(name, checkpoint, episode, device=device, layers=layers, stride=stride,
                                seed=seed, steps=steps, tokenizer_path=tokenizer_path)
        runs.append(run)
        configs.append(config)
    annotations = annotations or configs[0].data_index
    camera = camera or configs[0].cameras[0]
    labels = Annotations(annotations, episode, camera) if annotations else None
    frames = sorted(set.intersection(*(set(run.frames) for run in runs)))
    if not frames:
        raise ValueError("The models share no source frames for this episode")
    views = max(len(view.images) for run in runs for view in run.frames.values())
    return write_video(Renderer(runs, labels, episode, views=views), frames, destination, fps)
