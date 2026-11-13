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
import json
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

import openpi.models.pi0 as pi0
from openpi.policies.eskill_dataset import EpisodeDataset, EpisodeReader
from openpi.policies.policy_config import create_trained_policy, load_train_config
from openpi.transforms import resize_geometry
from scripts.preprocess.libero.render import BG, BORDER, GREEN, MUTED, PANEL, WHITE, font


PANEL_SIZE, LABEL_WIDTH, HEADER, GAP = 320, 240, 90, 12
ALPHA = .5


def percent(value):
    """Keep small but nonzero structural attention visible."""
    return f'{value:.3%}' if 0 < value < .01 else f'{value:.1%}'


def default_layers(depth):
    """Third quartile of the action expert, where GuidedVLA reports the most object-focused heads."""
    return list(range(depth // 2, max(depth // 2 + 1, 3 * depth // 4)))


class AttentionCapture:
    """Action-query softmax over image keys, last denoising step, mean over heads and action steps."""

    def __init__(self, model, layers, *, language=None):
        self.model, self.layers = model, set(layers)
        self.language, self._prefix_layer = language, None
        self.image_keys = None
        self._layer = None
        self._maps = []
        self.inputs, self.prediction = None, None
        self.prefix_keys = self.condition_keys = 0

    @contextmanager
    def active(self):
        original_attend, original_velocity = pi0.attend, self.model.velocity
        original_encode, original_predict = self.model.encode_observation, self.model.predict_condition

        def encode(inputs):
            self.inputs = inputs
            if self.language is not None:
                self.image_keys = inputs['images'].shape[1] * (self.model.config.image_size // self.model.config.patch_size) ** 2
                self.language.prepare(inputs, self.image_keys)
                self._prefix_layer = 0
            try:
                prefix = original_encode(inputs)
            finally:
                self._prefix_layer = None
            self.prefix_keys = prefix.mask.shape[1]
            self.image_keys = self.prefix_keys - inputs['tokens'].shape[1]
            return prefix

        def predict(*args, **kwargs):
            condition, self.prediction = original_predict(*args, **kwargs)
            return condition, self.prediction

        def attend(q, k, v, mask):
            if self._layer is not None:
                if self._layer in self.layers:
                    self._record(q, k, mask)
                self._layer += 1
            elif self._prefix_layer is not None:
                if self._prefix_layer in self.layers:
                    self.language.record(q, k, mask)
                self._prefix_layer += 1
            return original_attend(q, k, v, mask)

        def velocity(noisy_actions, time, condition):
            # Every call is one flow step; later steps overwrite earlier ones.
            self._layer, self._maps = 0, []
            self.condition_keys = condition.mask.shape[1]
            try:
                return original_velocity(noisy_actions, time, condition)
            finally:
                self._layer = None

        with patch.object(pi0, 'attend', attend), patch.object(self.model, 'velocity', velocity), \
             patch.object(self.model, 'encode_observation', encode), patch.object(self.model, 'predict_condition', predict):
            yield self

    def _record(self, q, k, mask):
        k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
        scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
        probs = scores.softmax(-1).mean((1, 2))  # [B, K]
        self._maps.append(probs)

    def result(self):
        return self.distribution()[:self.image_keys]

    def distribution(self):
        if len(self._maps) != len(self.layers):
            raise RuntimeError(f"Captured {len(self._maps)} of {len(self.layers)} expert layers")
        return torch.stack(self._maps).mean(0)[0].cpu().numpy()

    def key_mass(self):
        values = self.distribution()
        return dict(image=float(values[:self.image_keys].sum()),
                    text=float(values[self.image_keys:self.prefix_keys].sum()),
                    motion=float(values[self.prefix_keys:self.condition_keys].sum()),
                    action=float(values[self.condition_keys:].sum()))

    def motion_attention(self):
        if self.condition_keys == self.prefix_keys:
            return None
        return self.distribution()[self.prefix_keys:self.condition_keys].reshape(2, self.model.config.motion_queries)

    def frame_view(self):
        images = ((self.inputs['images'][0].float().cpu().numpy().transpose(0, 2, 3, 1) + 1) * 127.5)
        images = images.round().clip(0, 255).astype(np.uint8)
        grid = self.model.config.image_size // self.model.config.patch_size
        values = self.result().reshape(len(images), grid, grid)
        maps = [value / max(value.sum(), 1e-12) for value in values]
        return FrameView(list(images), maps, self.key_mass(), values.sum((1, 2)).tolist(), self.motion_attention(),
                         self.language.result() if self.language is not None else None)


def load_config(checkpoint):
    return load_train_config(torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True))


@dataclass
class FrameView:
    images: list          # Model-input RGB per view, resized to image_size.
    attention: list       # Per-view [grid, grid] maps, each renormalized to sum 1.
    key_mass: dict        # Unconditional mass on image/text/Motion/action keys, summing to one.
    view_mass: list       # Per-camera share of all action attention, before heatmap normalization.
    motion_attention: object = None  # [2, queries], unconditional; absent arms have zero mass.
    language_attention: object = None  # Target/Object/Instruction VLM queries, independently of Action.


@dataclass
class ModelRun:
    name: str
    image_size: int
    frames: dict = field(default_factory=dict)
    cameras: tuple = ()


def run_model(name, checkpoint, episode, *, device, layers=None, stride=1, seed=0, steps=10,
              tokenizer_path=None, data_index=None, inference_precision=None):
    config = load_config(checkpoint)
    # Inputs only: the same episode and views the checkpoint was trained on, without Event targets.
    dataset = EpisodeDataset(data_index or config.data_index, action_horizon=config.model.action_horizon,
                             cameras=config.cameras, auxiliary=False, episode_ids={episode},
                             include_state=config.model.discrete_state_input)
    if len(dataset) == 0:
        raise ValueError(f"{name}: episode {episode} is not in {config.data_index}")
    policy = create_trained_policy(checkpoint, device=device, tokenizer_path=tokenizer_path,
                                   inference_precision=inference_precision)
    model, m = policy.model, policy.model.config
    layers = default_layers(m.depth) if layers is None else layers
    if not all(0 <= layer < m.depth for layer in layers):
        raise ValueError(f"{name}: layers must be within [0, {m.depth})")
    capture = AttentionCapture(model, layers)
    run = ModelRun(name, m.image_size, cameras=config.cameras)
    with capture.active():
        for index in range(len(dataset)):
            sample = dataset[index]
            frame = int(sample["metadata"]["frame"])
            if frame % stride or frame in run.frames:
                continue
            observation = sample["inputs"]
            generator = torch.Generator().manual_seed(seed + frame)
            noise = torch.randn(m.action_horizon, m.action_dim, generator=generator).numpy()
            policy.infer(observation, noise=noise, steps=steps)
            run.frames[frame] = capture.frame_view()
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
        title = f"{self.episode}   frame {frame}   {task}"
        size = 22
        while size > 12 and draw.textlength(title, font=font(size)) > self.width - 2 * GAP:
            size -= 1
        draw.text((GAP, 10), title, fill=WHITE, font=font(size))
        if self.annotations:
            skills = "   ".join(f'{arm}: {event["skill"] if event else "—"}'
                                for arm, event in self.annotations.events(frame))
            draw.text((GAP, 38), skills, fill=MUTED, font=font(18))
        draw.text((GAP, 65), 'CONDITIONAL HEATMAPS | Q/K softmax, last FM step; averaged layers/heads/action queries',
                  fill=MUTED, font=font(13))
        for row, run in enumerate(self.runs):
            top = HEADER + row * (PANEL_SIZE + GAP)
            view = run.frames[frame]
            boxes = self.annotations.object_boxes(frame, run.image_size) if self.annotations else []
            annotated = (run.cameras.index(self.annotations.camera) if self.annotations and self.annotations.camera in run.cameras
                         else 0 if not run.cameras else None)
            mass = box_mass(view.attention[annotated], boxes, run.image_size) if annotated is not None else None
            draw.rectangle((GAP, top, LABEL_WIDTH - GAP, top + PANEL_SIZE), fill=PANEL, outline=BORDER)
            draw.text((GAP * 2, top + 14), run.name, fill=WHITE, font=font(20, "bold"))
            if mass is not None:
                draw.text((GAP * 2, top + 48), f"OBJ / VIEW: {percent(mass)}", fill=GREEN, font=font(16, 'mono'))
                draw.text((GAP * 2, top + 74), f"OBJ / ALL: {percent(mass * view.view_mass[annotated])}",
                          fill=GREEN, font=font(16, 'mono'))
            for i, (name, share) in enumerate(view.key_mass.items()):
                draw.text((GAP * 2, top + 120 + i * 30), f'{name.upper():6s} {percent(share)}',
                          fill=WHITE, font=font(17, 'mono'))
            if view.motion_attention is not None:
                draw.text((GAP * 2, top + 256), 'MOTION BY ARM', fill=MUTED, font=font(14, 'mono'))
                draw.text((GAP * 2, top + 282), ' / '.join(percent(v) for v in view.motion_attention.sum(1)),
                          fill=WHITE, font=font(16, 'mono'))
            for col, (image, attention) in enumerate(zip(view.images, view.attention)):
                left = LABEL_WIDTH + col * (PANEL_SIZE + GAP)
                panel = overlay(image, attention)
                camera = run.cameras[col] if run.cameras else f'view {col}'
                ImageDraw.Draw(panel).text((8, 8), f'{camera}: {percent(view.view_mass[col])} / ALL',
                                           fill=WHITE, font=font(16, 'mono'), stroke_width=2, stroke_fill=BG)
                if col == annotated and boxes:
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
           stride=1, seed=0, steps=10, fps=10., tokenizer_path=None, data_index=None, inference_precision=None):
    """`models` is a list of (name, checkpoint); one row per model in the output video."""
    if not models:
        raise ValueError("Give at least one model")
    runs, configs = [], []
    for name, checkpoint in models:
        run, config = run_model(name, checkpoint, episode, device=device, layers=layers, stride=stride,
                                seed=seed, steps=steps, tokenizer_path=tokenizer_path,
                                data_index=data_index, inference_precision=inference_precision)
        runs.append(run)
        configs.append(config)
    annotations = annotations or data_index or configs[0].data_index
    camera = camera or configs[0].cameras[0]
    labels = Annotations(annotations, episode, camera) if annotations else None
    frames = sorted(set.intersection(*(set(run.frames) for run in runs)))
    if not frames:
        raise ValueError("The models share no source frames for this episode")
    views = max(len(view.images) for run in runs for view in run.frames.values())
    renderer = Renderer(runs, labels, episode, views=views)
    path = write_video(renderer, frames, destination, fps)
    Image.fromarray(renderer.frame(frames[0])).save(path.with_suffix('.jpg'))
    report = dict(episode=episode, frames=frames, seed=seed, fm_steps=steps,
                  checkpoints={name: str(checkpoint) for name, checkpoint in models},
                  layers=layers if layers is not None else {run.name: default_layers(c.model.depth)
                                                           for run, c in zip(runs, configs)},
                  inference_precision=inference_precision or {run.name: c.precision for run, c in zip(runs, configs)},
                  method='Reconstructed float32 Q/K softmax; last FM step; mean selected layers/heads/action queries',
                  heatmaps='Conditional on each view; absolute shares shown separately',
                  models={run.name: {str(frame): dict(key_mass=run.frames[frame].key_mass,
                      view_mass=run.frames[frame].view_mass,
                      motion_attention=None if run.frames[frame].motion_attention is None
                      else run.frames[frame].motion_attention.tolist()) for frame in frames} for run in runs})
    path.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    arrays = dict(frames=np.array(frames))
    for i, run in enumerate(runs):
        arrays[f'model_{i}_images'] = np.stack([run.frames[n].images for n in frames])
        arrays[f'model_{i}_attention'] = np.stack([run.frames[n].attention for n in frames])
        arrays[f'model_{i}_key_mass'] = np.array([list(run.frames[n].key_mass.values()) for n in frames])
        arrays[f'model_{i}_view_mass'] = np.array([run.frames[n].view_mass for n in frames])
        if run.frames[frames[0]].motion_attention is not None:
            arrays[f'model_{i}_motion_attention'] = np.stack([run.frames[n].motion_attention for n in frames])
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    return path
