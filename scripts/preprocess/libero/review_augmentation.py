"""Compare appearance/geometry recipes on ten trajectories using the training transform."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter
import torch

from openpi.augmentation import augment_eskill_batch
from openpi.training.config import TrainConfig, split_episodes
from openpi.training.data_loader import episode_dataset
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.render import BG, GREEN, MUTED, WHITE, Episode, font, render_video
from scripts.preprocess.libero.review_mirror import box_outline, review_batch, rgb_image
from scripts.preprocess.libero.review_motion import select_episodes


class AugmentationRenderer:
    def __init__(self, dataset, rows, config, color, crop, seed, rotation=None, mask_outline=False):
        self.dataset, self.rows, self.config = dataset, rows, config
        self.seed = seed
        self.mask_outline = mask_outline
        self.variants = [('a0', 'A0 ORIGINAL', None)]
        self.variants += ([('a2', 'A2 COLOR + CROP', crop), ('a3', 'A3 + ROTATION', rotation)] if rotation
                         else [('a1', 'A1 COLOR', color), ('a2', 'A2 COLOR + CROP', crop)])
        self.records = {}

    def frame(self, index):
        original = review_batch(self.dataset, self.rows[index], self.config.model)
        variants = [original]
        for _, _, recipe in self.variants[1:]:
            # Fix one draw per trajectory for a stable review; training draws per sample.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed)
                variants.append(augment_eskill_batch(original, recipe.image_augmentation))
        task = self.dataset.metadata[self.dataset.samples[self.rows[index]][0]]['instructions'][0]
        canvas = Image.new('RGB', (1920, 1340), BG)
        draw = ImageDraw.Draw(canvas)
        for line, text in enumerate(textwrap.wrap(task, 105)):
            draw.text((24, 12 + 32 * line), text, fill=WHITE, font=font(24, 'bold'))
        frame = int(original['metadata'][0]['frame'])
        draw.text((24, 86), f'FRAME {frame} | GREEN: CURRENT OBJECT REGION | CYAN: FUTURE EEF LABELS | NOT PREDICTIONS',
                  fill=MUTED, font=font(18, 'mono'))
        record = dict(frame=frame, action_row=original['metadata'][0]['action_row'])
        for column, ((tag, name, _), value) in enumerate(zip(self.variants, variants)):
            x = column * 640
            images, targets = value['inputs']['images'][0], value['targets']
            draw.text((x + 24, 134), name, fill=WHITE, font=font(24, 'bold'))
            main = rgb_image(images[0], 576)
            if targets['role_valid'][0, 0, 0]:
                weights = targets['role_weights'][0, 0, 0]
                if self.mask_outline:
                    mask = Image.fromarray((weights.numpy() > .5).astype(np.uint8) * 255).resize(
                        main.size, Image.Resampling.NEAREST)
                    edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(3)),
                                               mask.filter(ImageFilter.MinFilter(3)))
                    main.paste(GREEN, (0, 0, *main.size), edge)
                else:
                    box_outline(main, weights, GREEN)
            points = ImageDraw.Draw(main)
            for k, (xy, valid) in enumerate(zip(targets['trace'][0, 0], targets['trace_valid'][0, 0])):
                if valid:
                    px, py = xy.numpy() * 576 - .5
                    points.ellipse((px - 4, py - 4, px + 4, py + 4), fill='#70d8f4')
                    points.text((px + 6, py - 10), str(k + 1), fill=WHITE, font=font(16, 'mono'),
                                stroke_width=1, stroke_fill=BG)
            canvas.paste(main, (x + 24, 180))
            for role, label in enumerate(('CURRENT TARGET', 'FUTURE TARGET')):
                left = x + 24 + 288 * role
                draw.text((left, 784), label, fill=MUTED, font=font(18, 'mono'))
                canvas.paste(rgb_image(targets['role_images'][0, 0, role], 256), (left, 818))
            canvas.paste(rgb_image(images[1], 160), (x + 24, 1110))
            draw.text((x + 208, 1110), 'WRIST / COLOR ONLY', fill=MUTED, font=font(16, 'mono'))
            draw.text((x + 208, 1140), f'VALID TRACE: {int(targets["trace_valid"][0, 0].sum())}/8',
                      fill=WHITE, font=font(17, 'mono'))
            extra = value['metadata'][0].get('augmentation')
            if extra:
                draw.text((x + 208, 1172), 'CROP ' + ' '.join(f'{v:.3f}' for v in extra['crop']),
                          fill=MUTED, font=font(14, 'mono'))
                draw.text((x + 208, 1200), f'ROLL {extra["rotation_degrees"]:+.2f} DEG / CW+',
                          fill=MUTED, font=font(16, 'mono'))
                record[f'{tag}_crop'] = extra['crop']
                record[f'{tag}_rotation_degrees'] = extra['rotation_degrees']
                record[f'{tag}_color_factors'] = extra['color_factors']
                record[f'{tag}_counts'] = extra['counts']
            for key in ('trace', 'trace_valid', 'role_valid', 'gripper', 'gripper_valid'):
                record[f'{tag}_{key}'] = targets[key][0].numpy()
        draw.text((24, 1300), 'ACTION / SKILL / GRIPPER COMMANDS UNCHANGED; VAE TARGETS KEEP ORIGINAL COLOR',
                  fill=MUTED, font=font(18, 'mono'))
        self.records[index] = record
        return np.asarray(canvas)


def save_snapshots(dataset, base, color, crop, rotation, output, seed, count):
    """Half nearest-edge and half random saved frames, with independent augmentation draws."""
    candidates = []
    lookup = {(ep, row): i for i, (ep, row) in enumerate(dataset.samples)}
    for path in sorted(output.glob('*.npz')):
        with np.load(path) as labels:
            identity = path.stem.split('_', 1)[1]
            ep = next(i for i, meta in enumerate(dataset.metadata) if meta['episode_id'] == identity)
            points, valid = labels['a0_trace'], labels['a0_trace_valid']
            margin = np.minimum(points, 1 - points).min(-1)
            margin = np.where(valid, margin, np.inf).min((1, 2))
            candidates.extend((lookup[(ep, int(row))], float(edge))
                              for row, edge in zip(labels['action_row'], margin))
    ordered = sorted(candidates, key=lambda value: value[1])
    count = min(count, len(ordered))
    near = ordered[:count // 2]
    rng = np.random.default_rng(seed)
    pool = ordered[count // 2:]
    chosen = near + [pool[i] for i in rng.choice(len(pool), count - len(near), replace=False)]
    destination = output / 'samples'
    destination.mkdir(exist_ok=True)
    records = []
    for i, (row, margin) in enumerate(chosen):
        draw_seed = seed + 10000 + i
        renderer = AugmentationRenderer(dataset, [row], base, color, crop, draw_seed, rotation, mask_outline=True)
        frame = renderer.frame(0)
        record = renderer.records[0]
        path = destination / f'{i + 1:02d}.jpg'
        Image.fromarray(frame).save(path)
        np.savez_compressed(path.with_suffix('.npz'), **{
            key: value for key, value in record.items() if not key.endswith('_counts')})
        records.append(dict(image=str(path), episode_id=dataset.metadata[dataset.samples[row][0]]['episode_id'],
                            frame=record['frame'], seed=draw_seed,
                            selection='near_edge' if i < len(near) else 'random',
                            source_edge_margin=margin if np.isfinite(margin) else None,
                            rotation_degrees=record.get('a3_rotation_degrees', 0.)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='experiments/eskill_libero40_direct/train.json', type=Path)
    parser.add_argument('--color', default='experiments/eskill_libero40_direct/color.json', type=Path)
    parser.add_argument('--crop', default='experiments/eskill_libero40_direct/color_crop.json', type=Path)
    parser.add_argument('--rotation', type=Path, help='Show A0/A2/A3 instead of A0/A1/A2')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--snapshots', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    base, color, crop = (TrainConfig.load(path) for path in (args.base, args.color, args.crop))
    rotation = TrainConfig.load(args.rotation) if args.rotation else None
    # Review selection uses the existing skill/task coverage rule, restricted to train.
    selected = set(split_episodes(base.data_index, base.validation_fraction, base.seed, base.split_path)['train'])
    entries = select_episodes(Path(base.data_index), args.count, episode_ids=selected)
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = episode_dataset(base, {row['episode_id'] for row in entries})
    results = []
    for i, entry in enumerate(entries):
        episode = next(j for j, meta in enumerate(dataset.metadata) if meta['episode_id'] == entry['episode_id'])
        rows = [j for j, (ep, _) in enumerate(dataset.samples) if ep == episode][::args.stride]
        name = f'{i + 1:02d}_{entry["episode_id"]}'
        renderer = AugmentationRenderer(dataset, rows, base, color, crop, args.seed + i, rotation)
        video = args.output / 'video' / (name + '.mp4')
        view = Episode(key=entry['episode_id'], frame_count=len(rows), playback_fps=10.)
        result = render_video(view, video, renderer=renderer)
        records = [renderer.records[j] for j in range(len(rows))]
        # Coordinates/validity and exact transform draws are saved beside the videos.
        arrays = {key: np.stack([row[key] for row in records])
                  for key in records[0] if not key.endswith('_counts')}
        np.savez_compressed(args.output / (name + '.npz'), **arrays)
        Image.fromarray(renderer.frame(len(rows) // 2)).save(args.output / (name + '.jpg'))
        for tag, _, _ in renderer.variants[1:]:
            counts = [row[f'{tag}_counts'] for row in records]
            result[f'{tag}_counts'] = {key: sum(row[key] for row in counts) for key in counts[0]}
        result.update(episode_id=entry['episode_id'], seed=args.seed + i, stride=args.stride,
                      skills=sorted({event['skill'] for event in dataset.metadata[episode]['events']}),
                      source=str(dataset.paths[episode]), source_sha256=digest(dataset.paths[episode]), video=str(video))
        results.append(result)
        print(json.dumps(dict(completed=i + 1, count=len(entries), video=str(video))), flush=True)
    snapshots = save_snapshots(dataset, base, color, crop, rotation, args.output, args.seed, args.snapshots)
    manifest = dict(subset='train', technical=True, seed=args.seed,
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    git_status=subprocess.check_output(['git', 'status', '--short'], text=True),
                    base=base.to_dict(), color=asdict(color.image_augmentation), crop=asdict(crop.image_augmentation),
                    rotation=asdict(rotation.image_augmentation) if rotation else None,
                    columns=[name for _, name, _ in renderer.variants],
                    episodes=results, snapshots=snapshots)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
