"""Review Stage-1 horizontal reflection using real Episodes and runtime target transforms."""
import argparse
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw
import torch

from openpi.models.pi0_config import ESkillConfig
from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.policies.libero_policy import LiberoInputs
from openpi.transforms import horizontal_mirror, mirror_eligible, resize_image, role_canvases
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.render import BG, BORDER, GREEN, MUTED, PANEL, WHITE, YELLOW, Episode, font, render_video
from scripts.preprocess.libero.review_motion import CLOSE, OPEN, select_episodes


POINT_FIELDS = ('trace', 'trace_valid', 'gripper', 'gripper_valid',
                'skill_labels', 'skill_valid', 'role_valid', 'future_cross_event')


def review_batch(dataset, row, config):
    """Use the training spatial transforms; tokenization/normalization are irrelevant to this review."""
    sample = dataset[row]
    inputs = LiberoInputs(include_state=False)(sample['inputs'])
    inputs['images'] = np.stack([resize_image(rgb, config.image_size).transpose(2, 0, 1)
                                for rgb in inputs['images']]).astype(np.float32) / 127.5 - 1
    inputs.pop('prompt')
    targets = dict(sample['targets'])
    targets.update(role_canvases(targets.pop('role_sources'), config.canvas_size,
                                future_bbox_expand_px=config.future_bbox_expand_px))
    return dict(inputs={k: torch.from_numpy(v[None]) for k, v in inputs.items()},
                targets={k: torch.from_numpy(v[None]) for k, v in targets.items()},
                metadata=[dict(sample['metadata'], mirror_eligible=mirror_eligible(sample['inputs']['prompt']))])


def pair(dataset, row, config):
    original = review_batch(dataset, row, config)
    mirrored = horizontal_mirror(original, torch.tensor([original['metadata'][0]['mirror_eligible']]))
    return original, mirrored


def save_targets(dataset, rows, config, destination):
    records = []
    for row in rows:
        original, mirrored = pair(dataset, row, config)
        record = dict(action_row=original['metadata'][0]['action_row'], frame=original['metadata'][0]['frame'],
                      mirrored=mirrored['metadata'][0]['mirrored'])
        for label, sample in (('original', original), ('mirror', mirrored)):
            fields = dict(**sample['inputs'], **sample['targets'])
            for key in POINT_FIELDS:
                record[f'{label}_{key}'] = fields[key][0].numpy()
        records.append(record)
    np.savez_compressed(destination, **{key: np.stack([row[key] for row in records]) for key in records[0]})


def rgb_image(value, size):
    rgb = ((value.detach().numpy().transpose(1, 2, 0) + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)


def box_outline(image, weight, color):
    y, x = np.where(weight.numpy() > 0)
    if len(x):
        scale = image.width / weight.shape[-1]
        box = (x.min()*scale, y.min()*scale, (x.max()+1)*scale-1, (y.max()+1)*scale-1)
        ImageDraw.Draw(image).rectangle(box, outline=color, width=2)


class MirrorRenderer:
    def __init__(self, dataset, rows, config, labels, task):
        self.dataset, self.rows, self.config, self.labels, self.task = dataset, rows, config, labels, task

    def frame(self, index):
        canvas = Image.new('RGB', (1920, 1340), BG)
        draw = ImageDraw.Draw(canvas)
        for line, text in enumerate(textwrap.wrap(self.task, 104)):
            draw.text((32, 16+line*35), text, fill=WHITE, font=font(26, 'bold'))
        batches = pair(self.dataset, self.rows[index], self.config)
        frame = int(self.labels['frame'][index])
        draw.text((32, 108), f'FRAME {frame} | ACTION ROW {self.labels["action_row"][index]} | GT TRAINING TARGETS / NOT PREDICTIONS',
                  fill=MUTED, font=font(18, 'mono'))
        for side, ((label, title), batch) in enumerate(zip((('original', 'ORIGINAL'), ('mirror', 'HORIZONTAL MIRROR')), batches)):
            x = side * 960
            flipped = bool(self.labels['mirrored'][index])
            if side and not flipped:
                title = 'ORIGINAL ONLY / DIRECTIONAL LANGUAGE'
            draw.rounded_rectangle((x+20, 155, x+940, 1275), radius=8, fill=PANEL, outline=BORDER)
            draw.text((x+40, 165), title, fill=WHITE, font=font(22, 'bold'))
            images, targets = batch['inputs']['images'][0], batch['targets']
            main = rgb_image(images[0], 576)
            box_outline(main, targets['role_weights'][0, 0, 0], GREEN)
            points = ImageDraw.Draw(main)
            get = lambda key: self.labels[f'{label}_{key}'][index, 0]
            previous = None
            for i, (xy, valid) in enumerate(zip(get('trace'), get('trace_valid'))):
                if not valid:
                    previous = None
                    continue
                px, py = xy * 576 - .5
                color = (CLOSE if get('gripper')[i] else OPEN) if get('gripper_valid')[i] else WHITE
                if previous is not None:
                    points.line((*previous, px, py), fill=WHITE, width=2)
                outline = YELLOW if get('future_cross_event')[i] else color
                points.ellipse((px-5, py-5, px+5, py+5), fill=color, outline=outline, width=2)
                points.text((px+7, py-14), str(i+1), fill=color, font=font(17, 'mono'), stroke_width=1, stroke_fill=BG)
                previous = (px, py)
            canvas.paste(main, (x+40, 210))
            canvas.paste(rgb_image(images[1], 256), (x+650, 210))
            draw.text((x+650, 478), 'WRIST / SAME FLIP', fill=MUTED, font=font(17, 'mono'))
            draw.text((x+650, 503), 'RGB + LANGUAGE ONLY', fill=WHITE, font=font(17, 'mono'))
            draw.text((x+650, 530), 'OFFSET  XY   CMD', fill=MUTED, font=font(17, 'mono'))
            for i, offset in enumerate(self.config.motion_offsets_frames):
                command = ('CLOSE' if get('gripper')[i] else 'OPEN') if get('gripper_valid')[i] else '--'
                color = CLOSE if command == 'CLOSE' else OPEN if command == 'OPEN' else MUTED
                status = ('X' if get('future_cross_event')[i] else 'OK') if get('trace_valid')[i] else '--'
                text = f'+{offset:02d}f    {status:2s}  {command}'
                draw.text((x+650, 570+36*i), text, fill=color, font=font(17, 'mono'))
            skill = int(get('skill_labels'))
            name = self.config.skill_names[skill] if get('skill_valid') else 'UNKNOWN'
            draw.text((x+40, 804), f'SKILL: {name} | WHITE: current EEF', fill=WHITE, font=font(20, 'bold'))
            for role, (title, color) in enumerate((('CURRENT OBJECT', GREEN), ('EVENT GOAL / +8px', YELLOW))):
                left = x+40+role*300
                draw.text((left, 850), title, fill=color, font=font(18, 'bold'))
                image = rgb_image(targets['role_images'][0, 0, role], 256)
                box_outline(image, targets['role_weights'][0, 0, role], color)
                canvas.paste(image, (left, 890))
                if not bool(targets['role_valid'][0, 0, role]):
                    draw.text((left+20, 1000), 'MISSING', fill=WHITE, font=font(24, 'bold'))
            draw.text((x+650, 920), 'CYAN: OPEN', fill=OPEN, font=font(19, 'bold'))
            draw.text((x+650, 960), 'PINK: CLOSE', fill=CLOSE, font=font(19, 'bold'))
            draw.text((x+40, 1170), 'Green: Current target extent, including original margin.', fill=MUTED, font=font(18))
            draw.text((x+40, 1210), f'X / yellow ring: cross Event within +{self.config.motion_min_horizon_frames}f. Goal: current Event.',
                      fill=MUTED, font=font(17))
        draw.text((32, 1294), 'MIRROR: 8D BODY STATE OFF | arm identity / gripper command unchanged | no mirrored Action loss',
                  fill=MUTED, font=font(19, 'mono'))
        return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Report root on data disk; videos go in video/')
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    entries = select_episodes(args.index, args.count)
    config = ESkillConfig()
    dataset = EpisodeDataset(args.index, action_horizon=config.action_horizon, cameras=('image', 'wrist_image'),
                             skill_names=config.skill_names, episode_ids={e['episode_id'] for e in entries})
    records = []
    for i, entry in enumerate(entries, 1):
        ep = next(j for j, meta in enumerate(dataset.metadata) if meta['episode_id'] == entry['episode_id'])
        rows = [j for j, (episode, _) in enumerate(dataset.samples) if episode == ep]
        name = f'{i:02d}_{entry["episode_id"]}'
        archive = args.output / f'{name}.npz'
        save_targets(dataset, rows, config, archive)
        with np.load(archive) as saved:
            labels = {key: saved[key] for key in saved.files}
        task = dataset.metadata[ep]['instructions'][0]
        renderer = MirrorRenderer(dataset, rows, config, labels, task)
        video = args.output / 'video' / f'{name}.mp4'
        view = Episode(key=entry['episode_id'], frame_count=len(rows), task=task, playback_fps=10.)
        result = render_video(view, video, renderer=renderer)
        crossing = (labels['original_future_cross_event'] & labels['original_trace_valid']).any((1, 2))
        best = int(np.flatnonzero(crossing)[0] if crossing.any() else labels['original_trace_valid'].sum((1, 2)).argmax())
        preview = args.output / f'{name}.jpg'
        Image.fromarray(renderer.frame(best)).save(preview)
        result.update(task=task, mirror_eligible=mirror_eligible(task), video=str(video), preview=str(preview),
                      episode=str(dataset.paths[ep]), episode_sha256=digest(dataset.paths[ep]),
                      labels=str(archive), labels_sha256=digest(archive),
                      skills=sorted({e['skill'] for e in dataset.metadata[ep]['events']}))
        result['cross_event_valid_points'] = int((labels['original_future_cross_event'] & labels['original_trace_valid']).sum())
        records.append(result)
        print(json.dumps(dict(completed=i, total=len(entries), video=str(video))), flush=True)
    all_meta = [json.loads((args.index.parent / entry['path']).read_text())
                for entry in json.loads(args.index.read_text())['episodes']]
    excluded = sorted({meta['instructions'][0] for meta in all_meta if not mirror_eligible(meta['instructions'][0])})
    manifest = dict(manual_review='pending', source='saved Episodes; shared runtime geometry transforms',
                    offsets_frames=list(config.motion_offsets_frames), future_bbox_expand_px=config.future_bbox_expand_px,
                    min_horizon_frames=config.motion_min_horizon_frames,
                    preview='eligible samples always mirrored on right; actual training probability=0.5',
                    mirror_body_state='8D Motion state projection, including bias, gated off',
                    excluded_tasks=excluded, samples=records)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
