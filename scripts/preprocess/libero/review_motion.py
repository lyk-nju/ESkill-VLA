"""Save and replay ten frame-based Motion targets using the shared LIBERO renderer."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from openpi.policies.eskill_dataset import EpisodeReader, event_motion_targets, gripper_semantics
from openpi.transforms import resize_image
from scripts.preprocess.artifacts import digest
from scripts.preprocess.libero.render import BG, BORDER, MUTED, PANEL, WHITE, YELLOW, Renderer, font, render_video
from scripts.preprocess.libero.view import from_saved

OPEN, CLOSE = '#7de0ed', '#ee91ba'
HOLD = '#b4a4ff'


def select_episodes(index, count, *, episode_ids=None):
    """Cover all recorded skills first, then spread remaining samples across suites/tasks."""
    records = [(entry, json.loads((index.parent / entry['path']).read_text()))
               for entry in json.loads(index.read_text())['episodes']
               if episode_ids is None or entry['episode_id'] in episode_ids]
    ordered = sorted(records, key=lambda row: hashlib.sha256(row[0]['episode_id'].encode()).hexdigest())
    selected = []
    skills, tasks = set(), set()
    for skill in ('Push', 'Turn', 'Open', 'Close', 'Pick', 'Place'):
        if skill in skills or len(selected) >= count:
            continue
        choice = next((row for row in ordered if row not in selected and row[1]['instructions'][0] not in tasks
                       and any(e['skill'] == skill for e in row[1]['events'])), None)
        if choice is not None:
            selected.append(choice)
            tasks.add(choice[1]['instructions'][0])
            skills.update(e['skill'] for e in choice[1]['events'])
    while len(selected) < min(count, len(records)):
        suites = Counter(row[0]['episode_id'].split('_no_noops')[0] for row in selected)
        candidates = [row for row in ordered if row not in selected and row[1]['instructions'][0] not in tasks]
        if not candidates:
            candidates = [row for row in ordered if row not in selected]
        choice = min(candidates, key=lambda row: suites[row[0]['episode_id'].split('_no_noops')[0]])
        selected.append(choice)
        tasks.add(choice[1]['instructions'][0])
    return [entry for entry, _ in selected]


def save_targets(reader, path, offsets, *, min_horizon_frames=10, terminal_hold=False):
    rows = []
    convention = gripper_semantics(reader.meta)
    for frame in range(reader.meta['frame_count']):
        targets, metadata = event_motion_targets(reader, frame, convention, offsets_frames=offsets,
                                                      min_horizon_frames=min_horizon_frames,
                                                      terminal_hold=terminal_hold)
        rows.append(dict(**targets, future_camera_rows=metadata['future_camera_rows'],
                         future_source_frame_indices=metadata['future_source_frame_indices'],
                         trace_source_frame_indices=metadata['trace_source_frame_indices']))
    arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(path, offsets_frames=offsets, min_horizon_frames=min_horizon_frames,
                        terminal_hold=terminal_hold, **arrays)


class MotionRenderer(Renderer):
    """Keep the shared task/object/goal panels; replace lower plots with training targets."""
    def __init__(self, episode, labels):
        super().__init__(episode)
        self.labels = labels

    def frame(self, index):
        canvas = Image.fromarray(super().frame(index))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((20, 744, 1900, 1275), fill=BG)
        offsets = self.labels['offsets_frames']
        draw.text((28, 754), f'SAVED TRAINING TARGETS / {len(offsets)} POINTS, STRIDE {offsets[0]} FRAMES / NOT MODEL PREDICTIONS',
                  font=font(22, 'bold'), fill=WHITE)
        draw.rounded_rectangle((28, 790, 666, 1268), radius=6, fill=PANEL, outline=BORDER)
        rgb = resize_image(self.episode.rgb(index), 224)
        image = Image.fromarray(rgb).resize((420, 420), Image.Resampling.BILINEAR)
        points = ImageDraw.Draw(image)
        labels = self.labels
        xy, valid = labels['trace'][index, 0], labels['trace_valid'][index, 0]
        grip, gv = labels['gripper'][index, 0], labels['gripper_valid'][index, 0]
        held = labels.get('trace_terminal_hold', np.zeros_like(labels['trace_valid']))[index, 0]
        sources = labels.get('trace_source_frame_indices', labels['future_source_frame_indices'])[index, 0]
        previous = None
        for i, (value, good) in enumerate(zip(xy, valid)):
            if not good:
                previous = None
                continue
            x, y = value * 420 - .5
            color = HOLD if held[i] else (CLOSE if grip[i] else OPEN) if gv[i] else WHITE
            if previous is not None:
                points.line((*previous, x, y), fill='#c9c9c9', width=2)
            outline = YELLOW if labels['future_cross_event'][index, 0, i] else BG
            points.ellipse((x-5, y-5, x+5, y+5), fill=color, outline=outline, width=2)
            # Include the observed endpoint in the same label as its repeated slots.
            repeated = held.any() and sources[i] == sources[held][0]
            earlier = (sources[:i] == sources[i]).any()
            if not (repeated and earlier):
                label = f'{i+1}-{len(xy)}' if repeated else str(i + 1)
                points.text((x+6, y-13), label, font=font(16, 'mono'), fill=color, stroke_width=1, stroke_fill=BG)
            previous = (x, y)
        canvas.paste(image, (48, 812))
        for y, label, color in ((850, 'NO CMD', WHITE), (908, 'OPEN CMD', OPEN), (966, 'CLOSE CMD', CLOSE)):
            draw.ellipse((486, y+4, 496, y+14), fill=color)
            draw.text((506, y), label, font=font(15, 'mono'), fill=color)
        draw.text((480, 1060), 'Yellow: cross Event', font=font(14, 'mono'), fill=YELLOW)
        draw.text((480, 1100), 'Purple: 2D HOLD', font=font(15, 'mono'), fill=HOLD)
        draw.text((48, 1240), '224px training canvas enlarged for review', font=font(15, 'mono'), fill=MUTED)
        draw.rounded_rectangle((688, 790, 1892, 1268), radius=6, fill=PANEL, outline=BORDER)
        mode = '2D TERMINAL HOLD' if labels.get('terminal_hold', False) else 'REAL TARGETS ONLY'
        draw.text((712, 806), f'EVENT + MINIMUM {int(labels["min_horizon_frames"])}f WINDOW / {mode}',
                  font=font(20, 'bold'), fill=WHITE)
        row_names = [(866, 'OFFSET'), (922, 'XY SOURCE'), (978, 'XY TARGET'), (1034, 'GRIPPER CMD'), (1090, 'STATUS')]
        for y, label in row_names:
            draw.text((710, y), label, font=font(16, 'mono'), fill=MUTED)
        for i, offset in enumerate(labels['offsets_frames']):
            x = 915 + i * 119
            source = int(labels.get('trace_source_frame_indices', labels['future_source_frame_indices'])[index, 0, i])
            reason = ('HOLD' if held[i] else 'EPISODE' if labels['future_episode_tail'][index, 0, i] else
                      'CROSS' if labels['future_cross_event'][index, 0, i] else
                      'EVENT' if labels['future_event_tail'][index, 0, i] else
                      'MISSING' if not valid[i] or not gv[i] else '-')
            values = [f'+{offset}f', str(source) if source >= 0 else '--',
                      'VALID' if valid[i] else '--', ('CLOSE' if grip[i] else 'OPEN') if gv[i] else '--', reason]
            for (y, _), text in zip(row_names, values):
                color = (HOLD if text == 'HOLD' else CLOSE if text == 'CLOSE' else OPEN if text == 'OPEN'
                         else YELLOW if text == 'CROSS' else WHITE)
                draw.text((x, y), text, font=font(17, 'mono'), fill=color)
        draw.text((712, 1170), f'Object/goal above: complete current Event. Motion below: up to +{offsets[-1]} saved frames.',
                  font=font(17, 'sans'), fill=MUTED)
        draw.text((712, 1210), 'HOLD = derived stop target in 2D loss; no future observation or gripper command.',
                  font=font(17, 'sans'), fill=MUTED)
        return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Report directory on the data disk')
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--motion-stride', type=int, default=5)
    parser.add_argument('--min-horizon-frames', type=int, default=10)
    parser.add_argument('--terminal-hold', action='store_true', help='Review derived 2D stop targets')
    args = parser.parse_args()
    offsets = np.arange(1, 9) * args.motion_stride
    args.output.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, entry in enumerate(select_episodes(args.index, args.count), 1):
        reader = EpisodeReader(args.index.parent / entry['path'])
        name = f'{i:02d}_{entry["episode_id"]}'
        target_path = args.output / (name + '.npz')
        save_targets(reader, target_path, offsets, min_horizon_frames=args.min_horizon_frames,
                     terminal_hold=args.terminal_hold)
        # Visualization deliberately reloads persisted targets, not a second implementation.
        with np.load(target_path) as archive:
            labels = {key: archive[key] for key in archive.files}
        view = from_saved(reader)
        renderer = MotionRenderer(view, labels)
        video = args.video_dir / (name + '.mp4')
        result = render_video(view, video, renderer=renderer)
        Image.fromarray(renderer.frame(view.frame_count // 2)).save(args.output / (name + '.jpg'))
        result.update(episode=str(reader.path), episode_sha256=digest(reader.path),
                      targets=str(target_path), targets_sha256=digest(target_path), video=str(video),
                      skills=sorted({e['skill'] for e in reader.meta['events']}))
        results.append(result)
        print(json.dumps(dict(completed=i, count=args.count, video=str(video))), flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        offsets_frames=offsets.tolist(), min_horizon_frames=args.min_horizon_frames,
        terminal_hold=args.terminal_hold,
        manual_review='pending', samples=results), ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
