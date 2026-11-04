"""Inspect frozen FLUX VAE reconstructions of saved episode role canvases."""
import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import textwrap
import time

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from openpi.models.grounding import load_vae
from openpi.policies.eskill_dataset import load_episode
from openpi.transforms import resize_image, role_canvas


ROLES = ('ee', 'obj', 'goal')
LABELS = ('CURRENT GRIPPER', 'CURRENT OBJECT', 'EVENT END OBJECT')
COLORS = ('#58adf5', '#79d9ad', '#e8bd5e')
WIDTH, HEIGHT = 1080, 1110


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def reconstruct(vae, canvas, weight):
    image = torch.from_numpy(canvas.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
    with torch.autocast('cpu', enabled=False):
        raw = vae.encode(image).latent_dist.mode()
        latent = (raw - vae.config.shift_factor) * vae.config.scaling_factor
        decoded = vae.decode(latent / vae.config.scaling_factor + vae.config.shift_factor).sample
    if tuple(latent.shape) != (1, 16, 32, 32) or not torch.isfinite(latent).all():
        raise ValueError('Unexpected or nonfinite FLUX latent')
    if not torch.isfinite(decoded).all():
        raise ValueError('Nonfinite FLUX reconstruction')
    rgb = ((decoded[0].permute(1, 2, 0).clamp(-1, 1) + 1) / 2).numpy()
    target = canvas.astype(np.float32) / 255
    error = rgb - target
    mse = np.square(error).mean(-1)
    mae = np.abs(error).mean(-1)
    metrics = dict(
        rgb_mse=float(mse.mean()),
        roi_mse=float((mse * weight).sum() / weight.sum()),
        roi_mae_255=float((mae * weight).sum() / weight.sum() * 255),
        background_mse=float((mse * (1 - weight)).sum() / max((1 - weight).sum(), 1)),
        roi_pixels=float(weight.sum()), latent_mean=float(latent.mean()),
        latent_std=float(latent.std()), latent_abs_max=float(latent.abs().max()),
    )
    result = np.rint(rgb * 255).astype(np.uint8)
    difference = np.rint(np.clip(np.abs(error) * 4, 0, 1) * 255).astype(np.uint8)
    return result, difference, metrics


def keyframes(meta, arm):
    frames = set()
    for event in meta['events']:
        if event['arm'] != arm:
            continue
        start, last = event['start_frame'], event['stop_frame'] - 1
        frames.update((start, (start + last) // 2, last))
    return sorted(frames)


def role_sources(reader, arm, frame, event):
    image = reader.rgb('image', frame)
    ee = reader.region(reader.meta['effector_regions'].get(arm, {}).get('image'), frame, include_mask=True)
    obj = reader.region(event['object_regions'].get('image'), frame, include_mask=True)
    goal = reader.goal(event['event_id'], 'image', include_mask=True)
    return [(frame, image, ee), (frame, image, obj),
            (goal['frame'], goal['rgb'], goal['region']) if goal else (None, None, None)]


def review_frame(reader, arm, frame, event, rows, number, ordinal, count):
    panel = Image.new('RGB', (WIDTH, HEIGHT), '#0b1218')
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 20)
    draw.text((16, 12), f'{number:02d} | {reader.meta["episode_id"]} | {arm}', font=title, fill='white')
    task = '\n'.join(textwrap.wrap(reader.meta['instructions'][0], 105))
    draw.multiline_text((16, 43), task, font=font, fill='#e9f0f5', spacing=4)
    label = f'Frame {frame} | {event["event_id"]} | {event["description"]}'
    draw.text((16, 94), label[:120], font=font, fill='#8395a3')
    columns = ('SOURCE RGB + BBOX', 'TARGET CANVAS', 'VAE RECONSTRUCTION', 'ABS RGB ERROR x4')
    for x, label in zip((16, 280, 544, 808), columns):
        draw.text((x, 126), label, font=font, fill='white')
    for role, row in enumerate(rows):
        y = 158 + role * 304
        metrics = row['metrics']
        status = f'ROI MAE {metrics["roi_mae_255"]:.2f}/255' if metrics else 'MISSING REGION'
        draw.text((16, y), f'{LABELS[role]} | source frame {row["frame"]} | {status}', font=font, fill=COLORS[role])
        if row['rgb'] is not None:
            source = Image.fromarray(row['rgb']).copy()
            if row['bbox'] is not None:
                box = row['bbox'].copy()
                box[2:] -= 1  # Saved xyxy upper bounds are exclusive.
                ImageDraw.Draw(source).rectangle(tuple(box), outline=COLORS[role], width=2)
            source = Image.fromarray(resize_image(np.asarray(source), 256))
            panel.paste(source, (16, y + 26))
        for x, name in ((280, 'canvas'), (544, 'reconstruction'), (808, 'difference')):
            if row[name] is not None:
                panel.paste(Image.fromarray(row[name]), (x, y + 26))
            else:
                draw.rectangle((x, y + 26, x + 255, y + 281), outline='#2b3b47')
                draw.text((x + 16, y + 140), 'MISSING', font=font, fill='#8395a3')
    draw.text((16, 1080), f'Keyframe {ordinal + 1}/{count} | 1 fps display | Frozen VAE only | Human review pending', font=font, fill='#8395a3')
    return np.asarray(panel)


def render_episode(vae, reader, number, output, video_dir):
    clips, records = [], []
    cache = {}
    for arm in reader.meta['arm_ids']:
        frames = keyframes(reader.meta, arm)
        name = f'{number:02d}_{reader.meta["episode_id"]}_{arm}_flux_vae.mp4'
        video = video_dir / name
        preview = output / 'preview' / (video.stem + '.png')
        with av.open(str(video), 'w') as container:
            stream = container.add_stream('libx264', rate=1)
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
            stream.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
            for ordinal, frame in enumerate(frames):
                event = reader.event_at(arm, frame)
                rows = []
                for role, (source_frame, rgb, region) in zip(ROLES, role_sources(reader, arm, frame, event)):
                    # Goal stays attached to this event's object; repeated targets reuse their exact result.
                    identity = None if role == 'ee' else event['event_id']
                    key = (arm, role, identity, source_frame)
                    if key not in cache:
                        canvas, weight, valid = role_canvas(rgb, region, 256, 4)
                        decoded, difference, metrics = reconstruct(vae, canvas, weight) if valid else (None, None, None)
                        cache[key] = dict(frame=source_frame, rgb=rgb, bbox=None if region is None else region['bbox'],
                                          canvas=canvas if valid else None, reconstruction=decoded, difference=difference,
                                          valid=bool(valid), metrics=metrics)
                    row = cache[key]
                    rows.append(row)
                    records.append(dict(arm=arm, frame=frame, event_id=event['event_id'], skill=event['skill'],
                                        role=role, source_frame=source_frame, valid=row['valid'], metrics=row['metrics']))
                image = review_frame(reader, arm, frame, event, rows, number, ordinal, len(frames))
                if ordinal == len(frames) // 2:
                    Image.fromarray(image).save(preview)
                for packet in stream.encode(av.VideoFrame.from_ndarray(image, format='rgb24')):
                    container.mux(packet)
                print(f'{number:02d} {arm} keyframe {ordinal + 1}/{len(frames)}', flush=True)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(video)) as container:
            count = sum(1 for decoded in container.decode(video=0)
                        if (decoded.width, decoded.height) == (WIDTH, HEIGHT))
        if count != len(frames):
            raise ValueError('Video decode or frame-count check failed')
        clips.append(dict(arm=arm, source_frames=frames, video=os.path.relpath(video, output),
                          preview='preview/' + preview.name, decoded_frames=count, sha256=digest(video)))
    return dict(episode_id=reader.meta['episode_id'], episode_path=str(reader.path),
                episode_sha256=digest(reader.path), clips=clips, records=records,
                unique_valid_canvases=sum(row['valid'] for row in cache.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--vae-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    source = json.loads((args.vae_path / 'source.json').read_text())
    for name, info in source['files'].items():
        if digest(args.vae_path / name) != info['sha256']:
            raise ValueError('VAE file differs from saved provenance: ' + name)
    vae = load_vae(args.vae_path)
    print('Loaded strict-matched FP32 FLUX VAE on CPU', flush=True)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    args.video_dir.mkdir(parents=True, exist_ok=True)
    (output / 'preview').mkdir(exist_ok=True)
    samples = []
    for number, item in enumerate(manifest['episodes'], 1):
        path = Path(item['path'])
        if not path.is_absolute():
            path = args.manifest.parent / path
        reader = load_episode(path)
        if reader.meta['episode_id'] != item['episode_id']:
            raise ValueError('Manifest episode identity mismatch')
        result = render_episode(vae, reader, number, output, args.video_dir)
        (output / f'{number:02d}_report.json').write_text(json.dumps(result, indent=2) + '\n')
        samples.append(result)
    metrics = defaultdict(list)
    for sample in samples:
        for row in sample['records']:
            if row['valid']:
                metrics[row['role']].append(row['metrics'])
    summary = {role: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
               for role, rows in metrics.items()}
    report = dict(manifest=str(args.manifest.resolve()), manifest_sha256=digest(args.manifest),
                  vae_source=source, vae_path=str(args.vae_path), device='cpu', precision='float32',
                  posterior='mode', canvas_size=256, source_margin_px=4,
                  latent_shape=[16, 32, 32], keyframes='event start / middle / last, source order',
                  display_fps=1, source_fps=None, manual_review='pending',
                  elapsed_seconds=time.monotonic() - start, mean_rgb_metrics=summary,
                  samples=samples, code_sha256=digest(__file__), torch_version=torch.__version__)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    items = ''.join(f'<h2>{html.escape(s["episode_id"])}</h2>' + ''.join(
        f'<video controls preload="none" width="900" poster="{c["preview"]}" src="{c["video"]}"></video>'
        for c in s['clips']) for s in samples)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FLUX VAE review</title>'
                                     f'<h1>{len(samples)}-episode FLUX VAE reconstruction review</h1>'
                                     '<p>Event start / middle / last; 1 fps display; human review pending.</p>' + items)
    print(json.dumps({'episodes': len(samples), 'elapsed_seconds': report['elapsed_seconds'],
                      'mean_rgb_metrics': summary}), flush=True)


if __name__ == '__main__':
    main()
