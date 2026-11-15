"""Current ESkill routing, two object roles, absolute 2D Motion and attention panel."""
import json
from bisect import bisect_right
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from scripts.analysis.attention_viewer import Annotations, ModelRun, Renderer as AttentionRenderer, overlay, percent, write_video
from scripts.analysis.predictions import collect
from scripts.analysis.language_attention import GROUPS
from scripts.preprocess.libero.render import BG, BORDER, GREEN, MUTED, PANEL, WHITE, YELLOW, font

CYAN = '#70d8f4'


class Renderer:
    width, height = 1920, 1340

    def __init__(self, frames, config, metadata, timeline_images=None):
        self.frames, self.config, self.metadata = frames, config, metadata
        self.sources = sorted(frames)
        self.timeline_images = timeline_images

    def source_frame(self, number):
        return self.sources[bisect_right(self.sources, number) - 1]

    def frame(self, number):
        source = self.source_frame(number)
        value = self.frames[source]
        p, gt, m = value.prediction, value.targets, self.config.model
        canvas = Image.new('RGB', (self.width, self.height), BG)
        draw = ImageDraw.Draw(canvas)

        def text(x, y, message, size=18, color=WHITE):
            draw.text((x, y), str(message), fill=color, font=font(size, 'mono'))

        def card(box):
            draw.rounded_rectangle(box, radius=6, fill=PANEL, outline=BORDER)

        def image(rgb, x, y, size):
            if rgb is None:
                draw.rectangle((x, y, x + size, y + size), outline=BORDER, fill=BG)
                text(x + 10, y + size // 2 - 10, 'UNAVAILABLE', 16, MUTED)
            else:
                canvas.paste(Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR), (x, y))

        card((24, 12, 1896, 124))
        title = self.metadata['task']
        while draw.textlength(title, font=font(28, 'bold')) > 1800:
            title = title[:-2]
        draw.text((44, 25), title, fill=WHITE, font=font(28, 'bold'))
        technical = 'TECHNICAL / RANDOM TINY | ' if self.metadata['technical'] else ''
        if self.timeline_images is None:
            status = f'OFFLINE DEMO FRAME {number}'
        else:
            status = f'CTRL {number} | INFER {source} | AGE {number-source} | {"UPDATE" if source == number else "HOLD"}'
        text(44, 76, f'{technical}STEP {self.metadata["step"]} | {status} | {self.metadata["precision"]}', 19)

        card((24, 142, 624, 772))
        title = 'MAIN / YELLOW=PRED / CYAN=GT' if self.timeline_images is None else f'LIVE MAIN / YELLOW=PLAN @ {source}'
        text(44, 152, title, 18)
        main_rgb = value.attention.images[0] if self.timeline_images is None else self.timeline_images[number]
        main = Image.fromarray(main_rgb).resize((576, 576), Image.Resampling.BILINEAR)
        pen = ImageDraw.Draw(main)
        for points, valid, color in ((gt['trace'][0], gt['trace_valid'][0], CYAN),
                                      (p['trace'][0], np.ones(m.motion_queries, bool), YELLOW)):
            previous = None
            for i, (xy, available) in enumerate(zip(points, valid)):
                if not available or not np.isfinite(xy).all() or not ((xy >= 0).all() and (xy < 1).all()):
                    previous = None
                    continue
                x, y = xy * 576 - .5
                if previous is not None:
                    pen.line((*previous, x, y), fill=color, width=2)
                pen.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
                pen.text((x + 5, y - 13), str(i + 1), fill=color, font=font(15, 'mono'), stroke_width=1, stroke_fill=BG)
                previous = (x, y)
        canvas.paste(main, (36, 184))
        if self.timeline_images is not None and number == source:
            draw.rectangle((36, 184, 612, 760), outline=GREEN, width=4)

        card((644, 142, 1096, 494))
        active = list(m.active_skill_ids)
        route = int(np.argmax(p['skill_prob'][0]))
        label = int(gt['skill_labels'][0])
        label_name = m.skill_names[label] if 0 <= label < len(m.skill_names) else 'MISSING'
        text(664, 154, f'PRED {m.skill_names[active[route]]}', 24, YELLOW)
        text(664, 190, f'GT   {label_name}', 22, CYAN if gt['skill_valid'][0] else MUTED)
        text(664, 224, 'HARD TOP-1 + SHARED / ROUTE PROB', 15, MUTED)
        for i, skill_id in enumerate(active):
            y = 258 + i * 36
            probability = float(p['skill_prob'][0, i])
            text(664, y, m.skill_names[skill_id], 17, YELLOW if route == i else WHITE)
            draw.rectangle((760, y + 4, 1014, y + 22), fill=BG)
            draw.rectangle((760, y + 4, 760 + 254 * probability, y + 22), fill=YELLOW if route == i else GREEN)
            text(1024, y, f'{probability:.0%}', 15)

        card((644, 514, 1096, 772))
        text(664, 532, 'ACTION -> MOTION TOKENS', 19)
        text(664, 568, 'LABELS=GLOBAL / HEIGHT=WITHIN ARM', 15, MUTED)
        attention = value.attention.motion_attention
        if attention is not None:
            for i, share in enumerate(attention[0]):
                x = 668 + i * 50
                height = 94 * float(share) / max(float(attention[0].max()), 1e-12)
                draw.rectangle((x, 712 - height, x + 30, 712), fill=GREEN)
                text(x, 604, percent(share), 10)
                text(x, 722, f'+{m.motion_offsets_frames[i]}', 13, MUTED)
        text(664, 748, 'GT mask never hides predicted tokens', 13, MUTED)

        card((1116, 142, 1896, 772))
        labels = ('GT TARGET', 'PREDICTED', 'GT VAE LOOP') if self.timeline_images is None else ('GT / NOT RECORDED', 'PREDICTED', 'GT VAE / N/A')
        for j, label in enumerate(labels):
            text(1132 + j * 250, 154, label, 17)
        for role, name in enumerate(('CURRENT OBJECT', 'FUTURE OBJECT / EVENT END')):
            top = 214 + role * 270
            text(1132, top - 30, name, 17, GREEN)
            decoded = p['role_rgb']
            reference = p['role_roundtrip']
            image(gt['role_rgb'][0, role] if gt['role_valid'][0, role] else None, 1132, top, 224)
            image(decoded[0, role] if decoded is not None else None, 1382, top, 224)
            image(reference[0, role] if reference is not None and gt['role_valid'][0, role] else None, 1632, top, 224)
        text(1132, 748, self.metadata['visual_readout'], 13, MUTED)

        card((24, 792, 624, 1304))
        text(44, 808, 'ACTION ATTENTION / CONDITIONAL MAP', 18)
        canvas.paste(overlay(value.attention.images[0], value.attention.attention[0]).resize((256, 256)), (40, 850))
        if len(value.attention.images) > 1:
            image(value.attention.images[1], 330, 850, 160)
            text(330, 1020, 'WRIST RGB', 16, MUTED)
        for i, (name, share) in enumerate(value.attention.key_mass.items()):
            text(330, 1064 + i * 37, f'{name.upper():6s} {percent(share)}', 19)
        for i, share in enumerate(value.attention.view_mass):
            text(44, 1136 + i * 32, f'VIEW {i} / ALL: {percent(share)}', 16)
        caption = 'Last FM step / selected layers, heads, queries' if self.timeline_images is None else f'INPUT + MAP HELD FROM CONTROL FRAME {source}'
        text(44, 1256, caption, 13, MUTED)

        card((644, 792, 1896, 1304))
        text(664, 808, 'VLM / TARGET - OBJECT - INSTRUCTION ATTENTION', 21)
        text(664, 844, 'Word queries -> main-image keys / conditional maps / selected VLM layers', 16, MUTED)
        language = value.attention.language_attention
        for i, name in enumerate(GROUPS):
            x = 664 + i * 406
            text(x, 878, name.upper(), 20, GREEN)
            item = language['layout'][name] if language is not None else {}
            phrase = item.get('phrase') or 'UNSPECIFIED'
            while draw.textlength(phrase, font=font(16, 'mono')) > 350:
                phrase = phrase[:-4] + '...'
            text(x, 906, phrase, 16, MUTED)
            if language is not None and language['valid'][i]:
                panel = overlay(value.attention.images[0], language['maps'][i, 0])
                canvas.paste(panel.resize((344, 344), Image.Resampling.BILINEAR), (x, 934))
                text(x, 1280, f'MAIN / ALL: {percent(language["view_mass"][i, 0])}', 14)
            else:
                draw.rectangle((x, 934, x + 344, 1278), outline=BORDER, fill=BG)
                text(x + 10, 1060, 'UNAVAILABLE', 18, MUTED)
                text(x + 10, 1092, item.get('status', 'NOT CAPTURED'), 13, MUTED)
        return np.asarray(canvas)


def render(checkpoint, episode, destination, *, fps=None, attention_video=None, rollout_record=None,
           recording_dir=None, **options):
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    if rollout_record is not None and path.with_suffix('.json').resolve() == Path(rollout_record).resolve():
        raise ValueError('Output sidecars would overwrite the input rollout record; use another video name or directory')
    timeline_images = None
    if rollout_record is not None:
        from scripts.analysis.rollout_predictions import load_rollout
        frames, config, metadata, timeline_images = load_rollout(rollout_record, recording_dir)
        episode = metadata['episode']
    else:
        frames, config, metadata = collect(checkpoint, episode, **options)
    if not frames:
        raise ValueError('No selected observation frames')
    renderer = Renderer(frames, config, metadata, timeline_images)
    numbers = sorted(frames)
    displayed = numbers if timeline_images is None else list(range(len(timeline_images)))
    fps = fps or (10. if timeline_images is None else metadata['control_fps'])
    write_video(renderer, displayed, path, fps)
    if timeline_images is not None:
        original = path.with_name(path.stem + '_original.mp4')
        write_video(OriginalRenderer(timeline_images), displayed, original, fps)
        metadata['original_video'] = str(original)
    Image.fromarray(renderer.frame(numbers[0])).save(path.with_suffix('.jpg'))
    arrays = dict(frames=np.array(numbers), offsets_frames=np.array(config.model.motion_offsets_frames))
    arrays['display_frames'] = np.array(displayed)
    arrays['display_source_frames'] = np.array([renderer.source_frame(n) for n in displayed])
    for name in ('skill_prob', 'trace', 'gripper_prob', 'actions', 'role_rgb', 'role_roundtrip'):
        if frames[numbers[0]].prediction[name] is not None:
            arrays['pred_' + name] = np.stack([frames[n].prediction[name] for n in numbers])
    for name in ('skill_labels', 'skill_valid', 'trace', 'trace_valid', 'gripper', 'gripper_valid',
                 'role_rgb', 'role_valid', 'future_cross_event', 'trace_terminal_hold', 'future_observed'):
        if name in frames[numbers[0]].targets:
            arrays['gt_' + name] = np.stack([frames[n].targets[name] for n in numbers])
    if 'trace_source_frame_indices' in frames[numbers[0]].metadata:
        arrays['gt_trace_source_frame_indices'] = np.stack([
            frames[n].metadata['trace_source_frame_indices'] for n in numbers])
    arrays['motion_attention'] = np.stack([frames[n].attention.motion_attention for n in numbers])
    arrays['arm_presence'] = np.stack([frames[n].presence for n in numbers])
    arrays['input_images'] = np.stack([frames[n].attention.images for n in numbers])
    arrays['image_attention'] = np.stack([frames[n].attention.attention for n in numbers])
    arrays['view_mass'] = np.array([frames[n].attention.view_mass for n in numbers])
    arrays['key_mass'] = np.array([list(frames[n].attention.key_mass.values()) for n in numbers])
    for name in ('maps', 'view_mass', 'probabilities', 'valid'):
        arrays['language_' + name] = np.stack([frames[n].attention.language_attention[name] for n in numbers])
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    metadata.update(frames=numbers, action_attention={str(n): dict(
        key_mass=frames[n].attention.key_mass, view_mass=frames[n].attention.view_mass,
        motion_attention=frames[n].attention.motion_attention.tolist()) for n in numbers},
        gt_usage=('Read only after action inference; never passed to the policy' if timeline_images is None
                  else 'Unavailable in live rollout; never supplied to the policy'),
        timeline='offline_demonstration_probes' if timeline_images is None else 'actual_rollout_control_frames',
        rendered_frame_count=len(displayed),
        fps=fps, language_groups=list(GROUPS),
        language_view_mass={str(n): frames[n].attention.language_attention['view_mass'].tolist() for n in numbers},
        attention_method='Reconstructed Q/K softmax; last FM step; mean selected layers/heads/action queries')
    path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    if attention_video is not None:
        name = 'TECHNICAL TINY' if metadata['technical'] else f'ESkill {metadata["step"]}'
        run = ModelRun(name, config.model.image_size, {n: frames[n].attention for n in numbers}, config.cameras)
        labels = Annotations(metadata['data_index'], episode, config.cameras[0]) if timeline_images is None else None
        attention = AttentionRenderer([run], labels, episode, views=len(frames[numbers[0]].attention.images))
        if timeline_images is not None:
            attention = HeldAttentionRenderer(attention, renderer.source_frame)
        output = write_video(attention, displayed, attention_video, fps)
        Image.fromarray(attention.frame(numbers[0])).save(output.with_suffix('.jpg'))
        output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return path


class HeldAttentionRenderer:
    """Use the actual source input and heatmap together between consecutive requests."""
    def __init__(self, renderer, source_frame):
        self.renderer, self.source_frame = renderer, source_frame
        self.width, self.height = renderer.width, renderer.height

    def frame(self, number):
        source = self.source_frame(number)
        canvas = Image.fromarray(self.renderer.frame(source))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 60, self.width, 88), fill=BG)
        draw.text((12, 65), f'CONTROL {number} | ATTENTION INPUT {source} | AGE {number-source} | '
                  f'{"UPDATE" if source == number else "HOLD"}', fill=GREEN, font=font(15, 'mono'))
        return np.asarray(canvas)


class OriginalRenderer:
    """The exact same control observations and final frame as the prediction panel."""
    def __init__(self, images):
        self.images = images
        self.height, self.width = images.shape[1:3]

    def frame(self, number):
        return self.images[number]
