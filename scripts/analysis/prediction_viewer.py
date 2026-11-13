"""Current ESkill routing, two object roles, absolute 2D Motion and attention panel."""
import json
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

    def __init__(self, frames, config, metadata):
        self.frames, self.config, self.metadata = frames, config, metadata

    def frame(self, number):
        value = self.frames[number]
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
        text(44, 76, f'{technical}STEP {self.metadata["step"]} | FRAME {number} | {self.metadata["precision"]} | RGB + LANGUAGE ONLY', 19)

        card((24, 142, 624, 772))
        text(44, 152, 'MAIN / YELLOW=PRED / CYAN=GT', 19)
        main = Image.fromarray(value.attention.images[0]).resize((576, 576), Image.Resampling.BILINEAR)
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
        for j, label in enumerate(('GT TARGET', 'PREDICTED', 'GT VAE LOOP')):
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
        text(44, 1256, 'Last FM step / selected layers, heads, queries', 13, MUTED)

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


def render(checkpoint, episode, destination, *, fps=10., attention_video=None, **options):
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    frames, config, metadata = collect(checkpoint, episode, **options)
    if not frames:
        raise ValueError('No selected observation frames')
    renderer = Renderer(frames, config, metadata)
    numbers = sorted(frames)
    write_video(renderer, numbers, path, fps)
    Image.fromarray(renderer.frame(numbers[0])).save(path.with_suffix('.jpg'))
    arrays = dict(frames=np.array(numbers), offsets_frames=np.array(config.model.motion_offsets_frames))
    for name in ('skill_prob', 'trace', 'gripper_prob', 'actions', 'role_rgb', 'role_roundtrip'):
        if frames[numbers[0]].prediction[name] is not None:
            arrays['pred_' + name] = np.stack([frames[n].prediction[name] for n in numbers])
    for name in ('skill_labels', 'skill_valid', 'trace', 'trace_valid', 'gripper', 'gripper_valid',
                 'role_rgb', 'role_valid', 'future_cross_event'):
        arrays['gt_' + name] = np.stack([frames[n].targets[name] for n in numbers])
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
        gt_usage='Read only after action inference; never passed to the policy',
        fps=fps, language_groups=list(GROUPS),
        language_view_mass={str(n): frames[n].attention.language_attention['view_mass'].tolist() for n in numbers},
        attention_method='Reconstructed Q/K softmax; last FM step; mean selected layers/heads/action queries')
    path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    if attention_video is not None:
        name = 'TECHNICAL TINY' if metadata['technical'] else f'ESkill {metadata["step"]}'
        run = ModelRun(name, config.model.image_size, {n: frames[n].attention for n in numbers}, config.cameras)
        labels = Annotations(metadata['data_index'], episode, config.cameras[0])
        attention = AttentionRenderer([run], labels, episode, views=len(frames[numbers[0]].attention.images))
        output = write_video(attention, numbers, attention_video, fps)
        Image.fromarray(attention.frame(numbers[0])).save(output.with_suffix('.jpg'))
        output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    return path
