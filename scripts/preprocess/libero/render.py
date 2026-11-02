"""Source-independent LIBERO review video: supplied values or explicit MISSING."""
from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1920, 1340
BG, PANEL, BORDER = '#0b1218', '#151f28', '#2b3b47'
WHITE, MUTED = '#e9f0f5', '#8395a3'
BLUE, GREEN, YELLOW = '#58adf5', '#79d9ad', '#e8bd5e'
MISSING = 'MISSING'


@dataclass
class Region:
    """Coordinates are pixel xyxy in the unmodified source RGB; upper bounds exclusive."""
    bbox: object = None
    mask: object = None


@dataclass
class Event:
    id: str
    start_frame: int | None = None
    stop_frame: int | None = None  # Display membership [start, stop); adapter owns conversion.
    goal_frame: int | None = None  # Actual source RGB index, never inferred by the renderer.
    skill: str | None = None
    object_name: str | None = None
    object_region: Callable[[int], Region | None] | None = None  # Same event object at every call.


@dataclass
class Episode:
    key: str
    frame_count: int
    rgb: Callable[[int], np.ndarray | None] | None = None
    source: str | None = None
    task: str | None = None
    events: list[Event] = field(default_factory=list)
    gripper_state: object = None  # [N,D], aligned to RGB; missing values may be NaN.
    actions: object = None        # [N,D], retain source dimension and values.
    eef_xyz: object = None        # [N,3], no integration/kinematics in this module.
    xyz_note: str | None = None
    gripper_region: Callable[[int], Region | None] | None = None
    source_fps: float | None = None
    playback_fps: float = 10.0
    annotation_label: str = 'REGIONS MISSING'
    review_status: str = 'pending'
    note: str | None = None


@lru_cache(maxsize=64)
def font(size, kind='sans'):
    path = ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf' if kind == 'mono' else
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if kind == 'bold' else
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    return ImageFont.truetype(path, size)


def value_text(value):
    if value is None:
        return MISSING
    values = np.asarray(value).reshape(-1)
    if not len(values):
        return MISSING
    return '[' + ', '.join(f'{v:.3f}' if np.isfinite(v) else MISSING for v in values) + ']'


def overlay(rgb, region, color):
    """Draw provided masks and/or boxes independently; bbox-only input remains visible."""
    if rgb is None:
        return None
    image = Image.fromarray(rgb).convert('RGB')
    if region is None:
        return image
    if region.mask is not None:
        mask = np.asarray(region.mask, dtype=bool)
        if mask.shape != rgb.shape[:2]:
            raise ValueError('Region mask must match source RGB resolution')
        pixels = np.asarray(image).copy()
        tint = np.asarray(Image.new('RGB', (1, 1), color))[0, 0]
        pixels[mask] = (.75*pixels[mask] + .25*tint).astype(np.uint8)
        image = Image.fromarray(pixels)
    if region.bbox is not None:
        x1, y1, x2, y2 = region.bbox
        if not np.isfinite(region.bbox).all() or not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
            raise ValueError('bbox must be valid source pixel xyxy, upper bounds exclusive')
        ImageDraw.Draw(image).rectangle((x1, y1, x2-1, y2-1), outline=color, width=2)
    return image


class Trajectory3D:
    """Fixed equal-scale source axes; cache Matplotlib axes and project moving overlays."""
    def __init__(self, xyz):
        self.image, self.points, self.bounds = None, None, None
        if xyz is None:
            return
        xyz = np.asarray(xyz, dtype=float)
        valid = np.isfinite(xyz).all(axis=1)
        if not valid.any():
            return
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator
        from mpl_toolkits.mplot3d import proj3d

        lo, hi = xyz[valid].min(axis=0), xyz[valid].max(axis=0)
        center = (lo+hi)/2
        span = max(float((hi-lo).max()), .01)*1.16
        self.bounds = np.column_stack((center-span/2, center+span/2))
        figure = Figure(figsize=(7.2, 3.9), dpi=100, facecolor=PANEL)
        canvas = FigureCanvasAgg(figure)
        axes = figure.add_axes((.01, .04, .96, .92), projection='3d', facecolor=PANEL)
        axes.view_init(elev=24, azim=-58)
        axes.set_box_aspect((1, 1, 1))
        for axis, name, color, limits in zip((axes.xaxis, axes.yaxis, axes.zaxis), 'XYZ',
                                              (BLUE, GREEN, YELLOW), self.bounds):
            axis.set_pane_color((.08, .12, .16, 1))
            axis.line.set_color(MUTED)
            axis.set_major_locator(MaxNLocator(nbins=3))
            axis.set_major_formatter(FormatStrFormatter('%.3f'))
            axis.set_tick_params(labelsize=10, colors=MUTED, pad=0)
            getattr(axes, 'set_'+name.lower()+'lim')(*limits)
            getattr(axes, 'set_'+name.lower()+'label')(name, color=color, fontsize=13, labelpad=3)
        # Matplotlib itself breaks the background line at missing source rows.
        masked = xyz.copy()
        masked[~valid] = np.nan
        axes.plot(*masked.T, color='#647786', linewidth=1.4)
        canvas.draw()
        self.image = Image.fromarray(np.asarray(canvas.buffer_rgba())[:, :, :3].copy())
        self.points = np.full((len(xyz), 2), np.nan)
        x, y, _ = proj3d.proj_transform(*xyz[valid].T, axes.get_proj())
        pixels = axes.transData.transform(np.column_stack((x, y)))
        pixels[:, 1] = self.image.height-pixels[:, 1]
        self.points[valid] = pixels
        figure.clear()

    def point(self, frame):
        if self.points is None or frame is None or not 0 <= frame < len(self.points):
            return None
        point = self.points[frame]
        return tuple(point) if np.isfinite(point).all() else None

    def frame(self, index, event):
        if self.image is None:
            return None
        image = self.image.copy()
        draw = ImageDraw.Draw(image)
        if event and event.start_frame is not None and event.stop_frame is not None:
            stop = event.stop_frame + (event.goal_frame == event.stop_frame)
            previous = None
            for frame in range(max(0, event.start_frame), min(stop, len(self.points))):
                point = self.point(frame)
                if point is not None and previous is not None:
                    draw.line((*previous, *point), fill=GREEN, width=3)
                previous = point  # Never connect across a missing source row.
        goal = self.point(event.goal_frame) if event else None
        if goal is not None:
            x, y = goal
            draw.ellipse((x-8, y-8, x+8, y+8), outline=YELLOW, width=3)
        current = self.point(index)
        if current is not None:
            x, y = current
            draw.ellipse((x-4, y-4, x+4, y+4), fill=WHITE)
        return image


class Renderer:
    def __init__(self, episode):
        self.episode = episode
        if episode.frame_count < 1 or not np.isfinite(episode.playback_fps) or episode.playback_fps <= 0:
            raise ValueError('Need at least one frame and positive playback fps')
        for name in ('gripper_state', 'actions', 'eef_xyz'):
            values = getattr(episode, name)
            if values is not None and (np.asarray(values).ndim != 2 or len(values) != episode.frame_count):
                raise ValueError(name + ' must be [N,D] aligned to RGB')
        if episode.eef_xyz is not None and np.asarray(episode.eef_xyz).shape[1] != 3:
            raise ValueError('eef_xyz must have three columns')
        self.trajectory = Trajectory3D(episode.eef_xyz)

    def active_event(self, frame):
        choices = [e for e in self.episode.events if e.start_frame is not None and e.stop_frame is not None
                   and e.start_frame <= frame < e.stop_frame]
        return choices[0] if len(choices) == 1 else None

    @staticmethod
    def region_status(region):
        if region is None or (region.bbox is None and region.mask is None):
            return MISSING
        if region.bbox is None:
            return 'MASK / BBOX MISSING'
        return 'BBOX AVAILABLE'

    def frame(self, index):
        episode = self.episode
        if not 0 <= index < episode.frame_count:
            raise IndexError(index)
        canvas = Image.new('RGB', (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)

        def text(x, y, label, size=20, color=WHITE, kind='sans'):
            draw.text((x, y), str(label) if label is not None else MISSING, font=font(size, kind), fill=color)

        def fit(x, y, label, width, size=28, minimum=18, color=WHITE, kind='sans'):
            label = str(label) if label is not None else MISSING
            while size > minimum and draw.textlength(label, font=font(size, kind)) > width:
                size -= 1
            while len(label) > 1 and draw.textlength(label, font=font(size, kind)) > width:
                label = label[:-2] + '…' if not label.endswith('…') else label[:-2] + '…'
            text(x, y, label, size, color, kind)

        def card(box):
            draw.rounded_rectangle(box, radius=6, fill=PANEL, outline=BORDER)

        def image_panel(image, box):
            x, y, w, h = box
            if image is None:
                draw.rectangle((x, y, x+w, y+h), fill=BG, outline=BORDER)
                text(x+24, y+h//2-20, MISSING, 32, YELLOW, 'mono')
                return
            scale = min(w/image.width, h/image.height)
            image = image.resize((round(image.width*scale), round(image.height*scale)), Image.Resampling.NEAREST)
            canvas.paste(image, (x+(w-image.width)//2, y+(h-image.height)//2))

        event = self.active_event(index)
        skill = event.skill if event else None
        rgb = episode.rgb(index) if episode.rgb else None
        gripper = episode.gripper_region(index) if episode.gripper_region else None
        obj = event.object_region(index) if event and event.object_region else None
        goal = event.goal_frame if event else None
        valid_goal = goal is not None and 0 <= goal < episode.frame_count
        goal_rgb = episode.rgb(goal) if valid_goal and episode.rgb else None
        goal_obj = event.object_region(goal) if valid_goal and event.object_region else None

        # Task header.
        card((28, 12, 1892, 108))
        draw.rectangle((28, 12, 33, 108), fill=GREEN)
        text(60, 27, 'TASK INSTRUCTION', 15, MUTED, 'mono')
        fit(60, 53, episode.task, 1515, 32, 20, kind='bold')
        draw.line((1634, 28, 1634, 92), fill=BORDER)
        text(1664, 30, 'ACTIVE SKILL', 15, MUTED, 'mono')
        draw.ellipse((1666, 72, 1676, 82), fill=GREEN if skill else YELLOW)
        fit(1690, 56, skill, 177, 31, 17, kind='bold')

        # Main camera card, exact source geometry for both RGB and overlays.
        card((28, 128, 666, 736))
        text(48, 140, '01', 17, GREEN, 'mono')
        text(90, 137, 'MAIN CAMERA', 23, kind='bold')
        rate = f'{episode.source_fps:g} FPS SOURCE' if episode.source_fps is not None else f'{episode.playback_fps:g} FPS DISPLAY'
        text(465, 144, rate, 14, MUTED, 'mono')
        draw.line((28, 180, 666, 180), fill=BORDER)
        current = overlay(rgb, gripper, BLUE)
        if current is not None:
            current = overlay(np.asarray(current), obj, GREEN)
        image_panel(current, (48, 193, 598, 518))
        fit(48, 714, episode.key, 598, 13, 10, MUTED, 'mono')

        # Event identity and clear skill, followed by scalar/numeric source values.
        card((688, 128, 1892, 370))
        text(716, 146, '02 / CURRENT EVENT', 15, MUTED, 'mono')
        fit(716, 184, skill, 310, 65, 24, kind='bold')
        text(716, 274, event.id if event else None, 21, MUTED, 'mono')
        left = f'{event.start_frame:03d}' if event and event.start_frame is not None else MISSING
        end = f'{goal:03d}' if valid_goal else MISSING
        text(716, 314, f'FRAME {left}', 14, MUTED, 'mono')
        text(931, 314, end, 14, MUTED, 'mono')
        draw.line((716, 350, 1030, 350), fill=BORDER, width=5)
        if event and event.stop_frame is not None and event.start_frame is not None:
            progress = (index-event.start_frame)/max(1, event.stop_frame-event.start_frame-1)
            x = 716 + round(314*progress)
            draw.line((716, 350, x, 350), fill=GREEN, width=5)
            draw.ellipse((x-5, 345, x+5, 355), fill=WHITE)
        draw.line((1057, 128, 1057, 370), fill=BORDER)
        for x, label in [(1084, 'MANIPULATED OBJECT'), (1360, 'GOAL FRAME'), (1560, 'GRIPPER STATE')]:
            text(x, 151, label, 14, MUTED, 'mono')
        fit(1084, 188, event.object_name if event else None, 248, 27, 17, kind='mono')
        text(1360, 188, end, 26, WHITE if valid_goal else YELLOW, 'mono')
        grip = episode.gripper_state[index] if episode.gripper_state is not None else None
        fit(1560, 188, value_text(grip), 303, 22, 16, kind='mono')
        draw.line((1084, 240, 1864, 240), fill=BORDER)
        text(1084, 254, 'ACTION VECTOR', 14, MUTED, 'mono')
        action = np.asarray(episode.actions[index]).reshape(-1) if episode.actions is not None else None
        dimension = len(action) if action is not None else 0
        text(1620, 254, f'{dimension}D / SOURCE VALUES' if dimension else MISSING, 14, MUTED, 'mono')
        if dimension:
            gap, tile = 8, (780-8*(dimension-1))/dimension
            for i, value in enumerate(action):
                x = 1084+i*(tile+gap)
                draw.rounded_rectangle((x, 290, x+tile, 336), radius=4, fill='#202d38', outline=BORDER)
                label = f'{value:.3f}' if np.isfinite(value) else MISSING
                fit(x+8, 302, label, tile-16, 17, 11, kind='mono')
        else:
            text(1084, 299, MISSING, 25, YELLOW, 'mono')

        # Goal image is from the active event's object, including at event switches.
        card((688, 392, 1892, 736))
        text(716, 407, '03', 17, GREEN, 'mono')
        text(758, 403, 'GOAL REFERENCE', 23, kind='bold')
        text(1604, 411, 'ACTUAL GOAL '+end, 15, MUTED, 'mono')
        draw.line((688, 444, 1892, 444), fill=BORDER)
        goal_image = overlay(goal_rgb, goal_obj, YELLOW)
        image_panel(goal_image, (710, 461, 264, 260))
        text(1002, 462, 'Same-object target', 30, kind='bold')
        fit(1002, 511, episode.annotation_label, 492, 17, 14, color=MUTED, kind='mono')
        for y, color, label in [(561, BLUE, 'Gripper'), (608, GREEN, 'Manipulated object'), (655, YELLOW, 'Object at actual goal')]:
            draw.rectangle((1004, y+9, 1016, y+21), outline=color, width=2)
            text(1032, y, label, 21)
        card((1538, 462, 1868, 717))
        text(1556, 480, 'ANNOTATION STATUS', 14, MUTED, 'mono')
        for y, label, region in [(518, 'GRIPPER', gripper), (554, 'OBJECT', obj), (590, 'GOAL OBJ', goal_obj)]:
            status = self.region_status(region)
            fit(1556, y, label+': '+status, 294, 16, 12, GREEN if status == 'BBOX AVAILABLE' else YELLOW, 'mono')
        draw.line((1556, 630, 1850, 630), fill=BORDER)
        fit(1556, 646, 'Manual review '+episode.review_status, 294, 19, 15, YELLOW, 'mono')
        fit(1556, 681, 'Availability is not correctness', 294, 14, 11, MUTED)

        # Spatial trajectory and scalar curves use the same supplied XYZ sequence.
        text(28, 753, 'EEF POSITION / 3D TRAJECTORY + XYZ OVER TIME', 22, kind='bold')
        fit(840, 759, episode.xyz_note, 1050, 15, 12, MUTED, 'mono')
        card((28, 790, 966, 1268))
        text(48, 805, '3D EEF TRAJECTORY', 23, kind='bold')
        text(48, 840, 'FIXED VIEW / EQUAL AXIS SCALE / SOURCE COORDINATES', 13, MUTED, 'mono')
        spatial = self.trajectory.frame(index, event)
        if spatial is None:
            text(220, 1025, MISSING, 35, YELLOW, 'mono')
        else:
            canvas.paste(spatial, (38, 864))
        for y, label, color in [(901, 'FULL PATH', '#647786'), (945, 'CURRENT EVENT', GREEN),
                                 (989, 'CURRENT', WHITE), (1033, 'EVENT GOAL', YELLOW)]:
            draw.line((770, y+13, 792, y+13), fill=color, width=3)
            text(807, y, label, 15, color, 'mono')
        if self.trajectory.point(index) is None:
            text(770, 1093, 'CURRENT: MISSING', 14, YELLOW, 'mono')
        if self.trajectory.point(goal) is None:
            text(770, 1120, 'GOAL: MISSING', 14, YELLOW, 'mono')
        text(770, 1186, 'NO GAP FILLING', 13, MUTED, 'mono')
        text(770, 1213, 'NO RESAMPLING', 13, MUTED, 'mono')
        for axis, color in enumerate([BLUE, GREEN, YELLOW]):
            x0, y0 = 986, 790+axis*168
            card((x0, y0, x0+906, y0+144))
            text(x0+18, y0+11, 'XYZ'[axis]+'  AXIS', 19, color, 'mono')
            values = np.asarray(episode.eef_xyz)[:, axis] if episode.eef_xyz is not None else None
            finite = np.isfinite(values) if values is not None else np.array([], dtype=bool)
            if not finite.any():
                text(x0+230, y0+72, MISSING, 26, YELLOW, 'mono')
                continue
            lo, hi = float(values[finite].min()), float(values[finite].max())
            current_value = f'{values[index]:.3f}' if finite[index] else MISSING
            fit(x0+730, y0+10, current_value, 150, 24, 17, color, 'mono')
            text(x0+18, y0+46, f'RANGE [{lo:.3f}, {hi:.3f}]', 13, MUTED, 'mono')
            previous = None
            for j, value in enumerate(values):
                if not finite[j]:
                    previous = None
                    continue
                point = (x0+18+j/max(1, episode.frame_count-1)*868,
                         y0+132-(value-lo)/max(hi-lo, 1e-7)*58)
                if previous is not None:
                    draw.line((*previous, *point), fill=color, width=2)
                previous = point
                if j == index:
                    draw.ellipse((point[0]-4, point[1]-3, point[0]+4, point[1]+3), fill=WHITE)

        # Whole-episode progress and event boundaries in source row order.
        text(28, 1296, f'FRAME {index:03d} / {episode.frame_count-1:03d}', 14, MUTED, 'mono')
        draw.line((244, 1316, 1564, 1316), fill=BORDER, width=6)
        for e in episode.events:
            if e.start_frame is not None and 0 <= e.start_frame < episode.frame_count:
                x = 244+1320*e.start_frame/max(1, episode.frame_count-1)
                draw.line((x, 1310, x, 1322), fill=MUTED)
        x = 244+1320*index/max(1, episode.frame_count-1)
        draw.line((244, 1316, x, 1316), fill='#a7cedd', width=6)
        draw.ellipse((x-5, 1311, x+5, 1321), fill=WHITE)
        fit(1600, 1295, episode.source or MISSING, 290, 14, 12, MUTED, 'mono')
        fit(1600, 1318, episode.note or 'SOURCE FRAME ORDER RETAINED', 290, 11, 9, MUTED, 'mono')
        return np.asarray(canvas)


def render_video(episode, destination, *, renderer=None):
    """Encode and fully decode to check frame count, source order and output timing."""
    path = Path(destination)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.mp4')
    renderer = Renderer(episode) if renderer is None else renderer
    try:
        with av.open(str(temporary), 'w') as container:
            stream = container.add_stream('libx264', rate=Fraction(str(episode.playback_fps)))
            stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
            stream.options = {'crf': '20', 'preset': 'fast', 'threads': '2'}
            for index in range(episode.frame_count):
                for packet in stream.encode(av.VideoFrame.from_ndarray(renderer.frame(index), format='rgb24')):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(temporary)) as container:
            count = 0
            for frame in container.decode(video=0):
                if frame.width != WIDTH or frame.height != HEIGHT or abs(float(frame.pts*frame.time_base)-count/episode.playback_fps) > 1e-6:
                    raise ValueError('Rendered frame geometry or timing differs from display input')
                count += 1
            if count != episode.frame_count:
                raise ValueError('Rendered frame count differs from source')
        temporary.rename(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {'key': episode.key, 'frames': episode.frame_count, 'width': WIDTH, 'height': HEIGHT,
            'playback_fps': episode.playback_fps, 'source_fps': episode.source_fps,
            'manual_review': episode.review_status}
