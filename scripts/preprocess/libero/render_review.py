"""Render any supported LIBERO source using the common review layout."""
import argparse
import html
import json
from pathlib import Path

from PIL import Image

from scripts.preprocess.libero.render import Renderer, render_video
from scripts.preprocess.libero.view import attach_regions, from_episode, from_rlds
from scripts.preprocess.libero.rlds import load_records
from scripts.preprocess.artifacts import artifact, digest, episodes, read_json, write_json
from scripts.preprocess.visual_grounding.render import Regions

from scripts.preprocess import DATA_ROOT

BASE = DATA_ROOT / "libero"


def inputs(source, manifest_path, annotations=None):
    manifest = read_json(manifest_path)
    if source == 'guidedvla':
        for episode in episodes('libero', manifest_path):
            view = from_episode(episode, source=source, source_fps=episode.fps)
            if annotations:
                record = read_json(artifact(annotations, 'regions', view.key))
                attach_regions(view, record, Regions(annotations, record))
            number = next(s['number'] for s in manifest['samples'] if s['episode_index'] == episode.entry['episode_index'])
            yield number, view
        return
    root = Path(manifest['raw_root'])
    for item in manifest['files']:
        if digest(root/item['path']) != item['sha256']:
            raise ValueError('Source shard SHA256 differs from sample manifest')
    for sample, data in load_records(root, manifest['samples']):
        view = from_rlds(source, sample, data)
        if annotations:
            record = read_json(artifact(annotations, 'regions', view.key))
            if record['source'].get('source_manifest_sha256') not in (None, digest(manifest_path)):
                raise ValueError('Regions refer to a different source manifest')
            attach_regions(view, record, Regions(annotations, record))
        yield sample['number'], view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['atomic-vla', 'guidedvla', 'lilo-vla', 'cyclevla'], required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--annotations', type=Path, help='Existing regions/ and masks/ root; omit for raw fields')
    parser.add_argument('--output', type=Path, help='Default: data/libero/<source>/check/test')
    parser.add_argument('--sample', type=int, help='Render only this fixed manifest sample number')
    parser.add_argument('--preview-only', action='store_true')
    parser.add_argument('--tag', default='', help='Artifact suffix for a separate render, for example 3d')
    args = parser.parse_args()
    manifest = args.manifest or BASE/args.source/'check/raw/manifest.json'
    output = args.output or BASE/args.source/'check/test'
    output.mkdir(parents=True, exist_ok=True)
    suffix = '_'+args.tag if args.tag else ''
    rows = []
    for number, view in inputs(args.source, manifest, args.annotations):
        if args.sample is not None and number != args.sample:
            continue
        if number == 1 or args.sample is not None:
            preview = output/f'render{suffix}_preview.png'
            if preview.exists():
                raise FileExistsError(preview)
            Image.fromarray(Renderer(view).frame(view.frame_count-1)).save(preview)
        if args.preview_only:
            rows.append({'number': number, 'key': view.key, 'frames': view.frame_count})
            continue
        path = output/'video'/f'{number:02d}_{view.key}_review{suffix}.mp4'
        row = render_video(view, path)
        row.update(number=number, video=str(path.relative_to(output)), sha256=digest(path),
                   events=[{'id': e.id, 'start_frame': e.start_frame, 'stop_frame': e.stop_frame,
                            'goal_frame': e.goal_frame, 'skill': e.skill, 'object_name': e.object_name}
                           for e in view.events])
        rows.append(row)
        print(json.dumps({'number': number, 'frames': view.frame_count, 'video': str(path)}), flush=True)
    if not rows:
        raise ValueError('No selected source samples')
    if not args.preview_only:
        code_root = Path(__file__).parent
        code = {p.name: digest(p) for p in [code_root/'render.py', code_root/'view.py', Path(__file__)]}
        write_json(output/f'render{suffix}_report.json', {'source_manifest': str(manifest.resolve()),
                   'source_manifest_sha256': digest(manifest), 'annotations': str(args.annotations) if args.annotations else None,
                   'samples': sorted(rows, key=lambda r: r['number']), 'code_sha256': code,
                   'manual_review': 'pending'})
        items = ''.join(f"<h2>{r['number']:02d} {html.escape(r['key'])}</h2><video controls preload='none' width='1280' src='{r['video']}'></video>" for r in sorted(rows, key=lambda r:r['number']))
        (output/f'render{suffix}_index.html').write_text('<!doctype html><meta charset="utf-8"><h1>LIBERO review / pending</h1>'+items)


if __name__ == '__main__':
    main()
