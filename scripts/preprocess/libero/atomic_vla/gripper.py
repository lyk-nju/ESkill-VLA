"""Verify AtomicVLA source commands and fill Episode gripper semantics, without copying arrays.

Run without --write to inspect; --write updates episode.json and its existing checksum entry. Source commands,
actual finger positions, images, Event labels and the unverified clock stay unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.artifacts import digest


REVISION = '6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551'
REFERENCES = (
    'https://github.com/ARISE-Initiative/robosuite/blob/v1.4.1/robosuite/models/grippers/panda_gripper.py#L39-L54',
    'https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/experiments/robot/libero/regenerate_libero_dataset.py#L121-L157',
    'https://github.com/moojink/rlds_dataset_builder/blob/6174b0b6bb69df6361f1117944952bf14afb0cc3/LIBERO_10/LIBERO_10_dataset_builder.py#L38-L46',
)


def gripper_convention():
    return dict(channels={'single': 6}, close_sign=1, alignment='observation_then_action',
                evidence='Panda source command: -1=open, +1=close; zero has no binary target. '
                         'Regeneration records observation before executing action; RLDS preserves the same row. '
                         'Episode export preserves raw RLDS actions, without normalization or sign inversion. '
                         + ' '.join(REFERENCES))


def numeric_digest(actions, state):
    """Content identity independent of NPZ compression and machine-specific paths."""
    checksum = hashlib.sha256()
    for value in (actions, state):
        checksum.update(json.dumps([value.dtype.str, value.shape]).encode())
        checksum.update(np.ascontiguousarray(value).tobytes())
    return checksum.hexdigest()


def prepare(reader, *, verified_sha256=None):
    """Check numeric source identity and observation pairing before changing metadata."""
    meta = reader.meta
    if (meta['source']['dataset'] != 'AtomicVLA / openvla/modified_libero_rlds'
            or meta['source']['revision'] != REVISION or meta['arm_ids'] != ['single']):
        raise ValueError(f"{meta['episode_id']}: expected the verified AtomicVLA LIBERO release")
    actions = reader.array(meta['actions']['values'])
    state = reader.array(meta['state']['values'])
    n = meta['frame_count']
    if actions.shape != (n, 7) or state.shape != (n, 15):
        raise ValueError('Expected one original 7D action and 15D joint + EEF state per saved frame')
    frames = np.arange(n)
    for sequence in (meta['state'], meta['actions']):
        np.testing.assert_array_equal(reader.array(sequence['frame_indices']), frames)
    for camera, pairing in meta['actions']['conditioning_image_indices'].items():
        np.testing.assert_array_equal(reader.array(pairing), frames)
        np.testing.assert_array_equal(reader.array(meta['cameras'][camera]['frame_indices']), frames)
    sources = [Path(item['uri']) for item in meta['source']['artifacts']
               if Path(item['uri']).suffix == '.npz' and Path(item['uri']).parent.name == 'source']
    if len(sources) != 1:
        raise ValueError(f"{meta['episode_id']}: expected one original RLDS numeric cache")
    numeric_sha256 = numeric_digest(actions, state)
    if verified_sha256 is None:
        with np.load(sources[0], allow_pickle=False) as raw:
            np.testing.assert_array_equal(actions, raw['actions'])
            np.testing.assert_array_equal(state[:, -8:], raw['state'])
    elif numeric_sha256 != verified_sha256:
        raise ValueError(f"{meta['episode_id']}: numeric content differs from verified manifest")
    command = actions[:, 6]
    finite = np.isfinite(command)
    if not np.isin(command[finite], [-1, 0, 1]).all():
        raise ValueError(f"{meta['episode_id']}: unexpected source command values")
    valid = reader.array(meta['actions']['valid'])[:, 6] & finite & (command != 0)
    declaration = gripper_convention()
    old = meta['actions'].get('gripper_convention')
    if old is not None and old != declaration:
        raise ValueError(f"{meta['episode_id']}: existing gripper declaration differs")
    meta['actions']['gripper_convention'] = declaration
    return dict(episode_id=meta['episode_id'], action_rows=n, source_cache=str(sources[0]),
                source_actions_equal=True, source_body_state_equal=True,
                numeric_sha256=numeric_sha256,
                verification='source_cache' if verified_sha256 is None else 'verified_manifest',
                close=int((valid & (command > 0)).sum()), open=int((valid & (command < 0)).sum()),
                invalid=int((~valid).sum()), zero=int((finite & (command == 0)).sum()),
                nonfinite=int((~finite).sum()), declaration_added=old is None)


def supplement(index_path, *, write=False, verified_manifest=None, export_manifest=None):
    index_path = Path(index_path)
    if verified_manifest is not None and export_manifest is not None:
        raise ValueError('Export a manifest from source-cache verification, not another manifest')
    verified = None
    if verified_manifest is not None:
        manifest = json.loads(Path(verified_manifest).read_text())
        if manifest['revision'] != REVISION or manifest['convention'] != gripper_convention():
            raise ValueError('Verified manifest has a different source revision or gripper convention')
        verified = manifest['episodes']
    pending, records, paths = [], [], []
    for entry in json.loads(index_path.read_text())['episodes']:
        reader = EpisodeReader(index_path.parent / entry['path'])
        identity = reader.meta['episode_id']
        if verified is not None and identity not in verified:
            raise ValueError(f'{identity}: absent from verified manifest')
        record = prepare(reader, verified_sha256=verified[identity] if verified is not None else None)
        records.append(record)
        paths.append(reader.path)
        if record['declaration_added']:
            pending.append((reader.path, reader.meta))
    # Validate the complete selection before updating any Episode.
    if export_manifest is not None:
        manifest_path = Path(export_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(dict(revision=REVISION, convention=gripper_convention(),
            verification='actions and body state compared with original RLDS numeric caches',
            episodes={record['episode_id']: record['numeric_sha256'] for record in records}),
            ensure_ascii=False, indent=2) + '\n')
    if write:
        for path, meta in pending:
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
            temporary.replace(path)
    checksum_updates = 0
    checksum_path = index_path.parent / 'SHA256SUMS'
    if write and checksum_path.exists():
        # Update only verified Episode metadata; keep image/array hashes as recorded.
        hashes = {path: digest(path) for path in paths}
        lines = []
        for line in checksum_path.read_text().splitlines():
            old, name = line.split(maxsplit=1)
            new = hashes.get((index_path.parent / name).resolve(), old)
            checksum_updates += int(new != old)
            lines.append(f'{new}  {name}\n')
        if checksum_updates:
            temporary = checksum_path.with_suffix('.tmp')
            temporary.write_text(''.join(lines))
            temporary.replace(checksum_path)
    counts = Counter()
    for record in records:
        counts.update({key: record[key] for key in ('action_rows', 'close', 'open', 'invalid', 'zero', 'nonfinite')})
    return dict(index=str(index_path.resolve()), episodes=len(records), **counts,
                updated_episodes=len(pending) if write else 0, pending_episodes=0 if write else len(pending),
                written=write, convention=gripper_convention(), references=REFERENCES,
                verification='source_cache' if verified is None else 'verified_manifest',
                checksum_entries_updated=checksum_updates,
                physical_clock='not modified or inferred', episodes_detail=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    verification = parser.add_mutually_exclusive_group()
    verification.add_argument('--verified-manifest', type=Path,
                              help='Check numeric content against a source-verified portable manifest')
    verification.add_argument('--export-manifest', type=Path,
                              help='Export a portable manifest after checking the original source caches')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = supplement(args.index, write=args.write, verified_manifest=args.verified_manifest,
                        export_manifest=args.export_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('episodes_detail', 'convention', 'references')}))


if __name__ == '__main__':
    main()
