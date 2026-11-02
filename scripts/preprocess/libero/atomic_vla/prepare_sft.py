"""Pack the fixed AtomicVLA source episodes for ordinary dual-camera π0.5 SFT."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.preprocess.libero.rlds import load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--raw-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = json.loads(args.index.read_text())['episodes']
    ids = {e['episode_id']: e for e in entries}
    samples = [s for s in manifest['samples'] if s['key'] in ids]
    assert len(samples) == len(ids) == 1693
    args.output.mkdir(parents=True, exist_ok=False)
    states, actions, offsets, episodes = [], [], [], []
    count = 0
    with (args.output / 'images.bin').open('wb') as stream:
        for sample, data in load_records(args.raw_root, samples):
            n = sample['frames']
            state = np.asarray(data['steps/observation/state'], np.float32).reshape(n, 8)
            action = np.asarray(data['steps/action'], np.float32).reshape(n, 7)
            image, wrist = data['steps/observation/image'], data['steps/observation/wrist_image']
            assert len(image) == len(wrist) == n and np.isfinite(state).all() and np.isfinite(action).all()
            meta = json.loads((args.index.parent / ids[sample['key']]['path']).read_text())
            source = Path(meta['cameras']['image']['rgb']['uri'])
            with source.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == meta['source']['checksums'][str(source)]
            with np.load(source, allow_pickle=False) as original:
                np.testing.assert_array_equal(state, original['state'])
                np.testing.assert_array_equal(action, original['actions'])
                assert b''.join(image) == original['image_bytes'].tobytes()
            task = data['steps/language_instruction'][0].decode()
            assert task.strip().replace('_', ' ') == meta['instructions'][0].strip().replace('_', ' ')
            locations = np.zeros((n, 2, 2), np.int64)
            for row, pair in enumerate(zip(image, wrist, strict=True)):
                for camera, encoded in enumerate(pair):
                    locations[row, camera] = stream.tell(), len(encoded)
                    stream.write(encoded)
            offsets.append(locations)
            states.append(state)
            actions.append(action)
            episodes.append(dict(episode_id=sample['key'], task=task, frames=n, start=count,
                                 source=dict(parent_trajectory_id=sample['key'], shard=sample['shard'],
                                             record_index=sample['record_index'], revision=manifest['revision'])))
            count += n
            if len(episodes) % 100 == 0:
                print(f'Packed {len(episodes)}/1693 episodes, {count} frames', flush=True)
    assert count == sum(e['frames'] for e in entries)
    values = dict(state=np.concatenate(states), actions=np.concatenate(actions), image_offsets=np.concatenate(offsets))
    for key, value in values.items():
        np.save(args.output / f'{key}.npy', value, allow_pickle=False)
    index = dict(format='libero_sft_v1', episodes=episodes, frames=count, cameras=['image', 'wrist_image'],
                 source_index=str(args.index), source_manifest=str(args.manifest), revision=manifest['revision'],
                 original_main_state_actions_verified=True, images='unaltered source encoded RGB bytes')
    (args.output / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    stats = {}
    for key in ('state', 'actions'):
        value = values[key]
        q01, q99 = np.quantile(value, [.01, .99], axis=0)
        stats[key] = dict(mean=value.mean(0).tolist(), std=value.std(0).tolist(), q01=q01.tolist(), q99=q99.tolist())
    stats.update(method='quantile', provenance=dict(episodes=len(episodes), frames=count, split='all_training',
                                                   source_index=str(args.index), source_commands='unchanged'))
    (args.output / 'norm_stats.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(f'Saved {len(episodes)} episodes / {count} paired frames to {args.output}', flush=True)


if __name__ == '__main__':
    main()
