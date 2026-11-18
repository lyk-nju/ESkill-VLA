"""Refresh saved interaction/copy flags without rendering or changing RGB/bboxes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from openpi.augmentation import COPY_MIN_VISIBLE_FRACTION
from openpi.policies.eskill_dataset import EpisodeReader
from .atomic_vla.scene_regions import interaction_locked, interaction_regions, copy_eligibility


def backfill(index, output, *, refresh_copy_eligibility=False):
    output.mkdir(parents=True, exist_ok=False)
    backups = output / 'episode_metadata_before'
    backups.mkdir()
    records, total_flags, newly_eligible = [], 0, 0
    bit_counts = np.array([i.bit_count() for i in range(256)],np.uint8)
    for entry in json.loads(index.read_text())['episodes']:
        path = index.parent / entry['path']
        reader = EpisodeReader(path)
        meta = reader.meta
        camera = meta['cameras']['image']
        grip = reader.array(meta['effector_regions']['single']['image']['boxes_xyxy'])
        near, _ = interaction_regions(grip, [], camera['height'], camera['width'])
        archive_path = path.parent / meta['online_grounding']['interaction_boxes']['uri']
        original = path.read_bytes()
        changed, flags = False, 0
        with zipfile.ZipFile(archive_path, 'a', compression=zipfile.ZIP_STORED) as archive:
            names = set(archive.namelist())
            pending = {}
            for key, entity in meta['scene_entities'].items():
                if entity['kind'] != 'object':
                    continue
                sequence = entity['regions']['image']
                frames = reader.array(sequence['frame_indices'])
                assert np.array_equal(frames,reader.array(camera['frame_indices']))
                value = interaction_locked(reader.array(sequence['boxes_xyxy']),near)
                array_key = key + '_interaction_locked'
                member = array_key + '.npy'
                if member in names:
                    with archive.open(member) as stream:
                        assert np.array_equal(np.load(stream,allow_pickle=False),value)
                else:
                    pending[member] = value
                ref = dict(uri=str(archive_path.relative_to(path.parent)),key=array_key,
                           shape=list(value.shape),dtype=str(value.dtype))
                if 'interaction_locked' in entity:
                    assert np.array_equal(reader.array(entity['interaction_locked']),value)
                else:
                    entity['interaction_locked'] = ref
                    changed = True
                if refresh_copy_eligibility:
                    # Reproduce native extraction's exact ratio from packed masks,
                    # avoiding a threshold change caused by stored float32 rounding.
                    visible = sequence['visible_masks']
                    full = sequence['masks']
                    visible_count = bit_counts[reader._raw(visible['uri'],visible['key'])].sum(1)
                    full_count = bit_counts[reader._raw(full['uri'],full['key'])].sum(1)
                    ratio = visible_count / np.maximum(full_count,1)
                    eligible = copy_eligibility(entity['kind'],value,ratio,full_count)
                    old_eligible = reader.array(entity['copy_eligible'])
                    newly_eligible += int((eligible & ~old_eligible).sum())
                    assert not (old_eligible & ~eligible).any()
                    copy_key = f'{key}_copy_eligible_visible_{100 * COPY_MIN_VISIBLE_FRACTION:g}pct'
                    copy_member = copy_key + '.npy'
                    if copy_member in names:
                        with archive.open(copy_member) as stream:
                            assert np.array_equal(np.load(stream,allow_pickle=False),eligible)
                    else:
                        pending[copy_member] = eligible
                    copy_ref = dict(uri=ref['uri'],key=copy_key,shape=list(eligible.shape),dtype=str(eligible.dtype))
                    if entity['copy_eligible'] != copy_ref:
                        entity['copy_eligible'] = copy_ref
                        changed = True
                flags += len(value)
            # Finish all reads before appending: ZIP's directory is rewritten during writes.
            for member,value in pending.items():
                with archive.open(member,'w') as stream:
                    np.lib.format.write_array(stream,value,allow_pickle=False)
            if changed:
                meta['online_grounding']['interaction_lock_rule'] = 'full bbox intersects gripper bbox expanded 4 source px; proximity proxy, not contact truth'
                if refresh_copy_eligibility:
                    meta['online_grounding']['copy_rule'] = (
                        f'movable object without category blacklist, visible/full >= {COPY_MIN_VISIBLE_FRACTION:.2f}, '
                        'no proximity to gripper; source/destination safety checked online')
        if changed:
            (backups / (meta['episode_id'] + '.json')).write_bytes(original)
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            temporary.replace(path)
        records.append(dict(episode_id=meta['episode_id'],updated=changed,object_frames=flags,
                            metadata_before_sha256=hashlib.sha256(original).hexdigest(),
                            metadata_after_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        total_flags += flags
    result = dict(episodes=len(records),updated=sum(r['updated'] for r in records),object_frames=total_flags,
                  newly_copy_eligible_frames=newly_eligible,refresh_copy_eligibility=refresh_copy_eligibility,
                  scope='append boolean flags to existing uncompressed NPZ; RGB/bbox/mask members preserved; no simulation',
                  records=records)
    (output / 'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--refresh-copy-eligibility',action='store_true',help='Remove the old container blacklist in saved flags')
    args = parser.parse_args()
    backfill(args.index,args.output,refresh_copy_eligibility=args.refresh_copy_eligibility)


if __name__ == '__main__':
    main()
