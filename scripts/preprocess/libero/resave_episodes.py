"""Re-save Episodes in place with the current derived fields.

Each Episode is rebuilt through the shared EpisodeWriter: per-frame Event indices and source-pixel
EEF projections are recomputed from the saved labels and referenced from episode.json. Arrays that
are no longer part of the format (precomputed future Motion targets) are dropped. images.npz and
every label are kept unchanged.
"""
import argparse
import json
from pathlib import Path

from openpi.policies.eskill_dataset import EpisodeReader
from scripts.preprocess.export import writer_from_saved

DROPPED = ("crosses_event", "eef_model_xy", "eef_uv", "eef_valid", "motion_frame_indices",
           "future_frame_indices", "future_model_xy", "future_observed", "future_padded",
           "future_source_frame_indices", "future_uv", "future_valid")


def resave(path):
    writer, meta = writer_from_saved(EpisodeReader(path), drop=DROPPED)
    meta.pop('motion_projection', None)  # Described the dropped arrays.
    writer.save(Path(path).parent, meta, exist_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, required=True, help='directory with index.json')
    args = parser.parse_args(argv)
    entries = json.loads((args.data/'index.json').read_text())['episodes']
    for number, entry in enumerate(entries, 1):
        resave(args.data/entry['path'])
        if number % 200 == 0 or number == len(entries):
            print(f'{number}/{len(entries)} episodes', flush=True)


if __name__ == '__main__':
    main()
