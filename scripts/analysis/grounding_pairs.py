"""Score two complete tasks against the same unaugmented Place frame."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from openpi.policies.current_grounding import create_current_grounding_policy
from openpi.policies.grounding_dataset import GroundingDataset
from openpi.policies.policy_config import load_train_config
from openpi.training.config import split_episodes
from openpi.transforms import normalized_image_boxes


def patch_distribution(boxes, side):
    """Normalized box overlap with each patch, using half-open normalized xyxy."""
    boxes = np.asarray(boxes, dtype=float)
    edges = np.linspace(0., 1., side + 1)
    lengths = np.maximum(0., np.minimum(boxes[:, 2:, None], edges[1:])
                         - np.maximum(boxes[:, :2, None], edges[:-1]))
    area = (lengths[:, 1, :, None] * lengths[:, 0, None, :]).reshape(len(boxes), -1)
    return area / np.maximum(area.sum(1, keepdims=True), 1e-15)


def score_pair(candidates, truth, predictions, side):
    ids = list(candidates)
    targets = patch_distribution(list(candidates.values()), side)
    roots = np.sqrt(targets)
    for key in truth:
        i = ids.index(key)
        if not targets[i].any() or np.isclose(roots @ roots[i], 1., atol=1e-9, rtol=0).sum() != 1:
            return None
    scores = np.sqrt(patch_distribution(predictions, side)) @ roots.T
    winners = []
    for row in scores:
        ties = np.flatnonzero(np.isclose(row, row.max(), atol=1e-9, rtol=0))
        winners.append(ids[ties[0]] if len(ties) == 1 and row.max() > 0 else None)
    return dict(chosen=winners, both_correct=winners == list(truth))


def evaluate(dataset, policy, *, limit, seed):
    if dataset.augment or limit < 1:
        raise ValueError('Paired evaluation requires unaugmented frames and a positive pair limit')
    sample_index = {sample: i for i, sample in enumerate(dataset.samples)}
    midpoints = []
    for episode in range(len(dataset.paths)):
        for event in dataset.reader(episode).meta['events']:
            if event['skill'] == 'Place':
                frame = (event['start_frame'] + event['stop_frame'] - 1) // 2
                if (episode, frame) in sample_index:
                    midpoints.append((sample_index[episode, frame], event['skill']))
    np.random.default_rng(seed).shuffle(midpoints)
    rows = []
    counts = Counter(examined=0, scored=0, ambiguous_gt=0, both_correct=0)
    for index, skill in midpoints:
        sample = dataset.sample(index, review=True)
        review, meta = sample['review'], sample['metadata']
        counts['examined'] += 1
        if 'paired_query' not in review or not meta['augmentation']['current_valid']:
            continue
        alternative, alt_id = review['paired_query']
        tasks, truth = [meta['task'], alternative], [review['target'], alt_id]
        rgb = review['original']
        size = dataset.config.model.image_size
        normalized = normalized_image_boxes(list(review['boxes'].values()), rgb.shape, size)
        boxes = dict(zip(review['boxes'], normalized, strict=True))
        predictions = [policy.infer([rgb], task)['bbox_xyxy'][0].cpu().tolist() for task in tasks]
        side = size // dataset.config.model.patch_size
        score = score_pair(boxes, truth, predictions, side)
        counts['ambiguous_gt' if score is None else 'scored'] += 1
        if score and score['both_correct']:
            counts['both_correct'] += 1
        rows.append(dict(episode=meta['episode_id'], frame=meta['frame'], skill=skill, tasks=tasks,
                         truth=truth, true_boxes=[boxes[k].tolist() for k in truth], predicted_boxes=predictions,
                         score=score))
        if len(rows) == limit:
            break
    return dict(counts=dict(counts), coverage=counts['scored'] / len(rows) if rows else None,
                paired_accuracy=(counts['both_correct']/counts['scored']
                if counts['scored'] else None), pairs=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-pairs', type=int, default=100)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--data-index')
    parser.add_argument('--split-path')
    parser.add_argument('--tokenizer-path')
    args = parser.parse_args()
    saved = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
    config = load_train_config(saved, device=args.device, tokenizer_path=args.tokenizer_path)
    del saved
    if config.training_stage != 'grounding' or config.model.current_grounding not in ('bbox', 'bbox_gac'):
        parser.error('Paired box evaluation expects a BBox or BBox+GAC checkpoint')
    index, split = args.data_index or config.data_index, args.split_path or config.split_path
    val = split_episodes(index, config.validation_fraction, config.seed, split)['val']
    policy = create_current_grounding_policy(args.checkpoint, device=args.device, tokenizer_path=args.tokenizer_path)
    dataset = GroundingDataset(index, config, policy.tokenizer, episode_ids=val, augment=False)
    result = evaluate(dataset, policy, limit=args.max_pairs, seed=config.seed)
    result.update(checkpoint=args.checkpoint, seed=config.seed, split=split, data_index=index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'pairs'}))


if __name__ == '__main__':
    main()
