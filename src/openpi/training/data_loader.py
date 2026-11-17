"""Dataset construction, transforms and batching; no model initialization."""
from itertools import groupby

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, Sampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from openpi.models.pi0_config import SKILLS
from openpi.policies.eskill_dataset import ARM_SLOTS, EpisodeDataset
from openpi.policies.libero_policy import LiberoInputs
from openpi.policies.robotwin_policy import RobotwinInputs
from openpi.training.config import split_episodes
from openpi.transforms import ModelTransforms, TokenizePrompt


class TransformedDataset:
    def __init__(self, dataset, adapter, transform=None):
        self.dataset, self.adapter, self.transform = dataset, adapter, transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["inputs"] = self.adapter(sample["inputs"])
        return self.transform(sample) if self.transform else sample


class ResumableBatchSampler(BatchSampler):
    """Skip consumed indices before workers read or decode any samples."""
    start_batch = 0
    epoch = 0
    sample_keys = False

    def __iter__(self):
        for batch_index, batch in enumerate(super().__iter__()):
            if batch_index < self.start_batch:
                continue
            yield ([(index, self.epoch, batch_index*self.batch_size+j) for j, index in enumerate(batch)]
                   if self.sample_keys else batch)

    def __len__(self):
        return max(0, super().__len__() - self.start_batch)


def skill_sampling_weights(dataset, max_weight=3.0):
    """Count skills at each sample's paired observation, without loading RGB or targets."""
    names = dataset.skills or {name.casefold(): i for i, (name, _) in enumerate(SKILLS)}
    active = dataset.active_skill_ids
    labels = np.full((len(dataset), 2), -1, np.int64)
    for episode, group in groupby(enumerate(dataset.samples), key=lambda item: item[1][0]):
        indices, rows = np.array([(index, row) for index, (_, row) in group]).T
        reader = dataset.reader(episode)
        meta = reader.meta
        paired = reader.array(meta['actions']['conditioning_image_indices'][dataset.camera])[rows]
        if np.any(paired < 0):
            raise ValueError(f"{meta['episode_id']}: action row has no paired image for {dataset.camera}")
        frames = reader.array(meta['cameras'][dataset.camera]['frame_indices'])[paired]
        event_labels = np.array([
            names.get(event['skill'].casefold(), -1)
            if event['boundary_valid'] and event['skill_valid'] and event['skill'] else -1
            for event in meta['events']], dtype=np.int64)
        event_labels[~np.isin(event_labels, list(active))] = -1
        for arm in meta['arm_ids']:
            events = reader.array(meta['event_index'][arm])[frames]
            valid = events >= 0
            labels[indices[valid], ARM_SLOTS[arm]] = event_labels[events[valid]]
    counts = np.bincount(labels[labels >= 0], minlength=len(names))
    class_weights = np.ones(len(names), np.float64)
    present = counts > 0
    if present.any():
        class_weights[present] = np.minimum(max_weight, np.sqrt(counts.max() / counts[present]))
    # Two-arm samples remain whole; use the rarer arm's weight, never their product.
    weights = np.where(labels >= 0, class_weights[np.maximum(labels, 0)], 1.).max(axis=1)
    weighted_arms = (weights[:, None] * (labels >= 0)).sum()
    summary = dict(method='capped_inverse_sqrt', max_weight=max_weight, samples=len(dataset),
                   samples_without_skill=int((labels < 0).all(axis=1).sum()),
                   unit='action rows at paired current observation; dual-arm weight=max', skills={})
    for name, index in names.items():
        if index not in active:
            continue
        summary['skills'][name] = dict(
            count=int(counts[index]), weight=float(class_weights[index]),
            source_fraction=float(counts[index] / max(1, counts.sum())),
            expected_fraction=float((weights[:, None] * (labels == index)).sum() / max(1, weighted_arms)))
    return torch.from_numpy(weights), summary


class SkillWeightedSampler(Sampler):
    """One seeded global draw with replacement, sharded by rank; replayable by epoch."""
    def __init__(self, weights, *, seed, num_replicas=1, rank=0, summary=None):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.seed, self.num_replicas, self.rank, self.epoch = seed, num_replicas, rank, 0
        self.num_samples = (len(weights) + num_replicas - 1) // num_replicas
        self.total_size = self.num_samples * num_replicas
        self.summary = summary

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank::self.num_replicas].tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def episode_dataset(config, episode_ids=None):
    """The same saved episodes for every recipe; ESkill also reads Event, region and Motion targets."""
    m = config.model
    if config.training_stage == 'grounding':
        from openpi.policies.grounding_dataset import GroundingDataset
        return GroundingDataset(config.data_index, config, TokenizePrompt(config.tokenizer_path, m.max_token_len),
                                episode_ids=episode_ids)
    auxiliary = config.model_type == "eskill"
    options = dict(skill_names=m.skill_names, motion_queries=m.motion_queries,
                   motion_stride=m.motion_stride, motion_min_horizon_frames=m.motion_min_horizon_frames,
                   motion_terminal_hold=m.motion_terminal_hold,
                   active_skill_ids=m.active_skill_ids) if auxiliary else {}
    return EpisodeDataset(
        config.data_index, action_horizon=m.action_horizon, cameras=config.cameras,
        auxiliary=auxiliary, action_padding=config.action_padding, image_size=m.image_size,
        episode_ids=episode_ids, frame_stride=config.frame_stride,
        include_state=m.discrete_state_input, **options)


def input_adapter(config):
    return {"libero": LiberoInputs, "robotwin": RobotwinInputs}[config.platform](
        include_state=config.model.discrete_state_input)


def create_dataset(config, split="train", *, transform=True):
    ids = split_episodes(config.data_index, config.validation_fraction, config.seed, config.split_path)
    dataset = episode_dataset(config, None if split == "all" else ids[split])
    if config.training_stage == 'grounding':
        dataset.augment = split == 'train'
        return dataset
    transforms = None
    if transform:
        transforms = ModelTransforms(
            config.model, config.norm_stats,
            TokenizePrompt(config.tokenizer_path, config.model.max_token_len))
    return TransformedDataset(dataset, input_adapter(config), transforms)


def collate_samples(samples):
    inputs = default_collate([s["inputs"] for s in samples])
    # Tokenization right-pads to a fixed limit; masked trailing tokens do no useful work.
    length = int(inputs["token_mask"].sum(-1).max())
    for key in ("tokens", "token_mask"):
        inputs[key] = inputs[key][:, :length]
    return dict(inputs=inputs,
                targets=default_collate([s["targets"] for s in samples]),
                metadata=[s["metadata"] for s in samples])


def create_data_loader(config, split="train"):
    dataset = create_dataset(config, split)
    if len(dataset) == 0:
        raise ValueError(f"No samples in {split} split")
    if split == 'val':
        world = dist.get_world_size() if dist.is_initialized() else 1
        count = min(len(dataset), config.val_max_batches * config.batch_size * world)
        count -= count % world
        if count <= 0:
            raise ValueError('Validation subset must contain at least one sample per rank')
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config.validation_seed))[:count].tolist()
        dataset = Subset(dataset, indices)
    # Keep sampling independent of worker seeding, including persistent-worker resets.
    if split == 'train' and config.skill_sampling:
        weights, summary = skill_sampling_weights(dataset.dataset, config.skill_sampling_max_weight)
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = SkillWeightedSampler(weights, seed=config.seed, num_replicas=world, rank=rank, summary=summary)
        summary.update(seed=config.seed, world_size=world, draws_per_epoch=sampler.total_size)
    elif dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=split == "train", seed=config.seed)
    else:
        sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(config.seed))
                   if split == "train" else SequentialSampler(dataset))
    from openpi.policies.grounding_dataset import collate_grounding
    batches = ResumableBatchSampler(sampler, config.batch_size, drop_last=False)
    batches.sample_keys = config.training_stage == 'grounding' and split == 'train'
    return DataLoader(
        dataset, batch_sampler=batches,
        num_workers=config.num_workers,
        collate_fn=collate_grounding if config.training_stage == 'grounding' else collate_samples,
        generator=torch.Generator().manual_seed(config.validation_seed if split == 'val' else config.seed),
        pin_memory=config.device.startswith("cuda"), persistent_workers=config.num_workers > 0,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
    )
