"""Dataset construction, transforms and batching; no model initialization."""
from itertools import islice

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from openpi.policies.eskill_dataset import EpisodeDataset
from openpi.policies.libero_policy import LiberoInputs, uses_arm_state
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

    def __iter__(self):
        return islice(super().__iter__(), self.start_batch, None)

    def __len__(self):
        return max(0, super().__len__() - self.start_batch)


def episode_dataset(config, episode_ids=None):
    """The same saved episodes for every recipe; ESkill also reads Event, region and Motion targets."""
    m = config.model
    return EpisodeDataset(
        config.data_index, action_horizon=m.action_horizon, cameras=config.cameras,
        auxiliary=config.model_type == "eskill", action_padding=config.action_padding,
        skill_names=m.skill_names, motion_queries=m.motion_queries, motion_stride=m.motion_stride,
        motion_tail_policy=m.motion_tail_policy, image_size=m.image_size,
        episode_ids=episode_ids, frame_stride=config.frame_stride)


def input_adapter(config):
    if config.platform == "libero":
        return LiberoInputs(use_arm_state=uses_arm_state(config.model_type, config.model))
    return {"robotwin": RobotwinInputs}[config.platform]()


def create_dataset(config, split="train", *, transform=True):
    ids = split_episodes(config.data_index, config.validation_fraction, config.seed, config.split_path)
    dataset = episode_dataset(config, None if split == "all" else ids[split])
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
    sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(config.seed))
               if split == "train" else SequentialSampler(dataset))
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=split == "train", seed=config.seed)
    return DataLoader(
        dataset, batch_sampler=ResumableBatchSampler(sampler, config.batch_size, drop_last=False),
        num_workers=config.num_workers, collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(config.validation_seed if split == 'val' else config.seed),
        pin_memory=config.device.startswith("cuda"), persistent_workers=config.num_workers > 0,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
    )
