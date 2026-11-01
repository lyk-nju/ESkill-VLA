"""Dataset construction, transforms and batching; no model initialization."""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data._utils.collate import default_collate

from openpi.policies.eskill_dataset import ESkillDataset
from openpi.policies.libero_dataset import LiberoDataset, LiberoSFTDataset
from openpi.policies.libero_policy import LiberoInputs, LiberoSFTInputs
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


def create_dataset(config, split="train", *, transform=True):
    ids = split_episodes(config.data_index, config.validation_fraction, config.seed)
    selected = None if split == "all" else ids[split]
    if config.data_format == "libero_sft":
        if config.model_type != "pi05" or config.platform != "libero":
            raise ValueError("libero_sft is an action-only LIBERO dataset")
        dataset = LiberoSFTDataset(config.data_index, action_horizon=config.model.action_horizon,
                                  episode_ids=selected)
    elif config.model_type == "pi05":
        dataset = LiberoDataset(
            config.data_index, action_horizon=config.model.action_horizon, data_format=config.data_format,
            episode_ids=selected, sample_manifest=config.sample_manifest, frame_stride=config.frame_stride,
            camera=config.camera)
    else:
        dataset = ESkillDataset(
            config.data_index, action_horizon=config.model.action_horizon,
            skill_names=config.model.skill_names, motion_queries=config.model.motion_queries,
            canvas_size=config.model.canvas_size, camera=config.camera,
            episode_ids=selected, frame_stride=config.frame_stride)
    adapter = {"libero": LiberoInputs, "robotwin": RobotwinInputs}[config.platform]()
    if config.data_format == "libero_sft":
        adapter = LiberoSFTInputs()
    transforms = None
    if transform:
        transforms = ModelTransforms(
            config.model, config.norm_stats,
            TokenizePrompt(config.tokenizer_path, config.model.max_token_len))
    return TransformedDataset(dataset, adapter, transforms)


def collate_samples(samples):
    return dict(inputs=default_collate([s["inputs"] for s in samples]),
                targets=default_collate([s["targets"] for s in samples]),
                metadata=[s["metadata"] for s in samples])


def create_data_loader(config, split="train"):
    dataset = create_dataset(config, split)
    if len(dataset) == 0:
        raise ValueError(f"No samples in {split} split")
    # Keep sampling independent of worker seeding, including persistent-worker resets.
    sampler = (RandomSampler(dataset, generator=torch.Generator().manual_seed(config.seed))
               if split == "train" else None)
    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=split == "train", seed=config.seed)
    return DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler,
        num_workers=config.num_workers, collate_fn=collate_samples,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=config.device.startswith("cuda"), persistent_workers=config.num_workers > 0,
        multiprocessing_context="spawn" if config.num_workers > 0 else None,
    )
