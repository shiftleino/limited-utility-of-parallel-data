from torch.utils.data import DataLoader, DistributedSampler
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from datasets import load_from_disk
from olmo.data.iterable_dataset import IterableDataset
from olmo.torch_util import barrier, get_global_rank, get_world_size
from olmo.exceptions import OLMoConfigurationError
from olmo.config import TrainConfig, DataConfig
from functools import partial

def collate_fn(batch, pad_token_id: int):
    input_ids = np.array([item['input_ids'] for item in batch])
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    attention_mask = (input_ids != pad_token_id).long()

    labels = input_ids.clone().detach()
    labels = torch.roll(labels, shifts=-1, dims=1)
    labels[:, -1] = pad_token_id

    labels[labels == pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_train_dataloader(
        train_config: TrainConfig,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        fs_local_rank: Optional[int] = None,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    
    hf_dataset = load_from_disk(train_config.data.dataset_path)

    dataset = IterableDataset(
        hf_dataset,
        train_config.global_train_batch_size,
        seed=seed,
        epoch=train_config.epoch or 0,
        shuffle=False,
        drop_last=train_config.data.drop_last,
        world_size=world_size,
        rank=rank,
        fs_local_rank=fs_local_rank,
        work_dir=work_dir,
        save_overwrite=train_config.save_overwrite,
    )
    barrier()
    out = DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=partial(collate_fn, pad_token_id=train_config.model.pad_token_id),
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    return out


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    hf_dataset = load_from_disk(data_config.dataset_path)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(hf_dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.dataset_path} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        hf_dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        hf_dataset,
        batch_size=batch_size,
        collate_fn=partial(collate_fn, pad_token_id=train_config.model.pad_token_id),
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )
