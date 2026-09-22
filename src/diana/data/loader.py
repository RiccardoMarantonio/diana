"""Dataloader factory.

Serves the training loop a DataLoader regardless of the data source:
``data_path == "synthetic"`` yields random-image tensors sized from the config
(smoke/profiling runs), while any real path routes to the MVTec dataset for
``config.category`` behind the same :func:`make_dataloader` contract.
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset

from diana.config import Config
from diana.data.mvtec import MVTecDataset
from diana.utils.device import worker_init_fn

SYNTHETIC_SENTINEL = "synthetic"


class SyntheticDataset(Dataset):
    """On-the-fly random images in [-1, 1]; deterministic per worker seed."""

    def __init__(self, config: Config, length: int = 512):
        self._config = config
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.rand(
            self._config.img_size,
            self._config.img_size,
            3,
        ).mul_(2.0).sub_(1.0).permute(2, 0, 1)


def make_dataloader(config: Config) -> DataLoader:
    if config.data_path == SYNTHETIC_SENTINEL:
        dataset: Dataset = SyntheticDataset(config)
    elif os.path.isdir(os.path.join(config.data_path, config.category, "train")):
        dataset = MVTecDataset(
            config.data_path,
            config.category,
            img_size=config.img_size,
            augment=config.augment_hflip,
        )
    else:
        raise RuntimeError(
            f"no dataset at {config.data_path!r}; pass --data_path synthetic for a "
            f"random-tensor smoke run, or fetch a category with: "
            f"python -m diana.data.download --categories {config.category}"
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        # Keep workers alive across epochs: forking + reaping worker processes
        # in every iteration slowly leaks pipes, and on systems with a modest
        # fd soft limit (e.g. macOS's default 256) long runs eventually die
        # with EMFILE ("Too many open files").
        persistent_workers=config.persistent_workers and config.num_workers > 0,
        pin_memory=config.pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )