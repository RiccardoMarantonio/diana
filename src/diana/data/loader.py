"""Dataloader factory.

Serves the training loop a DataLoader regardless of the data source:
``data_path == "synthetic"`` yields random-image tensors sized from the config
(smoke/profiling runs), while any real path routes to the MVTec dataset for
``config.category`` behind the same :func:`make_dataloader` contract.
"""

import os

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

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

    def __getitem__(self, index: int | tuple[int, int]) -> torch.Tensor:
        epoch, item = index if isinstance(index, tuple) else (0, index)
        generator = torch.Generator().manual_seed(self._config.seed + epoch * self._length + item)
        return torch.rand(
            self._config.img_size,
            self._config.img_size,
            3,
            generator=generator,
        ).mul_(2.0).sub_(1.0).permute(2, 0, 1)


class EpochSampler(Sampler):
    """Pass epoch with each index so persistent workers need no mutable RNG state."""

    def __init__(self, length: int, seed: int):
        self.length = length
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter((self.epoch, i) for i in torch.randperm(self.length, generator=generator).tolist())


def make_dataloader(config: Config) -> DataLoader:
    if config.data_path == SYNTHETIC_SENTINEL:
        dataset: Dataset = SyntheticDataset(config)
    elif os.path.isdir(os.path.join(config.data_path, config.category, "train")):
        dataset = MVTecDataset(
            config.data_path,
            config.category,
            img_size=config.img_size,
            augment=config.augment_hflip,
            seed=config.seed,
        )
    else:
        raise RuntimeError(
            f"no dataset at {config.data_path!r}; pass --data_path synthetic for a "
            f"random-tensor smoke run, or fetch a category with: "
            f"python -m diana.data.download --categories {config.category}"
        )
    if config.use_cuda_graphs and len(dataset) < config.batch_size:
        raise ValueError("CUDA graphs require at least one full batch; reduce batch_size")
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
        sampler=EpochSampler(len(dataset), config.seed),
        generator=torch.Generator().manual_seed(config.seed),
        drop_last=config.use_cuda_graphs,
    )