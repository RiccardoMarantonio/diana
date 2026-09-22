"""Dataloader factory.

Serves the training loop a DataLoader regardless of the data source. The
``"synthetic"`` sentinel yields random-image tensors sized from the config, so
the full train/smoke path is exercisable before the real MVTec dataset is
wired in behind the same :func:`make_dataloader` contract (Phase 4).
"""

import torch
from torch.utils.data import DataLoader, Dataset

from diana.config import Config
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
    else:
        # Real MVTec loading replaced by this branch in Phase 4.
        raise NotImplementedError(
            f"only data_path={SYNTHETIC_SENTINEL!r} is wired so far; got "
            f"{config.data_path!r}"
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )