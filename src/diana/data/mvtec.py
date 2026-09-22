"""MVTec AD dataset for 64x64-style diffusion training.

Layout expected under ``root`` (produced by ``diana.data.download``)::

    <root>/<category>/train/good/*.png       -- defect-free training images

Preprocessing follows the agreed contract: resize to twice ``img_size`` and
center-crop down to ``img_size`` (kills edge artifacts while keeping texture),
then map to [-1, 1].
"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
}


class MVTecDataset(Dataset):
    """Training split (defect-free ``good`` images) of one MVTec category."""

    def __init__(self, root: str, category: str, img_size: int = 64, augment: bool = False):
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"unknown MVTec category {category!r}; expected one of {sorted(MVTEC_CATEGORIES)}"
            )
        good_dir = os.path.join(root, category, "train", "good")
        if not os.path.isdir(good_dir):
            raise RuntimeError(
                f"no MVTec training images at {good_dir!r}; fetch them with: "
                f"python -m diana.data.download --categories {category}"
            )
        self._paths = sorted(
            os.path.join(good_dir, f)
            for f in os.listdir(good_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        )
        if not self._paths:
            raise RuntimeError(f"no image files found in {good_dir!r}")
        self._resize = 2 * img_size
        self._crop = img_size
        self._augment = augment

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self._paths[index]).convert("RGB")
        img = img.resize((self._resize, self._resize), Image.Resampling.BILINEAR)
        if self._resize != self._crop:
            left = top = (self._resize - self._crop) // 2
            img = img.crop((left, top, left + self._crop, top + self._crop))
        out = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
        out = out.permute(2, 0, 1)
        if self._augment and torch.rand(1).item() < 0.5:
            out = torch.flip(out, dims=[2])
        return out