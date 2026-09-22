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


def _load_image(path: str, resize: int, crop: int) -> torch.Tensor:
    """PIL decode -> RGB -> scale to `resize` -> center-crop to `crop` -> [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize((resize, resize), Image.Resampling.BILINEAR)
    if resize != crop:
        left = top = (resize - crop) // 2
        img = img.crop((left, top, left + crop, top + crop))
    out = torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0)
    return out.permute(2, 0, 1)


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
        out = _load_image(self._paths[index], self._resize, self._crop)
        if self._augment and torch.rand(1).item() < 0.5:
            out = torch.flip(out, dims=[2])
        return out


class MVTecEvalDataset(Dataset):
    """Test split of a category with ``label = 0`` for ``good`` and 1 otherwise.

    Same preprocessing as :class:`MVTecDataset`: defect-free and defective
    test images share the pipeline, so reconstruction residuals are comparable.
    """

    def __init__(self, root: str, category: str, img_size: int = 64):
        if category not in MVTEC_CATEGORIES:
            raise ValueError(
                f"unknown MVTec category {category!r}; expected one of {sorted(MVTEC_CATEGORIES)}"
            )
        test_dir = os.path.join(root, category, "test")
        if not os.path.isdir(test_dir):
            raise RuntimeError(
                f"no MVTec test images at {test_dir!r}; fetch them with: "
                f"python -m diana.data.download --categories {category}"
            )
        samples: list[tuple[str, int]] = []
        for sub in sorted(os.listdir(test_dir)):
            sub_dir = os.path.join(test_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            label = 0 if sub == "good" else 1
            for f in sorted(os.listdir(sub_dir)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    samples.append((os.path.join(sub_dir, f), label))
        if not samples:
            raise RuntimeError(f"no test images found in {test_dir!r}")
        self._samples = samples
        self._resize = 2 * img_size
        self._crop = img_size

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self._samples[index]
        return _load_image(path, self._resize, self._crop), label