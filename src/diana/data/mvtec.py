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


def _load_mask(path: str, resize: int, crop: int) -> torch.Tensor:
    """Binary segmentation mask through the *image-equivalent* transform.

    Same resize dimensions and center-crop offset as :func:`_load_image`, so
    mask pixels stay coordinate-aligned with the image; nearest-neighbour
    scaling keeps the mask exactly binary.
    """
    m = Image.open(path).resize((resize, resize), Image.Resampling.NEAREST)
    if resize != crop:
        left = top = (resize - crop) // 2
        m = m.crop((left, top, left + crop, top + crop))
    arr = np.asarray(m.convert("L"), dtype=np.float32)
    return torch.from_numpy((arr > 0.5 * 255).astype(np.float32)).unsqueeze(0)  # (1, H, W)


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
    """Test split of a category: ``(image, label, mask)`` per sample.

    ``label = 0`` for ``good`` images (zero mask) and 1 otherwise, with the
    binary ground-truth segmentation mask passed through the image-equivalent
    resize + center-crop so reconstruction residuals and GT pixels align.
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
        samples: list[tuple[str, int, str | None]] = []
        for sub in sorted(os.listdir(test_dir)):
            sub_dir = os.path.join(test_dir, sub)
            if not os.path.isdir(sub_dir):
                continue
            label = 0 if sub == "good" else 1
            gt_dir = os.path.join(root, category, "ground_truth", sub)
            for f in sorted(os.listdir(sub_dir)):
                if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    continue
                mask_path = None
                if label:
                    stem = os.path.splitext(f)[0]
                    for cand in (f"{stem}_mask.png", f"{stem}_mask.jpg", f"{stem}_mask.jpeg"):
                        if os.path.isfile(os.path.join(gt_dir, cand)):
                            mask_path = os.path.join(gt_dir, cand)
                            break
                samples.append((os.path.join(sub_dir, f), label, mask_path))
        if not samples:
            raise RuntimeError(f"no test images found in {test_dir!r}")
        self._samples = samples
        self._resize = 2 * img_size
        self._crop = img_size

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        path, label, mask_path = self._samples[index]
        image = _load_image(path, self._resize, self._crop)
        if mask_path is None:
            mask = torch.zeros(1, self._crop, self._crop)
        else:
            mask = _load_mask(mask_path, self._resize, self._crop)
        return image, label, mask