import hashlib
import os

import pytest
import torch

from diana.config import Config
from diana.data.download import (
    fetch_one,
    needs_download,
    plan_category,
    sha256_file,
)
from diana.data.loader import make_dataloader
from diana.data.mvtec import MVTecDataset

FAKE_TREE = [
    {"type": "file", "path": "hazelnut/train/good/000.png", "size": 42,
     "lfs": {"oid": "ab" * 32}},
    {"type": "directory", "path": "hazelnut/test"},
    {"type": "file", "path": "hazelnut/readme.txt", "size": 5, "lfs": {}},
]


class TestDownloadPlan:
    def test_filters_and_annotates(self, monkeypatch):
        monkeypatch.setattr(
            "diana.data.download._fetch_json", lambda url: FAKE_TREE
        )
        plan = plan_category("diana-test", "hazelnut", verify=False)
        assert [p["path"] for p in plan] == [
            "hazelnut/train/good/000.png", "hazelnut/readme.txt",
        ]
        assert plan[0]["sha256"] is None

    def test_verify_populates_sha(self, monkeypatch):
        monkeypatch.setattr(
            "diana.data.download._fetch_json", lambda url: FAKE_TREE
        )
        plan = plan_category("diana-test", "hazelnut", verify=True)
        assert plan[0]["sha256"] == "ab" * 32

    def test_unknown_category_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "diana.data.download._fetch_json",
            lambda url: pytest.fail("should not call network"),
        )
        with pytest.raises(ValueError, match="category"):
            plan_category("diana-test", "galaxy-nut", verify=False)


class TestDownloadHelpers:
    def test_needs_download_missing(self, tmp_path):
        p = os.path.join(tmp_path, "nope.png")
        assert needs_download(p, 100) is True
        assert needs_download(p, None) is True

    def test_needs_download_matches_size(self, tmp_path):
        p = os.path.join(tmp_path, "f.bin")
        with open(p, "wb") as f:
            f.write(b"x" * 10)
        assert needs_download(p, 10) is False
        assert needs_download(p, 11) is True
        assert needs_download(p, None) is False

    def test_sha256_file(self, tmp_path):
        p = os.path.join(tmp_path, "h.bin")
        with open(p, "wb") as f:
            f.write(b"abc")
        assert sha256_file(p) == hashlib.sha256(b"abc").hexdigest()

    def test_fetch_one_skips_complete(self, tmp_path):
        target = os.path.join(tmp_path, "hazelnut", "readme.txt")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(b"data")
        rel, state = fetch_one("repo-x", "hazelnut/readme.txt", str(tmp_path), 4, None)
        assert state == "ok"
        assert rel == "hazelnut/readme.txt"


class TestMVTecDataset:
    def _make_image(self, path, color=(128, 128, 128)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        from PIL import Image
        Image.new("RGB", (8, 8), color).save(path)

    def _build_root(self, tmp_path):
        root = str(tmp_path / "mvtec")
        for i in range(3):
            self._make_image(
                os.path.join(root, "hazelnut", "train", "good", f"{i:03d}.png")
            )
        return root

    def test_loads_images_and_crops(self, tmp_path):
        root = self._build_root(tmp_path)
        ds = MVTecDataset(root, "hazelnut", img_size=4)
        assert len(ds) == 3
        x = ds[0]
        assert x.shape == (3, 4, 4)
        assert x.dtype == torch.float32
        assert x.min() >= -1.0 and x.max() <= 1.0

    def test_resize_then_center_crop_contract(self, tmp_path):
        root = self._build_root(tmp_path)
        ds_large = MVTecDataset(root, "hazelnut", img_size=16, augment=False)
        assert ds_large[0].shape == (3, 16, 16)
        # gray image: the double-resize + center-crop must stay constant
        x = ds_large[0]
        assert torch.unique(x).numel() == 1

    def test_missing_data_gives_actionable_error(self, tmp_path):
        with pytest.raises(RuntimeError, match="download"):
            MVTecDataset(str(tmp_path), "hazelnut", img_size=16)

    def test_bad_category(self, tmp_path):
        with pytest.raises(ValueError, match="category"):
            MVTecDataset(str(tmp_path), "hazelnuts", img_size=16)

    def test_loader_wires_dataset(self, tmp_path):
        root = self._build_root(tmp_path)
        cfg = Config(data_path=root, category="hazelnut", img_size=16,
                     batch_size=2, num_workers=0)
        loader = make_dataloader(cfg)
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0].shape == (2, 3, 16, 16)

    def test_loader_synthetic_sentinel(self):
        cfg = Config(data_path="synthetic", category="x", img_size=16,
                     batch_size=2, num_workers=0)
        loader = make_dataloader(cfg)
        x = next(iter(loader))
        assert x.shape == (2, 3, 16, 16)


class TestPersistentWorkers:
    """Workers must live across epochs when possible; torch rejects
    persistent workers with num_workers == 0, so that combo auto-disables."""

    def test_enabled_with_workers(self):
        cfg = Config(data_path="synthetic", category="x", img_size=16,
                     num_workers=2)
        assert make_dataloader(cfg).persistent_workers is True

    def test_auto_disabled_with_zero_workers(self):
        cfg = Config(data_path="synthetic", category="x", img_size=16,
                     num_workers=0)
        assert make_dataloader(cfg).persistent_workers is False

    def test_explicitly_disabled(self):
        cfg = Config(data_path="synthetic", category="x", img_size=16,
                     num_workers=2, persistent_workers=False)
        assert make_dataloader(cfg).persistent_workers is False