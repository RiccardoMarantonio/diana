import os

import numpy as np
import torch
from PIL import Image

from diana.data.mvtec import _load_mask
from diana.models.ema import EMA
from diana.sample import (
    auc_sorted,
    auroc,
    load_checkpoint_run,
    make_grid,
    run_inference,
)
from diana.train import build_ddpm, build_optimizer, save_checkpoint
from diana.utils.amp import GradScaler
from tests.test_train import _cfg


def _make_png(path, color, size=128):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (size, size), color).save(path)


def _make_mask_png(path, box, size=128):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = Image.new("L", (size, size), 0)
    img.paste(255, box)
    img.save(path)


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc(torch.tensor([0.1, 0.2, 0.9, 0.8]), torch.tensor([0, 0, 1, 1])) == 1.0

    def test_inverted_is_zero(self):
        assert auroc(torch.tensor([0.9, 0.8, 0.1, 0.2]), torch.tensor([0, 0, 1, 1])) == 0.0

    def test_single_class_raises(self):
        try:
            auroc(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


class TestAucSorted:
    def test_agrees_with_pairwise_on_random(self):
        rng = torch.Generator().manual_seed(7)
        for n in (50, 200):
            scores = torch.rand(n, generator=rng)
            labels = torch.randint(0, 2, (n,), generator=rng)
            if len(torch.unique(labels)) < 2:
                labels[0] = 0
                labels[-1] = 1
            a = auroc(scores, labels)
            b = auc_sorted(scores, labels)
            assert abs(a - b) < 1e-6

    def test_ties_use_average_ranks(self):
        # scores: [1, 1, 0, 0], labels: [1, 0, 1, 0] -> each positive ties its
        # counterpart pair and beats nothing else -> AUC = 0.5
        assert auc_sorted(torch.tensor([1.0, 1.0, 0.0, 0.0]), torch.tensor([1, 0, 1, 0])) == 0.5


class TestMaskTransform:
    def test_alignment_with_image_crop(self, tmp_path):
        # Defect in the top-left corner; resized by /4 it sits at rows 0..4,
        # fully above the center-crop window (rows 8..), so it drops out.
        img = f"{tmp_path}/img.png"
        _make_png(img, (120, 120, 120))
        _make_mask_png(f"{tmp_path}/mask.png", (0, 0, 20, 20))
        m = _load_mask(f"{tmp_path}/mask.png", resize=32, crop=16)
        assert m.shape == (1, 16, 16)
        assert set(np.unique(m.numpy()).tolist()) <= {0.0, 1.0}
        assert float(m.sum()) == 0.0

    def test_mask_keeps_center_defect(self, tmp_path):
        # Central 24px block -> resized rows 13..18 -> inside the 16px crop.
        _make_png(f"{tmp_path}/img.png", (120, 120, 120))
        _make_mask_png(f"{tmp_path}/mask.png", (52, 52, 76, 76))
        m = _load_mask(f"{tmp_path}/mask.png", resize=32, crop=16)
        assert float(m.sum()) > 0.0
        assert set(np.unique(m.numpy()).tolist()) <= {0.0, 1.0}


class TestGrid:
    def test_make_grid_shape(self):
        o = torch.rand(3, 8, 8) * 2 - 1
        r = torch.rand(3, 8, 8) * 2 - 1
        a = torch.rand(8, 8)
        grid = make_grid([(o, r, a), (o, r, a)], cols=2)
        assert grid.shape == (16, 24, 3)
        assert grid.dtype == np.uint8


class TestCheckpointRoundtrip:
    def _write_checkpoint(self, tmp_path):
        cfg = _cfg(data_path="synthetic", checkpoint_dir=str(tmp_path))
        model = build_ddpm(cfg)
        opt = build_optimizer(cfg, model)
        ema = EMA(model, cfg.ema_decay)
        ema.update(model)
        ema.update(model)
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        run_dir = os.path.join(str(tmp_path), "run")
        save_checkpoint(run_dir, "best.pt", cfg, model, opt, scaler, ema, epoch=1, global_step=2, best_loss=0.1)
        return cfg, model, ema, run_dir

    def test_load_restores_ema_weights(self, tmp_path):
        cfg, model, ema, run_dir = self._write_checkpoint(tmp_path)
        cfg2, model2, ema2 = load_checkpoint_run(run_dir, tag="best")
        assert cfg2 == cfg
        assert ema2.steps == 2
        ema.apply(model)
        ema2.apply(model2)
        for pa, pb in zip(model.parameters(), model2.parameters()):
            assert torch.equal(pa, pb)


def _fake_mvtec(tmp_path):
    root = str(tmp_path / "mvtec")
    good = "hazelnut/test/good"
    crack = "hazelnut/test/crack"
    _make_png(f"{root}/{good}/a.png", (128, 128, 128))
    _make_png(f"{root}/{good}/b.png", (60, 60, 60))
    _make_png(f"{root}/{crack}/c.png", (222, 10, 10))
    _make_png(f"{root}/{crack}/d.png", (250, 250, 10))
    _make_mask_png(f"{root}/hazelnut/ground_truth/crack/c_mask.png", (40, 40, 60, 60))
    _make_mask_png(f"{root}/hazelnut/ground_truth/crack/d_mask.png", (40, 40, 60, 60))
    return root


class TestEndToEndInference:
    def test_scores_grid_and_pixel_auroc(self, tmp_path):
        root = _fake_mvtec(tmp_path)
        cfg = _cfg(data_path=root, category="hazelnut", checkpoint_dir=str(tmp_path), epochs=1)
        model = build_ddpm(cfg)
        opt = build_optimizer(cfg, model)
        ema = EMA(model, cfg.ema_decay)
        ema.update(model)
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        run_dir = os.path.join(str(tmp_path), "run")
        save_checkpoint(run_dir, "best.pt", cfg, model, opt, scaler, ema, epoch=0, global_step=0, best_loss=1.0)

        out = os.path.join(str(tmp_path), "grid.png")
        result = run_inference(run_dir, tag="best", t_start=8, num_steps=8, out=out, limit=4, pixel=True)
        assert os.path.isfile(out)
        assert set(result) >= {"image_auroc", "pixel_auroc", "n", "good_mean", "defect_mean"}
        assert result["n"] == 4
        assert result["pixel_auroc"] is not None
        assert 0.0 <= result["pixel_auroc"] <= 1.0
        assert os.path.isfile(os.path.join(run_dir, "eval.json"))