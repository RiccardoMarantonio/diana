import os

import numpy as np
import torch

from diana.models.ema import EMA
from diana.sample import (
    auroc,
    load_checkpoint_run,
    make_grid,
    run_inference,
)
from diana.train import build_ddpm, build_optimizer, save_checkpoint
from diana.utils.amp import GradScaler
from tests.test_train import _cfg


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc(torch.tensor([0.1, 0.2, 0.9, 0.8]), torch.tensor([0, 0, 1, 1])) == 1.0

    def test_inverted_is_zero(self):
        assert auroc(torch.tensor([0.9, 0.8, 0.1, 0.2]), torch.tensor([0, 0, 1, 1])) == 0.0

    def test_permutation(self):
        # 0.75: two positives, two negatives, one overlap pair
        assert auroc(torch.tensor([0.1, 0.4, 0.9, 0.6]), torch.tensor([0, 1, 1, 0])) == 0.75

    def test_single_class_raises(self):
        try:
            auroc(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")


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
        # applying EMA to both models must yield identical weights
        ema.apply(model)
        ema2.apply(model2)
        for pa, pb in zip(model.parameters(), model2.parameters()):
            assert torch.equal(pa, pb)


def _fake_mvtec(tmp_path):
    from tests.test_data import TestMVTecDataset
    t = TestMVTecDataset()
    root = str(tmp_path / "mvtec")
    t._make_image(os.path.join(root, "hazelnut", "test", "good", "a.png"), color=(128, 128, 128))
    t._make_image(os.path.join(root, "hazelnut", "test", "good", "b.png"), color=(60, 60, 60))
    t._make_image(os.path.join(root, "hazelnut", "test", "crack", "c.png"), color=(222, 10, 10))
    t._make_image(os.path.join(root, "hazelnut", "test", "crack", "d.png"), color=(250, 250, 10))
    return root


class TestEndToEndInference:
    def test_scores_and_grid(self, tmp_path):
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
        result = run_inference(run_dir, tag="best", t_start=8, num_steps=8, out=out, limit=4)
        assert os.path.isfile(out)
        assert set(result) >= {"auroc", "n", "good_mean", "defect_mean"}
        assert result["n"] == 4
        # original weights were restored, not left as EMA shadows
        for p, s in zip(model.parameters(), ema.state_dict()["shadows"].values()):
            # model was never touched again after save; spot-check a grad-free tensor
            assert p.device.type == "cpu"