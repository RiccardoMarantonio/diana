import os

import pytest
import torch

from diana.config import Config
from diana.data.loader import make_dataloader
from diana.models.ema import EMA
from diana.train import (
    BEST_CKPT,
    LAST_CKPT,
    RUN_CONFIG,
    build_ddpm,
    build_optimizer,
    load_checkpoint,
    lr_at_step,
    main,
    run_training,
    save_checkpoint,
    train_epoch,
)
from diana.utils.amp import GradScaler

SMALL: dict = {
    "data_path": "synthetic",
    "category": "testcat",
    "img_size": 16,
    "batch_size": 4,
    "num_workers": 0,
    "base_channels": 8,
    "channel_mults": [1, 2],
    "num_res_blocks": 1,
    "attention_resolutions": [8],
    "schedule_type": "linear",
    "num_timesteps": 32,
    "sample_timesteps": 8,
    "epochs": 2,
    "learning_rate": 1e-3,
    "lr_warmup_steps": 5,
    "grad_accum_steps": 2,
    "device": "cpu",
    "seed": 7,
    "grad_clip": 1.0,
}


def _cfg(**overrides):
    return Config(**{**SMALL, **overrides})


class TestLr:
    def test_warmup_ramps_then_flat(self):
        cfg = _cfg(lr_warmup_steps=4, learning_rate=0.1)
        assert lr_at_step(0, cfg) == 0.0
        assert lr_at_step(2, cfg) == pytest.approx(0.05)
        assert lr_at_step(10, cfg) == 0.1

    def test_zero_warmup_is_constant(self):
        cfg = _cfg(lr_warmup_steps=0, learning_rate=0.1)
        assert lr_at_step(0, cfg) == 0.1
        assert lr_at_step(999, cfg) == 0.1


class TestBuild:
    def test_build_ddpm(self):
        cfg = _cfg()
        model = build_ddpm(cfg)
        assert model.schedule.num_timesteps == 32

    def test_epoch_runs_and_steps_advance(self):
        cfg = _cfg()
        model = build_ddpm(cfg)
        opt = build_optimizer(cfg, model)
        scaler = GradScaler(torch.device("cpu"), enabled=False)
        ema = EMA(model, cfg.ema_decay)
        loader = make_dataloader(cfg)
        mean_loss, step = train_epoch(model, loader, opt, scaler, ema, cfg, epoch=0, global_step=0)
        assert torch.isfinite(torch.tensor(mean_loss))
        assert step == len(loader) // cfg.grad_accum_steps
        assert ema.steps == step  # once per optimizer step


class TestCheckpointing:
    def test_roundtrip_restores_state(self, tmp_path):
        cfg = _cfg()
        model_a = build_ddpm(cfg)
        opt_a = build_optimizer(cfg, model_a)
        ema_a = EMA(model_a, cfg.ema_decay)
        ema_a.update(model_a)
        ema_a.update(model_a)
        scaler_a = GradScaler(torch.device("cpu"), enabled=False)
        for i, p in enumerate(model_a.parameters()):
            p.data.fill_(float(i % 3))

        path = save_checkpoint(str(tmp_path), "ckpt", cfg, model_a, opt_a, scaler_a, ema_a, epoch=3, global_step=11, best_loss=0.25)
        assert os.path.exists(path)

        model_b = build_ddpm(cfg)
        opt_b = build_optimizer(cfg, model_b)
        ema_b = EMA(model_b, cfg.ema_decay)
        scaler_b = GradScaler(torch.device("cpu"), enabled=False)
        epoch, step, best = load_checkpoint(path, cfg, model_b, opt_b, scaler_b, ema_b)
        assert (epoch, step, best) == (3, 11, 0.25)

        for pa, pb in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(pa, pb)
        assert ema_b.steps == ema_a.steps


class TestEndToEnd:
    def test_run_training_writes_outputs(self, tmp_path):
        cfg = _cfg(checkpoint_dir=str(tmp_path), epochs=2, log_every_n_steps=100)
        results = run_training(cfg, torch.device("cpu"))
        assert len(results["losses"]) == 2
        assert results["best_loss"] == pytest.approx(min(results["losses"]))

        runs = [d for d in os.listdir(tmp_path) if d != ".DS_Store"]
        assert len(runs) == 1
        run_dir = os.path.join(tmp_path, runs[0])
        assert os.path.exists(os.path.join(run_dir, RUN_CONFIG))
        assert os.path.exists(os.path.join(run_dir, LAST_CKPT))
        assert os.path.exists(os.path.join(run_dir, BEST_CKPT))

    def test_resume_continuous_with_interrupted(self, tmp_path):
        """An interrupted run resumed from last.pt matches an uninterrupted run."""
        cfg_a = _cfg(checkpoint_dir=str(tmp_path / "a"), epochs=3)
        cfg_b = _cfg(checkpoint_dir=str(tmp_path / "b"), epochs=1)  # interrupted
        run_training(cfg_a, torch.device("cpu"))
        run_training(cfg_b, torch.device("cpu"))
        run_b = os.path.join(str(tmp_path / "b"), os.listdir(tmp_path / "b")[0])
        assert os.path.exists(os.path.join(run_b, LAST_CKPT))
        cfg_c = _cfg(
            checkpoint_dir=str(tmp_path / "b"),
            epochs=3,
            resume_from=run_b,
        )
        run_training(cfg_c, torch.device("cpu"))

        run_a = os.path.join(str(tmp_path / "a"), os.listdir(tmp_path / "a")[0])
        runs_c = [d for d in os.listdir(tmp_path / "b") if os.path.isdir(os.path.join(tmp_path / "b", d))]
        run_c = os.path.join(
            str(tmp_path / "b"),
            max(runs_c, key=lambda d: os.path.getmtime(os.path.join(tmp_path / "b", d))),
        )
        loss_a = torch.load(os.path.join(run_a, BEST_CKPT), map_location="cpu")["best_loss"]
        loss_c = torch.load(os.path.join(run_c, BEST_CKPT), map_location="cpu")["best_loss"]
        assert loss_a == pytest.approx(loss_c)

    def test_main_cli_smoke(self, tmp_path):
        cli = [
            "--data_path", "synthetic",
            "--category", "smoke",
            "--img_size", "16",
            "--batch_size", "2",
            "--num_workers", "0",
            "--base_channels", "8",
            "--channel_mults", "1", "2",
            "--num_res_blocks", "1",
            "--attention_resolutions", "8",
            "--num_timesteps", "16",
            "--sample_timesteps", "16",
            "--epochs", "1",
            "--lr_warmup_steps", "2",
            "--grad_accum_steps", "1",
            "--checkpoint_dir", str(tmp_path),
            "--save_every_n_epochs", "1",
            "--device", "cpu",
            "--seed", "3",
        ]
        main(cli)
        run_dir = os.path.join(str(tmp_path), os.listdir(tmp_path)[0])
        assert os.path.exists(os.path.join(run_dir, BEST_CKPT))