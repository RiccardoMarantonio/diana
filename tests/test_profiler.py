import json
import os

import torch

from diana.profiler import peak_memory_mb, profile_inference, profile_training
from diana.utils.device import resolve_device
from tests.test_train import _cfg


def test_profile_training_report_structure(tmp_path):
    cfg = _cfg(data_path="synthetic", batch_size=2, num_workers=0, grad_accum_steps=1)
    device = resolve_device("cpu")
    res = profile_training(cfg, device, steps=2, with_profiler=False)
    rep = res["report"]
    assert rep["mode"] == "train"
    assert rep["device"] == "cpu"
    assert rep["steps"] == 2
    assert rep["step_per_sec"] > 0
    assert "mean_loss" in rep and "peak_memory_mb" in rep


def test_profile_inference_report(tmp_path):
    cfg = _cfg(data_path="synthetic", batch_size=2, num_workers=0)
    res = profile_inference(cfg, resolve_device("cpu"), steps=1, run_dir=None,
                            t_start=8, num_steps=8)
    rep = res["report"]
    assert rep["mode"] == "inference"
    assert rep["images"] == 2
    assert rep["images_per_sec"] > 0


def test_sweep_json_roundtrip(tmp_path):
    from diana.profiler import sweep
    cfg = _cfg(data_path="synthetic", batch_size=2, num_workers=0)
    out = os.path.join(str(tmp_path), "prof.json")
    path = sweep(cfg, resolve_device("cpu"), steps=2, out=out)
    with open(path) as f:
        data = json.load(f)
    assert data["reports"][0]["mode"] == "train"
    assert data["reports"][0]["variant"] == "amp=0_bench=0"


def test_peak_memory_never_raises():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    for dev in devices:
        val = peak_memory_mb(resolve_device(dev))
        assert val is None or isinstance(val, float)

def test_profile_repeats_loader_to_requested_optimizer_steps():
    cfg = _cfg(batch_size=256, grad_accum_steps=2, img_size=4,
               channel_mults=[1], attention_resolutions=[4], base_channels=8)
    res = profile_training(cfg, torch.device("cpu"), 3, with_profiler=False)["report"]
    assert res["steps"] == 3
    assert res["images"] == 3 * 2 * 256
    assert res["images_per_sec"] > 0
    assert res["warmup_steps"] == 2


def test_sweep_applies_and_restores_backend_flags(monkeypatch, tmp_path):
    from diana import profiler
    seen = []
    previous = torch.backends.cudnn.benchmark

    def profile(cfg, device, steps, with_profiler):
        seen.append((cfg.cudnn_benchmark, torch.backends.cudnn.benchmark))
        return {"report": {"mode": "train", "device": "cuda", "step_per_sec": 1,
                           "peak_memory_mb": 0}, "op_table": ""}

    monkeypatch.setattr(profiler, "profile_training", profile)
    profiler.sweep(_cfg(), torch.device("cuda"), 2, str(tmp_path / "out.json"))
    assert seen == [(False, False), (True, True), (False, False), (True, True)]
    assert torch.backends.cudnn.benchmark == previous


def test_memory_units_are_platform_specific(monkeypatch):
    import resource
    from types import SimpleNamespace

    from diana import profiler
    monkeypatch.setattr(resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1048576))
    monkeypatch.setattr(profiler.sys, "platform", "darwin")
    assert peak_memory_mb(torch.device("cpu")) == 1.
    monkeypatch.setattr(profiler.sys, "platform", "linux")
    assert peak_memory_mb(torch.device("cpu")) == 1024.


def test_mps_synchronization(monkeypatch):
    from diana.profiler import synchronize
    calls = []
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append(True))
    synchronize(torch.device("mps"))
    assert calls == [True]
