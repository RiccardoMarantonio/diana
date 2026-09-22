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