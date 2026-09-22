"""Benchmark harness for the HPC spec (throughput, peak memory, profiler).

Subcommands:

* ``train``      -- time N optimizer steps of the real training spine on
                    synthetic data (no dataset wait, so the figure is compute
                    + dataloader pipeline). Reports step/s, peak memory, mean
                    loss, and a torch.profiler op table.
* ``inference``  -- images/s through a full
                    :func:`diana.diffusion.ddpm.DDPM.reconstruct` chain at
                    the eval dialect.
* ``sweep``      -- CUDA-only comparison matrix across the {amp, cudnn_bench
                    mark} HPC switches so the T4 config is chosen by numbers.

All output lands in a JSON report; ``--out`` defaults to
``runs/<category>-prof-<ts>.json`` plus a sibling ``.prof.txt`` op table.

Usage::

    python -m diana.profiler train --config configs/hazelnut.toml --steps 50 [--device cuda]
    python -m diana.profiler inference --config configs/hazelnut.toml [--run <run_dir>]
    python -m diana.profiler sweep --config configs/hazelnut.toml
"""

import argparse
import contextlib
import dataclasses
import itertools
import json
import os
import time
from datetime import UTC, datetime
from typing import Any

import torch

from diana.config import Config, parse_args
from diana.data.loader import make_dataloader
from diana.diffusion.ddpm import DDPM
from diana.models.ema import EMA
from diana.sample import load_checkpoint_run
from diana.train import build_ddpm, build_optimizer, train_epoch
from diana.utils.amp import GradScaler
from diana.utils.cudagraphs import CudaGraphStep
from diana.utils.device import configure_backends, resolve_device, set_seed


def peak_memory_mb(device: torch.device) -> float | None:
    """Peak/footprint memory in MiB for the run just finished; None if unknown."""
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
    if device.type == "mps":
        try:
            return round(torch.mps.current_allocated_memory() / 1024**2, 2)
        except Exception:  # noqa: BLE001 - API varies across torch builds
            return None
    if device.type == "cpu":
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(rss / 1024 if rss < 2**40 else rss / 1024**2, 2)  # macos KiB, linux bytes
    return None


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except RuntimeError:
            return  # reporting footprint, not a hard failure


def _op_table(device: torch.device, prof) -> str:
    if prof is None:
        return ""
    try:
        key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
        return prof.key_averages().table(sort_by=key, row_limit=15)
    except Exception:  # noqa: BLE001
        return ""


def profile_training(config: Config, device: torch.device, steps: int | None, with_profiler: bool = True) -> dict[str, Any]:
    """Time the real training spine on synthetic data for ``steps`` micro-batches."""
    if config.data_path != "synthetic":
        config.data_path = "synthetic"  # profiling is about the loop, not dataset I/O
    set_seed(config.seed)
    model = build_ddpm(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = GradScaler(device, enabled=config.use_amp)
    ema = EMA(model, config.ema_decay)
    graph = CudaGraphStep(model, device, scaler, enabled=config.use_cuda_graphs)

    loader = make_dataloader(config)
    steps = len(loader) if steps is None else steps
    stream = itertools.islice(loader, steps)

    _reset_peak(device)
    prof = None
    ctx: Any = contextlib.nullcontext()
    if with_profiler:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            prof = torch.profiler.profile(activities=activities)
            ctx = prof
        except Exception:  # noqa: BLE001
            prof = None

    wall0 = time.perf_counter()
    with ctx:
        mean_loss, gs = train_epoch(model, stream, optimizer, scaler, ema, config, 0, 0, graph)
    wall = time.perf_counter() - wall0

    report = {
        "mode": "train",
        "device": device.type,
        "amp": bool(scaler.enabled),
        "cudnn_benchmark": bool(config.cudnn_benchmark),
        "use_cuda_graphs": bool(config.use_cuda_graphs),
        "steps": gs,
        "seconds": round(wall, 4),
        "step_per_sec": round(gs / wall, 3) if wall else None,
        "mean_loss": round(float(mean_loss), 6),
        "peak_memory_mb": peak_memory_mb(device),
    }
    return {"report": report, "op_table": _op_table(device, prof)}


def profile_inference(
    config: Config,
    device: torch.device,
    steps: int,
    run_dir: str | None,
    t_start: int | None,
    num_steps: int | None,
) -> dict[str, Any]:
    """Images/s through the eval reconstruct chain (optionally a real checkpoint)."""
    set_seed(config.seed)
    model: DDPM
    if run_dir:
        cfg_from, model, _ = load_checkpoint_run(run_dir, tag="best")
        t_start = cfg_from.eval_t_start_effective
        num_steps = cfg_from.sample_timesteps
    else:
        model = build_ddpm(config)
        t_start = config.eval_t_start_effective if t_start is None else t_start
        num_steps = config.sample_timesteps if num_steps is None else num_steps
    model.to(device)
    model.eval()

    batch = config.batch_size
    x = torch.randn(batch, 3, config.img_size, config.img_size, device=device)
    with torch.no_grad():
        model.reconstruct(x, num_steps=num_steps, t_start=t_start)  # warmup
        if device.type == "cuda":
            torch.cuda.synchronize()
        _reset_peak(device)
        wall0 = time.perf_counter()
        for _ in range(steps):
            model.reconstruct(x, num_steps=num_steps, t_start=t_start)
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - wall0

    report = {
        "mode": "inference",
        "device": device.type,
        "category": config.category,
        "t_start": int(t_start),
        "num_steps": int(num_steps),
        "batch_size": batch,
        "images": batch * steps,
        "seconds": round(wall, 4),
        "images_per_sec": round(batch * steps / wall, 3) if wall else None,
        "peak_memory_mb": peak_memory_mb(device),
    }
    return {"report": report, "op_table": ""}


def _emit(results: list[dict[str, Any]], out: str | None) -> str:
    reports = [r["report"] for r in results]
    out = out or os.path.join(
        "runs",
        f"{reports[0]['category']}-prof-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}.json",
    )
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"reports": reports}, f, indent=2, sort_keys=True)
    for r, table in zip(results, [r["op_table"] for r in results]):
        if table:
            txt_path = os.path.splitext(out)[0] + ".prof.txt"
            with open(txt_path, "w") as f:
                f.write(table)
            print(f"[profiler] op table -> {txt_path}")
    for r in reports:
        rate_key = "step_per_sec" if r["mode"] == "train" else "images_per_sec"
        print(f"{r['mode']:>9} | {r['device']:>4} | {rate_key}={r[rate_key]} | mem={r['peak_memory_mb']} MiB")
    print(f"[profiler] report -> {out}")
    return out


def sweep(config: Config, device: torch.device, steps: int | None, out: str | None) -> str:
    """Compare {amp} x {cudnn_benchmark} combos; single row off-CUDA."""
    combos = [(False, False)] if device.type != "cuda" else list(itertools.product([False, True], [False, True]))
    results = []
    for amp, bench in combos:
        cfg = dataclasses.replace(config, use_amp=amp, cudnn_benchmark=bench)
        res = profile_training(cfg, device, steps, with_profiler=False)
        res["report"]["category"] = config.category
        res["report"]["variant"] = f"amp={int(amp)}_bench={int(bench)}"
        results.append(res)
    return _emit(results, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["train", "inference", "sweep"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run", type=str, default=None, help="Run dir for inference benchmarking")
    parser.add_argument("--t_start", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--no_profiler", action="store_true", help="Skip torch.profiler table")
    args = parser.parse_args(argv)

    cfg = parse_args(["--config", args.config])
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)
    configure_backends(device, benchmark=cfg.cudnn_benchmark)

    if args.mode == "train":
        res = profile_training(cfg, device, args.steps, with_profiler=not args.no_profiler)
        res["report"]["category"] = cfg.category
        _emit([res], args.out)
    elif args.mode == "inference":
        res = profile_inference(cfg, device, args.steps or 3, args.run, args.t_start, args.num_steps)
        _emit([res], args.out)
    else:
        sweep(cfg, device, args.steps, args.out)


if __name__ == "__main__":
    main()