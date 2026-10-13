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
import math
import os
import sys
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
from diana.utils.inference_graph import InferenceGraph


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
        return round(rss / (1024**2 if sys.platform == "darwin" else 1024), 2)
    return None


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_scope(device: torch.device) -> str:
    return {"cpu": "process_lifetime_peak_rss", "mps": "current_allocated",
            "cuda": "measured_interval_peak_allocated"}[device.type]


def repeat_batches(loader):
    """Repeat data without caching tensors or exhausting a single epoch."""
    while True:
        yield from loader


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
    """Time exactly ``steps`` optimizer updates after two warmup updates."""
    config = dataclasses.replace(config, data_path="synthetic")
    configure_backends(device, benchmark=config.cudnn_benchmark)
    set_seed(config.seed)
    model = build_ddpm(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = GradScaler(device, enabled=config.use_amp)
    ema = EMA(model, config.ema_decay)
    graph = CudaGraphStep(model, device, scaler, enabled=config.use_cuda_graphs)

    loader = make_dataloader(config)
    if len(loader) == 0:
        raise ValueError("profiling loader is empty")
    steps = math.ceil(len(loader) / config.grad_accum_steps) if steps is None else steps
    if steps < 1:
        raise ValueError("steps must be positive")
    batches = repeat_batches(loader)
    warmup_steps = 2
    train_epoch(model, itertools.islice(batches, warmup_steps * config.grad_accum_steps),
                optimizer, scaler, ema, config, 0, 0, graph)
    image_count = 0

    def measured_batches():
        nonlocal image_count
        for batch in itertools.islice(batches, steps * config.grad_accum_steps):
            image_count += len(batch)
            yield batch

    synchronize(device)
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
        mean_loss, gs = train_epoch(model, measured_batches(), optimizer, scaler, ema, config, 0, warmup_steps, graph)
    synchronize(device)
    gs -= warmup_steps
    wall = time.perf_counter() - wall0

    report = {
        "mode": "train",
        "device": device.type,
        "amp": bool(scaler.enabled),
        "cudnn_benchmark": device.type == "cuda" and bool(torch.backends.cudnn.benchmark),
        "use_cuda_graphs": graph.active,
        "steps": gs,
        "warmup_steps": warmup_steps,
        "images": image_count,
        "images_per_sec": round(image_count / wall, 3) if wall else None,
        "seconds": round(wall, 4),
        "step_per_sec": round(gs / wall, 3) if wall else None,
        "mean_loss": round(float(mean_loss), 6),
        "peak_memory_mb": peak_memory_mb(device),
        "memory_scope": memory_scope(device),
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
    if steps < 1:
        raise ValueError("steps must be positive")
    set_seed(config.seed)
    model: DDPM
    if run_dir:
        cfg_from, model, ema = load_checkpoint_run(run_dir, tag="best")
        ema.apply(model)
        config = dataclasses.replace(cfg_from, batch_size=config.batch_size,
                                     use_cuda_graphs=config.use_cuda_graphs)
        t_start = cfg_from.eval_t_start_effective if t_start is None else t_start
        num_steps = cfg_from.sample_timesteps if num_steps is None else num_steps
    else:
        model = build_ddpm(config)
        t_start = config.eval_t_start_effective if t_start is None else t_start
        num_steps = config.sample_timesteps if num_steps is None else num_steps
    model.to(device)
    model.eval()
    denoiser = InferenceGraph(model.unet, enabled=config.use_cuda_graphs)

    batch = config.batch_size
    x = torch.randn(batch, 3, config.img_size, config.img_size, device=device)
    with torch.no_grad():
        model.reconstruct(x, num_steps=num_steps, t_start=t_start, denoiser=denoiser)  # warmup
        synchronize(device)
        _reset_peak(device)
        wall0 = time.perf_counter()
        for _ in range(steps):
            model.reconstruct(x, num_steps=num_steps, t_start=t_start, denoiser=denoiser)
        synchronize(device)
        wall = time.perf_counter() - wall0

    report = {
        "mode": "inference",
        "use_cuda_graphs": denoiser.graph is not None,
        "device": device.type,
        "category": config.category,
        "t_start": int(t_start),
        "num_steps": int(num_steps),
        "batch_size": batch,
        "images": batch * steps,
        "seconds": round(wall, 4),
        "images_per_sec": round(batch * steps / wall, 3) if wall else None,
        "peak_memory_mb": peak_memory_mb(device),
        "memory_scope": memory_scope(device),
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
    previous = torch.backends.cudnn.benchmark
    try:
        for amp, bench in combos:
            cfg = dataclasses.replace(config, use_amp=amp, cudnn_benchmark=bench)
            configure_backends(device, benchmark=bench)
            res = profile_training(cfg, device, steps, with_profiler=False)
            res["report"]["category"] = config.category
            res["report"]["variant"] = f"amp={int(amp)}_bench={int(bench)}"
            results.append(res)
    finally:
        torch.backends.cudnn.benchmark = previous
    return _emit(results, out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["train", "inference", "sweep"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="Measured optimizer updates (train/sweep) or reconstruction batches (inference)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run", type=str, default=None, help="Run dir for inference benchmarking")
    parser.add_argument("--t_start", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--use_cuda_graphs", action="store_true", help="Capture training or denoiser forward on CUDA")
    parser.add_argument("--no_profiler", action="store_true", help="Skip torch.profiler table")
    args = parser.parse_args(argv)

    cfg = parse_args(["--config", args.config])
    if args.use_cuda_graphs:
        cfg = dataclasses.replace(cfg, use_cuda_graphs=True)
    if args.device:
        cfg.device = args.device
    device = resolve_device(cfg.device)
    configure_backends(device, benchmark=cfg.cudnn_benchmark)

    if args.mode == "train":
        res = profile_training(cfg, device, args.steps, with_profiler=not args.no_profiler)
        res["report"]["category"] = cfg.category
        _emit([res], args.out)
    elif args.mode == "inference":
        res = profile_inference(cfg, device, 3 if args.steps is None else args.steps, args.run, args.t_start, args.num_steps)
        _emit([res], args.out)
    else:
        sweep(cfg, device, args.steps, args.out)


if __name__ == "__main__":
    main()