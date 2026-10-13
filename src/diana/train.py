"""Training entrypoint: config -> device -> model -> loop -> checkpoints.

Run directory management (the Colab-resume backbone): every run writes into
``<checkpoint_dir>/<category>-<timestamp>/`` holding a `config.json`
provenance snapshot, a rotating `last.pt` every ``save_every_n_epochs``, and a
`best.pt` by per-epoch loss. `--resume <dir>` reloads the whole bundle --
weights, optimizer, AMP scaler, EMA shadows, step counters -- from `last.pt`,
so an ephemeral Colab session can pick up exactly where it died.
"""

import dataclasses
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime

import torch
from torch import nn

from diana.config import Config, parse_args
from diana.data.loader import make_dataloader
from diana.diffusion.ddpm import DDPM
from diana.diffusion.schedule import DiffusionSchedule
from diana.models.ema import EMA
from diana.models.unet import UNet
from diana.utils.amp import GradScaler, autocast
from diana.utils.cudagraphs import CudaGraphStep
from diana.utils.device import configure_backends, resolve_device, set_seed

LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"
RUN_CONFIG = "config.json"


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------


def build_ddpm(config: Config) -> DDPM:
    """Assemble the UNet + schedule pair the config describes (on CPU)."""
    schedule = DiffusionSchedule(
        schedule_type=config.schedule_type,
        num_timesteps=config.num_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule_param=config.schedule_param,
    )
    unet = UNet(
        img_size=config.img_size,
        in_channels=3,
        base_channels=config.base_channels,
        channel_mults=config.channel_mults,
        num_res_blocks=config.num_res_blocks,
        attention_resolutions=config.attention_resolutions,
        dropout=config.dropout,
    )
    return DDPM(unet, schedule, objective=config.objective)


def build_optimizer(config: Config, model: nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def lr_at_step(step: int, config: Config) -> float:
    """Linear warmup to ``learning_rate`` over ``lr_warmup_steps``, then flat."""
    if config.lr_warmup_steps == 0:
        return config.learning_rate
    warm = min(1.0, step / config.lr_warmup_steps)
    return config.learning_rate * warm


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------


def save_checkpoint(
    run_dir: str,
    tag: str,
    config: Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    epoch: int,
    global_step: int,
    best_loss: float,
) -> str:
    """Write one checkpoint bundle to ``<run_dir>/<tag>.pt``."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, tag)
    payload = {
        "config": dataclasses.asdict(config),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
        # RNG state so a resume re-enters the same stochastic stream instead
        # of starting a fresh (re-seeded) one -- otherwise epoch-count vs
        # continuous-run trajectories diverge. The timestep/noise draws happen
        # on the device, so on CUDA its generator must travel too (the CPU
        # state alone does not cover it). MPS has its own generator too.
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "mps_rng_state": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
    }
    # Publish only a complete checkpoint; failed writes leave the previous one intact.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=run_dir, suffix=".tmp", delete=False) as stream:
            temp_path = stream.name
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and os.path.exists(temp_path):
            os.unlink(temp_path)
    return path


def load_checkpoint(
    path: str,
    config: Config,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
) -> tuple[int, int, float]:
    """Restore every checkpointed piece into the live objects; return counters."""
    payload = torch.load(path, map_location="cpu")
    # Refuse silent changes to the learned distribution or network semantics.
    immutable = ("category", "img_size", "base_channels", "channel_mults", "num_res_blocks",
                 "attention_resolutions", "dropout", "num_timesteps", "schedule_type",
                 "beta_start", "beta_end", "schedule_param", "objective")
    changed = [key for key in immutable if payload["config"].get(key) != getattr(config, key)]
    if changed:
        raise ValueError(f"resume configuration differs from checkpoint: {', '.join(changed)}")
    torch.set_rng_state(payload["rng_state"])
    cuda_state = payload.get("cuda_rng_state")  # absent in older checkpoints
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state)
    mps_state = payload.get("mps_rng_state")
    if mps_state is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(mps_state)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    ema.load_state_dict(payload["ema"])
    return payload["epoch"], payload["global_step"], payload["best_loss"]


def resolve_resume_path(resume_from: str) -> str:
    """Checkpoint file to load: a directory means its ``last.pt``; a file is used as-is."""
    if os.path.isdir(resume_from):
        return os.path.join(resume_from, LAST_CKPT)
    return resume_from


def _provenance(config: Config, run_dir: str) -> None:
    """Write the effective config to the run dir so resumes are exact."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, RUN_CONFIG), "w") as f:
        json.dump(dataclasses.asdict(config), f, indent=2, sort_keys=True)


# ----------------------------------------------------------------------
# The loop
# ----------------------------------------------------------------------


def train_epoch(
    model: DDPM,
    loader: Iterable[torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    config: Config,
    epoch: int,
    global_step: int,
    graph: CudaGraphStep | None = None,
) -> tuple[float, int]:
    """Train one epoch, including partial accumulation groups and small batches.

    Gradients are normalized by the number of samples in each update, so a
    partial final batch has the same weighting as the corresponding large batch.
    Graphed training preserves the addresses used by captured backward kernels.
    """
    model.train()
    sampler = getattr(loader, "sampler", None)
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
    device = next(model.parameters()).device
    accum = config.grad_accum_steps
    epoch_loss = torch.zeros((), device=device)
    micro = 0
    group_samples = 0
    optimizer.zero_grad(set_to_none=not (graph is not None and graph.enabled))

    def update(samples: int, graphed: bool) -> None:
        nonlocal global_step
        if not graphed:
            # Loss scaling stays bounded for AMP; correct the last short group
            # before unscale/clip, without touching the scaler's live scale.
            factor = accum * config.batch_size / samples
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(factor)
        for group in optimizer.param_groups:
            group["lr"] = lr_at_step(global_step, config)
        scaler.clip_grad_norm_(optimizer, config.grad_clip, sync=False)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
        optimizer.zero_grad(set_to_none=not (graph is not None and graph.enabled))
        global_step += 1

    for micro, batch in enumerate(loader, 1):
        x = batch.to(device, non_blocking=config.pin_memory)
        graphed = graph is not None and graph.enabled
        if graphed:
            assert graph is not None
            loss = graph.step(x)
        else:
            with autocast(device, enabled=config.use_amp):
                loss = model.loss(x)
            scaler.backward(loss * (len(x) / (accum * config.batch_size)))
        epoch_loss.add_(loss.detach())
        group_samples += len(x)
        if graphed or micro % accum == 0:
            update(group_samples, graphed)
            group_samples = 0
            if global_step % config.log_every_n_steps == 0:
                print(f"[epoch {epoch}] step {global_step} | loss {float(loss.detach()):.4f} "
                      f"| lr {optimizer.param_groups[0]['lr']:.2e}")

    if micro == 0:
        raise ValueError("training loader is empty; reduce batch_size or provide more images")
    if group_samples:
        update(group_samples, False)
    return float(epoch_loss / micro), global_step


def run_training(config: Config, device: torch.device) -> dict:
    """Drive the full training schedule, checkpointing on the cadence."""
    torch.set_grad_enabled(True)
    set_seed(config.seed)
    model = build_ddpm(config).to(device)
    optimizer = build_optimizer(config, model)
    scaler = GradScaler(device, enabled=config.use_amp)
    ema = EMA(model, config.ema_decay)

    run_dir = os.path.join(
        config.checkpoint_dir,
        f"{config.category}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S-%f')}",
    )
    _provenance(config, run_dir)

    graph = CudaGraphStep(model, device, scaler, enabled=config.use_cuda_graphs)

    global_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if config.resume_from:
        resume_path = resolve_resume_path(config.resume_from)
        start_epoch, global_step, best_loss = load_checkpoint(
            resume_path, config, model, optimizer, scaler, ema,
        )
        start_epoch += 1  # checkpoint holds the finished epoch
        print(f"[resume] continuing from {resume_path}: epoch {start_epoch}, step {global_step}")
        # best.pt must hold the weights of the best epoch, which live in the
        # previous run's best.pt -- not in last.pt. Carry it over so the new
        # run dir is self-contained even if the tail never beats it.
        prev_best = os.path.join(os.path.dirname(resume_path), BEST_CKPT)
        if os.path.exists(prev_best):
            shutil.copy2(prev_best, os.path.join(run_dir, BEST_CKPT))

    loader = make_dataloader(config)
    print(
        f"[start] device={device.type} amp={'on' if scaler.enabled else 'off'} "
        f"cuda_graphs={'on' if graph.enabled else 'off'} "
        f"steps/epoch={math.ceil(len(loader) / config.grad_accum_steps)} "
        f"checkpoints -> {run_dir}"
    )
    results: dict = {"losses": [], "best_loss": best_loss}
    for epoch in range(start_epoch, config.epochs):
        loss, global_step = train_epoch(
            model, loader, optimizer, scaler, ema, config, epoch, global_step, graph
        )
        results["losses"].append(loss)
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(
                run_dir, BEST_CKPT, config, model, optimizer, scaler, ema,
                epoch, global_step, best_loss,
            )
        if (epoch + 1) % config.save_every_n_epochs == 0 or epoch == config.epochs - 1:
            save_checkpoint(
                run_dir, LAST_CKPT, config, model, optimizer, scaler, ema,
                epoch, global_step, best_loss,
            )
        print(f"[epoch {epoch}] mean loss {loss:.4f} | best {best_loss:.4f}")

    results["best_loss"] = best_loss
    results["amp_steps_skipped"] = scaler.steps_skipped()
    if scaler.enabled:
        print(f"[amp] optimizer steps skipped for inf/nan grads: {results['amp_steps_skipped']}")
    return results


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    device = resolve_device(config.device)
    configure_backends(device, benchmark=config.cudnn_benchmark)
    run_training(config, device)


if __name__ == "__main__":
    main()