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
import os
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
        # RNG state so a resume re-enters the same data stream instead of
        # starting a fresh (re-seeded) one -- otherwise epoch-count vs
        # continuous-run trajectories diverge.
        "rng_state": torch.get_rng_state(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
    }
    torch.save(payload, path)
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
    torch.set_rng_state(payload["rng_state"])
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scaler.load_state_dict(payload["scaler"])
    ema.load_state_dict(payload["ema"])
    return payload["epoch"], payload["global_step"], payload["best_loss"]


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
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    ema: EMA,
    config: Config,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    """One epoch, accumulating grads over ``grad_accum_steps`` micro-batches.

    Returns ``(mean_loss, global_step)``. Each micro-loss is divided by the
    accumulation factor up front so the summed gradient is the gradient of the
    mean loss, while reported loss stays on the true (unscaled) scale.
    """
    device = next(model.parameters()).device
    accum = config.grad_accum_steps
    epoch_loss = 0.0
    ran = 0
    optimizer.zero_grad(set_to_none=True)

    for micro, batch in enumerate(loader, 1):
        x = batch.to(device, non_blocking=config.pin_memory)
        with autocast(device, enabled=config.use_amp):
            loss = model.loss(x)
        forward_loss = float(loss.detach())
        epoch_loss += forward_loss

        scaler.backward(loss / accum)

        if micro % accum == 0:
            for group in optimizer.param_groups:
                group["lr"] = lr_at_step(global_step, config)
            scaler.clip_grad_norm_(optimizer, config.grad_clip)
            stepped = scaler.step(optimizer)
            scaler.update()
            if stepped:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            ran += 1

            if global_step % config.log_every_n_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[epoch {epoch}] step {global_step} | loss {forward_loss:.4f} "
                    f"| lr {lr:.2e} | clipped grad_norm to {config.grad_clip}"
                )
        del loss, x

    return epoch_loss / max(1, ran), global_step


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
        f"{config.category}-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}",
    )
    _provenance(config, run_dir)

    global_step = 0
    start_epoch = 0
    best_loss = float("inf")
    if config.resume_from:
        resume_dir = config.resume_from if os.path.isdir(config.resume_from) else os.path.dirname(config.resume_from)
        start_epoch, global_step, best_loss = load_checkpoint(
            os.path.join(resume_dir, LAST_CKPT),
            config, model, optimizer, scaler, ema,
        )
        start_epoch += 1  # checkpoint holds the finished epoch
        print(f"[resume] continuing from epoch {start_epoch}, step {global_step}")

    loader = make_dataloader(config)
    print(
        f"[start] device={device.type} amp={'on' if scaler.enabled else 'off'} "
        f"steps/epoch={len(loader) // config.grad_accum_steps} "
        f"checkpoints -> {run_dir}"
    )

    results: dict = {"losses": [], "best_loss": best_loss}
    for epoch in range(start_epoch, config.epochs):
        loss, global_step = train_epoch(
            model, loader, optimizer, scaler, ema, config, epoch, global_step
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
    return results


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    device = resolve_device(config.device)
    configure_backends(device, benchmark=config.cudnn_benchmark)
    run_training(config, device)


if __name__ == "__main__":
    main()