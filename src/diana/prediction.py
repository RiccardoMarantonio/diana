"""Shared single-image scoring, calibration protocol and demo rendering."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from diana.config import VALID_DEVICES
from diana.data.mvtec import _load_image
from diana.sample import _tile, load_checkpoint_run
from diana.utils.device import resolve_device, set_seed


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def valid_threshold(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("threshold must be finite and non-negative")
    return value


def check_outputs(outputs: list[Path], protected: list[Path]) -> None:
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or set(resolved) & {p.resolve() for p in protected}:
        raise ValueError("output paths must be distinct and must not overwrite input files")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run", required=True, help="Directory containing best.pt / last.pt")
    parser.add_argument("--tag", choices=["best", "last"], default="best")
    parser.add_argument("--device", choices=sorted(VALID_DEVICES), default="auto")
    parser.add_argument("--t_start", type=int, help="Default: checkpoint evaluation noise level")
    parser.add_argument("--num_steps", type=int, help="Default: checkpoint sampling grid size")
    parser.add_argument("--seed", type=int, help="Default: checkpoint seed; reset for each image")
    parser.add_argument("--no_ema", action="store_true", help="Use live weights instead of EMA")


class ImagePredictor:
    """Load once, reconstruct one image at a time with an independent RNG reset."""

    def __init__(
        self, run: str, tag: str = "best", device: str = "auto",
        t_start: int | None = None, num_steps: int | None = None,
        seed: int | None = None, no_ema: bool = False,
    ) -> None:
        self.checkpoint = Path(run) / f"{tag}.pt"
        self.cfg, self.model, ema = load_checkpoint_run(run, tag)
        self.device = resolve_device(device)
        self.t_start = self.cfg.eval_t_start_effective if t_start is None else t_start
        self.num_steps = self.cfg.sample_timesteps if num_steps is None else num_steps
        self.seed = self.cfg.seed if seed is None else seed
        if not 0 <= self.t_start < self.cfg.num_timesteps:
            raise ValueError(f"t_start must be in [0, {self.cfg.num_timesteps})")
        if not 0 < self.num_steps <= self.cfg.num_timesteps:
            raise ValueError(f"num_steps must be in (0, {self.cfg.num_timesteps}]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not no_ema:
            ema.apply(self.model)
        self.model.to(self.device).eval()
        self.protocol = {
            "checkpoint_sha256": file_digest(self.checkpoint),
            "category": self.cfg.category, "img_size": self.cfg.img_size,
            "preprocessing": "rgb_resize_2x_center_crop_normalize_minus1_1_v1",
            "score": "mean_squared_rgb_reconstruction_error_v1",
            "rng": "reset_per_image_v1", "seed": self.seed,
            "t_start": self.t_start, "num_steps": self.num_steps,
            "ema_applied": not no_ema, "device": self.device.type,
        }

    @torch.inference_mode()
    def reconstruct(self, image: Path) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        original = _load_image(str(image), 2 * self.cfg.img_size, self.cfg.img_size)
        # Per-image RNG and batch size 1 keep calibration and prediction aligned,
        # regardless of the image's position in a directory or previous calls.
        set_seed(self.seed)
        x = original.unsqueeze(0).to(self.device)
        reconstruction = self.model.reconstruct(x, num_steps=self.num_steps, t_start=self.t_start)
        anomaly = (x - reconstruction).square().mean(dim=1)
        score = float(anomaly.mean().item())
        if not math.isfinite(score):
            raise ValueError(f"non-finite anomaly score for {image}")
        return score, original, reconstruction[0].cpu(), anomaly[0].cpu()

    def calibrated_threshold(self, path: Path, image: Path) -> float:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("protocol") != self.protocol:
            raise ValueError("calibration does not match this checkpoint, device or sampling settings; recalibrate")
        references = data.get("references")
        if not isinstance(references, list) or len(references) < 2 or any(
            not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str)
            for entry in references
        ):
            raise ValueError("calibration must contain at least two image reference hashes")
        if file_digest(image) in {entry["sha256"] for entry in references}:
            raise ValueError("prediction image was used for calibration; choose a separate image")
        return valid_threshold(float(data["threshold"]))


def colorize_anomaly(anomaly: torch.Tensor, color_max: float) -> np.ndarray:
    """Map absolute squared errors to colors, with no per-image min/max shift."""
    if not math.isfinite(color_max) or color_max <= 0:
        raise ValueError("color_max must be finite and positive")
    scaled = (anomaly / color_max).clamp(0, 1).numpy()
    green = np.clip(1 - np.abs(scaled - 0.5) * 2, 0, 1)
    return (np.stack([scaled, green, 1 - scaled], axis=-1) * 255).astype(np.uint8)


def save_prediction_image(
    path: Path, original: torch.Tensor, reconstruction: torch.Tensor,
    anomaly: torch.Tensor, defective: bool, score: float, threshold: float,
    color_max: float | None = None,
) -> None:
    shared = color_max is not None
    color_max = max(float(anomaly.max().item()), 1e-12) if color_max is None else color_max
    grid = Image.fromarray(np.concatenate([
        _tile(original), _tile(reconstruction), colorize_anomaly(anomaly, color_max),
    ], axis=1))
    grid = grid.resize((768, 256), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (768, 360), "white")
    canvas.paste(grid, (0, 28))
    draw = ImageDraw.Draw(canvas)
    map_title = "Anomaly map (shared scale)" if shared else "Anomaly map (absolute scale)"
    for x, title in zip((8, 264, 520), ("Input (center crop)", "Reconstruction", map_title)):
        draw.text((x, 8), title, fill="black")
    verdict = "DIFETTOSA" if defective else "SANA"
    draw.text((8, 296), f"{verdict} | score={score:.6g} | threshold={threshold:.6g}", fill="black")
    draw.text((8, 324), "Colors: squared error per pixel. Decision: mean error over the image.", fill="black")
    ramp = torch.linspace(0, color_max, 224).unsqueeze(0)
    legend = Image.fromarray(colorize_anomaly(ramp, color_max)).resize((224, 12))
    canvas.paste(legend, (520, 320))
    for x, value in ((520, 0.0), (608, color_max / 2), (688, color_max)):
        draw.text((x, 340), f"{value:.4g}", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
