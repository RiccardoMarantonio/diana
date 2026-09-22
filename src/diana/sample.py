"""Anomaly-detection inference: reconstruct test images, score the residual.

Pipeline (the SDEdit workhorse from ``ddpm.reconstruct``):

1. Load a run directory's checkpoint and rebuild the model from the config
   snapshot it carries (that's why ``train.py`` serializes config).
2. Swap EMA shadows *into* the model (sampling uses the EMA weights).
3. Noise each test image up to ``t_start`` and denoise it back down.
4. Anomaly map = pixelwise squared residual, image score = its mean.
5. Image-level AUROC vs the good/anomalous split; save a visual grid.

``t_start`` is the healing-vs-faithfulness dial. On the 1000-step linear
schedule, ᾱ drops to ~0.37 at t=100 and ~0.007 at t=500, so the useful
operating range is roughly t in [50, 200]; the default is 100.

Usage::

    python -m diana.sample --run runs/hazelnut-<ts> --t_start 100
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from diana.config import Config
from diana.data.mvtec import MVTecEvalDataset
from diana.diffusion.ddpm import DDPM
from diana.models.ema import EMA
from diana.train import build_ddpm
from diana.utils.device import resolve_device


def load_checkpoint_run(run_dir: str, tag: str = "best") -> tuple[Config, DDPM, EMA]:
    """Rebuild config, model and EMA from a training checkpoint."""
    path = os.path.join(run_dir, f"{tag}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no {tag}.pt in {run_dir!r}")
    payload = torch.load(path, map_location="cpu")
    cfg = Config(**payload["config"])
    model = build_ddpm(cfg)
    model.load_state_dict(payload["model"])
    ema = EMA(model, cfg.ema_decay)
    ema.load_state_dict(payload["ema"])
    return cfg, model, ema


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Two-class AUC via rank comparison (no sklearn dependency).

    AUC = P(random defect score > random good score), ties counted 0.5.
    Intended for the image-level score vector (small n).
    """
    pos = scores[labels == 1].cpu()
    neg = scores[labels == 0].cpu()
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("need at least one positive and one negative label")
    tally = 0.0
    for p in pos:
        tally += int((neg < p).sum()) + 0.5 * int((neg == p).sum())
    return tally / (len(pos) * len(neg))


def auc_sorted(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Same AUC, but O(n log n) with average-rank tie handling.

    Ranks are assigned ascending by score and averaged within tied scores
    (Mann-Whitney U). Used for pixel-wise scoring where the pair loop in
    :func:`auroc` would be quadratic in the number of pixels.
    """
    y = labels.to(torch.bool)
    p = int(y.sum())
    q = int((~y).sum())
    if p == 0 or q == 0:
        raise ValueError("need at least one positive and one negative label")
    n = y.numel()
    s = scores.cpu().double()

    order = torch.argsort(s, stable=True)  # ascending, input indices
    sorted_s = s[order]

    is_boundary = torch.ones(n, dtype=torch.bool)
    is_boundary[1:] = sorted_s[1:] > sorted_s[:-1]
    group = torch.cumsum(is_boundary.to(torch.long), 0) - 1  # 0-based, sorted order
    naive_sorted = torch.arange(1, n + 1, dtype=torch.double)  # ascending ranks in sorted order
    num_groups = int(group.max()) + 1
    rank_sum = torch.zeros(num_groups, dtype=torch.double).scatter_add_(0, group, naive_sorted)
    count = torch.bincount(group).to(torch.double)
    avg_sorted = (rank_sum / count)[group]  # average rank per tied group, sorted order
    ranks = torch.empty_like(avg_sorted)
    ranks[order] = avg_sorted  # back to input order

    r_pos = ranks[y].sum()
    u = r_pos - p * (p + 1) / 2.0
    return float(u / (p * q))


def _tile(t: torch.Tensor) -> np.ndarray:
    """[-1,1] (C,H,W) -> uint8 (H,W,3), for PIL."""
    return ((t.clamp(-1, 1).permute(1, 2, 0).numpy() + 1) / 2 * 255).astype(np.uint8)


def _colorize(amap: torch.Tensor) -> np.ndarray:
    """Scale a (H,W) map to [0,1] and apply a blue->green->red heat ramp."""
    h = amap - amap.min()
    if h.max().item() < 1e-9:
        reds = np.zeros((*amap.shape, 3), dtype=np.uint8)
        return reds
    h = h / h.max()
    r = h.clamp(0, 1).numpy()
    g = (1 - (h - 0.5).abs() * 2).clamp(0, 1).numpy()
    b = (1 - h).clamp(0, 1).numpy()
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def make_grid(shown: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], cols: int = 4) -> np.ndarray:
    """Rows of orig | recon | heatmap tiles joined into one image (H, W, 3)."""
    tiles = [np.concatenate([_tile(o), _tile(r), _colorize(a)], axis=1) for o, r, a in shown]
    rows = [np.concatenate(tiles[i : i + cols], axis=0) for i in range(0, len(tiles), cols)]
    return np.concatenate(rows, axis=1)


def run_inference(
    run_dir: str,
    tag: str,
    t_start: int,
    num_steps: int,
    out: str,
    limit: int | None,
    pixel: bool = False,
) -> dict:
    cfg, model, ema = load_checkpoint_run(run_dir, tag)
    device = resolve_device(cfg.device)
    model.to(device)
    ema.apply(model)  # sampling scores come from the EMA weights
    try:
        ds = MVTecEvalDataset(cfg.data_path, cfg.category, img_size=cfg.img_size)
        indices = list(range(len(ds))) if limit is None else np.linspace(0, len(ds) - 1, limit).astype(int)
        batch = cfg.batch_size

        all_scores, all_labels = [], []
        pixel_scores: list[torch.Tensor] = []
        pixel_masks: list[torch.Tensor] = []
        shown: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(0, len(indices), batch):
            idx = indices[i : i + batch]
            rows = [ds[j] for j in idx]
            x = torch.stack([r[0] for r in rows]).to(device)
            labels = torch.tensor([r[1] for r in rows])
            masks = torch.stack([r[2] for r in rows])

            recon = model.reconstruct(x, num_steps=num_steps, t_start=t_start)
            amap = (x - recon).pow(2).mean(dim=1)
            scores = amap.mean(dim=(1, 2))

            all_scores.append(scores.cpu())
            all_labels.append(labels)
            if pixel:
                sel = labels == 1
                if sel.any():
                    pixel_scores.append(amap[sel].cpu().flatten())
                    pixel_masks.append(masks[sel].cpu().flatten())
            for k in range(x.shape[0]):
                if len(shown) < 8:
                    shown.append((x[k].cpu(), recon[k].cpu(), amap[k].cpu()))
            del x, recon, amap

        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        image_auc = auroc(scores, labels)

        pixel_auc = None
        if pixel:
            if pixel_scores:
                pixel_auc = auc_sorted(
                    torch.cat(pixel_scores), torch.cat(pixel_masks)
                )
            else:
                print("[warn] --pixel given but no defective images selected; skipping pixel AUROC")

        out = out or os.path.join(run_dir, f"anomaly_{tag}_t{t_start}.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        Image.fromarray(make_grid(shown)).save(out)

        good = float(scores[labels == 0].mean())
        bad = float(scores[labels == 1].mean())
        report = {
            "tag": tag, "t_start": t_start, "num_steps": num_steps,
            "device": device.type, "n": len(scores),
            "image_auroc": round(image_auc, 6),
            "pixel_auroc": round(pixel_auc, 6) if pixel_auc is not None else None,
            "good_mean": round(good, 6), "defect_mean": round(bad, 6),
            "grid": os.path.relpath(out),
        }
        eval_json = os.path.join(run_dir, "eval.json")
        with open(eval_json, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)

        print(f"[score] t_start={t_start} steps={num_steps} device={device.type}")
        print(f"[auc] image-level AUROC = {image_auc:.4f} on {len(scores)} test images")
        if pixel_auc is not None:
            print(f"[auc] pixel-level AUROC = {pixel_auc:.4f} vs ground-truth masks")
        print(f"[separ] mean anomaly  good={good:.4f}  defect={bad:.4f}")
        print(f"[out] grid -> {out}")
        print(f"[out] eval.json -> {eval_json}")
        return report
    finally:
        ema.restore(model)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="Run directory holding best.pt / last.pt")
    parser.add_argument("--tag", default="best", choices=["best", "last"])
    parser.add_argument("--t_start", type=int, default=None,
                        help="SDEdit noising level; higher = stronger healing (default: checkpointed eval_t_start)")
    parser.add_argument("--num_steps", type=int, default=None, help="Sampling strides (default: checkpointed config)")
    parser.add_argument("--out", default=None, help="Output grid PNG path")
    parser.add_argument("--limit", type=int, default=64, help="Max test images to score")
    parser.add_argument("--pixel", action="store_true",
                        help="Also score pixel-level AUROC against ground-truth masks (defective images only)")
    args = parser.parse_args(argv)

    cfg, _, _ = load_checkpoint_run(args.run, args.tag)
    num_steps = args.num_steps or cfg.sample_timesteps
    t_start = args.t_start if args.t_start is not None else cfg.eval_t_start_effective
    run_inference(args.run, args.tag, t_start, num_steps, args.out, args.limit, pixel=args.pixel)


if __name__ == "__main__":
    main()