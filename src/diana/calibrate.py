"""Calibrate a reusable anomaly threshold on separate, known healthy images."""

import argparse
from pathlib import Path

import torch

from diana.prediction import (
    ImagePredictor,
    add_model_arguments,
    check_outputs,
    file_digest,
    write_json,
)


def threshold_from_scores(scores: list[float], quantile: float) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1 (exclusive)")
    values = torch.tensor(scores, dtype=torch.float64)
    if len(scores) < 2 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("calibration requires at least two finite, non-negative scores")
    return float(torch.quantile(values, quantile, interpolation="higher").item())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--images", required=True, type=Path, help="Directory of held-out healthy images (recursive)")
    parser.add_argument("--limit", type=int, help="Maximum reference images (at least 2; default: all)")
    parser.add_argument("--exclude-image", type=Path, action="append", default=[],
                        help="Exclude a prediction image and byte-identical copies (repeatable)")
    parser.add_argument("--quantile", type=float, default=0.95, help="Healthy-score quantile (default: 0.95)")
    parser.add_argument("--out", type=Path, default=Path("demo/threshold.json"))
    args = parser.parse_args(argv)
    try:
        threshold_from_scores([0.0, 1.0], args.quantile)
        if args.limit is not None and args.limit < 2:
            raise ValueError("limit must be at least 2")
        if not args.images.is_dir():
            raise ValueError(f"calibration directory not found: {args.images}")
        images = sorted(p for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp",
        })
        if args.exclude_image:
            excluded = {file_digest(p) for p in args.exclude_image}
            images = [p for p in images if file_digest(p) not in excluded]
        if args.limit is not None:
            images = images[:args.limit]
        if len(images) < 2:
            raise ValueError("calibration requires at least two separate healthy images")
        protected = images + [Path(args.run) / f"{args.tag}.pt"]
        protected.extend(args.exclude_image)
        check_outputs([args.out], protected)
        digests = [file_digest(p) for p in images]
        if len(set(digests)) != len(digests):
            raise ValueError("calibration images contain duplicate files")
        predictor = ImagePredictor(**{k: getattr(args, k) for k in (
            "run", "tag", "device", "t_start", "num_steps", "seed", "no_ema",
        )})
        references = []
        scores: list[float] = []
        for i, (image, digest) in enumerate(zip(images, digests), start=1):
            score, *_ = predictor.reconstruct(image)
            scores.append(score)
            references.append({"path": str(image.resolve()), "sha256": digest, "score": score})
            print(f"[{i}/{len(images)}] {image.name}: {score:.6g}", flush=True)
        threshold = threshold_from_scores(scores, args.quantile)
        write_json(args.out, {
            "schema_version": 1, "protocol": predictor.protocol, "threshold": threshold,
            "quantile": args.quantile, "method": "healthy_quantile_higher",
            "references": references,
        })
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Soglia: {threshold:.6g}\nCalibrazione: {args.out}")


if __name__ == "__main__":
    main()
