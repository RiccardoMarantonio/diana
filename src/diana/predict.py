"""Classify a single image and save input, reconstruction and anomaly map."""

import argparse
from pathlib import Path

import numpy as np

from diana.prediction import (
    ImagePredictor,
    add_model_arguments,
    check_outputs,
    save_prediction_image,
    valid_threshold,
    write_json,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--image", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--threshold", type=float, help="Explicit anomaly score threshold; not a probability")
    source.add_argument("--calibration", type=Path, help="JSON created by diana.calibrate")
    parser.add_argument("--out", type=Path, default=Path("demo/prediction.png"))
    parser.add_argument("--report", type=Path, default=Path("demo/prediction.json"))
    args = parser.parse_args(argv)
    try:
        if not args.image.is_file():
            raise ValueError(f"image not found: {args.image}")
        if args.threshold is not None:
            valid_threshold(args.threshold)
        protected = [args.image, Path(args.run) / f"{args.tag}.pt"]
        if args.calibration:
            protected.append(args.calibration)
        arrays_path = args.report.with_suffix(".npz")
        check_outputs([args.out, args.report, arrays_path], protected)
        predictor = ImagePredictor(**{k: getattr(args, k) for k in (
            "run", "tag", "device", "t_start", "num_steps", "seed", "no_ema",
        )})
        threshold = (
            predictor.calibrated_threshold(args.calibration, args.image)
            if args.calibration else valid_threshold(float(args.threshold))
        )
        score, original, reconstruction, anomaly = predictor.reconstruct(args.image)
        defective = score > threshold
        save_prediction_image(args.out, original, reconstruction, anomaly, defective, score, threshold)
        arrays_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arrays_path, original=original.numpy(), reconstruction=reconstruction.numpy(),
                            anomaly=anomaly.numpy())
        write_json(args.report, {
            "image": str(args.image.resolve()), "protocol": predictor.protocol,
            "score": score, "threshold": threshold, "is_defective": defective,
            "decision_rule": "score > threshold", "verdict": "difettosa" if defective else "sana",
            "threshold_source": str(args.calibration.resolve()) if args.calibration else "explicit",
            "visualization": str(args.out.resolve()),
            "arrays": str(arrays_path.resolve()),
            "heatmap_scale": {"min": 0.0, "max": max(float(anomaly.max().item()), 1e-12), "scope": "image"},
        })
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Esito: {'DIFETTOSA' if defective else 'SANA'} (ritaglio centrale)")
    print(f"Punteggio anomalia: {score:.6g}\nSoglia: {threshold:.6g}")
    print(f"Immagine: {args.out}\nReport: {args.report}")


if __name__ == "__main__":
    main()
