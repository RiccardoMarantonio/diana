"""Render demo predictions using a shared absolute heatmap scale and numeric legends."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from diana.prediction import check_outputs, save_prediction_image, write_json


def render_summary(out: Path, cases: list[tuple[str, str, Path]]) -> dict[str, Any]:
    if not cases:
        raise ValueError("at least one prediction is required")
    predictions = []
    arrays = []
    for _, _, report in cases:
        data = json.loads(report.read_text())
        with np.load(data["arrays"], allow_pickle=False) as saved:
            tensors = tuple(torch.from_numpy(saved[key].copy()) for key in ("original", "reconstruction", "anomaly"))
        original, reconstruction, anomaly = tensors
        size = data["protocol"]["img_size"]
        if original.shape != (3, size, size) or reconstruction.shape != original.shape or anomaly.shape != (size, size):
            raise ValueError(f"invalid prediction array shapes: {report}")
        if not all(torch.isfinite(tensor).all() for tensor in tensors) or (anomaly < 0).any():
            raise ValueError(f"invalid prediction array values: {report}")
        if predictions and (data["protocol"] != predictions[0]["protocol"] or data["threshold"] != predictions[0]["threshold"]):
            raise ValueError("summary requires matching prediction protocols and thresholds")
        predictions.append(data)
        arrays.append(tensors)

    color_max = max(1e-12, *(float(tensors[2].max().item()) for tensors in arrays))
    scale = {"min": 0.0, "max": color_max, "scope": "demo_shared"}
    protected = [Path(data[key]) for data in predictions for key in ("image", "arrays")]
    outputs = [out / "summary.json", out / "summary.png"]
    outputs.extend(report for _, _, report in cases)
    outputs.extend(Path(data["visualization"]) for data in predictions)
    check_outputs(outputs, protected)

    results = []
    cards = []
    print("\nRiepilogo:")
    print(f"{'Prova':<15} {'Atteso':<12} {'Predetto':<12} {'Punteggio':<12} Esito")
    for (name, expected, report), data, tensors in zip(cases, predictions, arrays):
        correct = data["verdict"] == expected if expected != "unknown" else None
        outcome = ("OK" if correct else "ERRORE") if correct is not None else "-"
        print(f'{name:<15} {expected:<12} {data["verdict"]:<12} {data["score"]:<12.6g} {outcome}')
        save_prediction_image(Path(data["visualization"]), *tensors, data["is_defective"],
                              data["score"], data["threshold"], color_max=color_max)
        data["heatmap_scale"] = scale
        write_json(report, data)
        results.append({
            "name": name, "expected": expected if expected != "unknown" else None,
            "predicted": data["verdict"], "correct": correct,
            "score": data["score"], "threshold": data["threshold"],
            "image": data["image"], "report": str(report.resolve()),
        })
        with Image.open(data["visualization"]) as image:
            card = Image.new("RGB", (image.width, image.height + 28), "white")
            card.paste(image, (0, 28))
        ImageDraw.Draw(card).text((8, 8), f"{name} | atteso: {expected} | {outcome}", fill="black")
        cards.append(card)

    labelled = [r for r in results if r["correct"] is not None]
    summary = {"results": results, "correct": sum(r["correct"] is True for r in labelled),
               "labelled": len(labelled), "heatmap_scale": scale}
    write_json(out / "summary.json", summary)
    columns = 2 if len(cards) > 1 else 1
    overview = Image.new("RGB", (columns * cards[0].width,
                                ((len(cards) + columns - 1) // columns) * cards[0].height), "white")
    for index, card in enumerate(cards):
        overview.paste(card, ((index % columns) * card.width, (index // columns) * card.height))
    overview.save(out / "summary.png")
    if labelled:
        print(f'Classificazioni corrette: {summary["correct"]}/{summary["labelled"]} (esempi dimostrativi)')
    print(f'Scala colori condivisa: 0 - {color_max:.6g} (errore quadratico per pixel)')
    print(f'Demo completata: {out / "summary.png"}')
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--case", nargs=3, action="append", required=True, metavar=("NAME", "EXPECTED", "REPORT"))
    args = parser.parse_args(argv)
    try:
        render_summary(args.out_dir, [(name, expected, Path(report)) for name, expected, report in args.case])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
