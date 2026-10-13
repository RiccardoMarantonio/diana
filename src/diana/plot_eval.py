"""Create a CSV and SVG chart from saved evaluation JSON reports.

Usage::

    python -m diana.plot_eval --input runs/hazelnut-<timestamp>/evaluations
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


def load_reports(source: Path) -> list[dict[str, Any]]:
    paths = sorted(source.glob("eval_t*.json")) if source.is_dir() else [source]
    if not paths:
        raise ValueError(f"no eval_t*.json reports found in {source}")
    reports = [json.loads(path.read_text()) for path in paths]
    return sorted(reports, key=lambda row: int(row["t_start"]))


def write_csv(reports: list[dict[str, Any]], path: Path) -> None:
    fields = ["t_start", "num_steps", "n", "image_auroc", "pixel_auroc", "good_mean", "defect_mean"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in reports)


def write_svg(reports: list[dict[str, Any]], path: Path) -> None:
    width, height = 800, 460
    left, right, top, bottom = 80, 30, 55, 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    ts = [float(row["t_start"]) for row in reports]
    min_t, max_t = min(ts), max(ts)
    if min_t == max_t:
        min_t -= 1
        max_t += 1
    xs = [left + (t - min_t) / (max_t - min_t) * plot_width for t in ts]
    y_ticks = []
    for value in (0, 0.2, 0.4, 0.6, 0.8, 1.0):
        y = top + plot_height * (1 - value)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>'
            f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" font-size="12">{value:.1f}</text>'
        )
    # Map metric values onto the same plot rectangle used by the axes.
    def series(key: str) -> list[float | None]:
        return [float(row[key]) if row.get(key) is not None else None for row in reports]

    # Generate polylines with coordinates aligned to the chart axes.
    def line(key: str, color: str) -> str:
        pts = []
        for x, value in zip(xs, series(key)):
            if value is not None:
                y = top + plot_height * (1 - value)
                pts.append(f"{x:.1f},{y:.1f}")
        circles = "".join(
            f'<circle cx="{x:.1f}" cy="{top + plot_height * (1-value):.1f}" r="5" fill="{color}"/>'
            for x, value in zip(xs, series(key)) if value is not None
        )
        poly = f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="3"/>' if pts else ""
        return poly + circles

    category = html.escape(str(reports[0].get("category", "model")))
    labels = "".join(
        f'<text x="{x:.1f}" y="{top+plot_height+25}" text-anchor="middle" font-size="12">{int(t)}</text>'
        for x, t in zip(xs, ts)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="20" font-weight="bold">{category} evaluation by t_start</text>
<g font-family="sans-serif" fill="#374151">{''.join(y_ticks)}</g>
<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" y2="{top+plot_height}" stroke="#374151"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#374151"/>
{labels}
<text x="{left+plot_width/2}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="14">t_start</text>
<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="14">AUROC</text>
{line('image_auroc', '#2563eb')}
{line('pixel_auroc', '#dc2626')}
<g font-family="sans-serif" font-size="13"><line x1="{left+10}" y1="{height-52}" x2="{left+36}" y2="{height-52}" stroke="#2563eb" stroke-width="3"/><text x="{left+43}" y="{height-48}">Image AUROC</text><line x1="{left+170}" y1="{height-52}" x2="{left+196}" y2="{height-52}" stroke="#dc2626" stroke-width="3"/><text x="{left+203}" y="{height-48}">Pixel AUROC</text></g>
</svg>'''
    path.write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", required=True, help="Evaluation JSON file or directory of sweep reports")
    parser.add_argument("--out", default=None, help="Output basename or directory (default: input directory)")
    args = parser.parse_args()
    source = Path(args.input)
    reports = load_reports(source)
    out = Path(args.out) if args.out else (source if source.is_dir() else source.parent)
    if out.suffix:
        base = out.with_suffix("")
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)
        base = out / "evaluation"
    write_csv(reports, base.with_suffix(".csv"))
    write_svg(reports, base.with_suffix(".svg"))
    print(f"[out] evaluation CSV -> {base.with_suffix('.csv')}")
    print(f"[out] evaluation chart -> {base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
