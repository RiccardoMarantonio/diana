"""Exercise the complete demo with a tiny real checkpoint and image dataset."""

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from diana.calibrate import main as calibrate_main
from diana.demo_summary import render_summary
from diana.prediction import colorize_anomaly
from tests.test_sample import TestSeededSampling as _SeededRun


@pytest.fixture
def demo_project(tmp_path):
    project = tmp_path / "project with spaces"
    project.mkdir()
    shutil.copy(Path(__file__).resolve().parents[1] / "demo.sh", project / "demo.sh")
    python = project / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/bash\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    source = Path(_SeededRun()._ckpt(project))
    run = project / "runs/hazelnut-001"
    run.parent.mkdir()
    shutil.move(source, run)
    config = torch.load(run / "best.pt", map_location="cpu")["config"]
    (run / "config.json").write_text(json.dumps(config))
    Image.new("RGB", (64, 64), (30, 30, 30)).save(project / "mvtec/hazelnut/test/good/extra.png")
    Image.new("RGB", (64, 64), (90, 90, 90)).save(project / "mvtec/hazelnut/test/good/extra2.png")
    return project


def run_demo(project, *args):
    return subprocess.run(
        ["bash", str(project / "demo.sh"), *args], cwd=project.parent,
        capture_output=True, text=True, timeout=90, check=False,
    )


def test_demo_discovers_data_reuses_threshold_and_excludes_target(demo_project):
    project = demo_project
    result = run_demo(project, "--category", "hazelnut", "--device", "cpu", "--reference-limit", "2")
    assert result.returncode == 0, result.stderr
    output = project / "demo/hazelnut"
    calibration = output / "threshold.json"
    first = calibration.read_bytes()
    data = json.loads(first)
    assert len(data["references"]) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["labelled"] == 4
    assert [r["expected"] for r in summary["results"]] == ["difettosa", "difettosa", "sana", "sana"]
    assert len({r["image"] for r in summary["results"]}) == 4
    assert not ({r["image"] for r in summary["results"]} & {r["path"] for r in data["references"]})
    assert summary["correct"] == sum(r["predicted"] == r["expected"] for r in summary["results"])
    assert Image.open(output / "summary.png").size == (1536, 776)
    maxima = []
    for name in ("defective_1", "defective_2", "healthy_1", "healthy_2"):
        assert (output / f"{name}.png").is_file()
        report = json.loads((output / f"{name}.json").read_text())
        assert report["threshold"] == data["threshold"]
        assert report["heatmap_scale"] == summary["heatmap_scale"]
        with np.load(report["arrays"], allow_pickle=False) as arrays:
            maxima.append(float(arrays["anomaly"].max()))
    assert summary["heatmap_scale"]["min"] == 0
    assert summary["heatmap_scale"]["max"] == max(maxima)
    cases = [(r["name"], r["expected"], Path(r["report"])) for r in summary["results"]]
    rerendered = render_summary(output, cases)
    assert rerendered["results"] == summary["results"]

    result = run_demo(project, "--category", "hazelnut", "--device", "cpu")
    assert result.returncode == 0, result.stderr
    assert "Riutilizzo" in result.stdout
    assert calibration.read_bytes() == first

    target = data["references"][0]["path"]
    result = run_demo(
        project, "--category", "hazelnut", "--device", "cpu", "--image", target,
    )
    assert result.returncode == 0, result.stderr
    assert "ricalcolo escludendole" in result.stdout
    updated = json.loads(calibration.read_text())
    assert target not in {entry["path"] for entry in updated["references"]}
    assert json.loads((output / "prediction.json").read_text())["image"] == target
    single_summary = json.loads((output / "summary.json").read_text())
    assert len(single_summary["results"]) == 1
    assert single_summary["labelled"] == 0
    assert single_summary["results"][0]["expected"] is None


def test_demo_invalid_arguments_fail_before_creating_outputs(tmp_path):
    script = Path(__file__).resolve().parents[1] / "demo.sh"
    for args in (["--device", "invalid"], ["--image"], ["--reference-limit", "zero"], ["--unknown"]):
        result = subprocess.run(
            ["bash", str(script), *args], cwd=tmp_path, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 2
        assert "Errore" in result.stderr


def test_calibration_rejects_limit_below_two():
    with pytest.raises(SystemExit) as exc:
        calibrate_main(["--run", "missing", "--images", "missing", "--limit", "1"])
    assert exc.value.code == 2


def test_absolute_colors_are_comparable_between_images():
    a = colorize_anomaly(torch.tensor([[0.0, 0.2, 1.0]]), 1.0)
    b = colorize_anomaly(torch.tensor([[0.1, 0.2, 0.3]]), 1.0)
    np.testing.assert_array_equal(a[0, 1], b[0, 1])
    np.testing.assert_array_equal(a[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(a[0, 2], [255, 0, 0])
