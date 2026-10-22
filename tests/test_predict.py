import json
from pathlib import Path

import pytest
from PIL import Image

from diana.calibrate import main as calibrate_main
from diana.calibrate import threshold_from_scores
from diana.predict import main as predict_main
from diana.prediction import ImagePredictor, check_outputs
from tests.test_sample import TestSeededSampling as _SeededRun


@pytest.fixture
def run(tmp_path):
    return _SeededRun()._ckpt(tmp_path)


def image(path: Path, color: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 60), color).save(path)
    return path


@pytest.mark.parametrize("quantile", [0, 1, float("nan"), -0.1])
def test_invalid_quantiles(quantile):
    with pytest.raises(ValueError, match="quantile"):
        threshold_from_scores([1, 2], quantile)


def test_healthy_quantile_and_invalid_scores():
    assert threshold_from_scores([0.1, 0.2, 0.3, 0.4], 0.95) == 0.4
    for scores in ([], [0.1], [-1, 1], [1, float("nan")]):
        with pytest.raises(ValueError):
            threshold_from_scores(scores, 0.95)


def test_score_matches_residual_and_is_independent_of_previous_images(run, tmp_path):
    predictor = ImagePredictor(run, t_start=8)
    a, b = image(tmp_path / "a.png", 128), image(tmp_path / "b.png", 220)
    first, original, reconstruction, anomaly = predictor.reconstruct(a)
    predictor.reconstruct(b)
    second, *_ = predictor.reconstruct(a)
    assert first == second
    assert first == pytest.approx((original - reconstruction).square().mean().item())
    assert original.shape == reconstruction.shape == (3, 16, 16)
    assert anomaly.shape == (16, 16)
    assert not predictor.model.training
    assert not reconstruction.requires_grad


@pytest.mark.parametrize("threshold,defective", [(0.0, True), (1e6, False)])
def test_single_image_cli_outputs(run, tmp_path, threshold, defective, capsys):
    source = image(tmp_path / "input.png", 128)
    out, report = tmp_path / "demo/out.png", tmp_path / "demo/result.json"
    predict_main([
        "--run", run, "--image", str(source), "--threshold", str(threshold),
        "--t_start", "8", "--out", str(out), "--report", str(report),
    ])
    result = json.loads(report.read_text())
    assert result["is_defective"] is defective
    assert result["threshold_source"] == "explicit"
    assert result["protocol"]["ema_applied"] is True
    assert Image.open(out).size == (768, 360)
    assert ("DIFETTOSA" if defective else "SANA") in capsys.readouterr().out


def test_calibration_prediction_and_protocol_mismatch(run, tmp_path):
    reference = tmp_path / "healthy"
    a = image(reference / "a.png", 120)
    image(reference / "nested/b.png", 140)
    calibration = tmp_path / "calibration.json"
    calibrate_main([
        "--run", run, "--images", str(reference), "--t_start", "8", "--out", str(calibration),
    ])
    data = json.loads(calibration.read_text())
    assert len(data["references"]) == 2
    assert data["threshold"] == max(r["score"] for r in data["references"])
    predictor = ImagePredictor(run, t_start=8)
    target = image(tmp_path / "target.png", 240)
    assert predictor.calibrated_threshold(calibration, target) == data["threshold"]
    out, report = tmp_path / "out.png", tmp_path / "report.json"
    predict_main([
        "--run", run, "--image", str(target), "--calibration", str(calibration),
        "--t_start", "8", "--out", str(out), "--report", str(report),
    ])
    assert json.loads(report.read_text())["threshold_source"] == str(calibration.resolve())
    copy = tmp_path / "renamed.png"
    copy.write_bytes(a.read_bytes())
    with pytest.raises(ValueError, match="used for calibration"):
        predictor.calibrated_threshold(calibration, copy)
    changed = ImagePredictor(run, t_start=9)
    with pytest.raises(ValueError, match="does not match"):
        changed.calibrated_threshold(calibration, target)


@pytest.mark.parametrize("threshold", ["-1", "nan", "inf"])
def test_invalid_threshold_fails_before_loading_checkpoint(tmp_path, threshold):
    source = image(tmp_path / "input.png", 128)
    with pytest.raises(SystemExit) as exc:
        predict_main(["--run", "missing", "--image", str(source), f"--threshold={threshold}"])
    assert exc.value.code == 2


def test_missing_threshold_is_not_silently_invented():
    with pytest.raises(SystemExit):
        predict_main(["--run", "missing", "--image", "missing.png"])


def test_outputs_cannot_overwrite_inputs(tmp_path):
    source = image(tmp_path / "input.png", 128)
    with pytest.raises(ValueError, match="overwrite"):
        check_outputs([source], [source])
    with pytest.raises(ValueError, match="distinct"):
        check_outputs([tmp_path / "x", tmp_path / "x"], [])


def test_duplicate_reference_images_rejected(tmp_path):
    source = image(tmp_path / "healthy/a.png", 128)
    (source.parent / "b.png").write_bytes(source.read_bytes())
    with pytest.raises(SystemExit):
        calibrate_main(["--run", "missing", "--images", str(source.parent)])


def test_changed_checkpoint_rejected(run, tmp_path):
    predictor = ImagePredictor(run, t_start=8)
    calibration = tmp_path / "threshold.json"
    data = {"schema_version": 1, "protocol": dict(predictor.protocol), "threshold": 1, "references": []}
    data["protocol"]["checkpoint_sha256"] = "other checkpoint"
    calibration.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="does not match"):
        predictor.calibrated_threshold(calibration, image(tmp_path / "target.png", 128))
