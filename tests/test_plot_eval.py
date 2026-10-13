import csv
import json
import xml.etree.ElementTree as ET

import pytest

from diana.plot_eval import load_reports, write_csv, write_svg


def test_reports_sort_and_outputs_preserve_missing_metrics(tmp_path):
    rows = [{"t_start": 150, "num_steps": 250, "n": 110, "category": "a<&", "image_auroc": .7,
                 "pixel_auroc": None, "good_mean": .1, "defect_mean": .2},
            {"t_start": 50, "num_steps": 250, "n": 110, "category": "a<&", "image_auroc": .6,
                 "pixel_auroc": .8, "good_mean": .1, "defect_mean": .2}]
    for row in rows:
        (tmp_path / f"eval_t{row['t_start']}.json").write_text(json.dumps(row))
    reports = load_reports(tmp_path)
    assert [r["t_start"] for r in reports] == [50, 150]
    write_csv(reports, tmp_path / "evaluation.csv")
    with (tmp_path / "evaluation.csv").open() as f:
        output = list(csv.DictReader(f))
    assert output[1]["pixel_auroc"] == ""
    write_svg(reports, tmp_path / "evaluation.svg")
    root = ET.parse(tmp_path / "evaluation.svg").getroot()
    assert root.tag.endswith("svg")
    assert "a<& evaluation" in "".join(root.itertext())


def test_empty_report_directory_rejected(tmp_path):
    with pytest.raises(ValueError, match="no eval"):
        load_reports(tmp_path)
