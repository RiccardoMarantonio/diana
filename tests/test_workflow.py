"""Exercise orchestration without downloading data or launching a GPU job."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_workflow_repairs_data_resumes_and_preserves_canonical_report(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    shutil.copy(repo / "t4.sh", tmp_path / "t4.sh")
    cfg = tmp_path / "renamed.toml"
    cfg.write_text(f'data_path = "{tmp_path}/data"\ncategory = "hazelnut"\ncheckpoint_dir = "{tmp_path}/runs"\n')
    (tmp_path / "data/hazelnut").mkdir(parents=True)  # interrupted download
    binary = tmp_path / "bin"
    binary.mkdir()
    uv = binary / "uv"
    uv.write_text(f"#!{sys.executable}\n" + '''import sys, json, os, subprocess
from pathlib import Path
args = sys.argv[1:]
with open("calls.jsonl", "a") as f:
    f.write(json.dumps(args) + "\\n")
if args == ["sync"]:
    sys.exit(0)
if args[:2] == ["run", "python"]:
    sys.exit(subprocess.call([sys.executable] + args[2:]))
module = args[2]
if module == "diana.train":
    run = Path("runs/hazelnut-001")
    run.mkdir(parents=True, exist_ok=True)
    (run / "last.pt").write_text("checkpoint")
    (run / "best.pt").write_text("checkpoint")
elif module == "diana.sample":
    run = Path(args[args.index("--run") + 1])
    exploratory = "--report" in args
    report = Path(args[args.index("--report") + 1]) if exploratory else run / "eval.json"
    report.write_text(json.dumps({"image_auroc": .99 if exploratory else .5}))
''')
    uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"}
    for _ in range(2):
        result = subprocess.run(["bash", "t4.sh", "2", "200", str(cfg)], cwd=tmp_path,
                                env=env, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    downloads = [c for c in calls if "diana.data.download" in c]
    training = [c for c in calls if "diana.train" in c]
    assert len(downloads) == 2
    assert downloads[0][-2:] == ["--dest", str(tmp_path / "data")]
    assert "--resume_from" not in training[0]
    assert "--resume_from" in training[1]
    assert json.loads((tmp_path / "runs/hazelnut-001/eval.json").read_text())["image_auroc"] == .5
    assert json.loads((tmp_path / "runs/hazelnut-001/evaluations/final.json").read_text())["image_auroc"] == .5
