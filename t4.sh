#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-100}"
LIMIT="${2:-200}"
CONFIG_ARG="${3:-configs/hazelnut.toml}"
T_STARTS="${T_STARTS:-100 150 200}"

ulimit -n 65536 2>/dev/null || true
cd "$(dirname "$0")"

if [[ "$CONFIG_ARG" == "all" ]]; then
    CONFIGS=(configs/*.toml)
else
    CONFIGS=("$CONFIG_ARG")
fi

uv sync

run_config() {
    local config="$1"

    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config" >&2
        return 2
    fi

    local category data_path checkpoint_dir
    category="$(uv run python -c 'from diana.config import parse_args; import sys; print(parse_args(["--config", sys.argv[1]]).category)' "$config")"
    data_path="$(uv run python -c 'from diana.config import parse_args; import sys; print(parse_args(["--config", sys.argv[1]]).data_path)' "$config")"
    checkpoint_dir="$(uv run python -c 'from diana.config import parse_args; import sys; print(parse_args(["--config", sys.argv[1]]).checkpoint_dir)' "$config")"
    # The downloader verifies existing files and repairs interrupted downloads.
    uv run -m diana.data.download --categories "$category" --dest "$data_path"

    echo "== benchmark: $category"
    uv run -m diana.profiler sweep --config "$config" --steps 50

    local resume
    resume="$(uv run python -c 'from pathlib import Path; import sys; paths=[p for p in Path(sys.argv[1]).glob(sys.argv[2]+"-*/last.pt") if p.is_file()]; print(max(paths, key=lambda p:p.stat().st_mtime) if paths else "")' "$checkpoint_dir" "$category")"
    local train_args=(--config "$config" --use_amp --cudnn_benchmark --epochs "$EPOCHS")
    if [[ -n "$resume" && "${FRESH:-0}" != "1" ]]; then
        train_args+=(--resume_from "$resume")
        echo "== resuming $category from $resume"
    fi
    echo "== training $category: $EPOCHS total epochs"
    uv run -m diana.train "${train_args[@]}"

    local run
    run="$(uv run python -c 'from pathlib import Path; import sys; paths=[p.parent for p in Path(sys.argv[1]).glob(sys.argv[2]+"-*/best.pt")]; print(max(paths, key=lambda p:p.stat().st_mtime) if paths else "")' "$checkpoint_dir" "$category")"
    if [[ -z "$run" ]]; then
        echo "No run directory found for category $category" >&2
        return 1
    fi
    run="${run%/}"
    echo "== evaluation on newest run: $run"

    local reports="$run/evaluations"
    mkdir -p "$reports"
    echo "== canonical evaluation at the configured t_start"
    uv run -m diana.sample --run "$run" --limit "$LIMIT" --pixel
    cp "$run/eval.json" "$reports/final.json"
    for t in $T_STARTS; do
        echo "== evaluate $category at t_start=$t"
        uv run -m diana.sample --run "$run" --limit "$LIMIT" --t_start "$t" --pixel \
            --report "$reports/eval_t${t}.json"
    done

    uv run -m diana.plot_eval --input "$reports" --out "$reports"
    echo "== evaluation CSV: $reports/evaluation.csv"
    echo "== evaluation graph: $reports/evaluation.svg"
}

for config in "${CONFIGS[@]}"; do
    run_config "$config"
done
