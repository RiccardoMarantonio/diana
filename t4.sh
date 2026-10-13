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

git pull --ff-only
uv sync

run_config() {
    local config="$1"
    local category
    category="$(basename "$config" .toml)"

    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config" >&2
        return 2
    fi

    if [[ ! -d "data/mvtec/$category" ]]; then
        uv run -m diana.data.download --categories "$category"
    fi

    echo "== benchmark: $category"
    uv run -m diana.profiler sweep --config "$config" --steps 50

    echo "== training $category: $EPOCHS epochs"
    uv run -m diana.train --config "$config" --use_amp --cudnn_benchmark --epochs "$EPOCHS"

    local run
    run="$(ls -td "runs/$category-"*/ 2>/dev/null | head -1)"
    if [[ -z "$run" ]]; then
        echo "No run directory found for category $category" >&2
        return 1
    fi
    run="${run%/}"
    echo "== evaluation on newest run: $run"

    local reports="$run/evaluations"
    mkdir -p "$reports"
    for t in $T_STARTS; do
        echo "== evaluate $category at t_start=$t"
        uv run -m diana.sample --run "$run" --limit "$LIMIT" --t_start "$t" --pixel
        cp "$run/eval.json" "$reports/eval_t${t}.json"
    done

    local best
    best="$(uv run python -c 'import json, pathlib, sys; rows=[json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1]).glob("eval_t*.json")]; row=max(rows, key=lambda r: r["image_auroc"]); print(row["t_start"], row["image_auroc"])' "$reports")"
    local best_t best_au
    read -r best_t best_au <<< "$best"
    cp "$reports/eval_t${best_t}.json" "$run/eval.json"
    echo "== best t_start=$best_t (image AUROC=$best_au)"

    uv run -m diana.plot_eval --input "$reports" --out "$reports"
    echo "== evaluation CSV: $reports/evaluation.csv"
    echo "== evaluation graph: $reports/evaluation.svg"
}

for config in "${CONFIGS[@]}"; do
    run_config "$config"
done
