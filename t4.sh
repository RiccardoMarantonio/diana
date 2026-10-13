#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-100}"
LIMIT="${2:-200}"
CONFIG_ARG="${3:-configs/hazelnut.toml}"
T_STARTS="${T_STARTS:-100 150 200}"
DEVICE="${DEVICE:-auto}"
CUDA_GRAPHS="${CUDA_GRAPHS:-0}"

case "$DEVICE" in
    auto|cuda|mps|cpu) ;;
    *) echo "DEVICE must be one of: auto, cuda, mps, cpu (got '$DEVICE')" >&2; exit 2 ;;
esac
case "$CUDA_GRAPHS" in
    0|1) ;;
    *) echo "CUDA_GRAPHS must be 0 or 1 (got '$CUDA_GRAPHS')" >&2; exit 2 ;;
esac

ulimit -n 65536 2>/dev/null || true
cd "$(dirname "$0")"

if [[ "$CONFIG_ARG" == "all" ]]; then
    CONFIGS=(configs/*.toml)
else
    CONFIGS=("$CONFIG_ARG")
fi

uv sync

# An explicit CUDA request must fail before downloading data or starting a run
# if this environment (for example, the selected Colab runtime) has no GPU.
if [[ "$DEVICE" == "cuda" ]]; then
    uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA non disponibile"' \
        || { echo "CUDA richiesta ma non disponibile in questo runtime." >&2; exit 1; }
fi

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

    echo "== benchmark: $category (device=$DEVICE)"
    uv run -m diana.profiler sweep --config "$config" --device "$DEVICE" --steps 50

    local resume
    resume="$(uv run python -c 'from pathlib import Path; import sys; paths=[p for p in Path(sys.argv[1]).glob(sys.argv[2]+"-*/last.pt") if p.is_file()]; print(max(paths, key=lambda p:p.stat().st_mtime) if paths else "")' "$checkpoint_dir" "$category")"
    local train_args=(--config "$config" --device "$DEVICE" --use_amp --cudnn_benchmark --epochs "$EPOCHS")
    if [[ "$CUDA_GRAPHS" == "1" ]]; then
        train_args+=(--use_cuda_graphs)
    fi
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
    local profiling="$run/profiling"
    mkdir -p "$reports"
    mkdir -p "$profiling"
    echo "== canonical evaluation at the configured t_start"
    uv run -m diana.sample --run "$run" --limit "$LIMIT" --pixel
    cp "$run/eval.json" "$reports/final.json"
    for t in $T_STARTS; do
        echo "== evaluate $category at t_start=$t"
        uv run -m diana.sample --run "$run" --limit "$LIMIT" --t_start "$t" --pixel \
            --report "$reports/eval_t${t}.json"
    done

    echo "== inference throughput (eager)"
    uv run -m diana.profiler inference --config "$config" --run "$run" \
        --device "$DEVICE" --steps 5 --out "$profiling/inference_eager.json"
    if [[ "$DEVICE" == "cuda" ]]; then
        echo "== inference throughput (CUDA graphs)"
        uv run -m diana.profiler inference --config "$config" --run "$run" \
            --device cuda --use_cuda_graphs --steps 5 \
            --out "$profiling/inference_cuda_graphs.json"
    fi

    uv run -m diana.plot_eval --input "$reports" --out "$reports"
    echo "== evaluation CSV: $reports/evaluation.csv"
    echo "== evaluation graph: $reports/evaluation.svg"
}

for config in "${CONFIGS[@]}"; do
    run_config "$config"
done
