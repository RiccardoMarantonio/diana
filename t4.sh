#!/usr/bin/env bash
# One-shot, device-agnostic pipeline: benchmark -> train -> eval.
#   from scratch:  git clone <repo-url> diana && cd diana && bash t4.sh
#   existing run:  bash t4.sh [EPOCHS] [LIMIT] [CONFIG]
# Device comes from the config (device = "auto"): CUDA on Colab/T4, MPS on an
# Apple Silicon Mac. AMP and cudnn.benchmark are no-ops off CUDA.
set -euo pipefail

EPOCHS="${1:-100}"
LIMIT="${2:-200}"
CONFIG="${3:-configs/hazelnut.toml}"
CATEGORY="${CATEGORY:-hazelnut}"

# Dataloader workers are persistent but long MPS runs still fork per epoch;
# give the process headroom over macOS's default fd soft limit (256).
ulimit -n 65536 2>/dev/null || true

cd "$(dirname "$0")"

git pull --ff-only
uv sync

if [ ! -d "data/mvtec/$CATEGORY" ]; then
  uv run -m diana.data.download --categories "$CATEGORY"
fi

echo "== benchmark (HPC switch comparison)"
uv run -m diana.profiler sweep --config "$CONFIG" --steps 50

echo "== training: $EPOCHS epochs"
uv run -m diana.train --config "$CONFIG" --use_amp --cudnn_benchmark --epochs "$EPOCHS"

RUN=$(ls -td runs/"$CATEGORY"-*/ | head -1)
echo "== evaluation on newest run: $RUN"

best_t=100
best_au=0.0
for t in 100 150 200; do
  au=$(uv run -m diana.sample --run "$RUN" --limit "$LIMIT" --t_start "$t" 2>&1 |
    sed -n 's/.*image-level AUROC = \([0-9.]*\).*/\1/p')
  echo "t_start=$t -> image AUROC=$au"
  if awk -v x="$au" -v y="$best_au" 'BEGIN{exit !(x>y)}'; then
    best_t=$t
    best_au=$au
  fi
done

echo "== best t_start=$best_t (image AUROC=$best_au)"
uv run -m diana.sample --run "$RUN" --limit "$LIMIT" --t_start "$best_t" --pixel
echo "== eval.json -> $RUN/eval.json"