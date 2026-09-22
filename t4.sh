#!/usr/bin/env bash
# One-shot Colab T4 pipeline: setup -> benchmark -> train -> eval.
# Paste `!bash t4.sh` into a cell (runtime: GPU T4).
set -euo pipefail
cd "$(dirname "$0")"

git pull --ff-only
uv sync

uv run -m diana.data.download --categories hazelnut
uv run -m diana.profiler sweep --config configs/hazelnut.toml --device cuda --steps 50
uv run -m diana.train --config configs/hazelnut.toml --device cuda --use_amp --cudnn_benchmark --epochs 100

RUN=$(ls -td runs/hazelnut-*/ | head -1)
echo "== newest run: $RUN"

best_t=100
best_au=0.0
for t in 100 150 200; do
  au=$(uv run -m diana.sample --run "$RUN" --limit 200 --t_start "$t" 2>&1 |
    sed -n 's/.*image-level AUROC = \([0-9.]*\).*/\1/p')
  echo "t_start=$t -> image AUROC=$au"
  if awk -v x="$au" -v y="$best_au" 'BEGIN{exit !(x>y)}'; then
    best_t=$t
    best_au=$au
  fi
done

echo "== best t_start=$best_t (image AUROC=$best_au)"
uv run -m diana.sample --run "$RUN" --limit 200 --t_start "$best_t" --pixel
echo "== eval.json -> $RUN/eval.json"