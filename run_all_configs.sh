#!/usr/bin/env bash
set -euo pipefail

# Run the complete train/evaluate workflow once for every config in configs/.
# Usage: ./run_all_configs.sh [epochs] [max_test_images]
EPOCHS="${1:-100}"
LIMIT="${2:-200}"
DEVICE="${DEVICE:-auto}"
CUDA_GRAPHS="${CUDA_GRAPHS:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "Device: $DEVICE | training CUDA graphs: $CUDA_GRAPHS"
echo "Set DEVICE=cuda to require a CUDA GPU; set CUDA_GRAPHS=1 to enable graphs during training."

shopt -s nullglob
CONFIGS=("$SCRIPT_DIR"/configs/*.toml)
if ((${#CONFIGS[@]} == 0)); then
    echo "No TOML configs found in $SCRIPT_DIR/configs" >&2
    exit 1
fi

for config in "${CONFIGS[@]}"; do
    echo "===== Full evaluation: $(basename "$config" .toml) ====="
    bash "$SCRIPT_DIR/t4.sh" "$EPOCHS" "$LIMIT" "$config"
done

echo "All ${#CONFIGS[@]} configs completed."
