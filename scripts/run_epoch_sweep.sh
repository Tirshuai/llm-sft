#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/shuaiz/llm-sft-project"
CONDA_INIT="/home/shuaiz/miniconda3/etc/profile.d/conda.sh"

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_epoch_sweep.sh EPOCH [EPOCH ...]" >&2
  echo "Example: bash scripts/run_epoch_sweep.sh 1 3 6" >&2
  exit 2
fi

source "$CONDA_INIT"
conda activate llm_sft
cd "$PROJECT_DIR"

for epoch_count in "$@"; do
  if ! [[ "$epoch_count" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid epoch count: $epoch_count" >&2
    exit 2
  fi
  run_dir="outputs/lora/epoch_compare/epochs_${epoch_count}"
  report_path="reports/lora_epoch_${epoch_count}.md"
  echo "=== Starting ${epoch_count} epoch(s) ==="
  python src/training/train.py \
    --config configs/lora.yaml \
    --epochs "$epoch_count" \
    --output-dir "$run_dir" \
    --report-path "$report_path" \
    --save-epoch-checkpoints
done
