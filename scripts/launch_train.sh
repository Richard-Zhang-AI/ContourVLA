#!/usr/bin/env bash
# Portable single-node launcher for supervised fine-tuning (stage A).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/sft.yaml}"
NPROC="${NPROC:-1}"
cd "$ROOT_DIR"
exec "$PYTHON" -m torch.distributed.run \
  --standalone --nproc_per_node="$NPROC" \
  scripts/train.py --config "$CONFIG" "$@"
