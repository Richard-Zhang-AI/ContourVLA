#!/usr/bin/env bash
# Sharded GRES evaluation. All paths are supplied by the caller.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
if [[ $# -lt 4 ]]; then
  echo "usage: $0 <tag> <config> <checkpoint> <split> [shards] [dataset]" >&2
  exit 2
fi
TAG="$1"; CONFIG="$2"; CHECKPOINT="$3"; SPLIT="$4"
SHARDS="${5:-1}"; DATASET="${6:-}"
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || { echo "shards must be a positive integer" >&2; exit 2; }
OUT="${EVAL_DIR:-$ROOT_DIR/eval_gres}/$TAG"
mkdir -p "$OUT"
cd "$ROOT_DIR"
DATASET_ARGS=(); [[ -n "$DATASET" ]] && DATASET_ARGS=(--dataset "$DATASET")
for ((shard=0; shard<SHARDS; shard++)); do
  "$PYTHON" scripts/evaluate_gres.py --config "$CONFIG" --checkpoint "$CHECKPOINT" \
    --split "$SPLIT" "${DATASET_ARGS[@]}" --shard "$shard" --num-shards "$SHARDS" \
    --output "$OUT/shard${shard}.json"
done
"$PYTHON" scripts/merge_gres_shards.py "$OUT"/shard*.json | tee "$OUT/merged.json"
