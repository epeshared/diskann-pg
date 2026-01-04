#!/usr/bin/env bash
set -euo pipefail

# Minimal runner for pq_reorder_bench.py
#
# Usage:
#   ./run_pq_reorder_bench.sh \
#     --base-f32 ../data/rand_float_768D_1M_norm1.0.bin \
#     --query-f32 ../data/rand_float_768D_10K_norm1.0.bin

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=${PYTHON:-python3}

# Try to auto-detect DiskANN build apps if user didn't pass --diskann-apps.
DEFAULT_APPS=""
for c in \
  "/home/xtang/DiskANN-epeshared/build/apps" \
  "/home/xtang/DiskANN/build/apps" \
  "/home/xtang/DiskANN/build/tests"; do
  if [[ -x "$c/build_disk_index" && -x "$c/search_disk_index" ]]; then
    DEFAULT_APPS="$c"
    break
  fi
done

ARGS=("$@")

if [[ -n "$DEFAULT_APPS" ]]; then
  ARGS=("--diskann-apps" "$DEFAULT_APPS" "${ARGS[@]}")
fi

exec "$PY" "$ROOT_DIR/pq_reorder_bench.py" "${ARGS[@]}"
