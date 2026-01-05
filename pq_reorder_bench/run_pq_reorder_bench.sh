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

numactl -C 0-31 $PY  pq_reorder_bench.py \
  --diskann-apps /home/xtang/DiskANN-epeshared/build/apps \
  --base-f32 tmpdata/base_f32_1M.bin \
  --query-f32 tmpdata/query_f32_10K.bin \
  --dist mips \
  --beamwidth 8 \
  --K 10 \
  --Ls 200 \
  --PQ-disk-bytes 32 \
  --QD 32 \
  --threads 32 \
  --gt-shared float \
  "$@"
