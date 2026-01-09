#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for DiskANN build_disk_index for the exported train.*.bin files.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}

APPS_DIR=""
BUILD_BIN=""
DATA_PATH=""
INDEX_PREFIX=""

DIST="cosine"
DTYPE="bf16"
THREADS="$(nproc)"

# Required by build_disk_index
B_SEARCH_DRAM_GB="4"   # search_DRAM_budget
M_BUILD_DRAM_GB="32"   # build_DRAM_budget

# Optional build params
R=64
LBUILD=100
PQ_DISK_BYTES=0
APPEND_REORDER=0

usage() {
  cat <<'EOF'
Usage:
  ./run_build_disk_index.sh --data-path <train.(f32|bf16|int8).bin> --index-prefix <out_prefix> [options]

Options:
  --diskann-apps <dir>   DiskANN build apps dir containing build_disk_index (optional)
  --build-bin <path>     Explicit path to build_disk_index binary (optional)
  --dist <l2|mips|cosine>          Default: cosine
  --dtype <float|bf16|int8>        Default: bf16
  --threads <t>                    Default: nproc
  --B <gb>                         search_DRAM_budget (required by DiskANN) default: 4
  --M <gb>                         build_DRAM_budget (required by DiskANN)  default: 32
  --R <deg>                        max_degree default: 64
  --Lbuild <L>                     Lbuild default: 100
  --PQ-disk-bytes <n>              PQ_disk_bytes default: 0
  --append-reorder-data            Enable append_reorder_data (requires PQ-disk-bytes > 0)

Example:
  ./run_build_disk_index.sh \
    --data-path ./bins/train.bf16.bin \
    --index-prefix ./index/dbpedia_bf16 \
    --dist cosine --dtype bf16 --threads 32 --B 4 --M 32 --R 64 --Lbuild 100
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

_auto_apps_dir() {
  if [[ -n "${DISKANN_APPS_DIR:-}" ]]; then
    echo "${DISKANN_APPS_DIR}"; return 0
  fi
  for c in \
    "/home/xtang/DiskANN-epeshared/build/apps" \
    "/home/xtang/DiskANN-epeshared/build-avx512/apps" \
    "/home/xtang/DiskANN-epeshared/build-amx/apps"; do
    if [[ -x "$c/build_disk_index" ]]; then
      echo "$c"; return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --diskann-apps) APPS_DIR="$2"; shift 2;;
    --build-bin) BUILD_BIN="$2"; shift 2;;
    --data-path) DATA_PATH="$2"; shift 2;;
    --index-prefix) INDEX_PREFIX="$2"; shift 2;;

    --dist) DIST="$2"; shift 2;;
    --dtype|--data-type) DTYPE="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;

    --B) B_SEARCH_DRAM_GB="$2"; shift 2;;
    --M) M_BUILD_DRAM_GB="$2"; shift 2;;

    --R) R="$2"; shift 2;;
    --Lbuild|--L) LBUILD="$2"; shift 2;;
    --PQ-disk-bytes) PQ_DISK_BYTES="$2"; shift 2;;
    --append-reorder-data) APPEND_REORDER=1; shift;;

    --help|-h) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$DATA_PATH" ]] || die "--data-path is required"
[[ -n "$INDEX_PREFIX" ]] || die "--index-prefix is required"
[[ -f "$DATA_PATH" ]] || die "data file not found: $DATA_PATH"

if [[ -z "$BUILD_BIN" ]]; then
  if [[ -z "$APPS_DIR" ]]; then
    APPS_DIR="$(_auto_apps_dir || true)"
  fi
  [[ -n "$APPS_DIR" ]] || die "Could not auto-detect DiskANN apps dir; pass --diskann-apps"
  BUILD_BIN="$APPS_DIR/build_disk_index"
fi

[[ -x "$BUILD_BIN" ]] || die "build_disk_index not executable: $BUILD_BIN"

CMD=(
  "$BUILD_BIN"
  --data_type "$DTYPE"
  --dist_fn "$DIST"
  --index_path_prefix "$INDEX_PREFIX"
  --data_path "$DATA_PATH"
  --search_DRAM_budget "$B_SEARCH_DRAM_GB"
  --build_DRAM_budget "$M_BUILD_DRAM_GB"
  --num_threads "$THREADS"
  --max_degree "$R"
  --Lbuild "$LBUILD"
  --PQ_disk_bytes "$PQ_DISK_BYTES"
)

if [[ "$APPEND_REORDER" -eq 1 ]]; then
  CMD+=(--append_reorder_data)
fi

echo "[build_disk_index] Cmd: ${CMD[*]}"
"${CMD[@]}"

# DiskANN will emit <index_prefix>_disk.index and related files.
