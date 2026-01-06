#!/usr/bin/env bash
set -euo pipefail

# Unified runner for pq_reorder_bench.py (AVX512 vs AMX).
#
# Default (no args): run the full python pipeline using the AVX512 build.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}

ISA="avx512"          # amx | avx512
MODE="full"           # full | search-only
APPS_DIR=""            # can override via --diskann-apps or DISKANN_APPS_DIR
CPU_LIST="0-31"         # used when numactl exists
RUN_ROOT="./data"      # run root for search-only (contains <dtype>/index and data)

# search-only defaults
DTYPE="bf16"
DIST="mips"
K=24
L=200
W=8
SEARCH_THREADS=32

# full-mode defaults (python)
BASE_F32="tmpdata/base_1536_f32_1M.bin"
QUERY_F32="tmpdata/query_1536_f32_10K.bin"
FULL_THREADS=32
LS_LIST=(200)
GT_SHARED="none"        # none | float
DTYPES=(bf16)
PQ_DISK_BYTES=32
QD=32

PASSTHRU=()

usage() {
  cat <<'EOF'
Usage:
  ./run_pq_reorder_bench.sh [--isa amx|avx512] [--mode full|search-only]
                            [--search-only] [--run-root <dir>]
                            [--diskann-apps <apps_dir>] [--cpu-list <list>]
                            [--dtype <bf16|float|int8>] [--dist <l2|mips|cosine>]
                            [--K <k>] [--L <l>] [--W <beamwidth>] [--threads <t>]
                            [extra args passed to pq_reorder_bench.py in full mode]

Examples:
  # Full pipeline (python) with AVX512 build
  ./run_pq_reorder_bench.sh --isa avx512

  # Search only, reuse ./data/<dtype>/index and ./data/data/query.<dtype>.bin
  ./run_pq_reorder_bench.sh --isa avx512 --search-only --run-root ./data --dtype bf16 --K 24 --L 200 --W 8 --threads 32

Notes:
  - If --diskann-apps is not provided:
      avx512 -> /home/xtang/DiskANN-epeshared/build/apps
      amx    -> /home/xtang/DiskANN-epeshared/build-amx/apps
  - You can also set DISKANN_APPS_DIR to override the apps directory.
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --isa) ISA="$2"; shift 2;;
    --mode) MODE="$2"; shift 2;;
    --search-only) MODE="search-only"; shift;;
    --run-root|--out-root) RUN_ROOT="$2"; shift 2;;
    --diskann-apps) APPS_DIR="$2"; shift 2;;
    --cpu-list) CPU_LIST="$2"; shift 2;;

    --dtype) DTYPE="$2"; shift 2;;
    --dist) DIST="$2"; shift 2;;
    --K) K="$2"; shift 2;;
    --L) L="$2"; shift 2;;
    --Ls)
      shift
      LS_LIST=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        LS_LIST+=("$1")
        shift
      done
      ;;
    --W|--beamwidth) W="$2"; shift 2;;
    --threads|-T)
      SEARCH_THREADS="$2"
      FULL_THREADS="$2"
      shift 2
      ;;

    --base-f32) BASE_F32="$2"; shift 2;;
    --query-f32) QUERY_F32="$2"; shift 2;;
    --gt-shared) GT_SHARED="$2"; shift 2;;
    --dtypes)
      shift
      DTYPES=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        DTYPES+=("$1")
        shift
      done
      ;;
    --PQ-disk-bytes) PQ_DISK_BYTES="$2"; shift 2;;
    --QD) QD="$2"; shift 2;;

    --help|-h) usage; exit 0;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

if [[ -z "${APPS_DIR}" ]]; then
  if [[ -n "${DISKANN_APPS_DIR:-}" ]]; then
    APPS_DIR="${DISKANN_APPS_DIR}"
  else
    case "${ISA}" in
      amx) APPS_DIR="/home/xtang/DiskANN-epeshared/build-amx/apps";;
      avx512) APPS_DIR="/home/xtang/DiskANN-epeshared/build/apps";;
      *) die "unknown --isa '${ISA}' (expected amx or avx512)";;
    esac
  fi
fi

[[ -d "$APPS_DIR" ]] || die "apps dir not found: $APPS_DIR"

NUMACTL_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
  NUMACTL_PREFIX=(numactl -C "${CPU_LIST}")
fi

if [[ "${MODE}" == "search-only" ]]; then
  [[ -d "$RUN_ROOT" ]] || die "run root not found: $RUN_ROOT"

  DISK_INDEX=$(ls -t "$RUN_ROOT/$DTYPE/index/"*_disk.index 2>/dev/null | head -n 1 || true)
  [[ -n "$DISK_INDEX" ]] || die "cannot find *_disk.index under $RUN_ROOT/$DTYPE/index/"
  INDEX_PREFIX="${DISK_INDEX%_disk.index}"

  QUERY_FILE="$RUN_ROOT/data/query.$DTYPE.bin"
  [[ -f "$QUERY_FILE" ]] || die "query file not found: $QUERY_FILE"

  LOG_DIR="$RUN_ROOT/logs"
  mkdir -p "$LOG_DIR"

  RESULT_PREFIX="$RUN_ROOT/$DTYPE/results/res"
  mkdir -p "$(dirname "$RESULT_PREFIX")"

  CMD=(
    "${NUMACTL_PREFIX[@]}"
    "$APPS_DIR/search_disk_index"
    --data_type "$DTYPE"
    --dist_fn "$DIST"
    --index_path_prefix "$INDEX_PREFIX"
    --query_file "$QUERY_FILE"
    --result_path "$RESULT_PREFIX"
    -K "$K"
    -L "$L"
    -W "$W"
    --num_nodes_to_cache 0
    -T "$SEARCH_THREADS"
    --use_reorder_data
  )

  GT_FILE="$RUN_ROOT/gt/gt_float_K${K}.bin"
  if [[ -f "$GT_FILE" ]]; then
    CMD+=(--gt_file "$GT_FILE")
  fi

  echo "[search-only] ISA: $ISA"
  echo "[search-only] Run root: $RUN_ROOT"
  echo "[search-only] Apps: $APPS_DIR"
  echo "[search-only] Cmd: ${CMD[*]}"
  "${CMD[@]}" 2>&1 | tee "$LOG_DIR/search_${DTYPE}.log"
  exit ${PIPESTATUS[0]}
fi

# full mode: run python pipeline (derive args from the variables above)
PY_CMD=("${NUMACTL_PREFIX[@]}" "$PY" "$ROOT_DIR/pq_reorder_bench.py")

CMD=(
  "${PY_CMD[@]}"
  --diskann-apps "$APPS_DIR"
  --base-f32 "$BASE_F32"
  --query-f32 "$QUERY_F32"
  --dist "$DIST"
  --beamwidth "$W"
  --K "$K"
  --Ls "${LS_LIST[@]}"
  --PQ-disk-bytes "$PQ_DISK_BYTES"
  --QD "$QD"
  --threads "$FULL_THREADS"
  --gt-shared "$GT_SHARED"
  --dtypes "${DTYPES[@]}"
)

CMD+=("${PASSTHRU[@]}")

echo "[full] ISA: $ISA"
echo "[full] Apps: $APPS_DIR"
echo "[full] Cmd: ${CMD[*]}"
"${CMD[@]}"

exit 0

