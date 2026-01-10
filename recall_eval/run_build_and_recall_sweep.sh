#!/usr/bin/env bash
set -euo pipefail

# Combined runner:
# - Reuse an existing disk index for the same --max-degree/--Lbuild combination if present
# - Otherwise build a new disk index
# - Then run recall sweep and produce summary.csv + png

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PY="$ROOT_DIR/../.venv/bin/python"
if [[ -x "$DEFAULT_PY" ]]; then
  PY=${PYTHON:-$DEFAULT_PY}
else
  PY=${PYTHON:-python3}
fi

DATASET=""
BINS_DIR=""
DIST="cosine"
DTYPE="bf16"
THREADS="$(nproc)"

# Build params
B_SEARCH_DRAM_GB=4
M_BUILD_DRAM_GB=32
MAX_DEGREE=64
LBUILD=100

# Search params
K=10
RECALL_ATS=()
LS_LIST=(50 100 150 200)
WS_LIST=(2 4 8)
NUM_NODES_TO_CACHE=0
IO_LIMIT=""
USE_REORDER=0

# Optional
DISKANN_APPS_DIR=""
OUT_ROOT=""          # where to place index cache + sweep outputs

usage() {
  cat <<'EOF'
Usage:
  ./run_build_and_recall_sweep.sh --dataset <dbpedia-openai-1000k-angular.hdf5> --bins-dir <dir> [options]

Required:
  --dataset <path>     HDF5 dataset
  --bins-dir <dir>     Output of prepare_hdf5_bins.py (contains train.*.bin)

Build options:
  --dist <l2|mips|cosine>        Default: cosine
  --dtype <float|bf16|int8>      Default: bf16
  --threads <t>                  Default: nproc
  --B <gb>                       search_DRAM_budget (GB) default: 4
  --M <gb>                       build_DRAM_budget (GB)  default: 32
  --max-degree <deg>             max graph out-degree default: 64
  --Lbuild <L>                   build-time L default: 100
  --diskann-apps <dir>           DiskANN apps dir (optional)

Search/sweep options:
  --K <k>                        Default: 10
  --recall-ats <k1 k2 ...>        Optional; sweep multiple recall_at values
  --Ls <l1 l2 ...>               Default: 50 100 150 200
  --Ws <w1 w2 ...>               Default: 2 4 8
  --num-nodes-to-cache <n>       Default: 0
  --search-io-limit <n>          Optional
  --use-reorder-data             Pass --use_reorder_data to search

Output:
  --out-root <dir>               Optional; default: ./runs

Notes:
  - Index cache key includes dtype+dist+max-degree+Lbuild.
  - If <prefix>_disk.index exists, build is skipped.
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --bins-dir) BINS_DIR="$2"; shift 2;;
    --dist) DIST="$2"; shift 2;;
    --dtype|--data-type) DTYPE="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;

    --B) B_SEARCH_DRAM_GB="$2"; shift 2;;
    --M) M_BUILD_DRAM_GB="$2"; shift 2;;
    --max-degree) MAX_DEGREE="$2"; shift 2;;
    --Lbuild) LBUILD="$2"; shift 2;;
    --diskann-apps) DISKANN_APPS_DIR="$2"; shift 2;;

    --K) K="$2"; shift 2;;
    --recall-ats)
      shift
      RECALL_ATS=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        RECALL_ATS+=("$1")
        shift
      done
      ;;
    --Ls)
      shift
      LS_LIST=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        LS_LIST+=("$1")
        shift
      done
      ;;
    --Ws)
      shift
      WS_LIST=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        WS_LIST+=("$1")
        shift
      done
      ;;
    --num-nodes-to-cache) NUM_NODES_TO_CACHE="$2"; shift 2;;
    --search-io-limit) IO_LIMIT="$2"; shift 2;;
    --use-reorder-data) USE_REORDER=1; shift;;

    --out-root) OUT_ROOT="$2"; shift 2;;

    --help|-h) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$DATASET" ]] || die "--dataset is required"
[[ -n "$BINS_DIR" ]] || die "--bins-dir is required"
[[ -d "$BINS_DIR" ]] || die "bins dir not found: $BINS_DIR"

if [[ -z "$OUT_ROOT" ]]; then
  OUT_ROOT="$ROOT_DIR/runs"
fi
mkdir -p "$OUT_ROOT"

# pick train bin based on dtype
TRAIN_BIN="$BINS_DIR/train.f32.bin"
case "$DTYPE" in
  float) TRAIN_BIN="$BINS_DIR/train.f32.bin";;
  bf16) TRAIN_BIN="$BINS_DIR/train.bf16.bin";;
  int8) TRAIN_BIN="$BINS_DIR/train.int8.bin";;
  *) die "unsupported --dtype: $DTYPE";;
 esac
[[ -f "$TRAIN_BIN" ]] || die "train bin not found for dtype=$DTYPE: $TRAIN_BIN (run prepare_hdf5_bins.py first)"

DATASET_TAG="$(basename "$DATASET" | sed 's/\.hdf5$//')"
INDEX_DIR="$OUT_ROOT/index_cache/$DATASET_TAG/$DTYPE/$DIST/maxdeg_${MAX_DEGREE}_Lbuild_${LBUILD}"
mkdir -p "$INDEX_DIR"
INDEX_PREFIX="$INDEX_DIR/index"

INDEX_FILE="${INDEX_PREFIX}_disk.index"

if [[ -f "$INDEX_FILE" ]]; then
  echo "[pipeline] Reusing existing index: $INDEX_FILE"
else
  echo "[pipeline] Building new index: $INDEX_FILE"
  BUILD_CMD=(
    "$ROOT_DIR/run_build_disk_index.sh"
    --data-path "$TRAIN_BIN"
    --index-prefix "$INDEX_PREFIX"
    --dist "$DIST" --dtype "$DTYPE" --threads "$THREADS"
    --B "$B_SEARCH_DRAM_GB" --M "$M_BUILD_DRAM_GB"
    --max-degree "$MAX_DEGREE" --Lbuild "$LBUILD"
  )
  if [[ -n "$DISKANN_APPS_DIR" ]]; then
    BUILD_CMD+=(--diskann-apps "$DISKANN_APPS_DIR")
  fi
  echo "[pipeline] Build cmd: ${BUILD_CMD[*]}"
  PYTHON="$PY" "${BUILD_CMD[@]}"
fi

# Now run recall sweep.
SWEEP_OUT="$OUT_ROOT/sweeps/$(date +%Y%m%d_%H%M%S)_${DATASET_TAG}_${DTYPE}_${DIST}_maxdeg${MAX_DEGREE}_Lbuild${LBUILD}"

SWEEP_CMD=(
  "$ROOT_DIR/run_recall_sweep.sh"
  --dataset "$DATASET"
  --index-prefix "$INDEX_PREFIX"
  --dist "$DIST" --dtype "$DTYPE"
  --K "$K"
  --Ls "${LS_LIST[@]}"
  --Ws "${WS_LIST[@]}"
  --threads "$THREADS"
  --num-nodes-to-cache "$NUM_NODES_TO_CACHE"
  --out-dir "$SWEEP_OUT"
  --max-degree "$MAX_DEGREE"
  --Lbuild "$LBUILD"
  --B "$B_SEARCH_DRAM_GB"
  --M "$M_BUILD_DRAM_GB"
)

if [[ ${#RECALL_ATS[@]} -gt 0 ]]; then
  SWEEP_CMD+=(--recall-ats "${RECALL_ATS[@]}")
fi

if [[ -n "$IO_LIMIT" ]]; then
  SWEEP_CMD+=(--search-io-limit "$IO_LIMIT")
fi
if [[ "$USE_REORDER" -eq 1 ]]; then
  SWEEP_CMD+=(--use-reorder-data)
fi

echo "[pipeline] Sweep cmd: ${SWEEP_CMD[*]}"
PYTHON="$PY" "${SWEEP_CMD[@]}"

echo "[pipeline] Done. Outputs in: $SWEEP_OUT"
