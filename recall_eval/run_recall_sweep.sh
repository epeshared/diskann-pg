#!/usr/bin/env bash
set -euo pipefail

# Minimal wrapper to run recall_sweep_disk_index.py with sane defaults.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}

DATASET=""
INDEX_PREFIX=""
SEARCH_BIN=""

DIST="cosine"
DTYPE="bf16"
K=10
THREADS="$(nproc)"
NUM_NODES_TO_CACHE=0

LS_LIST=(50 100 150 200)
WS_LIST=(2 4 8)

OUT_DIR=""
USE_REORDER=0
IO_LIMIT=""

usage() {
  cat <<'EOF'
Usage:
  ./run_recall_sweep.sh --dataset <dbpedia-openai-1000k-angular.hdf5> \
                        --index-prefix <prefix_without__disk.index> [options]

Options:
  --search-bin <path>        Path to DiskANN search_disk_index (optional; auto-detect if omitted)
  --dist <l2|mips|cosine>    Default: cosine
  --dtype <float|bf16|int8>  Default: bf16
  --K <k>                    Default: 10
  --Ls <l1 l2 ...>           Default: 50 100 150 200
  --Ws <w1 w2 ...>           Default: 2 4 8
  --threads <t>              Default: nproc
  --num-nodes-to-cache <n>   Default: 0
  --search-io-limit <n>      Optional
  --use-reorder-data         Pass --use_reorder_data to search_disk_index
  --out-dir <dir>            Optional; default: recall_eval/runs/<timestamp>

Example:
  ./run_recall_sweep.sh \
    --dataset /path/to/dbpedia-openai-1000k-angular.hdf5 \
    --index-prefix /path/to/index_run0_disk \
    --dist cosine --dtype bf16 --K 10 \
    --Ls 50 100 150 200 --Ws 2 4 8 --threads 32
EOF
}

die() { echo "Error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --index-prefix) INDEX_PREFIX="$2"; shift 2;;
    --search-bin) SEARCH_BIN="$2"; shift 2;;

    --dist) DIST="$2"; shift 2;;
    --dtype|--data-type) DTYPE="$2"; shift 2;;
    --K) K="$2"; shift 2;;

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

    --threads) THREADS="$2"; shift 2;;
    --num-nodes-to-cache) NUM_NODES_TO_CACHE="$2"; shift 2;;
    --search-io-limit) IO_LIMIT="$2"; shift 2;;
    --use-reorder-data) USE_REORDER=1; shift;;
    --out-dir) OUT_DIR="$2"; shift 2;;

    --help|-h) usage; exit 0;;
    *) die "Unknown arg: $1";;
  esac
done

[[ -n "$DATASET" ]] || die "--dataset is required"
[[ -n "$INDEX_PREFIX" ]] || die "--index-prefix is required"

CMD=(
  "$PY" "$ROOT_DIR/recall_sweep_disk_index.py"
  --dataset "$DATASET"
  --index-prefix "$INDEX_PREFIX"
  --dist "$DIST"
  --data-type "$DTYPE"
  --K "$K"
  --Ls "${LS_LIST[@]}"
  --Ws "${WS_LIST[@]}"
  --threads "$THREADS"
  --num-nodes-to-cache "$NUM_NODES_TO_CACHE"
)

if [[ -n "$SEARCH_BIN" ]]; then
  CMD+=(--search-bin "$SEARCH_BIN")
fi

if [[ -n "$IO_LIMIT" ]]; then
  CMD+=(--search-io-limit "$IO_LIMIT")
fi

if [[ "$USE_REORDER" -eq 1 ]]; then
  CMD+=(--use-reorder-data)
fi

if [[ -n "$OUT_DIR" ]]; then
  CMD+=(--out-dir "$OUT_DIR")
fi

echo "[recall_eval] Cmd: ${CMD[*]}"
"${CMD[@]}"
