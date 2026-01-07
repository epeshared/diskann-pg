#!/usr/bin/env bash
set -euo pipefail

# Unified runner for pq_reorder_bench.py (AVX512 vs AMX).
#
# Default (no args): run the full python pipeline using the AVX512 build.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PYTHON:-python3}

ISA="avx512"          # amx | avx512
APPS_DIR=""            # can override via --diskann-apps or DISKANN_APPS_DIR
INDEX_ROOT="/home/xtang/diskann-pg/build_disk_index_bench/runs/avx512"    # root containing <dtype>/index/*_disk.index
QUERY_FILE="/home/xtang/diskann-pg/bench_data/tmpdata/query_1536_bf16_10K.bin"                                                         # query .bin file path (or template with {dtype})

# defaults
DTYPES=(bf16)
DIST="mips"
K=24
W=8
SEARCH_THREADS=32

LS_LIST=(200)
NUM_NODES_TO_CACHE=0

PASSTHRU=()

usage() {
  cat <<'EOF'
Usage:
  ./run_pq_reorder_bench.sh [--isa amx|avx512] [--index-root <dir>]
                            [--query-file <path>]
                            [--diskann-apps <apps_dir>]
                            [--dtypes <...>] [--dist <l2|mips|cosine>]
                            [--K <k>] [--Ls <l1 l2 ...>] [--W <beamwidth>] [--threads <t>]
                            [extra args passed to pq_reorder_bench.py]

Examples:
  # Search only
  ./run_pq_reorder_bench.sh --isa avx512 --index-root ./data --query-file ./data/data/query.bf16.bin --dtypes bf16 --K 24 --Ls 200 --W 8 --threads 32

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
    --run-root|--index-root|--out-root) INDEX_ROOT="$2"; shift 2;;
    --data-dir|--query-dir|--query-file) QUERY_FILE="$2"; shift 2;;
    --diskann-apps) APPS_DIR="$2"; shift 2;;

    --dtype) DTYPES=("$2"); shift 2;;
    --dtypes)
      shift
      DTYPES=()
      while [[ $# -gt 0 && "$1" != -* ]]; do
        DTYPES+=("$1")
        shift
      done
      ;;
    --dist) DIST="$2"; shift 2;;
    --K) K="$2"; shift 2;;
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
      shift 2
      ;;

    --num-nodes-to-cache) NUM_NODES_TO_CACHE="$2"; shift 2;;

    --help|-h) usage; exit 0;;
    *) PASSTHRU+=("$1"); shift;;
  esac
done

# Default query file:
# - If not specified, try build_disk_index_bench/data/1M/ with common naming.
if [[ -z "$QUERY_FILE" ]]; then
  BASE="$ROOT_DIR/../build_disk_index_bench/data/1M"
  if [[ -d "$BASE" ]]; then
    # Prefer explicit query.<dtype>.bin if present.
    if [[ -f "$BASE/query.bf16.bin" ]]; then
      QUERY_FILE="$BASE/query.bf16.bin"
    else
      # Otherwise pick the newest generator-style bf16 query file.
      CAND=$(ls -t "$BASE"/query_*_bf16_*.bin 2>/dev/null | head -n 1 || true)
      if [[ -n "$CAND" ]]; then
        QUERY_FILE="$CAND"
      fi
    fi
  fi
fi

if [[ -z "${APPS_DIR}" ]]; then
  if [[ -n "${DISKANN_APPS_DIR:-}" ]]; then
    APPS_DIR="${DISKANN_APPS_DIR}"
  else
    case "${ISA}" in
      amx) APPS_DIR="/home/xtang/DiskANN-epeshared/build-amx/apps";;
      avx512) APPS_DIR="/home/xtang/DiskANN-epeshared/build-avx512/apps";;
      *) die "unknown --isa '${ISA}' (expected amx or avx512)";;
    esac
  fi
fi

[[ -d "$APPS_DIR" ]] || die "apps dir not found: $APPS_DIR"

[[ -d "$INDEX_ROOT" ]] || die "index root not found: $INDEX_ROOT"
[[ -n "$QUERY_FILE" ]] || die "query file is required (pass --query-file)"
[[ -f "$QUERY_FILE" ]] || die "query file not found: $QUERY_FILE"

PY_CMD=("$PY" "$ROOT_DIR/pq_reorder_bench.py")

CMD=(
  "${PY_CMD[@]}"
  --diskann-apps "$APPS_DIR"
  --index-root "$INDEX_ROOT"
  --query-file "$QUERY_FILE"
  --dist "$DIST"
  --beamwidth "$W"
  --K "$K"
  --Ls "${LS_LIST[@]}"
  --num-nodes-to-cache "$NUM_NODES_TO_CACHE"
  --threads "$SEARCH_THREADS"
  --dtypes "${DTYPES[@]}"
)

CMD+=("${PASSTHRU[@]}")

echo "[search] ISA: $ISA"
echo "[search] Index root: $INDEX_ROOT"
echo "[search] Query file: $QUERY_FILE"
echo "[search] Apps: $APPS_DIR"
echo "[search] Cmd: ${CMD[*]}"
"${CMD[@]}"

exit 0

