#!/usr/bin/env bash
set -euo pipefail

# Shared data generation helper for diskann-pg benchmarks.
# Generates float32 base/query .bin using DiskANN's utils/rand_data_gen.
# Optional: also convert generated float32 .bin to bf16 .bin.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

APPS_DIR="${DISKANN_APPS_DIR:-/home/xtang/DiskANN-epeshared/build-avx512/apps}"
OUT_DIR="tmpdata"
DIM=1536
BASE_N=1000000
QUERY_N=10000
NORM=1.0
ALSO_BF16=0

fmt_count() {
  local n="$1"
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n % 1000000 == 0 )) && (( n >= 1000000 )); then
    echo "$(( n / 1000000 ))M"
    return
  fi
  if [[ "$n" =~ ^[0-9]+$ ]] && (( n % 1000 == 0 )) && (( n >= 1000 )); then
    echo "$(( n / 1000 ))K"
    return
  fi
  echo "$n"
}

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --diskann-apps <dir>   DiskANN apps dir (contains utils/rand_data_gen). Default: $APPS_DIR
  --out-dir <dir>        Output directory. Default: $OUT_DIR
  --dim <D>              Dimension. Default: $DIM
  --base-n <N>           Base vector count. Default: $BASE_N
  --query-n <N>          Query vector count. Default: $QUERY_N
  --norm <x>             L2 norm used by rand_data_gen. Default: $NORM
  --also-bf16            Also write bf16 versions of the generated float32 files.
  -h, --help             Show help.

Outputs:
  <out-dir>/base_<D>_f32_<base-n>.bin
  <out-dir>/query_<D>_f32_<query-n>.bin
  (optional)
  <out-dir>/base_<D>_bf16_<base-n>.bin
  <out-dir>/query_<D>_bf16_<query-n>.bin
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --diskann-apps) APPS_DIR="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --dim) DIM="$2"; shift 2;;
    --base-n) BASE_N="$2"; shift 2;;
    --query-n) QUERY_N="$2"; shift 2;;
    --norm) NORM="$2"; shift 2;;
    --also-bf16) ALSO_BF16=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

RAND="$APPS_DIR/utils/rand_data_gen"
if [[ ! -x "$RAND" ]]; then
  echo "ERROR: rand_data_gen not found or not executable: $RAND" >&2
  echo "Set DISKANN_APPS_DIR or pass --diskann-apps." >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

BASE_TAG="$(fmt_count "$BASE_N")"
QUERY_TAG="$(fmt_count "$QUERY_N")"

BASE_F32="$OUT_DIR/base_${DIM}_f32_${BASE_TAG}.bin"
QUERY_F32="$OUT_DIR/query_${DIM}_f32_${QUERY_TAG}.bin"

"$RAND" --data_type float --output_file "$BASE_F32" -D "$DIM" -N "$BASE_N" --norm "$NORM"
"$RAND" --data_type float --output_file "$QUERY_F32" -D "$DIM" -N "$QUERY_N" --norm "$NORM"

echo "Wrote: $BASE_F32"
echo "Wrote: $QUERY_F32"

if [[ $ALSO_BF16 -eq 1 ]]; then
  BASE_BF16="$OUT_DIR/base_${DIM}_bf16_${BASE_TAG}.bin"
  QUERY_BF16="$OUT_DIR/query_${DIM}_bf16_${QUERY_TAG}.bin"

  python3 -c "import sys; sys.path.insert(0, '$(cd "$ROOT_DIR/.." && pwd)'); from bench_data.bin_utils import convert_f32_bin_to_bf16_bin; from pathlib import Path; convert_f32_bin_to_bf16_bin(Path('$BASE_F32'), Path('$BASE_BF16'))"
  python3 -c "import sys; sys.path.insert(0, '$(cd "$ROOT_DIR/.." && pwd)'); from bench_data.bin_utils import convert_f32_bin_to_bf16_bin; from pathlib import Path; convert_f32_bin_to_bf16_bin(Path('$QUERY_F32'), Path('$QUERY_BF16'))"

  echo "Wrote: $BASE_BF16"
  echo "Wrote: $QUERY_BF16"
fi
