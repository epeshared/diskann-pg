#!/usr/bin/env bash
set -euo pipefail

# Wrapper for bench_build_disk_index.py
# You likely want to: source ../env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (can be overridden by passing the same flags on the command line).
# Tip: set DISKANN_BDI_DATA_PATH to avoid editing this file.
DEFAULT_DATA_TYPE=${DISKANN_BDI_DATA_TYPE:-bf16}
DEFAULT_DIST_FN=${DISKANN_BDI_DIST_FN:-mips}
DEFAULT_DATA_PATH=${DISKANN_BDI_DATA_PATH:-/home/xtang/diskann-pg/bench_data/tmpdata/base_1536_bf16_1M.bin}
DEFAULT_SEARCH_DRAM_BUDGET=${DISKANN_BDI_SEARCH_DRAM_BUDGET:-64}
DEFAULT_BUILD_DRAM_BUDGET=${DISKANN_BDI_BUILD_DRAM_BUDGET:-64}
DEFAULT_NUM_THREADS=${DISKANN_BDI_NUM_THREADS:-32}
DEFAULT_MAX_DEGREE=${DISKANN_BDI_MAX_DEGREE:-64}
DEFAULT_LBUILD=${DISKANN_BDI_LBUILD:-100}
DEFAULT_RUNS=${DISKANN_BDI_RUNS:-3}

# If the user did not pass --data-path, ensure the default path exists.
has_data_path=0
for arg in "$@"; do
	if [[ "$arg" == "--data-path" ]]; then
		has_data_path=1
		break
	fi
done

if [[ $has_data_path -eq 0 ]]; then
	if [[ ! -f "$DEFAULT_DATA_PATH" ]]; then
		echo "ERROR: default --data-path does not exist: $DEFAULT_DATA_PATH" >&2
		echo "Set DISKANN_BDI_DATA_PATH or pass --data-path <file>." >&2
		exit 2
	fi
fi

exec python3 "$SCRIPT_DIR/bench_build_disk_index.py" \
	--data-type "$DEFAULT_DATA_TYPE" \
	--dist-fn "$DEFAULT_DIST_FN" \
	--data-path "$DEFAULT_DATA_PATH" \
	--search-dram-budget "$DEFAULT_SEARCH_DRAM_BUDGET" \
	--build-dram-budget "$DEFAULT_BUILD_DRAM_BUDGET" \
	--num-threads "$DEFAULT_NUM_THREADS" \
	--max-degree "$DEFAULT_MAX_DEGREE" \
	--lbuild "$DEFAULT_LBUILD" \
	--runs "$DEFAULT_RUNS" \
	"$@"
