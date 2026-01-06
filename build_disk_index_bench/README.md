# build_disk_index_bench

Benchmarks DiskANN `build_disk_index` build time by running two separate DiskANN builds (e.g. AVX512 vs AMX) with identical CLI arguments.

## Defaults

- AVX512 build dir: `/home/xtang/DiskANN-epeshared/build-avx512`
- AMX build dir: `/home/xtang/DiskANN-epeshared/build-amx`

Both are expected to contain `apps/build_disk_index`.

## Usage

From this folder:

```bash
# If you need oneAPI/ICC runtime libs, do this first:
source ../env.sh

# Example: bf16 + l2
./run_build_disk_index_bench.sh \
  --data-type bf16 \
  --dist-fn l2 \
  --data-path /path/to/base.bf16.bin \
  --search-dram-budget 64 \
  --build-dram-budget 64 \
  --num-threads 32 \
  --max-degree 64 \
  --lbuild 100 \
  --runs 3
```

Outputs (indexes + logs) are written under `build_disk_index_bench/runs/`.

If you need synthetic `.bin` data, use the shared generator in `diskann-pg/bench_data/`:

```bash
cd diskann-pg/bench_data

# Generate float32 base/query (and optionally bf16 versions)
./generate_data.sh --dim 768 --base-n 1000000 --query-n 10000 --also-bf16
```

## Arguments

This benchmark runner ultimately executes DiskANN’s `apps/build_disk_index` twice (AVX512 build and AMX build). Most flags below map 1:1 to `build_disk_index`’s CLI.

### Build selection

- `--avx512-build-dir`
  - Meaning: DiskANN CMake build directory for the AVX512 binary.
  - Default: `/home/xtang/DiskANN-epeshared/build-avx512`
  - Must contain: `apps/build_disk_index`
- `--amx-build-dir`
  - Meaning: DiskANN CMake build directory for the AMX binary.
  - Default: `/home/xtang/DiskANN-epeshared/build-amx`
  - Must contain: `apps/build_disk_index`

### Required `build_disk_index` parameters

- `--data-type`
  - Meaning: input vector element type.
  - Examples: `float`, `bf16`/`bfloat16`, `uint8`, `int8`
  - Maps to DiskANN: `--data_type`
- `--dist-fn`
  - Meaning: distance/metric used during build.
  - Values: `l2`, `mips`, `cosine`
  - Maps to DiskANN: `--dist_fn`
- `--data-path`
  - Meaning: path to the input `.bin` file.
  - Maps to DiskANN: `--data_path`
- `--search-dram-budget`
  - Meaning: search-time DRAM budget (GB). Used by DiskANN to decide compression level.
  - Maps to DiskANN: `-B` / `--search_DRAM_budget`
- `--build-dram-budget`
  - Meaning: build-time DRAM budget (GB).
  - Maps to DiskANN: `-M` / `--build_DRAM_budget`

### Common optional `build_disk_index` parameters

- `--num-threads`
  - Meaning: number of threads.
  - Maps to DiskANN: `-T` / `--num_threads`
- `--max-degree`
  - Meaning: graph max degree `R`.
  - Maps to DiskANN: `-R` / `--max_degree`
- `--lbuild`
  - Meaning: graph build complexity `L`.
  - Maps to DiskANN: `-L` / `--Lbuild`
- `--pq-disk-bytes`
  - Meaning: compress vectors on SSD to this many bytes; `0` means store full vectors.
  - Maps to DiskANN: `--PQ_disk_bytes`
- `--build-pq-bytes`
  - Meaning: PQ bytes used during graph build (`build_PQ_bytes`).
  - Maps to DiskANN: `--build_PQ_bytes`
- `--qd`
  - Meaning: quantized dimension (`QD`) used by DiskANN compression.
  - Maps to DiskANN: `--QD`
- `--use-opq`
  - Meaning: enable OPQ.
  - Maps to DiskANN: `--use_opq`
- `--append-reorder-data`
  - Meaning: include full-precision reorder data in the disk index (only relevant when using disk PQ).
  - Maps to DiskANN: `--append_reorder_data`
- `--codebook-prefix`
  - Meaning: path prefix for a pre-trained codebook.
  - Maps to DiskANN: `--codebook_prefix`

### Benchmark-only parameters

- `--runs`
  - Meaning: number of repeated runs per ISA.
  - Note: each run uses a fresh output prefix (index rebuilt each time).
- `--out-root`
  - Meaning: base output directory for logs and generated index files.

### Passing through extra DiskANN flags

Any extra flags not covered above can be appended after a `--` separator and will be forwarded to `build_disk_index` as-is.

Example:

```bash
./run_build_disk_index_bench.sh \
  --data-type bf16 \
  --dist-fn l2 \
  --data-path /path/to/base.bf16.bin \
  --search-dram-budget 64 \
  --build-dram-budget 64 \
  --runs 1 \
  -- \
  --label_type uint
```

## Notes

- This measures wall-clock time of the `build_disk_index` process (includes IO).
- The script rebuilds the index for each run to keep runs comparable.
- To override the DiskANN build dirs:

```bash
./run_build_disk_index_bench.sh \
  --avx512-build-dir /home/xtang/DiskANN-epeshared/build-avx512 \
  --amx-build-dir /home/xtang/DiskANN-epeshared/build-amx \
  ...
```
