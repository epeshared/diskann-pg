# bench_data

Shared data utilities for `diskann-pg` benchmarks.

## generate_data.sh

Generates DiskANN `.bin` float32 base/query data using DiskANN `utils/rand_data_gen`.

Outputs are named with both datatype and a readable count tag (e.g. `1M`, `10K`):

- `base_<D>_f32_<N>.bin`, `query_<D>_f32_<N>.bin`
- (with `--also-bf16`) `base_<D>_bf16_<N>.bin`, `query_<D>_bf16_<N>.bin`

```bash
# default: D=1536, base=1M, query=10K, out-dir=tmpdata
./generate_data.sh --diskann-apps /home/xtang/DiskANN-epeshared/build/apps

# custom
./generate_data.sh --out-dir /home/xtang/diskann-pg/tmpdata --dim 768 --base-n 1000000 --query-n 10000

# also produce bf16 versions
./generate_data.sh --out-dir ./tmpdata --dim 768 --base-n 1000000 --query-n 10000 --also-bf16
```

## Python conversion helpers

- `bench_data/bin_utils.py` provides chunked conversion:
  - float32 `.bin` -> bf16 `.bin`
  - float32 `.bin` -> int8 `.bin` (with optional scale)

These are used by `pq_reorder_bench` and can be reused by other benchmarks.
