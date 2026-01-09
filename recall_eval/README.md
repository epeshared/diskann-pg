# recall_eval

这个目录用于 **DiskANN disk index 的召回率(correctness)验证**：

- 输入数据集：`dbpedia-openai-1000k-angular.hdf5`
- 运行 DiskANN 的 `search_disk_index`，可调整多种搜索参数
- 输出：`summary.csv` + `recall_heatmap.png`(或 `recall_curve.png`)

## 前置条件

- 已经构建好 DiskANN 的 `search_disk_index`（在 `DiskANN-epeshared/build*/apps/search_disk_index`）
- Python 依赖：`numpy`、`h5py`、`matplotlib`

如果缺依赖：

- `pip install -r requirements.txt`

或手动安装：

- `pip install numpy h5py matplotlib`

## 快速开始

```bash
cd /home/xtang/diskann-pg/recall_eval

./run_recall_sweep.sh \
  --dataset /path/to/dbpedia-openai-1000k-angular.hdf5 \
  --index-prefix /path/to/index_run0_disk \
  --dist cosine \
  --dtype bf16 \
  --K 10 \
  --Ls 50 100 150 200 \
  --Ws 2 4 8 \
  --threads 32
```

## 从 dbpedia HDF5 到“建索引 + sweep”的一条龙

dbpedia 的 HDF5 keys（已验证）:

- `train`: (990000, 1536) float32
- `test`: (10000, 1536) float32
- `neighbors`: (10000, 100) int64
- `distances`: (10000, 100) float64

1) 导出 DiskANN 所需的 `.bin`（train/test/gt），并生成 bf16/int8 版本：

```bash
/home/xtang/diskann-pg/.venv/bin/python ./prepare_hdf5_bins.py \
  --dataset /mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5 \
  --out-dir ./dbpedia_bins
```

会生成：

- `dbpedia_bins/train.f32.bin`, `dbpedia_bins/test.f32.bin`
- `dbpedia_bins/train.bf16.bin`, `dbpedia_bins/test.bf16.bin`
- `dbpedia_bins/train.int8.bin`, `dbpedia_bins/test.int8.bin`
- `dbpedia_bins/gt.uint32.f32.K100.bin`

2) 建 DiskANN disk index（示例：bf16 + cosine）：

```bash
./run_build_disk_index.sh \
  --data-path ./dbpedia_bins/train.bf16.bin \
  --index-prefix ./dbpedia_index/dbpedia_bf16_cosine \
  --dist cosine --dtype bf16 --threads 32 \
  --B 4 --M 32 --R 64 --Lbuild 100
```

完成后会生成：

- `dbpedia_index/dbpedia_bf16_cosine_disk.index`（以及同前缀的其它文件）

3) 跑 recall sweep（注意 index-prefix 不带 `_disk.index` 后缀）：

```bash
./run_recall_sweep.sh \
  --dataset /mnt/nvme2n1p1/xtang/ann-data/dbpedia-openai-1000k-angular.hdf5 \
  --index-prefix ./dbpedia_index/dbpedia_bf16_cosine \
  --dist cosine --dtype bf16 --K 10 \
  --Ls 50 100 150 200 --Ws 2 4 8 --threads 32
```

运行结束后会生成：

- `runs/<timestamp>/summary.csv`
- `runs/<timestamp>/recall_heatmap.png`（当同时 sweep 多个 L 和多个 W）
  - 否则输出 `runs/<timestamp>/recall_curve.png`

## 参数说明

脚本主体：`recall_sweep_disk_index.py`

- `--dataset`：HDF5 数据集路径
- `--index-prefix`：DiskANN index 前缀（不带 `_disk.index` 后缀）
- `--data-type`：`float|bf16|int8`（会从 HDF5 queries 自动生成对应 dtype 的 `.bin` 查询文件）
- `--dist`：`l2|mips|cosine`
- `--K`：计算 Recall@K，同时作为 `search_disk_index --recall_at`
- `--Ls`：要 sweep 的 `search_list` 参数列表（每个 L 都会输出结果，并计算召回率）
- `--Ws`：要 sweep 的 `beamwidth` 列表（每个 W 会单独运行一次 `search_disk_index`）
- 其它：`--threads`、`--num-nodes-to-cache`、`--search-io-limit`、`--use-reorder-data`

## HDF5 数据格式假设

脚本会自动尝试从 HDF5 中推断 key：

- queries: `test|queries|query|Q|xq`
- gt ids: `neighbors|knn|gt|groundtruth|truth`
- gt distances(可选): `distances|dists|gt_distances|truth_distances`

如果你的 HDF5 key 不同，可以显式指定：

```bash
python3 recall_sweep_disk_index.py \
  --dataset ... \
  --queries-key test \
  --gt-ids-key neighbors \
  --gt-dists-key distances \
  --index-prefix ...
```
