# diskann-pg/pq_reorder_bench

这个目录放一些可复现的 benchmark 脚本。

## PQ + reorder (float / bf16 / int8)

脚本：`pq_reorder_bench.py`

> 重要：该脚本 **只测试 SSD/disk index** 路径（`build_disk_index` + `search_disk_index`）。
> 不包含 in-memory index（`build_memory_index`/`search_memory_index`）的性能评估。

它会：

- 输入 float32 的 base/query（DiskANN `.bin` 格式）
- 自动生成 bf16/int8 版本的数据文件（chunked 转换，不要求一次性加载到内存）
- 对 float/bf16：构建 `PQ_disk_bytes>0 + append_reorder_data` 的 disk index，并在 search 时开启 `use_reorder_data`
- 对 int8：DiskANN 目前不支持 reorder（`search_disk_index`/`build_disk_index` 会报错），因此会跑 **PQ-only** 的 disk index
- 产出每个 dtype 的 build/search 时间、search 输出表格解析，以及 `summary.json`

> 注意：DiskANN 的 **reorder data**（也就是 `append_reorder_data`/`use_reorder_data`）目前只支持 `float/bf16`。
> 所以本 benchmark 的 int8 路径是 **PQ-only**（仍然会 build/search，但不会启用 reorder）。

---

## 1) 如何运行

### 1.1 前置条件

- 你已经编译了 DiskANN，并且能找到 `build_disk_index` / `search_disk_index` / `utils/rand_data_gen` 等二进制。
- 依赖：`python3` + `numpy`

安装 numpy：

```bash
pip install numpy
```

### 1.2 数据格式要求（DiskANN .bin）

本脚本要求 `--base-f32`/`--query-f32` 是 DiskANN 的 `.bin` 格式：

- 前 8 字节：`uint32 npts` + `uint32 dim`
- 后面：`npts * dim` 个元素
- 这里必须是 float32（`--base-f32` / `--query-f32` 都是 float32）

如果你没有数据，可用 DiskANN 自带的 `rand_data_gen` 生成：

```bash
APPS=/home/xtang/DiskANN-epeshared/build/apps
mkdir -p tmpdata

$APPS/utils/rand_data_gen --data_type float --output_file tmpdata/base_f32.bin -D 768 -N 1000000 --norm 1.0
$APPS/utils/rand_data_gen --data_type float --output_file tmpdata/query_f32.bin -D 768 -N 10000 --norm 1.0
```

### 1.3 最常用运行方式（推荐）

用 wrapper 脚本：

```bash
cd diskann-pg/pq_reorder_bench
chmod +x run_pq_reorder_bench.sh

./run_pq_reorder_bench.sh \
  --base-f32 tmpdata/rand_float_768D_1M_norm1.0.bin \
  --query-f32 tmpdata/rand_float_768D_10K_norm1.0.bin \
  --dist l2 \
  --K 10 \
  --Ls 10 20 30 40 50 100 \
  --PQ-disk-bytes 32 \
  --threads 32
```

运行成功后会在 stdout 打印一个路径，指向本次 run 的 `summary.json`。

### 1.4 直接运行 Python（更灵活）

```bash
cd diskann-pg/pq_reorder_bench

python3 pq_reorder_bench.py \
  --diskann-apps /home/xtang/DiskANN-epeshared/build/apps \
  --base-f32 tmpdata/base_f32.bin \
  --query-f32 tmpdata/query_f32.bin \
  --dist l2 \
  --K 10 \
  --Ls 10 20 30 \
  --PQ-disk-bytes 32 \
  --threads 32
```

### 1.5 Groundtruth（可选）

- 默认：如果你不传 `--gt` 且不加 `--skip-gt`，脚本会对每个 dtype 调 `compute_groundtruth` 生成自己的 gt。
- 如果你已经有 gt 文件：用 `--gt /path/to/gt.bin`（脚本会对三个 dtype 复用同一个 gt 路径）。
- 如果你只关心吞吐/延迟，不想花时间算 gt：加 `--skip-gt`。

---

## 2) 运行参数含义（完整）

### 2.1 输入/路径相关

- `--diskann-apps`：DiskANN 的 apps 目录（包含 `build_disk_index` / `search_disk_index`）。不传时会尝试自动探测。
- `--base-f32`：float32 base 向量 `.bin`
- `--query-f32`：float32 query 向量 `.bin`
- `--workdir`：输出根目录，默认 `diskann-pg/pq_reorder_bench/out`
- `--tag`：run 目录后缀标签，便于区分实验

### 2.2 距离函数

- `--dist`：`l2` / `cosine` / `mips`
  - DiskANN 限制：`mips` 的 disk index 路径不支持 `int8`。
  - `bf16+mips`：上游 DiskANN 默认可能未支持；如果你在本仓库中打了相关补丁（确保 MIPS 预处理文件 dtype 一致），则可以跑通。

### 2.3 Index build（DiskANN build_disk_index）

> 本节的 build 参数只适用于 **disk index（SSD）**：脚本调用的是 `build_disk_index`，不会调用 `build_memory_index`。

这些参数会传给 `build_disk_index`：

- `--R`：图的最大度（`-R/--max_degree`）
- `--Lbuild`：build 阶段的搜索 list（`-L/--Lbuild`）
- `--B`：search_DRAM_budget（GB）
- `--M`：build_DRAM_budget（GB）
- `--PQ-disk-bytes`：`--PQ_disk_bytes`，SSD 上的 PQ 压缩字节数（要启用 reorder 必须 > 0）
- `--QD`：`--QD`，强制覆盖 DiskANN 的 quantized dimension（PQ chunk 数）。
- `--threads`：并行线程数（build/search 都用这个）

> 强烈建议（尤其是小数据集/quickcheck）：显式设置 `--QD`（例如 `--QD 32`）。
>
> 原因：DiskANN 默认会从 `--B/--search_DRAM_budget` 反推一个 chunk 数；在点数很小但 `B` 不小（例如 `B=0.25GB`）时，会推导出非常大的 chunk 数，
> 进而导致 PQ 训练（kmeans/kmeans++，perf 里常见热点 `kmeanspp_selecting_pivots`）耗时非常长。
> 对于 `--dist mips`，DiskANN 还会先把 base 预处理成 $d+1$ 维（例如 768 -> 769），会进一步放大这个问题。
>
> 经验值：通常可以先设 `--QD = --PQ-disk-bytes`（比如 `--PQ-disk-bytes 32` 配 `--QD 32`），然后再按需要调优。

脚本的策略：

- dtype=float/bf16：build 时会追加 `--append_reorder_data`，让 index 里包含用于 reorder 的“全精度数据”。
- dtype=int8：不会加 `--append_reorder_data`（DiskANN 不支持 int8 reorder）。

### 2.4 Search（DiskANN search_disk_index）

> 本节的 search 参数只适用于 **disk index（SSD）**：脚本调用的是 `search_disk_index`，不会调用 `search_memory_index`。

这些参数会传给 `search_disk_index`：

- `--K`：返回 topK
- `--Ls`：一组 L 值（search_list），脚本会逐个 L 跑并解析表格输出
- `--beamwidth`：`-W/--beamwidth`
- `--num-nodes-to-cache`：`--num_nodes_to_cache`

脚本的策略：

- dtype=float/bf16：search 时会加 `--use_reorder_data`（启用 PQ+reorder）。
- dtype=int8：不会加 `--use_reorder_data`（PQ-only）。

### 2.5 数据转换（float32 -> bf16 / int8）

- `--chunk-elems`：转换时每次处理的元素数量（默认 `1<<20`），用于控制峰值内存。
- `--int8-scale`：int8 量化 scale。
  - 不指定时：脚本会扫描 base/query 的 max-abs，自动设定 `scale = max_abs/127`。
  - 量化公式：`q = round(x / scale)`，并 clamp 到 `[-127, 127]`。

### 2.6 跳过步骤（调试用）

- `--skip-build`：只做转换/gt（如果没 skip），不 build
- `--skip-search`：build 完不 search
- `--skip-gt`：不算 groundtruth、不传 `--gt_file`

### 2.7 一致性检测

- `--consistency-L`：用哪个 L 的结果做一致性对比。
  - 默认：`max(--Ls)`

---

## 3) 如何解读运行后的结果

### 3.1 输出目录结构

每次运行会生成一个目录：

`<workdir>/<timestamp>_<tag>/`

里面包含：

- `summary.json`：核心汇总结果（你主要看这个）
- `logs/`：每个 dtype 的 build/search（以及可能的 gt）原始输出
- `data/`：转换后的 `bf16/int8` 数据文件
- `<dtype>/index/`：构建出来的 disk index 文件（前缀形如 `disk_index_<dtype>_...`）
- `<dtype>/results/`：search 结果文件（每个 L 一份）

其中 `<dtype>` 取值：`float` / `bf16` / `int8`。

### 3.2 summary.json 字段说明

`summary.json` 大致结构：

- `meta`：本次实验的输入参数（K/Ls/R/Lbuild/B/M/PQ bytes/threads/dist 等）
- `convert`：int8 的量化 scale 等信息（bf16 不需要额外参数）
- `gt`：每个 dtype 用的 gt 路径（如果 `--skip-gt` 则为 null）
- `results`：每个 dtype 的 build/search 指标
- `consistency`：和 float baseline 的一致性统计

#### results.<dtype>

- `build_time_s`：从脚本角度测到的 build wall time（秒）
- `search_time_s`：search 整体 wall time（秒）
- `reorder`：该 dtype 是否启用 reorder（float/bf16 为 true；int8 为 false）
- `search_rows`：从 `search_disk_index` 的表格输出中解析出的每个 L 的核心指标

`search_rows` 的每一行包含：

- `L`：search list size
- `beamwidth`
- `qps`：queries/sec（越大越好）
- `mean_us`：平均延迟（微秒，越小越好）
- `p999_us`：P99.9 延迟（微秒，越小越好）

> 如果你没有提供/生成 gt，DiskANN 的表格不会打印 Recall 列；脚本也不会在 `search_rows` 里包含 recall。

#### consistency

一致性比较是“同一组 query 的 topK 结果”对比：

- baseline 固定为 `float`
- 对 `bf16` 和 `int8` 分别统计：
  - `topk_overlap_mean`：
    $$\frac{|TopK_{float} \cap TopK_{dtype}|}{K}$$
    越接近 1 表示返回集合越一致。
  - `positional_match_mean`：
    $$\frac{\#\{i \in [0,K): id_{float}[i] = id_{dtype}[i]\}}{K}$$
    越接近 1 表示排名顺序越一致。

一般解读建议：

- bf16 vs float：`topk_overlap_mean` 通常应较高（取决于数据/参数/是否启用 BF16 SIMD 等）。
- int8（PQ-only）vs float（PQ+reorder）：差异可能会更大，这是预期现象（算法路径都不同）。

### 3.3 常见问题 / 注意事项

- “为什么 int8 没有 reorder？”：这是 DiskANN CLI 的限制（reorder data 仅支持 float/bf16）。
- “bf16 build 为什么特别慢，perf 热点是 kmeans++？”：大概率是 PQ 训练在跑（kmeans/kmeans++）。
  如果你是小数据集/快速验证，请显式设 `--QD`（例如 `--QD 32`），避免 DiskANN 从 `--B` 推导出巨大的 chunk 数导致训练时间爆炸。
- “跑得很慢/文件很大”：`--PQ-disk-bytes` 越大、`R/Lbuild` 越大，build 时间和 index 体积都会上升。
- “想复现实验”：保留对应 run 目录即可（里面包含转换后的数据、index、日志、summary）。

### 运行示例

```bash
cd diskann-pg/pq_reorder_bench
chmod +x run_pq_reorder_bench.sh

./run_pq_reorder_bench.sh \
  --base-f32 tmpdata/rand_float_768D_1M_norm1.0.bin \
  --query-f32 tmpdata/rand_float_768D_10K_norm1.0.bin \
  --dist l2 \
  --K 10 \
  --Ls 10 20 30 40 50 100 \
  --PQ-disk-bytes 32 \
  --threads 32
```

输出会打印 `summary.json` 的路径，默认在 `diskann-pg/pq_reorder_bench/out/<timestamp>_pq_reorder/summary.json`。

### 依赖

- `python3`
- `numpy`

如果缺 numpy：

```bash
pip install numpy
```

### DiskANN apps 路径

脚本会尝试自动找：

- `/home/xtang/DiskANN-epeshared/build/apps`
- `/home/xtang/DiskANN/build/apps`
- `/home/xtang/DiskANN/build/tests`

如果你的实际路径不同，请显式传：

```bash
./run_pq_reorder_bench.sh --diskann-apps /path/to/DiskANN/build/apps ...
```
