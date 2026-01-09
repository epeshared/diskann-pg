#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray as NDArray
else:  # pragma: no cover
    NDArray = Any  # type: ignore


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench_data.bin_utils import read_bin_header, require_numpy  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _must_exist(path: Path, what: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def _try_import_h5py():
    try:
        import h5py  # noqa: F401

        return h5py
    except Exception as e:
        raise RuntimeError(
            "This script requires h5py. Install with: pip install h5py"
        ) from e


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        return matplotlib
    except Exception as e:
        raise RuntimeError(
            "This script requires matplotlib. Install with: pip install matplotlib"
        ) from e


def _load_hdf5_arrays(
    dataset_path: Path,
    *,
    queries_key: Optional[str] = None,
    gt_ids_key: Optional[str] = None,
    gt_dists_key: Optional[str] = None,
) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
    np = require_numpy()
    h5py = _try_import_h5py()

    _must_exist(dataset_path, "HDF5 dataset")

    with h5py.File(dataset_path, "r") as f:
        keys = set(f.keys())

        def pick(candidates: Sequence[str]) -> Optional[str]:
            for k in candidates:
                if k in keys:
                    return k
            return None

        qk = queries_key or pick(["test", "queries", "query", "Q", "xq"])
        gik = gt_ids_key or pick(["neighbors", "knn", "gt", "groundtruth", "truth"])
        gdk = gt_dists_key or pick(["distances", "dists", "gt_distances", "truth_distances"])

        if qk is None:
            raise KeyError(
                f"Could not infer queries key in {dataset_path}. Available keys: {sorted(keys)}. "
                "Pass --queries-key explicitly."
            )
        if gik is None:
            raise KeyError(
                f"Could not infer ground-truth neighbor ids key in {dataset_path}. Available keys: {sorted(keys)}. "
                "Pass --gt-ids-key explicitly."
            )

        queries = np.asarray(f[qk], dtype=np.float32)
        gt_ids = np.asarray(f[gik], dtype=np.uint32)
        gt_dists = None
        if gdk is not None and gdk in f:
            try:
                gt_dists = np.asarray(f[gdk], dtype=np.float32)
            except Exception:
                gt_dists = None

    if queries.ndim != 2:
        raise ValueError(f"queries must be 2D, got shape={queries.shape}")
    if gt_ids.ndim != 2:
        raise ValueError(f"gt_ids must be 2D, got shape={gt_ids.shape}")
    if gt_ids.shape[0] != queries.shape[0]:
        raise ValueError(
            f"Mismatch: queries.shape[0]={queries.shape[0]} vs gt_ids.shape[0]={gt_ids.shape[0]}"
        )
    if gt_dists is not None:
        if gt_dists.shape != gt_ids.shape:
            _log(
                f"Warning: gt_dists shape {gt_dists.shape} != gt_ids shape {gt_ids.shape}; ignoring gt_dists"
            )
            gt_dists = None

    return queries, gt_ids, gt_dists


def _numpy_to_diskann_bin_f32(array: NDArray, out_path: Path) -> None:
    np = require_numpy()
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"array must be 2D, got shape={array.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    npts = np.uint32(array.shape[0])
    dim = np.uint32(array.shape[1])
    with out_path.open("wb") as f:
        f.write(npts.tobytes())
        f.write(dim.tobytes())
        f.write(array.tobytes(order="C"))


def _write_gt_bin(gt_ids: NDArray, gt_dists: Optional[NDArray], out_path: Path) -> None:
    np = require_numpy()
    if gt_ids.dtype != np.uint32:
        gt_ids = gt_ids.astype(np.uint32, copy=False)
    if gt_ids.ndim != 2:
        raise ValueError(f"gt_ids must be 2D, got shape={gt_ids.shape}")

    if gt_dists is None:
        gt_dists = np.zeros_like(gt_ids, dtype=np.float32)
    else:
        if gt_dists.dtype != np.float32:
            gt_dists = gt_dists.astype(np.float32, copy=False)
        if gt_dists.shape != gt_ids.shape:
            raise ValueError(f"gt_dists shape {gt_dists.shape} != gt_ids shape {gt_ids.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    nq = np.uint32(gt_ids.shape[0])
    k = np.uint32(gt_ids.shape[1])
    with out_path.open("wb") as f:
        f.write(nq.tobytes())
        f.write(k.tobytes())
        f.write(gt_ids.tobytes(order="C"))
        f.write(gt_dists.tobytes(order="C"))


def _read_ids_bin(path: Path) -> NDArray:
    np = require_numpy()
    header = read_bin_header(path)
    nq, k = header.npts, header.dim
    with path.open("rb") as f:
        f.seek(8)
        arr = np.fromfile(f, dtype=np.uint32, count=nq * k)
        if arr.size != nq * k:
            raise IOError(f"Short read: {path}")
    return arr.reshape((nq, k))


def _recall_at_k(result_ids: NDArray, gt_ids: NDArray, k: int) -> float:
    if result_ids.shape[0] != gt_ids.shape[0]:
        raise ValueError(
            f"Mismatch: result_ids.shape[0]={result_ids.shape[0]} vs gt_ids.shape[0]={gt_ids.shape[0]}"
        )
    if result_ids.shape[1] < k:
        raise ValueError(f"result_ids has only {result_ids.shape[1]} columns, need k={k}")
    if gt_ids.shape[1] < k:
        raise ValueError(f"gt_ids has only {gt_ids.shape[1]} columns, need k={k}")

    found = 0
    for i in range(result_ids.shape[0]):
        found += len(set(map(int, result_ids[i, :k])).intersection(map(int, gt_ids[i, :k])))
    return float(found) / float(result_ids.shape[0] * k)


def _parse_search_table(stdout: str) -> List[Dict[str, float]]:
    """Parse the per-L table printed by DiskANN search_disk_index.

    Expected columns (with gt):
      L, Beamwidth, QPS, Mean Latency, 99.9 Latency, Mean IOs, Mean IO (us), CPU (s), Recall@K
    """
    rows: List[Dict[str, float]] = []
    for line in stdout.splitlines():
        toks = line.strip().split()
        if len(toks) < 5:
            continue
        if not (toks[0].isdigit() and toks[1].isdigit()):
            continue
        try:
            row: Dict[str, float] = {
                "L": float(toks[0]),
                "beamwidth": float(toks[1]),
                "qps": float(toks[2]),
                "mean_us": float(toks[3]),
                "p999_us": float(toks[4]),
            }
            if len(toks) >= 8:
                row["mean_ios"] = float(toks[5])
                row["mean_io_us"] = float(toks[6])
                row["cpu_s"] = float(toks[7])
            if len(toks) >= 9:
                row["diskann_recall"] = float(toks[8])
            rows.append(row)
        except ValueError:
            continue
    return rows


def _auto_find_search_bin(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        _must_exist(p, "search_disk_index binary")
        return p

    # Try sibling DiskANN builds commonly used in this workspace.
    candidates = [
        _REPO_ROOT.parent / "DiskANN-epeshared" / "build" / "apps" / "search_disk_index",
        _REPO_ROOT.parent / "DiskANN-epeshared" / "build-avx512" / "apps" / "search_disk_index",
        _REPO_ROOT.parent / "DiskANN-epeshared" / "build-amx" / "apps" / "search_disk_index",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not auto-detect search_disk_index. Pass --search-bin /path/to/search_disk_index"
    )


def _ensure_query_bin_for_dtype(
    queries_f32: NDArray,
    *,
    dtype: str,
    out_dir: Path,
) -> Path:
    """Write query bin file for the given DiskANN dtype."""
    from bench_data import bin_utils

    dtype = dtype.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    f32_path = out_dir / "queries.f32.bin"
    _numpy_to_diskann_bin_f32(queries_f32, f32_path)

    if dtype in ("float", "f32"):
        return f32_path

    if dtype in ("bf16", "bfloat16"):
        bf16_path = out_dir / "queries.bf16.bin"
        bin_utils.convert_f32_bin_to_bf16_bin(f32_path, bf16_path)
        return bf16_path

    if dtype in ("int8", "i8"):
        i8_path = out_dir / "queries.int8.bin"
        bin_utils.convert_f32_bin_to_int8_bin(f32_path, i8_path)
        return i8_path

    raise ValueError(f"Unsupported data_type: {dtype}")


def _write_csv(rows: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_heatmap(
    *,
    Ls: List[int],
    Ws: List[int],
    grid: Dict[Tuple[int, int], float],
    out_path: Path,
    title: str,
) -> None:
    _try_import_matplotlib()
    import numpy as np
    import matplotlib.pyplot as plt

    Z = np.full((len(Ws), len(Ls)), np.nan, dtype=np.float32)
    for yi, W in enumerate(Ws):
        for xi, L in enumerate(Ls):
            v = grid.get((L, W))
            if v is not None:
                Z[yi, xi] = float(v)

    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(Ls)), max(4, 0.5 * len(Ws))))
    im = ax.imshow(Z, aspect="auto", origin="lower")
    ax.set_xticks(range(len(Ls)))
    ax.set_xticklabels([str(x) for x in Ls], rotation=45, ha="right")
    ax.set_yticks(range(len(Ws)))
    ax.set_yticklabels([str(y) for y in Ws])
    ax.set_xlabel("L (search_list)")
    ax.set_ylabel("W (beamwidth)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recall@K")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_lines(
    *,
    Ls: List[int],
    by_W: Dict[int, List[Tuple[int, float]]],
    out_path: Path,
    title: str,
) -> None:
    _try_import_matplotlib()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for W, pts in sorted(by_W.items()):
        pts_sorted = sorted(pts, key=lambda x: x[0])
        xs = [p[0] for p in pts_sorted]
        ys = [p[1] for p in pts_sorted]
        ax.plot(xs, ys, marker="o", label=f"W={W}")

    ax.set_xlabel("L (search_list)")
    ax.set_ylabel("Recall@K")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recall correctness evaluation for DiskANN disk index using an HDF5 dataset. "
            "Runs search_disk_index with parameter sweeps, computes Recall@K, and saves a plot + CSV."
        )
    )

    ap.add_argument(
        "--dataset",
        required=True,
        help="Path to dbpedia-openai-1000k-angular.hdf5 (or similar).",
    )
    ap.add_argument("--queries-key", default=None, help="HDF5 key for queries (default: auto)")
    ap.add_argument("--gt-ids-key", default=None, help="HDF5 key for ground-truth neighbor ids (default: auto)")
    ap.add_argument("--gt-dists-key", default=None, help="HDF5 key for ground-truth distances (default: auto)")

    ap.add_argument(
        "--index-prefix",
        required=True,
        help=(
            "DiskANN disk index prefix (without _disk.index). Example: /path/to/index_run0_disk"
        ),
    )

    ap.add_argument(
        "--search-bin",
        default=None,
        help="Optional path to DiskANN search_disk_index binary (auto-detect if omitted).",
    )

    ap.add_argument("--dist", default="cosine", choices=["l2", "mips", "cosine"], help="Distance function")
    ap.add_argument(
        "--data-type",
        default="bf16",
        choices=["float", "bf16", "int8"],
        help="Index/query dtype for search_disk_index",
    )

    ap.add_argument("--K", type=int, default=10, help="Recall@K")
    ap.add_argument("--Ls", type=int, nargs="+", default=[50, 100, 150, 200], help="search_list L values")
    ap.add_argument(
        "--Ws",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="beamwidth W values (each W is a separate run)",
    )

    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)
    ap.add_argument(
        "--search-io-limit",
        type=int,
        default=None,
        help="Optional search_io_limit (max IOs). If omitted, DiskANN default (uint32::max).",
    )
    ap.add_argument(
        "--use-reorder-data",
        action="store_true",
        help="Pass --use_reorder_data to search_disk_index",
    )

    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <this_dir>/runs/<timestamp>",
    )

    args = ap.parse_args()

    np = require_numpy()

    dataset_path = Path(args.dataset).expanduser().resolve()
    index_prefix = Path(args.index_prefix).expanduser().resolve()
    search_bin = _auto_find_search_bin(args.search_bin)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (_THIS_DIR / "runs" / time.strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Dataset: {dataset_path}")
    _log(f"Index prefix: {index_prefix}")
    _log(f"search_disk_index: {search_bin}")
    _log(f"Output: {out_dir}")

    queries, gt_ids, gt_dists = _load_hdf5_arrays(
        dataset_path,
        queries_key=args.queries_key,
        gt_ids_key=args.gt_ids_key,
        gt_dists_key=args.gt_dists_key,
    )

    if gt_ids.shape[1] < int(args.K):
        raise ValueError(f"Ground truth has only {gt_ids.shape[1]} neighbors per query; need K={args.K}")

    # Prepare query bin in requested dtype and gt bin for DiskANN.
    bin_dir = out_dir / "bins"
    query_bin = _ensure_query_bin_for_dtype(queries, dtype=args.data_type, out_dir=bin_dir)
    gt_bin = bin_dir / f"gt.uint32.f32.K{gt_ids.shape[1]}.bin"
    _write_gt_bin(gt_ids, gt_dists, gt_bin)

    # Sweep.
    Ls = [int(x) for x in args.Ls]
    Ws = [int(x) for x in args.Ws]

    results: List[Dict[str, object]] = []
    recall_grid: Dict[Tuple[int, int], float] = {}
    recall_by_W: Dict[int, List[Tuple[int, float]]] = {}

    for W in Ws:
        _log(f"Running search sweep: W={W}, Ls={Ls}")
        run_dir = out_dir / f"W{W}"
        run_dir.mkdir(parents=True, exist_ok=True)

        result_prefix = run_dir / "result"
        cmd = [
            str(search_bin),
            "--data_type",
            args.data_type,
            "--dist_fn",
            args.dist,
            "--index_path_prefix",
            str(index_prefix),
            "--result_path",
            str(result_prefix),
            "--query_file",
            str(query_bin),
            "--gt_file",
            str(gt_bin),
            "--recall_at",
            str(int(args.K)),
            "--search_list",
            *[str(x) for x in Ls],
            "--beamwidth",
            str(int(W)),
            "--num_threads",
            str(int(args.threads)),
            "--num_nodes_to_cache",
            str(int(args.num_nodes_to_cache)),
        ]

        if args.search_io_limit is not None:
            cmd += ["--search_io_limit", str(int(args.search_io_limit))]

        if args.use_reorder_data:
            cmd += ["--use_reorder_data"]

        log_path = run_dir / "search_stdout.txt"
        _log(f"Cmd: {' '.join(cmd)}")

        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
        log_path.write_text(proc.stdout or "", encoding="utf-8")
        if proc.returncode != 0:
            raise RuntimeError(f"search_disk_index failed (W={W}) rc={proc.returncode}. See {log_path}")

        table_rows = _parse_search_table(proc.stdout or "")
        table_by_L = {int(r["L"]): r for r in table_rows if "L" in r}

        # Compute recall ourselves from saved result ids.
        recall_by_W.setdefault(W, [])
        for L in Ls:
            ids_path = Path(str(result_prefix) + f"_{L}_idx_uint32.bin")
            _must_exist(ids_path, f"result ids for L={L}")
            res_ids = _read_ids_bin(ids_path)
            recall = _recall_at_k(res_ids, gt_ids, int(args.K))

            recall_grid[(L, W)] = recall
            recall_by_W[W].append((L, recall))

            row: Dict[str, object] = {
                "L": int(L),
                "W": int(W),
                "recall_at": int(args.K),
                "recall": float(recall),
                "data_type": args.data_type,
                "dist": args.dist,
                "threads": int(args.threads),
                "num_nodes_to_cache": int(args.num_nodes_to_cache),
                "search_io_limit": int(args.search_io_limit) if args.search_io_limit is not None else "default",
            }

            tr = table_by_L.get(int(L))
            if tr:
                for k in ("qps", "mean_us", "p999_us", "mean_ios", "mean_io_us", "cpu_s", "diskann_recall"):
                    if k in tr:
                        row[k] = float(tr[k])

            results.append(row)
            _log(f"W={W} L={L} recall@{args.K}={recall:.6f}")

    csv_path = out_dir / "summary.csv"
    _write_csv(results, csv_path)

    title = f"Recall@{args.K} sweep (dtype={args.data_type}, dist={args.dist})"
    if len(Ws) > 1 and len(Ls) > 1:
        img_path = out_dir / "recall_heatmap.png"
        _plot_heatmap(Ls=Ls, Ws=Ws, grid=recall_grid, out_path=img_path, title=title)
    else:
        img_path = out_dir / "recall_curve.png"
        _plot_lines(Ls=Ls, by_W=recall_by_W, out_path=img_path, title=title)

    _log(f"Wrote: {csv_path}")
    _log(f"Wrote: {img_path}")
    _log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
