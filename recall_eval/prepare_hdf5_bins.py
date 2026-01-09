#!/usr/bin/env python3

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench_data.bin_utils import require_numpy  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _try_import_h5py():
    try:
        import h5py  # noqa: F401

        return h5py
    except Exception as e:
        raise RuntimeError("This script requires h5py. Install with: pip install h5py") from e


def _write_diskann_bin_f32(array, out_path: Path) -> None:
    np = require_numpy()
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={array.shape}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    npts = np.uint32(array.shape[0])
    dim = np.uint32(array.shape[1])
    with out_path.open("wb") as f:
        f.write(npts.tobytes())
        f.write(dim.tobytes())
        f.write(array.tobytes(order="C"))


def _write_gt_bin(gt_ids, gt_dists, out_path: Path) -> None:
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


def _convert_dtype_bins(*, train_f32: Path, test_f32: Path, out_dir: Path) -> None:
    from bench_data import bin_utils

    # bf16
    bin_utils.convert_f32_bin_to_bf16_bin(train_f32, out_dir / "train.bf16.bin")
    bin_utils.convert_f32_bin_to_bf16_bin(test_f32, out_dir / "test.bf16.bin")

    # int8
    # For int8, we compute scale independently for train/test (simple and robust).
    # If you need a shared scale, add a flag and reuse the train scale.
    bin_utils.convert_f32_bin_to_int8_bin(train_f32, out_dir / "train.int8.bin")
    bin_utils.convert_f32_bin_to_int8_bin(test_f32, out_dir / "test.int8.bin")


def _read_dataset(
    dataset_path: Path,
    *,
    train_key: str,
    test_key: str,
    neighbors_key: str,
    distances_key: Optional[str],
):
    np = require_numpy()
    h5py = _try_import_h5py()

    with h5py.File(dataset_path, "r") as f:
        train = np.asarray(f[train_key], dtype=np.float32)
        test = np.asarray(f[test_key], dtype=np.float32)
        neighbors = np.asarray(f[neighbors_key], dtype=np.uint32)
        dists = None
        if distances_key and distances_key in f:
            try:
                dists = np.asarray(f[distances_key], dtype=np.float32)
            except Exception:
                dists = None

    return train, test, neighbors, dists


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare DiskANN .bin files (train/test/gt) from an ann-benchmarks style HDF5 dataset. "
            "Default keys match dbpedia-openai-1000k-angular.hdf5 (train/test/neighbors/distances)."
        )
    )
    ap.add_argument("--dataset", required=True, help="Path to .hdf5 dataset")
    ap.add_argument("--out-dir", required=True, help="Output directory for .bin files")

    ap.add_argument("--train-key", default="train")
    ap.add_argument("--test-key", default="test")
    ap.add_argument("--neighbors-key", default="neighbors")
    ap.add_argument("--distances-key", default="distances")

    ap.add_argument(
        "--no-convert",
        action="store_true",
        help="Only write float32 bins; do not generate bf16/int8 bins",
    )

    args = ap.parse_args()

    dataset_path = Path(args.dataset).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Reading dataset: {dataset_path}")
    train, test, neighbors, dists = _read_dataset(
        dataset_path,
        train_key=args.train_key,
        test_key=args.test_key,
        neighbors_key=args.neighbors_key,
        distances_key=args.distances_key,
    )

    _log(f"train: shape={train.shape} dtype={train.dtype}")
    _log(f"test:  shape={test.shape} dtype={test.dtype}")
    _log(f"gt:    shape={neighbors.shape} dtype={neighbors.dtype}")

    train_f32 = out_dir / "train.f32.bin"
    test_f32 = out_dir / "test.f32.bin"
    gt_bin = out_dir / f"gt.uint32.f32.K{neighbors.shape[1]}.bin"

    _log(f"Writing: {train_f32}")
    _write_diskann_bin_f32(train, train_f32)

    _log(f"Writing: {test_f32}")
    _write_diskann_bin_f32(test, test_f32)

    _log(f"Writing: {gt_bin}")
    _write_gt_bin(neighbors, dists, gt_bin)

    if not args.no_convert:
        _log("Converting to bf16/int8...")
        _convert_dtype_bins(train_f32=train_f32, test_f32=test_f32, out_dir=out_dir)

    meta = {
        "dataset": str(dataset_path),
        "train_key": args.train_key,
        "test_key": args.test_key,
        "neighbors_key": args.neighbors_key,
        "distances_key": args.distances_key,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "gt_shape": list(neighbors.shape),
        "files": {
            "train_f32": str(train_f32),
            "test_f32": str(test_f32),
            "gt": str(gt_bin),
            "train_bf16": str(out_dir / "train.bf16.bin"),
            "test_bf16": str(out_dir / "test.bf16.bin"),
            "train_int8": str(out_dir / "train.int8.bin"),
            "test_int8": str(out_dir / "test.int8.bin"),
        },
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    _log(f"Wrote: {out_dir / 'meta.json'}")
    _log("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
