#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Allow importing shared utilities from diskann-pg/bench_data when this script is
# invoked directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench_data.bin_utils import read_bin_header as _read_bin_header, require_numpy as _require_numpy  # noqa: E402


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _find_diskann_apps_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"diskann apps dir not found: {p}")
        return p
    raise FileNotFoundError(
        "Could not auto-detect DiskANN apps directory. "
        "Pass --diskann-apps /path/to/DiskANN/build/apps"
    )


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, float, str]:
    start = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=True,
    )
    try:
        out, _ = proc.communicate()
    except KeyboardInterrupt:
        _log("Interrupted (Ctrl+C). Killing child process group...")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        raise

    elapsed = time.perf_counter() - start
    return int(proc.returncode or 0), elapsed, out or ""


def _parse_search_table(stdout: str) -> List[Dict[str, float]]:
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
            if len(toks) >= 9:
                row["recall"] = float(toks[-1])
            rows.append(row)
        except ValueError:
            continue
    return rows


def _read_ids_bin_np(path: Path):
    np = _require_numpy()
    header = _read_bin_header(path)
    nq, k = header.npts, header.dim
    with path.open("rb") as f:
        f.seek(8)
        arr = np.fromfile(f, dtype=np.uint32, count=nq * k)
        if arr.size != nq * k:
            raise IOError(f"Short read: {path}")
    return nq, k, arr.reshape((nq, k))


def _topk_overlap(a: List[int], b: List[int], k: int) -> float:
    return len(set(a[:k]).intersection(set(b[:k]))) / float(k)


def _positional_match(a: List[int], b: List[int], k: int) -> float:
    return sum(1 for i in range(k) if a[i] == b[i]) / float(k)


def _must_exist(p: Path, what: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")


def _expected_elem_bytes(data_type: str) -> int:
    dt = data_type.lower()
    if dt in ("float", "f32"):
        return 4
    if dt in ("bf16", "bfloat16"):
        return 2
    if dt in ("int8", "i8", "uint8", "u8"):
        return 1
    raise ValueError(f"Unsupported dtype: {data_type}")


def _infer_elem_bytes_from_bin(path: Path) -> Tuple[int, int, int]:
    """Return (npts, dim, elem_bytes) inferred from file size.

    DiskANN .bin layout: uint32 npts, uint32 dim, then contiguous payload.
    """
    file_size = path.stat().st_size
    if file_size < 8:
        raise ValueError(f"File too small to be a DiskANN .bin: {path} (size={file_size})")

    with path.open("rb") as f:
        header = f.read(8)
    npts, dim = struct.unpack("<II", header)
    payload = file_size - 8
    if npts == 0 or dim == 0:
        raise ValueError(f"Invalid .bin header in {path}: npts={npts}, dim={dim}")

    total_elems = int(npts) * int(dim)
    if total_elems <= 0:
        raise ValueError(f"Invalid element count inferred from header in {path}: npts={npts}, dim={dim}")

    if payload % total_elems != 0:
        raise ValueError(
            f"File size does not match header: {path} payload_bytes={payload} is not divisible by npts*dim={total_elems}"
        )
    elem_bytes = payload // total_elems
    return int(npts), int(dim), int(elem_bytes)


def _disk_index_has_reorder_data(index_prefix: Path) -> bool:
    """Returns True if <prefix>_disk.index metadata indicates reorder data exists."""
    index_path = Path(str(index_prefix) + "_disk.index")
    if not index_path.exists():
        return False

    with index_path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            return False
        nr, nc = struct.unpack("<II", header)
        if nr < 8 or nc != 1:
            return False
        meta = f.read(int(nr) * 8)
        if len(meta) != int(nr) * 8:
            return False
        vals = struct.unpack("<" + "Q" * int(nr), meta)

    reorder_exists = vals[7]
    return bool(reorder_exists)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Search-only benchmark for DiskANN disk index. "
            "Reads prebuilt indexes from --run-root and runs search_disk_index for selected dtypes."
        )
    )
    ap.add_argument(
        "--diskann-apps",
        required=True,
        help="Path to DiskANN build apps dir (contains search_disk_index).",
    )
    ap.add_argument(
        "--index-root",
        default=None,
        help=(
            "Directory containing prebuilt indexes. Expected layout: "
            "<index-root>/<dtype>/index/*_disk.index"
        ),
    )

    # Legacy name: --run-root (kept for compatibility)
    ap.add_argument(
        "--run-root",
        dest="index_root",
        default=None,
        help=argparse.SUPPRESS,
    )

    ap.add_argument(
        "--query-file",
        required=True,
        help=(
            "Query .bin file path. "
            "If you pass multiple dtypes, you can use a template containing '{dtype}', "
            "e.g. /path/to/query_{dtype}.bin or /path/to/query.<dtype>.bin"
        ),
    )

    ap.add_argument(
        "--workdir",
        default=None,
        help="Output work dir for logs/results/summary. Default: <run-root>/search_out",
    )
    ap.add_argument("--tag", default=None, help="Optional run tag")

    ap.add_argument("--dist", default="l2", choices=["l2", "mips", "cosine"], help="Distance function")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--Ls", type=int, nargs="+", default=[100])
    ap.add_argument("--beamwidth", type=int, default=2)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)

    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=["bf16"],
        choices=["float", "bf16", "int8"],
        help="Which dtypes to search. Default: bf16",
    )

    ap.add_argument(
        "--gt-file",
        default=None,
        help=(
            "Optional groundtruth file to pass to search_disk_index (--gt_file). "
            "If omitted, will auto-use <run-root>/gt/gt_float_K{K}.bin if present."
        ),
    )

    ap.add_argument(
        "--consistency-L",
        type=int,
        default=None,
        help="Which L result to use for cross-dtype consistency; default=max(Ls)",
    )

    args = ap.parse_args()
    selected_dtypes = list(dict.fromkeys(args.dtypes))

    # mips + int8 not supported
    if args.dist == "mips" and "int8" in selected_dtypes:
        _log("Warning: skipping dtype=int8 for --dist mips (DiskANN disk index does not support int8+mips).")
        selected_dtypes = [d for d in selected_dtypes if d != "int8"]

    if not args.index_root:
        raise ValueError("--index-root is required")

    apps = _find_diskann_apps_dir(args.diskann_apps)
    index_root = Path(args.index_root).expanduser().resolve()
    _must_exist(index_root, "index_root")

    query_file_arg = str(args.query_file)
    if "{dtype}" not in query_file_arg and len(selected_dtypes) != 1:
        raise ValueError("When --query-file does not contain '{dtype}', you must specify exactly one dtype via --dtypes")

    def _find_latest_index_prefix(dtype: str) -> Path:
        # Support multiple layouts:
        # 1) <index-root>/<dtype>/index/*_disk.index
        # 2) <index-root>/<dtype>/*_disk.index
        # 3) <index-root>/*_disk.index
        search_dirs = [
            index_root / dtype / "index",
            index_root / dtype,
            index_root,
        ]
        candidates: List[Path] = []
        for d in search_dirs:
            if d.exists() and d.is_dir():
                candidates.extend(d.glob("*_disk.index"))

        if not candidates:
            tried = ", ".join(str(d) for d in search_dirs)
            raise FileNotFoundError(
                f"No *_disk.index found for dtype={dtype} under any of: {tried}. "
                "Hint: point --index-root to the directory that contains the *_disk.index (or <dtype>/index/*_disk.index)."
            )

        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
        disk_index = candidates[0]
        prefix = Path(str(disk_index)[: -len("_disk.index")])
        return prefix

    def _resolve_query_file(dtype: str) -> Path:
        s = query_file_arg.replace("{dtype}", dtype)
        p = Path(s).expanduser().resolve()
        _must_exist(p, f"query_file for dtype={dtype}")
        return p

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "pq_search"
    script_dir = Path(__file__).resolve().parent
    base_workdir = Path(args.workdir).expanduser().resolve() if args.workdir else (script_dir / "runs")
    out_root = base_workdir / f"{run_id}_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Index root: {index_root}")
    _log(f"Query file: {args.query_file}")
    _log(f"Output dir: {out_root}")
    _log(f"DiskANN apps: {apps}")

    gt_file: Optional[Path] = None
    if args.gt_file:
        gt_file = Path(args.gt_file).expanduser().resolve()
        _must_exist(gt_file, "gt_file")
    else:
        auto_gt = index_root / "gt" / f"gt_float_K{args.K}.bin"
        if auto_gt.exists():
            gt_file = auto_gt

    # Search
    results: Dict[str, Dict[str, object]] = {}
    for dtype in selected_dtypes:
        index_prefix = _find_latest_index_prefix(dtype)
        query_file = _resolve_query_file(dtype)

        # Validate query file dtype by inferring element-size.
        try:
            npts, dim, inferred_elem_bytes = _infer_elem_bytes_from_bin(query_file)
            expected_bytes = _expected_elem_bytes(dtype)
            if inferred_elem_bytes != expected_bytes:
                raise ValueError(
                    "Query file dtype mismatch:\n"
                    f"  dtype: {dtype} (expected elem_bytes={expected_bytes})\n"
                    f"  query_file: {query_file}\n"
                    f"  header: npts={npts} dim={dim}\n"
                    f"  inferred elem_bytes from file size: {inferred_elem_bytes}"
                )
        except Exception as e:
            raise RuntimeError(f"Failed validating query file '{query_file}' for dtype={dtype}: {e}")

        dtype_dir = out_root / dtype
        dtype_dir.mkdir(parents=True, exist_ok=True)
        result_prefix = dtype_dir / "results" / "res"
        result_prefix.parent.mkdir(parents=True, exist_ok=True)

        want_reorder = dtype in ("float", "bf16") and _disk_index_has_reorder_data(index_prefix)
        if dtype in ("float", "bf16") and not want_reorder:
            _log(f"Note: index has no reorder data; running PQ-only (dtype={dtype}).")
        cmd = [
            str(apps / "search_disk_index"),
            "--data_type",
            dtype,
            "--dist_fn",
            args.dist,
            "--index_path_prefix",
            str(index_prefix),
            "--query_file",
            str(query_file),
            "--result_path",
            str(result_prefix),
            "-K",
            str(args.K),
            "-L",
            *[str(x) for x in args.Ls],
            "-W",
            str(args.beamwidth),
            "--num_nodes_to_cache",
            str(args.num_nodes_to_cache),
            "-T",
            str(args.threads),
        ]
        if gt_file is not None:
            cmd.extend(["--gt_file", str(gt_file)])
        if want_reorder:
            cmd.append("--use_reorder_data")

        _log(f"=== dtype={dtype} (reorder={'on' if want_reorder else 'off'}) ===")
        _log("Search cmd: " + " ".join(cmd))
        rc, elapsed, out = _run(cmd)
        (logs_dir / f"search_{dtype}.log").write_text(out)
        _log(f"Search {dtype}: rc={rc}, time_s={elapsed:.3f}, log={logs_dir / f'search_{dtype}.log'}")
        if rc != 0:
            raise RuntimeError(f"search_disk_index failed for {dtype} (rc={rc}). See logs.")

        search_rows = _parse_search_table(out)
        if search_rows:
            _log(f"Parsed {len(search_rows)} search row(s) for {dtype}: Ls={[int(r['L']) for r in search_rows]}")
        else:
            _log(f"Warning: failed to parse any search table rows for {dtype}; see log")

        results[dtype] = {
            "dtype": dtype,
            "query": str(query_file),
            "index_prefix": str(index_prefix),
            "reorder": bool(want_reorder),
            "search_time_s": float(elapsed),
            "search_rows": search_rows,
        }

    # Consistency (cross-dtype)
    consistency = None
    if "float" in selected_dtypes and any(d in selected_dtypes for d in ("bf16", "int8")):
        consistency_L = args.consistency_L if args.consistency_L is not None else max(args.Ls)
        if consistency_L not in args.Ls:
            raise ValueError(f"--consistency-L {consistency_L} must be one of --Ls {args.Ls}")

        def ids_path(dtype: str) -> Path:
            dtype_dir = out_root / dtype
            return dtype_dir / "results" / f"res_{consistency_L}_idx_uint32.bin"

        consistency = {"baseline": "float", "L": consistency_L, "K": args.K, "by_dtype": {}}

        base_ids_file = ids_path("float")
        if base_ids_file.exists() and any(ids_path(d).exists() for d in ("bf16", "int8") if d in selected_dtypes):
            _log(f"Computing consistency vs float using L={consistency_L}")
            nq0, k0, ids0 = _read_ids_bin_np(base_ids_file)
            if k0 < args.K:
                raise ValueError(f"Result K ({k0}) is smaller than requested K ({args.K})")

            for dtype in [d for d in ("bf16", "int8") if d in selected_dtypes]:
                p = ids_path(dtype)
                if not p.exists():
                    continue
                nq, k, ids = _read_ids_bin_np(p)
                if nq != nq0 or k != k0:
                    raise ValueError(
                        f"Result shape mismatch for {dtype}: got (nq={nq},k={k}) vs float (nq={nq0},k={k0})"
                    )

                overlaps: List[float] = []
                pos_matches: List[float] = []
                for i in range(nq0):
                    a = ids0[i, : args.K].tolist()
                    b = ids[i, : args.K].tolist()
                    overlaps.append(_topk_overlap(a, b, args.K))
                    pos_matches.append(_positional_match(a, b, args.K))

                consistency["by_dtype"][dtype] = {
                    "topk_overlap_mean": sum(overlaps) / len(overlaps) if overlaps else None,
                    "positional_match_mean": sum(pos_matches) / len(pos_matches) if pos_matches else None,
                    "result_ids": str(p),
                }
                _log(
                    f"Consistency {dtype}: topk_overlap_mean={consistency['by_dtype'][dtype]['topk_overlap_mean']:.6g}, "
                    f"positional_match_mean={consistency['by_dtype'][dtype]['positional_match_mean']:.6g}"
                )
        else:
            _log("Skipping consistency: missing result id files")
    else:
        _log("Skipping consistency (need float + at least one of bf16/int8)")

    summary = {
        "meta": {
            "run_id": run_id,
            "tag": tag,
            "diskann_apps": str(apps),
            "index_root": str(index_root),
            "query_file": str(args.query_file),
            "dist": args.dist,
            "K": args.K,
            "Ls": args.Ls,
            "beamwidth": args.beamwidth,
            "num_nodes_to_cache": args.num_nodes_to_cache,
            "threads": args.threads,
            "gt_file": str(gt_file) if gt_file else None,
        },
        "results": results,
        "consistency": consistency,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _log(f"Wrote summary: {out_root / 'summary.json'}")
    print(str(out_root / "summary.json"))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _log("Interrupted by user; exiting.")
        sys.exit(130)
