#!/usr/bin/env python3

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Allow importing shared utilities from diskann-pg/bench_data when this script is
# invoked directly.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench_data.bin_utils import (  # noqa: E402
    BinHeader,
    convert_f32_bin_to_bf16_bin,
    convert_f32_bin_to_int8_bin,
    read_bin_header as _read_bin_header,
    require_numpy as _require_numpy,
)


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark PQ+reorder for float/bf16/int8 with consistency checks")
    ap.add_argument("--diskann-apps", default=None, help="Path to DiskANN build apps dir (contains build_disk_index)")

    ap.add_argument("--workdir", default=str(Path(__file__).resolve().parent / "out"), help="Output work dir")
    ap.add_argument("--tag", default=None, help="Optional run tag")

    ap.add_argument("--dist", default="l2", choices=["l2", "mips", "cosine"], help="Distance function")

    # Inputs: f32 optional now (方案B)
    ap.add_argument("--base-f32", default=None, help="Base vectors float32 .bin")
    ap.add_argument("--query-f32", default=None, help="Query vectors float32 .bin")
    ap.add_argument("--base-bf16", default=None, help="Base vectors bf16 .bin (DiskANN bf16: uint16 payload)")
    ap.add_argument("--query-bf16", default=None, help="Query vectors bf16 .bin (DiskANN bf16: uint16 payload)")
    ap.add_argument("--base-int8", default=None, help="Base vectors int8 .bin (DiskANN int8 payload)")
    ap.add_argument("--query-int8", default=None, help="Query vectors int8 .bin (DiskANN int8 payload)")

    ap.add_argument("--gt", default=None, help="Groundtruth file path (optional). If omitted, will compute per dtype")
    ap.add_argument(
        "--gt-shared",
        default="none",
        choices=["none", "float"],
        help=(
            "Which GT file to use when computing Recall@K for non-float dtypes. "
            "'none' computes/uses per-dtype GT (default). "
            "'float' evaluates bf16/int8 recall against the float32 GT file."
        ),
    )

    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--Lbuild", type=int, default=100)
    ap.add_argument("--B", type=float, default=0.25, help="search_DRAM_budget GB")
    ap.add_argument("--M", type=float, default=4.0, help="build_DRAM_budget GB")
    ap.add_argument("--PQ-disk-bytes", type=int, default=32, help="PQ bytes on SSD; must be >0 for reorder")
    ap.add_argument(
        "--QD",
        type=int,
        default=0,
        help=(
            "Override DiskANN 'quantized dimension' used for in-memory compression during build/search. "
            "If 0, DiskANN derives it from -B/--search_DRAM_budget, which can become very large for small datasets."
        ),
    )
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--Ls", type=int, nargs="+", default=[10, 20, 30, 40, 50, 100])
    ap.add_argument("--beamwidth", type=int, default=2)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)

    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--skip-search", action="store_true")

    ap.add_argument(
        "--dtypes",
        nargs="+",
        default=["float", "bf16", "int8"],
        choices=["float", "bf16", "int8"],
        help="Which dtypes to run. Default: float bf16 int8. Use e.g. --dtypes bf16 for quick-check.",
    )

    ap.add_argument("--int8-scale", type=float, default=None, help="Quantization scale for int8 (optional)")
    ap.add_argument("--chunk-elems", type=int, default=1 << 20, help="Conversion chunk size in elements")

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

    _log("Starting pq_reorder_bench (SSD/disk index only)")
    _log(
        "Args: "
        + " ".join(
            [
                f"dist={args.dist}",
                f"K={args.K}",
                f"Ls={args.Ls}",
                f"dtypes={selected_dtypes}",
                f"R={args.R}",
                f"Lbuild={args.Lbuild}",
                f"PQ_disk_bytes={args.PQ_disk_bytes}",
                f"QD={args.QD}",
                f"threads={args.threads}",
                f"B={args.B}",
                f"M={args.M}",
                f"skip_build={args.skip_build}",
                f"skip_search={args.skip_search}",
                f"skip_gt={args.skip_gt}",
            ]
        )
    )

    apps = _find_diskann_apps_dir(args.diskann_apps)
    _log(f"DiskANN apps: {apps}")

    # Resolve provided inputs
    base_f32 = Path(args.base_f32).expanduser().resolve() if args.base_f32 else None
    query_f32 = Path(args.query_f32).expanduser().resolve() if args.query_f32 else None
    base_bf16 = Path(args.base_bf16).expanduser().resolve() if args.base_bf16 else None
    query_bf16 = Path(args.query_bf16).expanduser().resolve() if args.query_bf16 else None
    base_int8 = Path(args.base_int8).expanduser().resolve() if args.base_int8 else None
    query_int8 = Path(args.query_int8).expanduser().resolve() if args.query_int8 else None

    # Validate availability by dtype
    def _need_pair(dtype: str) -> None:
        if dtype == "float":
            if base_f32 is None or query_f32 is None:
                raise ValueError("dtype=float requested but --base-f32/--query-f32 not provided")
            _must_exist(base_f32, "base-f32")
            _must_exist(query_f32, "query-f32")
        elif dtype == "bf16":
            # Either bf16 provided OR f32 provided to convert
            if base_bf16 is not None and query_bf16 is not None:
                _must_exist(base_bf16, "base-bf16")
                _must_exist(query_bf16, "query-bf16")
            elif base_f32 is not None and query_f32 is not None:
                _must_exist(base_f32, "base-f32")
                _must_exist(query_f32, "query-f32")
            else:
                raise ValueError("dtype=bf16 requested but neither (--base-bf16/--query-bf16) nor (--base-f32/--query-f32) provided")
        elif dtype == "int8":
            if base_int8 is not None and query_int8 is not None:
                _must_exist(base_int8, "base-int8")
                _must_exist(query_int8, "query-int8")
            elif base_f32 is not None and query_f32 is not None:
                _must_exist(base_f32, "base-f32")
                _must_exist(query_f32, "query-f32")
            else:
                raise ValueError("dtype=int8 requested but neither (--base-int8/--query-int8) nor (--base-f32/--query-f32) provided")

    for d in selected_dtypes:
        _need_pair(d)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "pq_reorder"
    out_root = Path(args.workdir).expanduser().resolve() / f"{run_id}_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Output dir: {out_root}")
    _log(f"Logs dir: {logs_dir}")

    # Print input headers where available
    if base_f32 and query_f32:
        base_hdr = _read_bin_header(base_f32)
        query_hdr = _read_bin_header(query_f32)
        _log(f"Input base_f32: {base_f32} (npts={base_hdr.npts}, dim={base_hdr.dim})")
        _log(f"Input query_f32: {query_f32} (npts={query_hdr.npts}, dim={query_hdr.dim})")
    if base_bf16 and query_bf16:
        base_hdr = _read_bin_header(base_bf16)
        query_hdr = _read_bin_header(query_bf16)
        _log(f"Input base_bf16: {base_bf16} (npts={base_hdr.npts}, dim={base_hdr.dim})")
        _log(f"Input query_bf16: {query_bf16} (npts={query_hdr.npts}, dim={query_hdr.dim})")
    if base_int8 and query_int8:
        base_hdr = _read_bin_header(base_int8)
        query_hdr = _read_bin_header(query_int8)
        _log(f"Input base_int8: {base_int8} (npts={base_hdr.npts}, dim={base_hdr.dim})")
        _log(f"Input query_int8: {query_int8} (npts={query_hdr.npts}, dim={query_hdr.dim})")

    # Prepare per-dtype data file mapping
    # - If user provides dtype-specific files, use them directly.
    # - Else, derive from f32 via conversion into out_root/data/.
    data: Dict[str, Dict[str, Path]] = {}

    if "float" in selected_dtypes:
        assert base_f32 is not None and query_f32 is not None
        data["float"] = {"base": base_f32, "query": query_f32}

    if "bf16" in selected_dtypes:
        if base_bf16 is not None and query_bf16 is not None:
            data["bf16"] = {"base": base_bf16, "query": query_bf16}
        else:
            # convert from f32 into out_root
            data["bf16"] = {
                "base": out_root / "data" / "base.bf16.bin",
                "query": out_root / "data" / "query.bf16.bin",
            }

    if "int8" in selected_dtypes:
        if base_int8 is not None and query_int8 is not None:
            data["int8"] = {"base": base_int8, "query": query_int8}
        else:
            data["int8"] = {
                "base": out_root / "data" / "base.int8.bin",
                "query": out_root / "data" / "query.int8.bin",
            }

    # Convert only when needed (bf16/int8 from f32)
    convert_meta: Dict[str, Dict[str, float]] = {}

    if "bf16" in selected_dtypes:
        if (base_bf16 is None or query_bf16 is None) and (base_f32 is not None and query_f32 is not None):
            if (not data["bf16"]["base"].exists()) or (not data["bf16"]["query"].exists()):
                _log("Converting float32 -> bf16 for base/query")
                convert_f32_bin_to_bf16_bin(base_f32, data["bf16"]["base"], chunk_elems=args.chunk_elems)
                convert_f32_bin_to_bf16_bin(query_f32, data["bf16"]["query"], chunk_elems=args.chunk_elems)
                _log(f"bf16 base: {data['bf16']['base']}")
                _log(f"bf16 query: {data['bf16']['query']}")
            else:
                _log("bf16 base/query already exist; skipping bf16 conversion")
        else:
            _log("bf16 inputs provided; skipping bf16 conversion")

    int8_scale = args.int8_scale
    if "int8" in selected_dtypes:
        if (base_int8 is None or query_int8 is None) and (base_f32 is not None and query_f32 is not None):
            if (not data["int8"]["base"].exists()) or (not data["int8"]["query"].exists()):
                _log("Converting float32 -> int8 for base/query")
                if int8_scale is None:
                    base_max = _compute_max_abs_f32_bin(base_f32, chunk_elems=args.chunk_elems)
                    query_max = _compute_max_abs_f32_bin(query_f32, chunk_elems=args.chunk_elems)
                    max_abs = max(base_max, query_max)
                    int8_scale = (max_abs / 127.0) if max_abs > 0 else 1.0
                    _log(f"Auto int8 scale: max_abs={max_abs:.6g} => scale={float(int8_scale):.6g}")
                else:
                    _log(f"User int8 scale: scale={float(int8_scale):.6g}")

                used_scale_base = convert_f32_bin_to_int8_bin(
                    base_f32, data["int8"]["base"], scale=float(int8_scale), chunk_elems=args.chunk_elems
                )
                used_scale_query = convert_f32_bin_to_int8_bin(
                    query_f32, data["int8"]["query"], scale=float(int8_scale), chunk_elems=args.chunk_elems
                )
                convert_meta["int8"] = {
                    "scale": float(int8_scale),
                    "used_scale_base": used_scale_base,
                    "used_scale_query": used_scale_query,
                }
                _log(f"int8 base: {data['int8']['base']}")
                _log(f"int8 query: {data['int8']['query']}")
            else:
                _log("int8 base/query already exist; skipping int8 conversion")
        else:
            _log("int8 inputs provided; skipping int8 conversion")

    # GT
    gt_paths: Dict[str, Optional[Path]] = {"float": None, "bf16": None, "int8": None}
    if args.gt:
        _log(f"Using provided groundtruth path for all dtypes: {args.gt}")
        p = Path(args.gt).expanduser().resolve()
        _must_exist(p, "gt")
        gt_paths = {"float": p, "bf16": p, "int8": p}
    elif not args.skip_gt:
        if args.gt_shared == "float":
            if "float" not in selected_dtypes:
                raise ValueError("--gt-shared=float requires dtype=float to be selected (or provide --gt)")
            _log("Computing groundtruth for float and reusing it for other dtypes (--gt-shared=float)")
            gt_out = out_root / "gt" / f"gt_float_K{args.K}.bin"
            gt_out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(apps / "utils" / "compute_groundtruth"),
                "--data_type",
                "float",
                "--dist_fn",
                args.dist,
                "--base_file",
                str(data["float"]["base"]),
                "--query_file",
                str(data["float"]["query"]),
                "--gt_file",
                str(gt_out),
                "--K",
                str(args.K),
            ]
            rc, elapsed, out = _run(cmd)
            (logs_dir / "compute_gt_float.log").write_text(out)
            _log(f"GT float: rc={rc}, time_s={elapsed:.3f}, log={logs_dir / 'compute_gt_float.log'}")
            if rc != 0:
                raise RuntimeError(f"compute_groundtruth failed for float (rc={rc}). See logs.")
            for dtype in gt_paths.keys():
                gt_paths[dtype] = gt_out
        else:
            _log("Computing groundtruth per dtype")
            for dtype in selected_dtypes:
                gt_out = out_root / "gt" / f"gt_{dtype}_K{args.K}.bin"
                gt_out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(apps / "utils" / "compute_groundtruth"),
                    "--data_type",
                    dtype,
                    "--dist_fn",
                    args.dist,
                    "--base_file",
                    str(data[dtype]["base"]),
                    "--query_file",
                    str(data[dtype]["query"]),
                    "--gt_file",
                    str(gt_out),
                    "--K",
                    str(args.K),
                ]
                rc, elapsed, out = _run(cmd)
                (logs_dir / f"compute_gt_{dtype}.log").write_text(out)
                _log(f"GT {dtype}: rc={rc}, time_s={elapsed:.3f}, log={logs_dir / f'compute_gt_{dtype}.log'}")
                if rc != 0:
                    raise RuntimeError(f"compute_groundtruth failed for {dtype} (rc={rc}). See logs.")
                gt_paths[dtype] = gt_out
    else:
        _log("Skipping groundtruth computation (--skip-gt)")

    # Build + search
    results: Dict[str, Dict[str, object]] = {}

    for dtype in selected_dtypes:
        dtype_dir = out_root / dtype
        dtype_dir.mkdir(exist_ok=True)

        index_prefix = dtype_dir / "index" / f"disk_index_{dtype}_R{args.R}_L{args.Lbuild}_PQ{args.PQ_disk_bytes}"
        index_prefix.parent.mkdir(parents=True, exist_ok=True)

        can_reorder = dtype in ("float", "bf16")
        want_reorder = can_reorder

        _log(f"=== dtype={dtype} (reorder={'on' if want_reorder else 'off'}) ===")

        # Build
        build_time_s = None
        if not args.skip_build:
            cmd = [
                str(apps / "build_disk_index"),
                "--data_type",
                dtype,
                "--dist_fn",
                args.dist,
                "--data_path",
                str(data[dtype]["base"]),
                "--index_path_prefix",
                str(index_prefix),
                "-R",
                str(args.R),
                "-L",
                str(args.Lbuild),
                "-B",
                str(args.B),
                "-M",
                str(args.M),
                "--PQ_disk_bytes",
                str(args.PQ_disk_bytes),
                "-T",
                str(args.threads),
            ]
            if args.QD and int(args.QD) > 0:
                cmd.extend(["--QD", str(int(args.QD))])
            if want_reorder:
                cmd.append("--append_reorder_data")

            _log("Build cmd: " + " ".join(cmd))
            rc, elapsed, out = _run(cmd)
            build_time_s = elapsed
            (logs_dir / f"build_{dtype}.log").write_text(out)
            _log(f"Build {dtype}: rc={rc}, time_s={elapsed:.3f}, log={logs_dir / f'build_{dtype}.log'}")
            if rc != 0:
                raise RuntimeError(f"build_disk_index failed for {dtype} (rc={rc}). See logs.")
        else:
            _log("Skipping build (--skip-build)")

        # Search
        search_rows: List[Dict[str, float]] = []
        search_time_s = None
        result_prefix = dtype_dir / "results" / "res"
        result_prefix.parent.mkdir(parents=True, exist_ok=True)

        if not args.skip_search:
            cmd = [
                str(apps / "search_disk_index"),
                "--data_type",
                dtype,
                "--dist_fn",
                args.dist,
                "--index_path_prefix",
                str(index_prefix),
                "--query_file",
                str(data[dtype]["query"]),
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
            if gt_paths[dtype] is not None:
                cmd.extend(["--gt_file", str(gt_paths[dtype])])
            if want_reorder:
                cmd.append("--use_reorder_data")

            _log("Search cmd: " + " ".join(cmd))
            rc, elapsed, out = _run(cmd)
            search_time_s = elapsed
            (logs_dir / f"search_{dtype}.log").write_text(out)
            _log(f"Search {dtype}: rc={rc}, time_s={elapsed:.3f}, log={logs_dir / f'search_{dtype}.log'}")
            if rc != 0:
                raise RuntimeError(f"search_disk_index failed for {dtype} (rc={rc}). See logs.")
            search_rows = _parse_search_table(out)
            if search_rows:
                _log(f"Parsed {len(search_rows)} search row(s) for {dtype}: Ls={[int(r['L']) for r in search_rows]}")
            else:
                _log(f"Warning: failed to parse any search table rows for {dtype}; see log")
        else:
            _log("Skipping search (--skip-search)")

        results[dtype] = {
            "dtype": dtype,
            "base": str(data[dtype]["base"]),
            "query": str(data[dtype]["query"]),
            "index_prefix": str(index_prefix),
            "reorder": bool(want_reorder),
            "build_time_s": build_time_s,
            "search_time_s": search_time_s,
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
            "dist": args.dist,
            "K": args.K,
            "Ls": args.Ls,
            "R": args.R,
            "Lbuild": args.Lbuild,
            "B": args.B,
            "M": args.M,
            "PQ_disk_bytes": args.PQ_disk_bytes,
            "QD": args.QD,
            "threads": args.threads,
        },
        "convert": convert_meta,
        "gt": {k: (str(v) if v else None) for k, v in gt_paths.items()},
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
