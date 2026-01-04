#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# DiskANN .bin format:
# uint32 npts, uint32 dim, then npts*dim elements of dtype.


@dataclass(frozen=True)
class BinHeader:
    npts: int
    dim: int


def _read_bin_header(path: Path) -> BinHeader:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid bin file (too small): {path}")
        npts, dim = struct.unpack("<II", header)
        return BinHeader(npts=npts, dim=dim)


def _write_bin_header(f, header: BinHeader) -> None:
    f.write(struct.pack("<II", header.npts, header.dim))


def _require_numpy():
    try:
        import numpy as np  # noqa: F401

        return np
    except Exception as e:
        raise RuntimeError(
            "This benchmark requires numpy for fast, chunked conversion. "
            "Install it with: pip install numpy"
        ) from e


def _iter_f32_payload_chunks(path: Path, header: BinHeader, chunk_elems: int):
    np = _require_numpy()
    total_elems = header.npts * header.dim
    with path.open("rb") as f:
        f.seek(8)
        remaining = total_elems
        while remaining > 0:
            this_elems = min(chunk_elems, remaining)
            arr = np.fromfile(f, dtype=np.float32, count=this_elems)
            if arr.size != this_elems:
                raise IOError(f"Short read when reading payload from {path}")
            yield arr
            remaining -= this_elems


def convert_f32_bin_to_bf16_bin(src_f32: Path, dst_bf16: Path, chunk_elems: int = 1 << 20) -> None:
    """Convert float32 .bin to bf16 .bin (payload is uint16 bf16 words).

    DiskANN's bf16 is bfloat16 (truncated float32 high 16 bits).
    """
    np = _require_numpy()

    header = _read_bin_header(src_f32)
    dst_bf16.parent.mkdir(parents=True, exist_ok=True)

    with dst_bf16.open("wb") as out:
        _write_bin_header(out, header)
        for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
            u32 = chunk.view(np.uint32)
            bf16_u16 = (u32 >> 16).astype(np.uint16, copy=False)
            bf16_u16.tofile(out)


def _compute_max_abs_f32_bin(src_f32: Path, chunk_elems: int = 1 << 20) -> float:
    np = _require_numpy()
    header = _read_bin_header(src_f32)
    max_abs = 0.0
    for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
        # chunk is float32
        local = float(np.max(np.abs(chunk)))
        if local > max_abs:
            max_abs = local
    return max_abs


def convert_f32_bin_to_int8_bin(
    src_f32: Path,
    dst_i8: Path,
    scale: Optional[float] = None,
    chunk_elems: int = 1 << 20,
) -> float:
    """Quantize float32 .bin to int8 .bin.

    Uses symmetric linear quantization: q = round(x / scale), clamped to [-127, 127].
    Returns the scale used.
    """
    np = _require_numpy()

    header = _read_bin_header(src_f32)
    dst_i8.parent.mkdir(parents=True, exist_ok=True)

    if scale is None:
        max_abs = _compute_max_abs_f32_bin(src_f32, chunk_elems=chunk_elems)
        scale = (max_abs / 127.0) if max_abs > 0 else 1.0

    with dst_i8.open("wb") as out:
        _write_bin_header(out, header)
        inv = 1.0 / float(scale)
        for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
            q = np.rint(chunk * inv)
            q = np.clip(q, -127, 127).astype(np.int8)
            q.tofile(out)

    return float(scale)


def _find_diskann_apps_dir(explicit: Optional[str]) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"diskann apps dir not found: {p}")
        return p

    # Heuristic: common build outputs.
    candidates = [
        Path("/home/xtang/DiskANN-epeshared/build/apps"),
        Path("/home/xtang/DiskANN/build/apps"),
        Path("/home/xtang/DiskANN/build/tests"),
    ]
    for c in candidates:
        if (c / "build_disk_index").exists() and (c / "search_disk_index").exists():
            return c

    raise FileNotFoundError(
        "Could not auto-detect DiskANN apps directory. "
        "Pass --diskann-apps /path/to/DiskANN/build/apps"
    )


def _run(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[int, float, str]:
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - start
    return proc.returncode, elapsed, proc.stdout


_QPS_LINE_RE = re.compile(
    r"^\s*(?P<L>\d+)\s+(?P<beam>\d+)\s+(?P<qps>[0-9.]+)\s+(?P<mean>[0-9.]+)\s+(?P<p999>[0-9.]+)",
    re.IGNORECASE,
)


def _parse_search_table(stdout: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for line in stdout.splitlines():
        m = _QPS_LINE_RE.match(line)
        if not m:
            continue
        rows.append(
            {
                "L": float(m.group("L")),
                "beamwidth": float(m.group("beam")),
                "qps": float(m.group("qps")),
                "mean_us": float(m.group("mean")),
                "p999_us": float(m.group("p999")),
            }
        )
    return rows


def _read_ids_bin(path: Path) -> Tuple[int, int, List[int]]:
    # uint32 header (nq, k), then nq*k uint32
    header = _read_bin_header(path)
    nq, k = header.npts, header.dim
    # read payload
    np = _require_numpy()
    with path.open("rb") as f:
        f.seek(8)
        arr = np.fromfile(f, dtype=np.uint32, count=nq * k)
        if arr.size != nq * k:
            raise IOError(f"Short read: {path}")
    return nq, k, arr.tolist()


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
    # a and b are length k
    return len(set(a[:k]).intersection(set(b[:k]))) / float(k)


def _positional_match(a: List[int], b: List[int], k: int) -> float:
    return sum(1 for i in range(k) if a[i] == b[i]) / float(k)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark PQ+reorder for float/bf16/int8 with consistency checks")
    ap.add_argument("--diskann-apps", default=None, help="Path to DiskANN build apps dir (contains build_disk_index)")

    ap.add_argument("--workdir", default=str(Path(__file__).resolve().parent / "out"), help="Output work dir")
    ap.add_argument("--tag", default=None, help="Optional run tag")

    ap.add_argument("--dist", default="l2", choices=["l2", "mips", "cosine"], help="Distance function")
    ap.add_argument("--base-f32", required=True, help="Base vectors float32 .bin")
    ap.add_argument("--query-f32", required=True, help="Query vectors float32 .bin")
    ap.add_argument("--gt", default=None, help="Groundtruth file path (optional). If omitted, will compute per dtype")

    ap.add_argument("--R", type=int, default=64)
    ap.add_argument("--Lbuild", type=int, default=100)
    ap.add_argument("--B", type=float, default=0.25, help="search_DRAM_budget GB")
    ap.add_argument("--M", type=float, default=4.0, help="build_DRAM_budget GB")
    ap.add_argument("--PQ-disk-bytes", type=int, default=32, help="PQ bytes on SSD; must be >0 for reorder")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)

    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--Ls", type=int, nargs="+", default=[10, 20, 30, 40, 50, 100])
    ap.add_argument("--beamwidth", type=int, default=2)
    ap.add_argument("--num-nodes-to-cache", type=int, default=0)

    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-gt", action="store_true")
    ap.add_argument("--skip-search", action="store_true")

    ap.add_argument("--int8-scale", type=float, default=None, help="Quantization scale for int8 (optional)")
    ap.add_argument("--chunk-elems", type=int, default=1 << 20, help="Conversion chunk size in elements")

    ap.add_argument(
        "--consistency-L",
        type=int,
        default=None,
        help="Which L result to use for cross-dtype consistency; default=max(Ls)",
    )

    args = ap.parse_args()

    apps = _find_diskann_apps_dir(args.diskann_apps)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag or "pq_reorder"
    out_root = Path(args.workdir).expanduser().resolve() / f"{run_id}_{tag}"
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    base_f32 = Path(args.base_f32).expanduser().resolve()
    query_f32 = Path(args.query_f32).expanduser().resolve()

    # Prepare per-dtype data files
    data: Dict[str, Dict[str, Path]] = {
        "float": {"base": base_f32, "query": query_f32},
        "bf16": {"base": out_root / "data" / "base.bf16.bin", "query": out_root / "data" / "query.bf16.bin"},
        "int8": {"base": out_root / "data" / "base.int8.bin", "query": out_root / "data" / "query.int8.bin"},
    }

    # Convert
    convert_meta: Dict[str, Dict[str, float]] = {}

    if not data["bf16"]["base"].exists() or not data["bf16"]["query"].exists():
        convert_f32_bin_to_bf16_bin(base_f32, data["bf16"]["base"], chunk_elems=args.chunk_elems)
        convert_f32_bin_to_bf16_bin(query_f32, data["bf16"]["query"], chunk_elems=args.chunk_elems)

    int8_scale = args.int8_scale
    if not data["int8"]["base"].exists() or not data["int8"]["query"].exists():
        if int8_scale is None:
            # Choose a single scale based on base+query maxabs, so queries/base are consistent.
            base_max = _compute_max_abs_f32_bin(base_f32, chunk_elems=args.chunk_elems)
            query_max = _compute_max_abs_f32_bin(query_f32, chunk_elems=args.chunk_elems)
            max_abs = max(base_max, query_max)
            int8_scale = (max_abs / 127.0) if max_abs > 0 else 1.0

        used_scale_base = convert_f32_bin_to_int8_bin(
            base_f32, data["int8"]["base"], scale=float(int8_scale), chunk_elems=args.chunk_elems
        )
        used_scale_query = convert_f32_bin_to_int8_bin(
            query_f32, data["int8"]["query"], scale=float(int8_scale), chunk_elems=args.chunk_elems
        )
        convert_meta["int8"] = {"scale": float(int8_scale), "used_scale_base": used_scale_base, "used_scale_query": used_scale_query}

    # GT
    gt_paths: Dict[str, Optional[Path]] = {"float": None, "bf16": None, "int8": None}
    if args.gt:
        gt_paths = {"float": Path(args.gt).expanduser().resolve(), "bf16": Path(args.gt).expanduser().resolve(), "int8": Path(args.gt).expanduser().resolve()}
    elif not args.skip_gt:
        for dtype in ["float", "bf16", "int8"]:
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
            if rc != 0:
                raise RuntimeError(f"compute_groundtruth failed for {dtype} (rc={rc}). See logs.")
            gt_paths[dtype] = gt_out

    # Build + search
    results: Dict[str, Dict[str, object]] = {}

    for dtype in ["float", "bf16", "int8"]:
        dtype_dir = out_root / dtype
        dtype_dir.mkdir(exist_ok=True)
        index_prefix = dtype_dir / "index" / f"disk_index_{dtype}_R{args.R}_L{args.Lbuild}_PQ{args.PQ_disk_bytes}"
        index_prefix.parent.mkdir(parents=True, exist_ok=True)

        can_reorder = dtype in ("float", "bf16")
        want_reorder = can_reorder

        # Build
        build_log = ""
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
            if want_reorder:
                cmd.append("--append_reorder_data")

            rc, elapsed, out = _run(cmd)
            build_log = out
            build_time_s = elapsed
            (logs_dir / f"build_{dtype}.log").write_text(out)
            if rc != 0:
                raise RuntimeError(f"build_disk_index failed for {dtype} (rc={rc}). See logs.")

        # Search
        search_rows: List[Dict[str, float]] = []
        search_log = ""
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

            rc, elapsed, out = _run(cmd)
            search_log = out
            search_time_s = elapsed
            (logs_dir / f"search_{dtype}.log").write_text(out)
            if rc != 0:
                raise RuntimeError(f"search_disk_index failed for {dtype} (rc={rc}). See logs.")
            search_rows = _parse_search_table(out)

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
    consistency_L = args.consistency_L if args.consistency_L is not None else max(args.Ls)
    if consistency_L not in args.Ls:
        raise ValueError(f"--consistency-L {consistency_L} must be one of --Ls {args.Ls}")

    def ids_path(dtype: str) -> Path:
        dtype_dir = out_root / dtype
        return dtype_dir / "results" / f"res_{consistency_L}_idx_uint32.bin"

    consistency: Dict[str, object] = {"baseline": "float", "L": consistency_L, "K": args.K, "by_dtype": {}}

    base_ids_file = ids_path("float")
    if base_ids_file.exists() and (ids_path("bf16").exists() or ids_path("int8").exists()):
        nq0, k0, ids0 = _read_ids_bin_np(base_ids_file)
        if k0 < args.K:
            raise ValueError(f"Result K ({k0}) is smaller than requested K ({args.K})")

        for dtype in ["bf16", "int8"]:
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

    # Persist raw summary (consistency computed in a later step)
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
            "threads": args.threads,
        },
        "convert": convert_meta,
        "gt": {k: (str(v) if v else None) for k, v in gt_paths.items()},
        "results": results,
        "consistency": consistency,
    }

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(str(out_root / "summary.json"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise
