#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import time
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class RunResult:
    isa: str
    run_idx: int
    rc: int
    wall_s: float
    cmd: List[str]
    log_path: Path


def _resolve_binary(build_dir: Path) -> Path:
    # CMake builds in this repo place apps under <build>/apps/<name>
    candidates = [
        build_dir / "apps" / "build_disk_index",
        build_dir / "apps" / "build_disk_index.exe",
    ]
    for p in candidates:
        if p.exists() and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError(
        f"Could not find an executable build_disk_index under build dir: {build_dir}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def _rm_prefix_outputs(prefix: Path) -> None:
    parent = prefix.parent
    stem = prefix.name
    if not parent.exists():
        return
    for p in parent.glob(stem + "*"):
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except FileNotFoundError:
            pass


def _run_one(
    *,
    isa: str,
    exe: Path,
    cmd_args: List[str],
    run_idx: int,
    env: Dict[str, str],
    log_path: Path,
) -> RunResult:
    cmd = [str(exe)] + cmd_args
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    with log_path.open("wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    end = time.perf_counter()

    return RunResult(isa=isa, run_idx=run_idx, rc=proc.returncode, wall_s=end - start, cmd=cmd, log_path=log_path)


def _fmt_seconds(x: float) -> str:
    return f"{x:.3f}s"


def _expected_elem_bytes(data_type: str) -> int:
    dt = data_type.lower()
    if dt in ("float", "f32"):
        return 4
    if dt in ("bf16", "bfloat16"):
        return 2
    if dt in ("uint8", "u8"):
        return 1
    if dt in ("int8", "i8"):
        return 1
    raise ValueError(f"Unsupported --data-type: {data_type}")


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
    return npts, dim, int(elem_bytes)


def main(argv: Optional[List[str]] = None) -> int:
    this_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DiskANN build_disk_index by running two different build directories "
            "(e.g., AVX512 vs AMX) with identical CLI arguments and comparing wall-clock time."
        )
    )

    parser.add_argument(
        "--avx512-build-dir",
        default="/home/xtang/DiskANN-epeshared/build-avx512",
        help="DiskANN CMake build directory containing apps/build_disk_index (AVX512 build).",
    )
    parser.add_argument(
        "--amx-build-dir",
        default="/home/xtang/DiskANN-epeshared/build-amx",
        help="DiskANN CMake build directory containing apps/build_disk_index (AMX build).",
    )

    # Required build_disk_index args
    parser.add_argument("--data-type", required=True, help="e.g. float|bf16|bfloat16|uint8|int8")
    parser.add_argument("--dist-fn", required=True, help="l2|mips|cosine")
    parser.add_argument("--data-path", required=True, help="Path to input .bin")
    parser.add_argument("--search-dram-budget", type=float, required=True, help="-B / search_DRAM_budget (GB)")
    parser.add_argument("--build-dram-budget", type=float, required=True, help="-M / build_DRAM_budget (GB)")

    # Common optional knobs
    parser.add_argument("--num-threads", type=int, default=None, help="-T / num_threads")
    parser.add_argument("--max-degree", type=int, default=None, help="-R / max_degree")
    parser.add_argument("--lbuild", type=int, default=None, help="-L / Lbuild")
    parser.add_argument("--pq-disk-bytes", type=int, default=None, help="--PQ_disk_bytes")
    parser.add_argument("--build-pq-bytes", type=int, default=None, help="--build_PQ_bytes")
    parser.add_argument("--qd", type=int, default=None, help="--QD")
    parser.add_argument("--use-opq", action="store_true", help="--use_opq")
    parser.add_argument("--append-reorder-data", action="store_true", help="--append_reorder_data")
    parser.add_argument(
        "--with-reorder",
        dest="append_reorder_data",
        action="store_true",
        help="Alias for --append-reorder-data (build disk index with reorder data)",
    )
    parser.add_argument("--codebook-prefix", default=None, help="--codebook_prefix")

    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of repeated runs per ISA (indexes will be rebuilt each run).",
    )
    parser.add_argument(
        "--out-root",
        default=str(this_dir / "runs"),
        help="Directory to place output indexes/logs (will create per-run subdirs).",
    )

    # Extra args passthrough
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to build_disk_index after a '--' separator.",
    )

    args = parser.parse_args(argv)

    avx_build = Path(args.avx512_build_dir)
    amx_build = Path(args.amx_build_dir)

    avx_exe = _resolve_binary(avx_build)
    amx_exe = _resolve_binary(amx_build)

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"data_path does not exist: {data_path}")

    # Fail fast on common foot-gun: passing a float32 .bin with --data-type bf16 (or vice versa).
    try:
        npts, dim, inferred_elem_bytes = _infer_elem_bytes_from_bin(data_path)
        expected_bytes = _expected_elem_bytes(args.data_type)
        if inferred_elem_bytes != expected_bytes:
            print(
                "ERROR: --data-type does not match input .bin element size\n"
                f"  data_path: {data_path}\n"
                f"  header: npts={npts} dim={dim}\n"
                f"  inferred element bytes from file size: {inferred_elem_bytes}\n"
                f"  expected element bytes for --data-type {args.data_type}: {expected_bytes}\n"
                "Fix: pass the correct --data-type for this file (e.g., float vs bf16), or use a matching input file.\n"
            )
            return 2
    except Exception as e:
        print(f"WARNING: could not validate input bin type/size: {e}")

    out_root = Path(args.out_root)
    run_root = out_root
    run_root.mkdir(parents=True, exist_ok=True)

    if args.append_reorder_data:
        if args.pq_disk_bytes is None or int(args.pq_disk_bytes) == 0:
            print(
                "ERROR: --append-reorder-data/--with-reorder requires disk PQ on SSD.\n"
                "Fix: pass --pq-disk-bytes <N> (N > 0), e.g. --pq-disk-bytes 64 (choose what you need).\n"
            )
            return 2

    # Build the shared CLI args for build_disk_index.
    def build_common_cli(index_prefix: Path) -> List[str]:
        cli: List[str] = [
            "--data_type",
            args.data_type,
            "--dist_fn",
            args.dist_fn,
            "--index_path_prefix",
            str(index_prefix),
            "--data_path",
            str(data_path),
            "-B",
            str(args.search_dram_budget),
            "-M",
            str(args.build_dram_budget),
        ]

        if args.num_threads is not None:
            cli += ["-T", str(args.num_threads)]
        if args.max_degree is not None:
            cli += ["-R", str(args.max_degree)]
        if args.lbuild is not None:
            cli += ["-L", str(args.lbuild)]
        if args.qd is not None:
            cli += ["--QD", str(args.qd)]
        if args.pq_disk_bytes is not None:
            cli += ["--PQ_disk_bytes", str(args.pq_disk_bytes)]
        if args.build_pq_bytes is not None:
            cli += ["--build_PQ_bytes", str(args.build_pq_bytes)]
        if args.append_reorder_data:
            cli += ["--append_reorder_data"]
        if args.use_opq:
            cli += ["--use_opq"]
        if args.codebook_prefix is not None:
            cli += ["--codebook_prefix", args.codebook_prefix]

        if args.extra_args:
            # argparse keeps the leading "--" if provided; strip it.
            extra = list(args.extra_args)
            if extra and extra[0] == "--":
                extra = extra[1:]
            cli += extra

        return cli

    base_env = dict(os.environ)
    # Keep env.sh behaviour if user sourced it; we do not mutate LD_LIBRARY_PATH here.
    if args.num_threads is not None:
        base_env.setdefault("OMP_NUM_THREADS", str(args.num_threads))

    results: List[RunResult] = []
    for isa, exe in ("avx512", avx_exe), ("amx", amx_exe):
        for run_idx in range(args.runs):
            suffix = "_reorder" if args.append_reorder_data else ""
            prefix = run_root / isa / f"index_run{run_idx}{suffix}"
            _rm_prefix_outputs(prefix)
            cmd_args = build_common_cli(prefix)
            log_path = run_root / isa / f"run{run_idx}.log"
            res = _run_one(isa=isa, exe=exe, cmd_args=cmd_args, run_idx=run_idx, env=base_env, log_path=log_path)
            results.append(res)
            print(f"[{isa}] run {run_idx}: rc={res.rc} wall={_fmt_seconds(res.wall_s)} log={log_path}")
            if res.rc != 0:
                print(f"[{isa}] failed. Command was: {' '.join(res.cmd)}")
                return res.rc

    def summarize(isa: str) -> Tuple[float, float, float]:
        times = [r.wall_s for r in results if r.isa == isa]
        return min(times), sum(times) / len(times), max(times)

    avx_min, avx_avg, avx_max = summarize("avx512")
    amx_min, amx_avg, amx_max = summarize("amx")

    print("\nSummary (wall-clock):")
    print(f"  avx512: min={_fmt_seconds(avx_min)} avg={_fmt_seconds(avx_avg)} max={_fmt_seconds(avx_max)}")
    print(f"  amx:    min={_fmt_seconds(amx_min)} avg={_fmt_seconds(amx_avg)} max={_fmt_seconds(amx_max)}")

    if amx_avg > 0:
        print(f"\nSpeedup (avg): avx512/amx = {avx_avg / amx_avg:.3f}x")

    print(f"\nArtifacts: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
