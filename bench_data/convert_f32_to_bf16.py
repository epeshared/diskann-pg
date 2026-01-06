#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bench_data.bin_utils import convert_f32_bin_to_bf16_bin  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Convert DiskANN float32 .bin to bf16 .bin (bfloat16 payload).")
    p.add_argument("--src", required=True, help="Source float32 .bin")
    p.add_argument("--dst", required=True, help="Destination bf16 .bin")
    p.add_argument("--chunk-elems", type=int, default=1 << 20, help="Chunk size in elements (default: 1<<20)")
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    convert_f32_bin_to_bf16_bin(src, dst, chunk_elems=args.chunk_elems)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
