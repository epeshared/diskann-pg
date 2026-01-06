import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# DiskANN .bin format:
# uint32 npts, uint32 dim, then npts*dim elements of dtype.


@dataclass(frozen=True)
class BinHeader:
    npts: int
    dim: int


def read_bin_header(path: Path) -> BinHeader:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"Invalid bin file (too small): {path}")
        npts, dim = struct.unpack("<II", header)
        return BinHeader(npts=npts, dim=dim)


def write_bin_header(f, header: BinHeader) -> None:
    f.write(struct.pack("<II", header.npts, header.dim))


def require_numpy():
    try:
        import numpy as np  # noqa: F401

        return np
    except Exception as e:
        raise RuntimeError(
            "This utility requires numpy for fast, chunked conversion. "
            "Install it with: pip install numpy"
        ) from e


def _iter_f32_payload_chunks(path: Path, header: BinHeader, chunk_elems: int):
    np = require_numpy()
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

    DiskANN's bf16 is bfloat16. We convert using round-to-nearest-even (RNE)
    rather than truncation to reduce quantization error.
    """
    np = require_numpy()

    header = read_bin_header(src_f32)
    dst_bf16.parent.mkdir(parents=True, exist_ok=True)

    with dst_bf16.open("wb") as out:
        write_bin_header(out, header)
        for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
            u32 = chunk.view(np.uint32)
            # RNE: bf16 = (u32 + 0x7FFF + lsb_of_upper16) >> 16
            lsb = (u32 >> 16) & 1
            bf16_u16 = ((u32 + np.uint32(0x7FFF) + lsb) >> 16).astype(np.uint16, copy=False)
            bf16_u16.tofile(out)


def _compute_max_abs_f32_bin(src_f32: Path, chunk_elems: int = 1 << 20) -> float:
    np = require_numpy()
    header = read_bin_header(src_f32)
    max_abs = 0.0
    for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
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
    np = require_numpy()

    header = read_bin_header(src_f32)
    dst_i8.parent.mkdir(parents=True, exist_ok=True)

    if scale is None:
        max_abs = _compute_max_abs_f32_bin(src_f32, chunk_elems=chunk_elems)
        scale = (max_abs / 127.0) if max_abs > 0 else 1.0

    with dst_i8.open("wb") as out:
        write_bin_header(out, header)
        inv = 1.0 / float(scale)
        for chunk in _iter_f32_payload_chunks(src_f32, header, chunk_elems=chunk_elems):
            q = np.rint(chunk * inv)
            q = np.clip(q, -127, 127).astype(np.int8)
            q.tofile(out)

    return float(scale)
