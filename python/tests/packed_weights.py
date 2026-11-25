"""Random packed matrices and a NumPy decoder written from the bit layouts."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_worker.native_mlx.packed_matmul import (
    FORMAT_BITS,
    TRELLIS_LUT_SHAPES,
    PackedRows,
)

_GROUP_COLUMNS = {"ht1": 4, "ht2": 4, "ht3": 2}
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def random_tables(rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Return one random table per trellis format."""

    return {
        fmt: rng.normal(size=shape).astype(np.float16)
        for fmt, shape in TRELLIS_LUT_SHAPES.items()
    }


def random_codes(
    rng: np.random.Generator, fmt: str, rows: int, columns: int
) -> np.ndarray:
    """Return ``(rows, row bytes)`` random codes of one block."""

    size = (rows, columns * FORMAT_BITS[fmt] // 8)
    return rng.integers(0, 256, size=size, dtype=np.uint8)


def random_rows(
    rng: np.random.Generator,
    fmt: str,
    rows: int,
    columns: int,
    tables: dict[str, np.ndarray],
    **placement: int,
) -> tuple[PackedRows, np.ndarray]:
    """Return one random packed matrix and the weights it stands for."""

    codes = random_codes(rng, fmt, rows, columns)
    levels = outer = None
    if fmt == "nvfp4":
        scales = rng.integers(0, 120, size=(rows, columns // 16), dtype=np.uint8)
        outer = rng.uniform(0.5, 2.0, size=(1,)).astype(np.float32)
    else:
        scales = rng.uniform(0.5, 2.0, size=(rows,)).astype(np.float16)
    if fmt.startswith("lm"):
        levels = rng.normal(size=(1 << FORMAT_BITS[fmt],)).astype(np.float32)
    dense = decode(fmt, codes, columns, scales, tables, levels, outer)
    packed = PackedRows(
        fmt,
        rows,
        mx.array(codes),
        mx.array(scales),
        levels=None if levels is None else mx.array(levels),
        outer=None if outer is None else mx.array(outer),
        **placement,
    )
    return packed, dense


def decode(
    fmt: str,
    codes: np.ndarray,
    columns: int,
    scales: np.ndarray,
    tables: dict[str, np.ndarray],
    levels: np.ndarray | None = None,
    outer: np.ndarray | None = None,
) -> np.ndarray:
    """Decode packed rows one weight at a time."""

    out = np.zeros((len(codes), columns), dtype=np.float32)
    for row, data in enumerate(codes):
        stream = int.from_bytes(data.tobytes(), "little")
        if fmt == "nvfp4":
            assert outer is not None
            for column in range(columns):
                code = stream >> (4 * column) & 15
                value = -_E2M1[code & 7] if code & 8 else _E2M1[code & 7]
                block = _e4m3(int(scales[row, column // 16]))
                out[row, column] = value * block * outer[0]
            continue
        bits = FORMAT_BITS[fmt]
        if fmt.startswith("lm"):
            assert levels is not None
            for column in range(columns):
                out[row, column] = levels[stream >> (bits * column) & ((1 << bits) - 1)]
        else:
            group = _GROUP_COLUMNS.get(fmt, 1)
            for first in range(0, columns, group):
                last = first + group - 1
                window = stream >> ((columns - 1 - last) * bits) & 0xFFF
                if group > 1:
                    entries = len(tables[fmt])
                    shift = 32 - entries.bit_length() + 1
                    window = (window * 0x9E3779B1 & 0xFFFFFFFF) >> shift
                out[row, first : last + 1] = tables[fmt][window]
        out[row] *= scales[row]
    return out


def _e4m3(code: int) -> float:
    exponent, mantissa = code >> 3 & 15, code & 7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
