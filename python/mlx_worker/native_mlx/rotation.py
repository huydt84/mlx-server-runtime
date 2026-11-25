"""Randomized Hadamard rotations used by packed checkpoints.

Packed weights are quantized in a rotated basis: an input ``x`` is multiplied
by random signs and a Hadamard matrix of its own width before it meets the
packed rows. Widths that are not powers of two use the Kronecker product of a
Sylvester matrix and a Paley matrix, so ``H = A (x) B`` and the rotation is
two small products instead of one dense one.

The kernels of ``kernels/rotation.metal`` apply a rotation together with the
element-wise steps around it in one launch, because a forward runs about two
hundred rotations one after another.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header

_SIMDGROUPS = 8
_THREADS = _SIMDGROUPS * 32
_PLAIN, _SIGMOID, _SOFT = range(3)

_INPUT_SOURCE = """
    uint index = threadgroup_position_in_grid.x;
    uint matrix = index / TOKENS;
    uint token = index % TOKENS;
    constexpr uint WIDTH = A * B;
    threadgroup float staged[INPUT == mlx_air::ROTATE_SOFT ? WIDTH : 1];
    uint row = inputs[matrix] * TOKENS + token;
    mlx_air::RotationSource<INPUT, A * B, X> source{
        x + ulong(row) * (INPUT == mlx_air::ROTATE_SOFT ? 2 * WIDTH : WIDTH),
        gate + ulong(token) * gate_layout[0] + gate_layout[1],
        signs + matrix * WIDTH, float2(caps[0], caps[1])};
    mlx_air::rotate_input<INPUT, A, B, SIMDGROUPS, RAW>(
        source, factor, y + ulong(index) * WIDTH,
        y + ulong(MATRICES * TOKENS + token) * WIDTH, staged, matrix,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""


@lru_cache(maxsize=16)
def hadamard_factors(width: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Return the Kronecker factors of the Hadamard matrix for ``width``.

    Args:
        width: Rotated vector width.

    Returns:
        The Sylvester factor and the second factor as nested tuples of +-1.

    Raises:
        ValueError: If no supported construction exists.
    """

    for power in range(int(math.log2(width)), -1, -1):
        if width % (1 << power):
            continue
        rest = width >> power
        if rest == 1 and power > 1:
            return _sylvester(power // 2), _sylvester(power - power // 2)
        if rest == 1:
            return _sylvester(power), ((1,),)
        if rest == 2:
            return _sylvester(power), _sylvester(1)
        if (rest - 1) % 4 == 3 and _is_prime(rest - 1):
            return _sylvester(power), _paley(rest - 1)
    raise ValueError(f"no Hadamard construction for width {width}")


@lru_cache(maxsize=16)
def _factor_arrays(width: int) -> tuple[mx.array, mx.array]:
    """Return both factors, the second one scaled to make the product unitary."""

    first, second = hadamard_factors(width)
    return (
        mx.array(first, dtype=mx.float32),
        mx.array(second, dtype=mx.float32) * (1.0 / math.sqrt(width)),
    )


def rotate_out(y: mx.array, signs: mx.array) -> mx.array:
    """Rotate packed outputs back: ``H^T y * signs / sqrt(width)``.

    Args:
        y: ``(..., width)`` products in the packed basis.
        signs: Signs broadcastable to ``y``.

    Returns:
        Float32 outputs.
    """

    width = int(y.shape[-1])
    first, second = _factor_arrays(width)
    blocks = y.astype(mx.float32).reshape(-1, first.shape[0], second.shape[0])
    rotated = first.T @ blocks @ second
    return rotated.reshape(y.shape) * signs


def rotate_packed(
    x: mx.array,
    signs: mx.array,
    *,
    inputs: mx.array | None = None,
    dtype: Any = mx.float16,
    gate: tuple[mx.array, int] | None = None,
    soft_caps: mx.array | None = None,
    keep_input: bool = False,
) -> mx.array:
    """Rotate inputs for several packed matrices in one launch.

    Each matrix reads ``H (x * signs) / sqrt(width)``, where ``x`` is the input,
    the input times ``sigmoid(gate)``, or the soft-capped product of the input's
    halves: ``g`` and ``u`` with caps ``a`` and ``b`` give
    ``tanh(g / a) * a * sigmoid(g) * tanh(u / b) * b``.

    Args:
        x: ``(sources, tokens, width)`` float32 inputs; twice the width with
            ``soft_caps``.
        signs: ``(matrices, width)`` float32 signs.
        inputs: ``(matrices,)`` uint32 source of each matrix; all read the
            first one when omitted.
        dtype: Type of the result.
        gate: ``(tokens, columns)`` float32 array and the column where the
            gate of ``width`` columns starts.
        soft_caps: Caps of the gate and of the gated half.
        keep_input: Also return the unrotated input as a last matrix.

    Returns:
        ``(matrices, tokens, width)`` rotated inputs, with one more leading
        entry when ``keep_input`` is set.
    """

    matrices, width = (int(size) for size in signs.shape)
    tokens = int(x.shape[1])
    first, second = _factor_arrays(width)
    mode = _SIGMOID if gate is not None else _SOFT if soft_caps is not None else _PLAIN
    gates, gate_start = (x, 0) if gate is None else gate
    planes = matrices + int(keep_input)
    return _input_kernel()(
        inputs=[
            x,
            gates,
            signs,
            second,
            _zeros(matrices) if inputs is None else inputs,
            _pair(0 if gate is None else int(gates.shape[1]), gate_start),
            _NO_CAPS if soft_caps is None else soft_caps,
        ],
        template=[
            ("INPUT", mode),
            ("A", int(first.shape[0])),
            ("B", int(second.shape[0])),
            ("SIMDGROUPS", _SIMDGROUPS),
            ("RAW", keep_input),
            ("TOKENS", tokens),
            ("MATRICES", matrices),
            ("X", x.dtype),
        ],
        grid=(matrices * tokens * _THREADS, 1, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(planes, tokens, width)],
        output_dtypes=[dtype],
    )[0]


def rotate_ranges(
    y: mx.array, ranges: Sequence[tuple[int, int]], signs: Sequence[mx.array | None]
) -> mx.array:
    """Rotate column ranges back from the packed basis in one launch.

    Equals :func:`rotate_out` on each range that has signs; the other ranges
    pass through.

    Args:
        y: ``(tokens, columns)`` float32 products.
        ranges: First and end column of each range, covering all columns.
        signs: Signs of each range.

    Returns:
        ``(tokens, columns)`` outputs.
    """

    layout = tuple(
        (first, end - first, flips is not None)
        for (first, end), flips in zip(ranges, signs, strict=True)
    )
    kernel, widths = _output_kernel(layout)
    tokens, columns = (int(size) for size in y.shape)
    return kernel(
        inputs=[
            y,
            *(flips for flips in signs if flips is not None),
            *(_inverse_factor(width) for width in widths),
        ],
        grid=(len(layout) * _THREADS, tokens, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(tokens, columns)],
        output_dtypes=[mx.float32],
    )[0]


_NO_CAPS = mx.ones((2,), dtype=mx.float32)


@lru_cache(maxsize=16)
def _inverse_factor(width: int) -> mx.array:
    """Return the second factor of the inverse rotation in row order."""

    return mx.contiguous(_factor_arrays(width)[1].T)


@lru_cache(maxsize=16)
def _zeros(count: int) -> mx.array:
    return mx.zeros((count,), dtype=mx.uint32)


@lru_cache(maxsize=64)
def _pair(first: int, second: int) -> mx.array:
    return mx.array((first, second), dtype=mx.uint32)


@lru_cache(maxsize=1)
def _input_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_rotate_packed",
        input_names=[
            "x",
            "gate",
            "signs",
            "factor",
            "inputs",
            "gate_layout",
            "caps",
        ],
        output_names=["y"],
        source=_INPUT_SOURCE,
        header=metal_header("rotation.metal"),
    )


@lru_cache(maxsize=64)
def _output_kernel(
    layout: tuple[tuple[int, int, bool], ...],
) -> tuple[Any, tuple[int, ...]]:
    """Return a kernel for one range layout and the widths whose factors it reads."""

    widths = tuple(sorted({width for _, width, rotated in layout if rotated}))
    cases = []
    names = []
    for index, (start, width, rotated) in enumerate(layout):
        where = f"in + {start}, "
        if rotated:
            first, second = hadamard_factors(width)
            names.append(f"signs_{index}")
            call = (
                f"rotate_output<{len(first)}, {len(second)}, {_SIMDGROUPS}>("
                f"{where}signs_{index}, factor_{width}, "
            )
        else:
            call = f"copy_output<{width}, {_SIMDGROUPS}>({where}"
        cases.append(f"case {index}: mlx_air::{call}out + {start}, simd, lane); break;")
    source = """
        uint simd = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        ulong offset = ulong(threadgroup_position_in_grid.y) * x_shape[1];
        const device float* in = x + offset;
        device float* out = y + offset;
        switch (threadgroup_position_in_grid.x) {
    """
    kernel = mx.fast.metal_kernel(
        name="native_rotate_ranges_" + "_".join(f"{w}{int(r)}" for _, w, r in layout),
        input_names=["x", *names, *(f"factor_{width}" for width in widths)],
        output_names=["y"],
        source=source + "\n".join(cases) + "\n}",
        header=metal_header("rotation.metal"),
    )
    return kernel, widths


def _sylvester(power: int) -> tuple[tuple[int, ...], ...]:
    size = 1 << power
    return tuple(
        tuple(-1 if (row & column).bit_count() & 1 else 1 for column in range(size))
        for row in range(size)
    )


def _paley(prime: int) -> tuple[tuple[int, ...], ...]:
    """Return the Paley type I Hadamard matrix of size ``prime + 1``."""

    squares = {value * value % prime for value in range(1, prime)}

    def character(value: int) -> int:
        value %= prime
        return 0 if not value else (1 if value in squares else -1)

    rows = [(1,) * (prime + 1)]
    for row in range(prime):
        core = tuple(
            character(column - row) + int(column == row) for column in range(prime)
        )
        rows.append((-1, *core))
    return tuple(rows)


def _is_prime(value: int) -> bool:
    return value > 1 and all(value % d for d in range(2, math.isqrt(value) + 1))
