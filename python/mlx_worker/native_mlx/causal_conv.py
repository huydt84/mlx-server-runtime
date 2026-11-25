"""Causal depthwise convolutions over unpadded token rows.

The kernel source is ``kernels/causal_conv.metal``. Tokens of every request are
concatenated, so a batch that mixes prefill chunks with single decode tokens
computes no padding; see :class:`TokenLayout`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header

CONV_LAGS = 4
"""Inputs one convolution reads, including the current token."""

_SIMD_WIDTH = 32


@dataclass
class TokenLayout:
    """Position of every token of a step in the concatenated token axis.

    Attributes:
        cache_lengths: Committed tokens per row before the step.
        token_lengths: Tokens each row appends.
    """

    cache_lengths: tuple[int, ...]
    token_lengths: tuple[int, ...]
    tokens: int = field(init=False)
    rows: mx.array = field(init=False)
    index: mx.array = field(init=False)
    starts: mx.array = field(init=False)
    lengths: mx.array = field(init=False)
    positions: mx.array = field(init=False)

    def __post_init__(self) -> None:
        self.tokens = sum(self.token_lengths)
        rows: list[int] = []
        index: list[int] = []
        starts: list[int] = []
        positions: list[int] = []
        for row, (cached, count) in enumerate(
            zip(self.cache_lengths, self.token_lengths, strict=True)
        ):
            starts.append(len(rows))
            rows.extend([row] * count)
            index.extend(range(count))
            positions.extend(range(cached, cached + count))
        self.rows = mx.array(rows, dtype=mx.int32)
        self.index = mx.array(index, dtype=mx.int32)
        self.starts = mx.array(starts, dtype=mx.int32)
        self.lengths = mx.array(self.token_lengths, dtype=mx.int32)
        self.positions = mx.array(positions, dtype=mx.int32)

    def last_tokens(self) -> mx.array:
        """Return the concatenated index of every row's last token."""

        return self.starts + self.lengths - 1


_CONV_SOURCE = """
    uint c = thread_position_in_grid.x;
    uint v = thread_position_in_grid.y;
    ulong stride = x_shape[1];
    if (v < uint(x_shape[0])) {
      float value = x[v * stride + c];
      for (int lag = 0; lag < LAGS; lag++) {
        value += weights[lag * C + c] * mlx_air::causal_input<LAGS, C>(
            x, stride, 0, history, rows, index, v, c, lag);
      }
      value *= scale[0];
      if (RESIDUAL) {
        value += residual[v * C + c];
      }
      out[v * C + c] = value;
    } else {
      uint slot = v - uint(x_shape[0]);
      next_history[slot * C + c] = mlx_air::causal_next_history<LAGS, C>(
          x, stride, 0, history, starts, lengths, slot / (LAGS - 1),
          slot % (LAGS - 1), c);
    }
"""

_KV_SOURCE = """
    mlx_air::causal_keys_values<LAGS, HEADS, HEAD_DIM, DYNAMIC>(
        x, x_shape[1], offset[0], history, rows, index, weights,
        dynamic_weights, params[0], params[1], params[2], out,
        threadgroup_position_in_grid.xy, thread_position_in_threadgroup.x);
"""

_HISTORY_SOURCE = """
    uint c = thread_position_in_grid.x;
    uint slot = thread_position_in_grid.y;
    next_history[slot * C + c] = mlx_air::causal_next_history<LAGS, C>(
        x, x_shape[1], offset[0], history, starts, lengths, slot / (LAGS - 1),
        slot % (LAGS - 1), c);
"""


@lru_cache(maxsize=1)
def _conv_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_causal_conv",
        input_names=[
            "x",
            "history",
            "rows",
            "index",
            "starts",
            "lengths",
            "weights",
            "scale",
            "residual",
        ],
        output_names=["out", "next_history"],
        source=_CONV_SOURCE,
        header=metal_header("causal_conv.metal"),
    )


@lru_cache(maxsize=1)
def _kv_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_causal_keys_values",
        input_names=[
            "x",
            "offset",
            "history",
            "rows",
            "index",
            "weights",
            "dynamic_weights",
            "params",
        ],
        output_names=["out"],
        source=_KV_SOURCE,
        header=metal_header("causal_conv.metal"),
    )


@lru_cache(maxsize=1)
def _history_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_causal_history",
        input_names=["x", "offset", "history", "starts", "lengths"],
        output_names=["next_history"],
        source=_HISTORY_SOURCE,
        header=metal_header("causal_conv.metal"),
    )


@lru_cache(maxsize=64)
def _floats(*values: float) -> mx.array:
    return mx.array(values, dtype=mx.float32)


@lru_cache(maxsize=64)
def _uints(*values: int) -> mx.array:
    return mx.array(values, dtype=mx.uint32)


def causal_conv(
    x: mx.array,
    history: mx.array,
    weights: mx.array,
    layout: TokenLayout,
    *,
    scale: float,
    residual: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Convolve, add the input, scale, and optionally add a residual.

    Args:
        x: ``(tokens, channels)`` float32 inputs.
        history: ``(rows, lags - 1, channels)`` float32 committed inputs.
        weights: ``(lags, channels)`` float32 weights, current token first.
        layout: Row of every token.
        scale: Factor applied to ``x + conv(x)``.
        residual: Optional ``(tokens, channels)`` float32 array to add.

    Returns:
        ``(tokens, channels)`` outputs and the history after the step.
    """

    tokens, channels = (int(size) for size in x.shape)
    rows = int(history.shape[0])
    slots = rows * (CONV_LAGS - 1)
    out, following = _conv_kernel()(
        inputs=[
            x,
            history,
            layout.rows,
            layout.index,
            layout.starts,
            layout.lengths,
            weights,
            _floats(scale),
            x if residual is None else residual,
        ],
        template=[
            ("LAGS", CONV_LAGS),
            ("C", channels),
            ("RESIDUAL", residual is not None),
        ],
        grid=(channels, tokens + slots, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(tokens, channels), tuple(history.shape)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return out, following


def causal_keys_values(
    x: mx.array,
    history: mx.array,
    weights: mx.array,
    layout: TokenLayout,
    *,
    offset: int,
    heads: int,
    head_dim: int,
    scale: float,
    clip: float,
    eps: float,
    dynamic_weights: mx.array | None = None,
    dtype: Any = mx.float16,
) -> tuple[mx.array, mx.array]:
    """Convolve key and value projections and normalize their heads.

    Args:
        x: ``(tokens, width)`` float32 projections; keys then values start at
            ``offset``.
        history: ``(rows, lags - 1, heads * head_dim)`` committed inputs.
        weights: ``(lags, heads * head_dim)`` weights, current token first.
        layout: Row of every token.
        offset: First key column of ``x``.
        heads: Key heads plus value heads.
        head_dim: Elements per head.
        scale: Factor applied to ``x + conv(x)``.
        clip: Largest norm of a key head.
        eps: Value normalization epsilon.
        dynamic_weights: Optional ``(tokens, heads, lags)`` weights added to
            every channel of a head.
        dtype: Output dtype.

    Returns:
        ``(tokens, 2, heads / 2, head_dim)`` keys and values and the history
        after the step.
    """

    tokens = int(x.shape[0])
    channels = heads * head_dim
    inputs = [x, _uints(offset), history]
    out = _kv_kernel()(
        inputs=[
            *inputs,
            layout.rows,
            layout.index,
            weights,
            _floats(0.0) if dynamic_weights is None else dynamic_weights,
            _floats(scale, clip, eps),
        ],
        template=[
            ("LAGS", CONV_LAGS),
            ("HEADS", heads),
            ("HEAD_DIM", head_dim),
            ("DYNAMIC", dynamic_weights is not None),
        ],
        grid=(heads * _SIMD_WIDTH, tokens, 1),
        threadgroup=(_SIMD_WIDTH, 1, 1),
        output_shapes=[(tokens, 2, heads // 2, head_dim)],
        output_dtypes=[dtype],
    )[0]
    following = _history_kernel()(
        inputs=[*inputs, layout.starts, layout.lengths],
        template=[("LAGS", CONV_LAGS), ("C", channels)],
        grid=(channels, int(history.shape[0]) * (CONV_LAGS - 1), 1),
        threadgroup=(256, 1, 1),
        output_shapes=[tuple(history.shape)],
        output_dtypes=[mx.float32],
    )[0]
    return mx.depends(out, following), following
