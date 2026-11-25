"""Metal kernels for fused steps of the Inkling-2 graph.

The kernel sources are ``kernels/inkling2.metal``. Each function equals a short
chain of MLX operations, named in its docstring, and exists because a forward
repeats that chain in every one of its layer visits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header

_SIMD_WIDTH = 32
_MIX_SIMDGROUPS = 6
_GATE_SHARES = 8
_GATE_SIMDGROUPS = 2
_PRODUCT_SIMDGROUPS = 2
_PRODUCT_ROWS = 2

_CLIP_SOURCE = """
    uint token = threadgroup_position_in_grid.x / HEADS;
    uint head = threadgroup_position_in_grid.x % HEADS;
    uint width = projected_shape[1];
    float factor = SCALED ? token_scale[token] : 1.0f;
    mlx_air::inkling2_clip_head<HEAD_DIM>(
        projected + token * width + head * HEAD_DIM, q_scale,
        Q_SCALE_PER_HEAD * head * HEAD_DIM, clips[0], factor,
        queries + (token * HEADS + head) * HEAD_DIM, thread_index_in_simdgroup);
    mlx_air::inkling2_clip_head<REL_DIM>(
        projected + token * width + REL_START + head * REL_DIM, rel_scale,
        REL_SCALE_PER_HEAD * head * REL_DIM, clips[1], factor,
        rel + (token * HEADS + head) * REL_DIM, thread_index_in_simdgroup);
"""

_ROUTE_SOURCE = """
    mlx_air::inkling2_route<EXPERTS, TOP>(
        logits, bias, route_scale[0], chosen, weights, thread_position_in_grid.x);
"""

_GATE_INDEX = """
    uint piece = threadgroup_position_in_grid.x;
    uint index = threadgroup_position_in_grid.y;
    uint tokens = message_shape[1];
    uint gate = index / tokens;
    uint row = SHARED_HIDDEN ? index % tokens : index;
"""

_GATE_DOWN_SOURCE = (
    _GATE_INDEX
    + """
    mlx_air::inkling2_gate_down<HIDDEN, RANK, PIECES, SIMDGROUPS, G>(
        hidden + row * HIDDEN, message + index * HIDDEN,
        down + gate * 2 * HIDDEN * RANK, low + index * RANK,
        piece, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
"""
)

_GATE_UP_SOURCE = (
    _GATE_INDEX
    + """
    mlx_air::inkling2_gate_up<HIDDEN, RANK, PIECES, SIMDGROUPS, G>(
        hidden + row * HIDDEN, message + index * HIDDEN, low + index * RANK,
        up + gate * 2 * HIDDEN * RANK, out + index * HIDDEN,
        piece, thread_position_in_threadgroup.x);
"""
)

_PRODUCTS_SOURCE = """
    uint token = threadgroup_position_in_grid.y;
    uint rows = weight_shape[0];
    uint group = threadgroup_position_in_grid.x * simdgroups_per_threadgroup +
        simdgroup_index_in_threadgroup;
    mlx_air::inkling2_row_products<WIDTH, GROUP>(
        x + ulong(token) * WIDTH, weight, y + ulong(token) * rows,
        group * GROUP, rows, thread_index_in_simdgroup);
"""


@lru_cache(maxsize=1)
def _products_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_inkling2_row_products",
        input_names=["x", "weight"],
        output_names=["y"],
        source=_PRODUCTS_SOURCE,
        header=metal_header("inkling2.metal"),
    )


@lru_cache(maxsize=16)
def _mix_kernel(states: int) -> Any:
    names = [f"state{index}" for index in range(states)]
    source = f"""
        threadgroup mlx_air::MixScratch<{states}, QUERIES, HIDDEN, SIMDGROUPS> scratch;
        metal::array<const device float*, {states}> states = {{{", ".join(names)}}};
        mlx_air::inkling2_mix<{states}, QUERIES, HIDDEN, SIMDGROUPS>(
            states, queries, 1e-6f, out, threadgroups_per_grid.x, scratch,
            threadgroup_position_in_grid.x, simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup, thread_position_in_threadgroup.x);
    """
    return mx.fast.metal_kernel(
        name=f"native_inkling2_mix_{states}",
        input_names=[*names, "queries"],
        output_names=["out"],
        source=source,
        header=metal_header("inkling2.metal"),
    )


@lru_cache(maxsize=1)
def _clip_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_inkling2_scaled_clip",
        input_names=["projected", "q_scale", "rel_scale", "clips", "token_scale"],
        output_names=["queries", "rel"],
        source=_CLIP_SOURCE,
        header=metal_header("inkling2.metal"),
    )


@lru_cache(maxsize=1)
def _route_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_inkling2_route",
        input_names=["logits", "bias", "route_scale"],
        output_names=["chosen", "weights"],
        source=_ROUTE_SOURCE,
        header=metal_header("inkling2.metal"),
    )


@lru_cache(maxsize=1)
def _gate_kernels() -> tuple[Any, Any]:
    header = metal_header("inkling2.metal")
    return (
        mx.fast.metal_kernel(
            name="native_inkling2_gate_down",
            input_names=["hidden", "message", "down"],
            output_names=["low"],
            source=_GATE_DOWN_SOURCE,
            header=header,
        ),
        mx.fast.metal_kernel(
            name="native_inkling2_gate_up",
            input_names=["hidden", "message", "low", "up"],
            output_names=["out"],
            source=_GATE_UP_SOURCE,
            header=header,
        ),
    )


@lru_cache(maxsize=16)
def _floats(*values: float) -> mx.array:
    return mx.array(values, dtype=mx.float32)


def mix_states(queries: mx.array, states: Sequence[mx.array]) -> mx.array:
    """Mix residual states with learned queries.

    Equals ``softmax(rms_norm(states) @ query / sqrt(hidden))`` over the states
    applied to the normalized states, per query.

    Args:
        queries: ``(mixes, hidden)`` float32 queries.
        states: Residual states of ``(tokens, hidden)`` float32, oldest first.

    Returns:
        ``(mixes, tokens, hidden)`` mixtures.
    """

    mixes, hidden = (int(size) for size in queries.shape)
    tokens = int(states[0].shape[0])
    threads = _MIX_SIMDGROUPS * _SIMD_WIDTH
    return _mix_kernel(len(states))(
        inputs=[*states, queries],
        template=[
            ("QUERIES", mixes),
            ("HIDDEN", hidden),
            ("SIMDGROUPS", _MIX_SIMDGROUPS),
        ],
        grid=(tokens * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(mixes, tokens, hidden)],
        output_dtypes=[mx.float32],
    )[0]


def scaled_clip(
    projected: mx.array,
    q_scale: mx.array,
    rel_scale: mx.array,
    *,
    heads: int,
    head_dim: int,
    rel_start: int,
    rel_dim: int,
    clips: tuple[float, float],
    token_scale: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Scale queries and distance states and clip each head's L2 norm.

    Args:
        projected: ``(tokens, columns)`` float32 projections. Queries start at
            column zero.
        q_scale: Query scale, ``(head_dim,)`` or ``(heads, head_dim)``.
        rel_scale: Distance-state scale, ``(rel_dim,)`` or ``(heads, rel_dim)``.
        heads: Query heads.
        head_dim: Width of one query.
        rel_start: First column of the distance states.
        rel_dim: Width of one distance state.
        clips: Largest norms of a query and of a distance state.
        token_scale: Optional ``(tokens,)`` factor applied after the clip.

    Returns:
        ``(tokens, heads, head_dim)`` queries and ``(tokens, heads, rel_dim)``
        distance states.
    """

    tokens = int(projected.shape[0])
    queries, rel = _clip_kernel()(
        inputs=[
            projected,
            q_scale,
            rel_scale,
            _floats(*clips),
            _floats(1.0) if token_scale is None else token_scale,
        ],
        template=[
            ("HEADS", heads),
            ("HEAD_DIM", head_dim),
            ("REL_DIM", rel_dim),
            ("REL_START", rel_start),
            ("Q_SCALE_PER_HEAD", int(q_scale.size) == heads * head_dim),
            ("REL_SCALE_PER_HEAD", int(rel_scale.size) == heads * rel_dim),
            ("SCALED", token_scale is not None),
        ],
        grid=(tokens * heads * _SIMD_WIDTH, 1, 1),
        threadgroup=(_SIMD_WIDTH, 1, 1),
        output_shapes=[(tokens, heads, head_dim), (tokens, heads, rel_dim)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return queries, rel


def route_experts(
    logits: mx.array, bias: mx.array, *, top: int, scale: float
) -> tuple[mx.array, mx.array]:
    """Choose the experts of every token.

    Equals a top-k over ``sigmoid(logits) + bias`` whose weights are the chosen
    sigmoids, normalized to sum to ``scale``.

    Args:
        logits: ``(tokens, experts)`` float32 router logits.
        bias: ``(experts,)`` float32 routing bias.
        top: Experts per token.
        scale: Sum of one token's weights.

    Returns:
        ``(tokens, top)`` uint32 experts, best first, and their weights.
    """

    tokens, experts = (int(size) for size in logits.shape)
    chosen, weights = _route_kernel()(
        inputs=[logits, bias, _floats(scale)],
        template=[("EXPERTS", experts), ("TOP", top)],
        grid=(tokens, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(tokens, top), (tokens, top)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return chosen, weights


def blend_messages(
    hidden: mx.array, messages: mx.array, down: mx.array, up: mx.array
) -> mx.array:
    """Blend hidden states with messages through low-rank gates.

    Equals ``gate = concat(rms_norm(hidden), rms_norm(messages)) @ down @ up``
    with products in the weights' dtype, split into ``forget`` and ``keep``, and
    ``(1 - tanh(forget)) * hidden + tanh(keep) * messages``.

    Args:
        hidden: ``(gates, tokens, hidden)`` float32 states, or one state for
            all gates.
        messages: ``(gates, tokens, hidden)`` float32 messages; the width must
            be a multiple of 32.
        down: ``(gates, 2 * hidden, rank)`` weights.
        up: ``(gates, rank, 2 * hidden)`` weights.

    Returns:
        ``(gates, tokens, hidden)`` blended states.
    """

    gates, tokens, width = (int(size) for size in messages.shape)
    rank = int(down.shape[-1])
    shares = math.gcd(_GATE_SHARES, width // _SIMD_WIDTH, rank)
    simdgroups = min(_GATE_SIMDGROUPS, shares)
    pieces = shares // simdgroups
    threads = simdgroups * _SIMD_WIDTH
    down_kernel, up_kernel = _gate_kernels()
    launch: dict[str, Any] = {
        "template": [
            ("HIDDEN", width),
            ("RANK", rank),
            ("PIECES", pieces),
            ("SIMDGROUPS", simdgroups),
            ("G", down.dtype),
            ("SHARED_HIDDEN", int(hidden.shape[0]) != gates),
        ],
        "grid": (pieces * threads, gates * tokens, 1),
        "threadgroup": (threads, 1, 1),
    }
    low = down_kernel(
        inputs=[hidden, messages, down],
        output_shapes=[(gates, tokens, rank)],
        output_dtypes=[down.dtype],
        **launch,
    )[0]
    return up_kernel(
        inputs=[hidden, messages, low, up],
        output_shapes=[(gates, tokens, width)],
        output_dtypes=[mx.float32],
        **launch,
    )[0]


def row_products(x: mx.array, weight: mx.array) -> mx.array:
    """Return ``x @ weight.T`` with results that do not depend on other tokens.

    MLX uses different kernels for one token and for several, and their
    rounding differs. A step that verifies proposed tokens must compute what
    one-token steps compute.

    Args:
        x: ``(tokens, width)`` float32 inputs; the width must be a multiple of 32.
        weight: ``(rows, width)`` float32 weights.

    Returns:
        ``(tokens, rows)`` float32 products.
    """

    tokens, width = (int(size) for size in x.shape)
    rows = int(weight.shape[0])
    groups = (rows + _PRODUCT_ROWS - 1) // _PRODUCT_ROWS
    simdgroups = min(_PRODUCT_SIMDGROUPS, groups)
    pieces = (groups + simdgroups - 1) // simdgroups
    threads = simdgroups * _SIMD_WIDTH
    return _products_kernel()(
        inputs=[x, weight],
        template=[("WIDTH", width), ("GROUP", _PRODUCT_ROWS)],
        grid=(pieces * threads, tokens, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(tokens, rows)],
        output_dtypes=[mx.float32],
    )[0]
