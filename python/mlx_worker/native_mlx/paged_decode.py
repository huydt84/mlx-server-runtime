"""Metal kernel for decode attention read directly from the paged KV pool.

The kernel source is ``kernels/paged_decode.metal``.

A decode step attends one new query token per row to that row's cached keys.
Gathering every row's pages into dense K/V before SDPA copies the whole cache
for every layer of every step, which costs more than the attention itself once
contexts grow. This kernel instead follows each row's block table inside the
attention loop, so rows of different lengths need neither a copy nor padding.

One threadgroup handles one (row, KV head) pair and every query head that
shares that KV head, so grouped-query models read each key and value once.
Each simdgroup accumulates an online softmax over a strided subset of the
row's keys; the simdgroups are then merged through threadgroup memory.

Layers run one after another, so a step with few rows would otherwise occupy
only a handful of GPU cores. Each row's keys are therefore split into ranges
handled by separate threadgroups, and a second small kernel merges the
partial softmax results. Even at short contexts the extra parallelism costs
less than the merge launch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header

_SIMD_WIDTH = 32
_THREADGROUP_MEMORY_BYTES = 32 * 1024
_MAX_SIMDGROUPS = 8
_TARGET_THREADGROUPS = 32
_MIN_KEYS_PER_SPLIT = 32
_MAX_SPLITS = 32

_SOURCE = """
    threadgroup mlx_air::PagedDecodeScratch<
        SIMDGROUPS, QUERY_HEADS / KV_HEADS, HEAD_DIM> scratch;
    mlx_air::paged_decode_attention<
        Q, KV, QUERY_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE, SIMDGROUPS, SPLITS>(
        queries, key_pages, value_pages, block_tables, block_tables_shape[1],
        lengths, scale[0], window[0], out, partial_out, partial_max,
        partial_sum, scratch, threadgroup_position_in_grid.xy,
        simdgroup_index_in_threadgroup, thread_index_in_simdgroup,
        thread_position_in_threadgroup.x);
"""

_MERGE_SOURCE = """
    mlx_air::paged_decode_merge<Q, HEAD_DIM, SPLITS>(
        partial_out, partial_max, partial_sum, out,
        threadgroup_position_in_grid.x, thread_position_in_threadgroup.x);
"""


@lru_cache(maxsize=1)
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_paged_decode_attention",
        input_names=[
            "queries",
            "key_pages",
            "value_pages",
            "block_tables",
            "lengths",
            "scale",
            "window",
        ],
        output_names=["out", "partial_out", "partial_max", "partial_sum"],
        source=_SOURCE,
        header=metal_header("paged_decode.metal"),
    )


@lru_cache(maxsize=1)
def _merge_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_paged_decode_merge",
        input_names=["partial_out", "partial_max", "partial_sum"],
        output_names=["out"],
        source=_MERGE_SOURCE,
        header=metal_header("paged_decode.metal"),
    )


@lru_cache(maxsize=64)
def _scalar(value: float, dtype: Any) -> mx.array:
    return mx.array([value], dtype=dtype)


def simdgroups_for(query_heads: int, kv_heads: int, head_dim: int) -> int:
    """Return how many simdgroups fit the merge buffers in threadgroup memory.

    Returns 0 when the geometry is unsupported and callers must fall back.
    """

    if head_dim % _SIMD_WIDTH or kv_heads <= 0 or query_heads % kv_heads:
        return 0
    group = query_heads // kv_heads
    simdgroups = _MAX_SIMDGROUPS
    while simdgroups:
        merge_bytes = 4 * simdgroups * group * (head_dim + 2)
        if merge_bytes <= _THREADGROUP_MEMORY_BYTES:
            return simdgroups
        simdgroups //= 2
    return 0


def splits_for(rows: int, kv_heads: int, max_keys: int) -> int:
    """Return the power-of-two key-range split count for one decode step."""

    groups = max(1, rows * kv_heads)
    limit = min(
        _MAX_SPLITS,
        max(1, _TARGET_THREADGROUPS // groups),
        max(1, max_keys // _MIN_KEYS_PER_SPLIT),
    )
    splits = 1
    while splits * 2 <= limit:
        splits *= 2
    return splits


def warmup_key_counts(kv_heads: int) -> tuple[int, ...]:
    """Return one single-row decode length per split count a step can use.

    Every split count compiles its own kernel variant on first use. Decoding
    once at each returned length, which sits inside the range of keys that
    selects that split count, compiles them all. Steps with more rows use a
    subset of the same split counts.

    Args:
        kv_heads: KV heads per layer.

    Returns:
        Ascending key counts, starting with the single-split range.
    """

    counts: list[int] = []
    splits = 1
    while splits <= _MAX_SPLITS:
        keys = (_MIN_KEYS_PER_SPLIT * 3 // 2) * splits
        if splits_for(1, kv_heads, keys) != splits:
            break
        counts.append(keys)
        splits *= 2
    return tuple(counts)


def attention_template(
    queries: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    *,
    simdgroups: int,
    splits: int,
) -> list[tuple[str, Any]]:
    """Return the template arguments of the attention kernel for one launch.

    The query (and output) dtype ``Q`` and the KV pool dtype ``KV`` are bound
    separately: a model's activations can be float16 while its cache is
    bfloat16, and each pairing compiles its own kernel variant.

    Args:
        queries: ``(rows, query_heads, 1, head_dim)`` decode queries.
        key_pages: ``(pages, page_size, kv_heads, head_dim)`` key pool.
        value_pages: Value pool; must share the key pool's dtype.
        simdgroups: Simdgroups per threadgroup, from ``simdgroups_for``.
        splits: Key-range splits per row, from ``splits_for``.

    Returns:
        ``(name, value)`` pairs for ``mx.fast.metal_kernel``'s ``template``.

    Raises:
        ValueError: If the key and value pools have different dtypes.
    """

    if value_pages.dtype != key_pages.dtype:
        raise ValueError(
            "paged decode attention needs key and value pools of one dtype, "
            f"got {key_pages.dtype} and {value_pages.dtype}"
        )
    _, query_heads, _, head_dim = (int(size) for size in queries.shape)
    _, page_size, kv_heads, _ = (int(size) for size in key_pages.shape)
    return [
        ("Q", queries.dtype),
        ("KV", key_pages.dtype),
        ("QUERY_HEADS", query_heads),
        ("KV_HEADS", kv_heads),
        ("HEAD_DIM", head_dim),
        ("PAGE_SIZE", page_size),
        ("SIMDGROUPS", simdgroups),
        ("SPLITS", splits),
    ]


def paged_decode_attention(
    queries: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_tables: mx.array,
    lengths: mx.array,
    *,
    max_length: int,
    scale: float,
    window_size: int | None = None,
) -> mx.array:
    """Attend one query token per row to its paged keys and values.

    Args:
        queries: ``(rows, query_heads, 1, head_dim)`` decode queries.
        key_pages: ``(pages, page_size, kv_heads, head_dim)`` key pool, already
            holding this step's keys.
        value_pages: Value pool with the same shape and dtype as
            ``key_pages``. The pool dtype may differ from the query dtype.
        block_tables: ``(rows, width)`` int32 page ids; entries past a row's
            pages are ignored.
        lengths: ``(rows,)`` int32 key count per row, including this step.
        max_length: Largest entry of ``lengths``, used to size the launch.
        scale: Softmax scale applied to the query.
        window_size: Optional sliding window; only the last ``window_size``
            keys of each row are attended.

    Returns:
        ``(rows, query_heads, 1, head_dim)`` attention output in the query
        dtype.

    Raises:
        ValueError: If the head geometry is unsupported or the key and value
            pools have different dtypes.
    """

    rows, query_heads, query_tokens, head_dim = (int(size) for size in queries.shape)
    _, _, kv_heads, pool_head_dim = (int(size) for size in key_pages.shape)
    simdgroups = simdgroups_for(query_heads, kv_heads, head_dim)
    if query_tokens != 1 or pool_head_dim != head_dim or not simdgroups:
        raise ValueError("paged decode attention geometry is unsupported")
    max_keys = min(max_length, window_size) if window_size else max_length
    splits = splits_for(rows, kv_heads, max_keys)
    partial_rows = rows * query_heads * splits if splits > 1 else 1
    out, partial_out, partial_max, partial_sum = _kernel()(
        inputs=[
            queries,
            key_pages,
            value_pages,
            block_tables,
            lengths,
            _scalar(float(scale), mx.float32),
            _scalar(int(window_size or 0), mx.int32),
        ],
        template=attention_template(
            queries,
            key_pages,
            value_pages,
            simdgroups=simdgroups,
            splits=splits,
        ),
        grid=(rows * kv_heads * simdgroups * _SIMD_WIDTH, splits, 1),
        threadgroup=(simdgroups * _SIMD_WIDTH, 1, 1),
        output_shapes=[
            queries.shape,
            (partial_rows * (head_dim if splits > 1 else 1),),
            (partial_rows,),
            (partial_rows,),
        ],
        output_dtypes=[queries.dtype, mx.float32, mx.float32, mx.float32],
    )
    if splits == 1:
        return out
    return _merge_kernel()(
        inputs=[partial_out, partial_max, partial_sum],
        template=[("Q", queries.dtype), ("HEAD_DIM", head_dim), ("SPLITS", splits)],
        grid=(rows * query_heads * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[queries.dtype],
    )[0]
