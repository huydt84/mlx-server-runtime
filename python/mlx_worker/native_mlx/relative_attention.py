"""Metal kernel for relative-bias attention with sinks over paged KV.

The kernel source is ``kernels/relative_attention.metal``. It extends the paged
decode kernel with a learned bias per key distance, one sink logit per query
head, any number of query tokens, and rows whose oldest pages were released.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

from .metal_source import metal_header
from .paged_decode import (
    _MIN_KEYS_PER_SPLIT,
    _merge_kernel,
    _scalar,
    simdgroups_for,
    splits_for,
)

_SIMD_WIDTH = 32
_EXACT_TOKENS = 8

_SOURCE = """
    threadgroup mlx_air::PagedDecodeScratch<
        SIMDGROUPS, QUERY_HEADS / KV_HEADS, HEAD_DIM> scratch;
    mlx_air::relative_paged_attention<
        P, QUERY_HEADS, KV_HEADS, HEAD_DIM, PAGE_SIZE, REL_DIM, EXTENT,
        SIMDGROUPS, SPLITS, MIN_KEYS>(
        queries, pages, block_tables, block_tables_shape[1], bases, token_rows,
        positions, rel, rel_proj, sinks, scale[0], window[0], out, partial_out, partial_max, partial_sum, scratch,
        threadgroup_position_in_grid.xy, simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup, thread_position_in_threadgroup.x);
"""


@lru_cache(maxsize=1)
def _kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_relative_paged_attention",
        input_names=[
            "queries",
            "pages",
            "block_tables",
            "bases",
            "token_rows",
            "positions",
            "rel",
            "rel_proj",
            "sinks",
            "scale",
            "window",
        ],
        output_names=["out", "partial_out", "partial_max", "partial_sum"],
        source=_SOURCE,
        header=metal_header("relative_attention.metal"),
    )


def relative_paged_attention(
    queries: mx.array,
    pages: mx.array,
    block_tables: mx.array,
    bases: mx.array,
    token_rows: mx.array,
    positions: mx.array,
    rel: mx.array,
    rel_proj: mx.array,
    sinks: mx.array,
    *,
    max_length: int,
    scale: float,
    window_size: int | None = None,
) -> mx.array:
    """Attend query tokens to paged keys with a distance bias and a sink.

    Args:
        queries: ``(tokens, query_heads, head_dim)`` float32 queries of all rows.
        pages: ``(pages, page_size, 2, kv_heads, head_dim)`` pool holding keys
            then values, including this step's append.
        block_tables: ``(rows, width)`` int32 page ids.
        bases: ``(rows,)`` int32 position of each table's first entry.
        token_rows: ``(tokens,)`` int32 row of each query token.
        positions: ``(tokens,)`` int32 position of each query token.
        rel: ``(tokens, query_heads, rel_dim)`` float32 distance states.
        rel_proj: ``(extent, rel_dim)`` float32 distance embeddings.
        sinks: ``(tokens, query_heads)`` float32 sink logits.
        max_length: Largest key count one query can attend to.
        scale: Factor applied to the query-key product.
        window_size: Optional sliding window.

    Returns:
        ``(tokens, query_heads * head_dim)`` float32 attention output.

    Raises:
        ValueError: If the geometry is unsupported.
    """

    tokens, query_heads, head_dim = (int(size) for size in queries.shape)
    _, page_size, kinds, kv_heads, pool_head_dim = (int(size) for size in pages.shape)
    extent, rel_dim = (int(size) for size in rel_proj.shape)
    simdgroups = simdgroups_for(query_heads, kv_heads, head_dim)
    if kinds != 2 or pool_head_dim != head_dim or not simdgroups:
        raise ValueError("relative paged attention geometry is unsupported")
    if not 0 < rel_dim <= _SIMD_WIDTH:
        raise ValueError("relative paged attention distance width is unsupported")
    max_keys = min(max_length, window_size) if window_size else max_length
    split_rows = 1 if tokens <= _EXACT_TOKENS else tokens
    splits = splits_for(split_rows, kv_heads, max_keys)
    partial_rows = tokens * query_heads * splits if splits > 1 else 1
    out_shape = (tokens, query_heads * head_dim)
    out, partial_out, partial_max, partial_sum = _kernel()(
        inputs=[
            queries,
            pages,
            block_tables,
            bases,
            token_rows,
            positions,
            rel,
            rel_proj,
            sinks,
            _scalar(float(scale), mx.float32),
            _scalar(int(window_size or 0), mx.int32),
        ],
        template=[
            ("P", pages.dtype),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
            ("HEAD_DIM", head_dim),
            ("PAGE_SIZE", page_size),
            ("REL_DIM", rel_dim),
            ("EXTENT", extent),
            ("SIMDGROUPS", simdgroups),
            ("SPLITS", splits),
            ("MIN_KEYS", _MIN_KEYS_PER_SPLIT),
        ],
        grid=(tokens * kv_heads * simdgroups * _SIMD_WIDTH, splits, 1),
        threadgroup=(simdgroups * _SIMD_WIDTH, 1, 1),
        output_shapes=[
            out_shape,
            (partial_rows * (head_dim if splits > 1 else 1),),
            (partial_rows,),
            (partial_rows,),
        ],
        output_dtypes=[mx.float32] * 4,
    )
    if splits == 1:
        return out
    return _merge_kernel()(
        inputs=[partial_out, partial_max, partial_sum],
        template=[("Q", mx.float32), ("HEAD_DIM", head_dim), ("SPLITS", splits)],
        grid=(tokens * query_heads * head_dim, 1, 1),
        threadgroup=(head_dim, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[mx.float32],
    )[0]
