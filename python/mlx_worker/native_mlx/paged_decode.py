"""Metal kernel for decode attention read directly from the paged KV pool.

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
only a handful of GPU cores. Long rows are therefore split into key ranges
handled by separate threadgroups, and a second small kernel merges the
partial softmax results.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import mlx.core as mx

_SIMD_WIDTH = 32
_THREADGROUP_MEMORY_BYTES = 32 * 1024
_MAX_SIMDGROUPS = 8
# Split a row's keys until a step launches about this many threadgroups...
_TARGET_THREADGROUPS = 32
# ...but give every split at least this many keys.
_MIN_KEYS_PER_SPLIT = 256
_MAX_SPLITS = 32

_SOURCE = """
    constexpr int EPL = HEAD_DIM / 32;
    constexpr int GQA = QUERY_HEADS / KV_HEADS;
    uint row = threadgroup_position_in_grid.x / KV_HEADS;
    uint kv_head = threadgroup_position_in_grid.x % KV_HEADS;
    uint split = threadgroup_position_in_grid.y;
    uint lane = thread_index_in_simdgroup;
    uint simd = simdgroup_index_in_threadgroup;
    uint tid = thread_position_in_threadgroup.x;

    threadgroup float maxes[SIMDGROUPS * GQA];
    threadgroup float sums[SIMDGROUPS * GQA];
    threadgroup float outputs[SIMDGROUPS * GQA * HEAD_DIM];

    float qv[GQA][EPL];
    float o[GQA][EPL];
    float max_score[GQA];
    float sum_exp[GQA];
    for (int g = 0; g < GQA; g++) {
        const device Q* q = queries
            + (row * QUERY_HEADS + kv_head * GQA + g) * HEAD_DIM + lane * EPL;
        for (int i = 0; i < EPL; i++) {
            qv[g][i] = static_cast<float>(q[i]) * scale[0];
            o[g][i] = 0.0f;
        }
        max_score[g] = -INFINITY;
        sum_exp[g] = 0.0f;
    }

    int length = lengths[row];
    int start = window[0] > 0 ? max(0, length - window[0]) : 0;
    int span = (length - start + SPLITS - 1) / SPLITS;
    int split_start = start + int(split) * span;
    int split_end = min(length, split_start + span);
    // Single-element inputs arrive in the constant address space.
    auto table = block_tables + row * block_tables_shape[1];
    for (int t = split_start + int(simd); t < split_end; t += SIMDGROUPS) {
        uint page = uint(table[t / PAGE_SIZE]);
        uint base = ((page * PAGE_SIZE + uint(t % PAGE_SIZE)) * KV_HEADS + kv_head)
            * HEAD_DIM + lane * EPL;
        float k[EPL];
        float v[EPL];
        for (int i = 0; i < EPL; i++) {
            k[i] = static_cast<float>(key_pages[base + i]);
            v[i] = static_cast<float>(value_pages[base + i]);
        }
        for (int g = 0; g < GQA; g++) {
            float score = 0.0f;
            for (int i = 0; i < EPL; i++) {
                score += qv[g][i] * k[i];
            }
            score = simd_sum(score);
            float new_max = max(max_score[g], score);
            float factor = fast::exp(max_score[g] - new_max);
            float weight = fast::exp(score - new_max);
            max_score[g] = new_max;
            sum_exp[g] = sum_exp[g] * factor + weight;
            for (int i = 0; i < EPL; i++) {
                o[g][i] = o[g][i] * factor + weight * v[i];
            }
        }
    }

    for (int g = 0; g < GQA; g++) {
        if (lane == 0) {
            maxes[g * SIMDGROUPS + simd] = max_score[g];
            sums[g * SIMDGROUPS + simd] = sum_exp[g];
        }
        for (int i = 0; i < EPL; i++) {
            outputs[(simd * GQA + g) * HEAD_DIM + lane * EPL + i] = o[g][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint first_head = row * QUERY_HEADS + kv_head * GQA;
    for (uint e = tid; e < GQA * HEAD_DIM; e += SIMDGROUPS * 32) {
        uint g = e / HEAD_DIM;
        float group_max = -INFINITY;
        for (int s = 0; s < SIMDGROUPS; s++) {
            group_max = max(group_max, maxes[g * SIMDGROUPS + s]);
        }
        float total = 0.0f;
        float acc = 0.0f;
        for (int s = 0; s < SIMDGROUPS; s++) {
            float m = maxes[g * SIMDGROUPS + s];
            float factor = m == -INFINITY ? 0.0f : fast::exp(m - group_max);
            total += sums[g * SIMDGROUPS + s] * factor;
            acc += outputs[(s * GQA + g) * HEAD_DIM + (e % HEAD_DIM)] * factor;
        }
        if (SPLITS == 1) {
            out[first_head * HEAD_DIM + e] =
                static_cast<Q>(total > 0.0f ? acc / total : 0.0f);
        } else {
            // Unnormalized partial result, scaled to this split's maximum.
            uint head = first_head + g;
            partial_out[(head * SPLITS + split) * HEAD_DIM + e % HEAD_DIM] = acc;
            if (e % HEAD_DIM == 0) {
                partial_max[head * SPLITS + split] = group_max;
                partial_sum[head * SPLITS + split] = total;
            }
        }
    }
"""

_MERGE_SOURCE = """
    uint head = threadgroup_position_in_grid.x;
    uint dim = thread_position_in_threadgroup.x;
    float global_max = -INFINITY;
    for (int s = 0; s < SPLITS; s++) {
        global_max = max(global_max, partial_max[head * SPLITS + s]);
    }
    float total = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < SPLITS; s++) {
        float m = partial_max[head * SPLITS + s];
        float factor = m == -INFINITY ? 0.0f : fast::exp(m - global_max);
        total += partial_sum[head * SPLITS + s] * factor;
        acc += partial_out[(head * SPLITS + s) * HEAD_DIM + dim] * factor;
    }
    out[head * HEAD_DIM + dim] = static_cast<Q>(total > 0.0f ? acc / total : 0.0f);
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
    )


@lru_cache(maxsize=1)
def _merge_kernel() -> Any:
    return mx.fast.metal_kernel(
        name="native_paged_decode_merge",
        input_names=["partial_out", "partial_max", "partial_sum"],
        output_names=["out"],
        source=_MERGE_SOURCE,
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
        value_pages: Value pool with the same shape as ``key_pages``.
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
        ValueError: If the head geometry is unsupported.
    """

    rows, query_heads, query_tokens, head_dim = (int(size) for size in queries.shape)
    _, page_size, kv_heads, pool_head_dim = (int(size) for size in key_pages.shape)
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
        template=[
            ("Q", queries.dtype),
            ("QUERY_HEADS", query_heads),
            ("KV_HEADS", kv_heads),
            ("HEAD_DIM", head_dim),
            ("PAGE_SIZE", page_size),
            ("SIMDGROUPS", simdgroups),
            ("SPLITS", splits),
        ],
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
