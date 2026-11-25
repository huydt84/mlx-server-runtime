"""Parity of the paged attention fast paths and decode kernel with dense SDPA."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.attention import (
    DenseReferenceAttentionBackend,
    PagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.cache import DenseKVCacheBackend, PagedKVCacheBackend
from mlx_worker.native_mlx.interfaces import ForwardMode
from mlx_worker.native_mlx.paged_decode import (
    paged_decode_attention,
    simdgroups_for,
    splits_for,
)

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="paged attention requires MLX Metal"
)

_QUERY_HEADS = 4
_KV_HEADS = 2


def _padded(rows: list[mx.array], width: int) -> mx.array:
    return mx.stack(
        [mx.pad(row, [(0, 0), (0, width - int(row.shape[1])), (0, 0)]) for row in rows]
    )


def _attend(backend, contexts_for, caches, token_lengths, mask, seed, head_dim):
    """Run one step on ``caches`` and return the attention output."""

    mx.random.seed(seed)
    width = max(token_lengths)
    queries = _padded(
        [
            mx.random.normal((_QUERY_HEADS, length, head_dim)).astype(mx.float16)
            for length in token_lengths
        ],
        width,
    )
    keys = _padded(
        [
            mx.random.normal((_KV_HEADS, length, head_dim)).astype(mx.float16)
            for length in token_lengths
        ],
        width,
    )
    values = _padded(
        [
            mx.random.normal((_KV_HEADS, length, head_dim)).astype(mx.float16)
            for length in token_lengths
        ],
        width,
    )
    reservation = backend.reserve_batch(tuple(caches), tuple(token_lengths))
    mode = ForwardMode.from_phases(
        tuple("decode" if length == 1 else "prefill" for length in token_lengths)
    )
    (context,) = contexts_for(reservation, mode)
    output = context.append_and_attend(
        queries, keys, values, scale=head_dim**-0.5, mask=mask
    )
    mx.eval(output)
    reservation.commit()
    return output


def _run(backend, contexts_for, cache_lengths, token_lengths, mask, head_dim):
    caches = []
    for index, cache_length in enumerate(cache_lengths):
        handle = backend.create(f"row-{index}")
        cache = backend.get(handle, f"row-{index}")
        if cache_length:
            _attend(
                backend,
                contexts_for,
                [cache],
                [cache_length],
                mask,
                100 + index,
                head_dim,
            )
        caches.append(cache)
    return _attend(backend, contexts_for, caches, token_lengths, mask, 7, head_dim)


@pytest.mark.parametrize("head_dim", [8, 32])
@pytest.mark.parametrize(
    ("cache_lengths", "token_lengths", "mask"),
    [
        ((0,), (5,), "causal"),
        ((7,), (4,), "causal"),
        ((6,), (1,), "causal"),
        ((6, 6), (1, 1), "causal"),
        ((6, 9), (1, 1), "causal"),
        ((6, 3), (1, 4), "causal"),
        ((0,), (6,), "sliding_window:3"),
        ((6, 2), (2, 3), "sliding_window:3"),
        ((20, 5, 9), (1, 1, 1), "causal"),
        ((20, 5, 9), (1, 1, 1), "sliding_window:4"),
    ],
)
def test_paged_attention_fast_paths_match_dense_reference(
    cache_lengths: tuple[int, ...],
    token_lengths: tuple[int, ...],
    mask: str,
    head_dim: int,
) -> None:
    paged_backend = PagedKVCacheBackend(1, _KV_HEADS, head_dim, 8, 1 << 20)
    paged = _run(
        paged_backend,
        PagedMetalAttentionBackend().contexts,
        cache_lengths,
        token_lengths,
        mask,
        head_dim,
    )
    dense = _run(
        DenseKVCacheBackend(num_layers=1),
        DenseReferenceAttentionBackend().contexts,
        cache_lengths,
        token_lengths,
        mask,
        head_dim,
    )
    for row, length in enumerate(token_lengths):
        assert mx.allclose(
            paged[row, :, :length, :].astype(mx.float32),
            dense[row, :, :length, :].astype(mx.float32),
            atol=2e-2,
        ).item(), (row, length)


def test_paged_attention_builds_one_mask_per_step() -> None:
    backend = PagedKVCacheBackend(3, _KV_HEADS, 8, 8, 1 << 20)
    caches = tuple(
        backend.get(backend.create(f"row-{index}"), f"row-{index}")
        for index in range(2)
    )
    reservation = backend.reserve_batch(caches, (2, 3))
    contexts = PagedMetalAttentionBackend().contexts(reservation, ForwardMode.PREFILL)
    assert len({id(context.step_masks) for context in contexts}) == 1
    reservation.abort()


@pytest.mark.parametrize(
    ("query_dtype", "kv_dtype"),
    [
        (mx.float16, mx.float16),
        (mx.bfloat16, mx.bfloat16),
        (mx.float16, mx.bfloat16),
        (mx.bfloat16, mx.float16),
    ],
)
@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "head_dim"), [(4, 2, 64), (7, 1, 128), (4, 1, 256)]
)
@pytest.mark.parametrize("window_size", [None, 5, 300])
@pytest.mark.parametrize("lengths", [[1, 4, 13, 40], [70], [700]])
def test_paged_decode_kernel_matches_sdpa_over_scattered_pages(
    query_dtype,
    kv_dtype,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    window_size: int | None,
    lengths: list[int],
) -> None:
    page_size, pages = 4, 200
    mx.random.seed(3)
    key_pages = mx.random.normal((pages, page_size, kv_heads, head_dim)).astype(
        kv_dtype
    )
    value_pages = mx.random.normal((pages, page_size, kv_heads, head_dim)).astype(
        kv_dtype
    )
    order = [int(page) for page in mx.random.permutation(pages).tolist()]
    tables: list[list[int]] = []
    for length in lengths:
        count = -(-length // page_size)
        tables.append(order[:count])
        order = order[count:]
    width = max(len(table) for table in tables)
    queries = mx.random.normal((len(lengths), query_heads, 1, head_dim)).astype(
        query_dtype
    )

    output = paged_decode_attention(
        queries,
        key_pages,
        value_pages,
        mx.array([table + [0] * (width - len(table)) for table in tables], mx.int32),
        mx.array(lengths, dtype=mx.int32),
        max_length=max(lengths),
        scale=head_dim**-0.5,
        window_size=window_size,
    )

    for row, (length, table) in enumerate(zip(lengths, tables, strict=True)):
        start = max(0, length - window_size) if window_size else 0

        def dense(pool, table=table, length=length, start=start):
            rows = pool[mx.array(table)].reshape(-1, kv_heads, head_dim)[:length]
            return rows[start:].transpose(1, 0, 2)[None].astype(mx.float32)

        expected = mx.fast.scaled_dot_product_attention(
            queries[row : row + 1].astype(mx.float32),
            dense(key_pages),
            dense(value_pages),
            scale=head_dim**-0.5,
        )
        assert output.dtype == query_dtype
        assert mx.allclose(
            output[row : row + 1].astype(mx.float32),
            expected,
            atol=2e-2,
        ).item(), (row, length)


def test_paged_decode_kernel_rejects_unsupported_geometry() -> None:
    assert simdgroups_for(4, 2, 8) == 0
    assert simdgroups_for(6, 4, 64) == 0
    assert simdgroups_for(32, 8, 128) == 8
    assert simdgroups_for(4, 1, 256) == 4


def test_paged_decode_splits_rows_only_when_few_threadgroups_run() -> None:
    assert splits_for(rows=1, kv_heads=8, max_keys=40) == 1
    assert splits_for(rows=1, kv_heads=8, max_keys=64) == 2
    assert splits_for(rows=1, kv_heads=8, max_keys=4096) == 4
    assert splits_for(rows=2, kv_heads=8, max_keys=4096) == 2
    assert splits_for(rows=4, kv_heads=8, max_keys=4096) == 1
    assert splits_for(rows=1, kv_heads=1, max_keys=100_000) == 32


def test_paged_reservation_copies_consecutive_slots_as_one_slice() -> None:
    backend = PagedKVCacheBackend(1, _KV_HEADS, 32, 8, 1 << 20)
    first = backend.get(backend.create("first"), "first")
    second = backend.get(backend.create("second"), "second")

    reservation = backend.reserve_batch((first,), (12,))
    assert reservation._slot_start == 0
    reservation.abort()

    reservation = backend.reserve_batch((first, second), (1, 1))
    assert reservation._slot_start is None
    reservation.abort()
