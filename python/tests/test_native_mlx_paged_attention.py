"""Parity of the paged attention mask and gather fast paths with dense SDPA."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_worker.native_mlx.attention import (
    DenseReferenceAttentionBackend,
    PagedMetalAttentionBackend,
)
from mlx_worker.native_mlx.cache import DenseKVCacheBackend, PagedKVCacheBackend
from mlx_worker.native_mlx.interfaces import ForwardMode

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="paged attention requires MLX Metal"
)

_QUERY_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8


def _padded(rows: list[mx.array], width: int) -> mx.array:
    return mx.stack(
        [
            mx.pad(row, [(0, 0), (0, width - int(row.shape[1])), (0, 0)])
            for row in rows
        ]
    )


def _attend(backend, contexts_for, caches, token_lengths, mask, seed):
    """Run one step on ``caches`` and return the output plus its inputs."""

    mx.random.seed(seed)
    width = max(token_lengths)
    queries = _padded(
        [
            mx.random.normal((_QUERY_HEADS, length, _HEAD_DIM)).astype(mx.float16)
            for length in token_lengths
        ],
        width,
    )
    keys = _padded(
        [
            mx.random.normal((_KV_HEADS, length, _HEAD_DIM)).astype(mx.float16)
            for length in token_lengths
        ],
        width,
    )
    values = _padded(
        [
            mx.random.normal((_KV_HEADS, length, _HEAD_DIM)).astype(mx.float16)
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
        queries, keys, values, scale=_HEAD_DIM**-0.5, mask=mask
    )
    mx.eval(output)
    reservation.commit()
    return output


def _run(backend, contexts_for, cache_lengths, token_lengths, mask):
    caches = []
    for index, cache_length in enumerate(cache_lengths):
        handle = backend.create(f"row-{index}")
        cache = backend.get(handle, f"row-{index}")
        if cache_length:
            _attend(backend, contexts_for, [cache], [cache_length], mask, 100 + index)
        caches.append(cache)
    return _attend(backend, contexts_for, caches, token_lengths, mask, 7)


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
    ],
)
def test_paged_attention_fast_paths_match_dense_reference(
    cache_lengths: tuple[int, ...],
    token_lengths: tuple[int, ...],
    mask: str,
) -> None:
    paged_backend = PagedKVCacheBackend(1, _KV_HEADS, _HEAD_DIM, 8, 1 << 20)
    paged = _run(
        paged_backend,
        PagedMetalAttentionBackend().contexts,
        cache_lengths,
        token_lengths,
        mask,
    )
    dense = _run(
        DenseKVCacheBackend(num_layers=1),
        DenseReferenceAttentionBackend().contexts,
        cache_lengths,
        token_lengths,
        mask,
    )
    for row, length in enumerate(token_lengths):
        assert mx.allclose(
            paged[row, :, :length, :].astype(mx.float32),
            dense[row, :, :length, :].astype(mx.float32),
            atol=2e-2,
        ).item(), (row, length)


def test_paged_attention_builds_one_mask_per_step() -> None:
    backend = PagedKVCacheBackend(3, _KV_HEADS, _HEAD_DIM, 8, 1 << 20)
    caches = tuple(
        backend.get(backend.create(f"row-{index}"), f"row-{index}")
        for index in range(2)
    )
    reservation = backend.reserve_batch(caches, (2, 3))
    contexts = PagedMetalAttentionBackend().contexts(reservation, ForwardMode.PREFILL)
    assert len({id(context.step_masks) for context in contexts}) == 1
    reservation.abort()
