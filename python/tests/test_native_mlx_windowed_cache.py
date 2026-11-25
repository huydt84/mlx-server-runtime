"""Two-pool windowed KV cache ownership and transactions."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_worker.native_mlx.cache import KVCacheGeometry
from mlx_worker.native_mlx.execution_backends import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    build_native_execution_backend,
)
from mlx_worker.native_mlx.windowed_attention import WindowedMetalAttentionBackend
from mlx_worker.native_mlx.windowed_cache import (
    MAX_PARTIAL_TOKENS,
    WindowedBatchReservation,
    WindowedPagedKVCacheBackend,
)

_LAYERS, _HEADS, _DIM, _PAGE, _WINDOW = 3, 2, 4, 8, 16
_CHANNELS = 5


def _backend(pages: int = 24) -> WindowedPagedKVCacheBackend:
    layer_page = 2 * _PAGE * _HEADS * _DIM * 2
    return WindowedPagedKVCacheBackend(
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=_DIM,
        page_size=_PAGE,
        budget_bytes=pages * 3 * layer_page,
        window_layers=frozenset((0, 2)),
        window_size=_WINDOW,
        max_append_tokens=8,
    )


def _stage(reservation: WindowedBatchReservation, value: float) -> None:
    """Stage recognizable keys and convolution inputs for every layer."""

    tokens = reservation.layout.tokens
    for layer in range(_LAYERS):
        kv = mx.broadcast_to(
            value + mx.arange(tokens)[:, None, None, None], (tokens, 2, _HEADS, _DIM)
        )
        reservation.stage_layer(layer, kv.astype(mx.float16))
        before = reservation.conv_history(layer, 0, _CHANNELS)
        inputs = mx.broadcast_to(
            value + mx.arange(tokens)[:, None], (tokens, _CHANNELS)
        ).astype(mx.float32)
        after = []
        start = 0
        for row, count in enumerate(reservation.append_lengths):
            joined = mx.concatenate([before[row], inputs[start : start + count]])
            after.append(joined[-3:])
            start += count
        reservation.stage_conv(layer, 0, before, inputs, mx.stack(after))


def _append(
    backend: WindowedPagedKVCacheBackend,
    handle: str,
    tokens: int,
    value: float,
    accepted: int | None = None,
) -> None:
    cache = backend.get(handle, _request_of(handle))
    reservation = backend.reserve_batch((cache,), (tokens,))
    _stage(reservation, value)
    reservation.commit(accepted=None if accepted is None else (accepted,))


def _request_of(handle: str) -> str:
    return handle.removeprefix("paged-windowed-kv-").split("-")[0]


def _stored(backend: WindowedPagedKVCacheBackend, handle: str, layer: int) -> list:
    """Return the first key element of every position the layer still holds."""

    cache = backend.get(handle, _request_of(handle))
    windowed = layer in backend.window_layers
    table = cache.window_table if windowed else cache.block_table
    base = cache.window_base if windowed else 0
    pages = np.array(backend.kv_pages[layer][:, :, 0, 0, 0])
    return [
        float(pages[table[(position - base) // _PAGE], (position - base) % _PAGE])
        for position in range(base, cache.length)
    ]


def test_window_pages_are_returned_once_no_query_can_reach_them() -> None:
    backend = _backend()
    handle = backend.create("a")
    for step in range(8):
        _append(backend, handle, 7, 100.0 * step)
    cache = backend.get(handle, "a")

    assert cache.length == 56
    assert len(cache.block_table) == 7
    assert cache.window_base == 40
    assert len(cache.window_table) == 2
    expected = [100.0 * step + token for step in range(8) for token in range(7)]
    assert _stored(backend, handle, 1) == expected
    assert _stored(backend, handle, 0) == expected[40:]
    metrics = backend.metrics()
    assert metrics["used_pages"] == 7
    assert metrics["window_used_pages"] == 2
    assert metrics["active_kv_bytes"] == (
        7 * backend.bytes_per_page + 2 * backend.window_bytes_per_page
    )


def test_release_returns_the_pages_of_both_pools() -> None:
    backend = _backend()
    handle = backend.create("a")
    _append(backend, handle, 8, 1.0)
    _append(backend, handle, 8, 2.0)

    backend.release(handle)
    backend.release(handle)

    assert backend.free_page_count() == backend.num_pages
    assert backend.metrics()["window_used_pages"] == 0
    with pytest.raises(ValueError, match="invalid cache handle"):
        backend.get(handle, "a")


def test_abort_leaves_pages_lengths_and_history_unchanged() -> None:
    backend = _backend()
    handle = backend.create("a")
    _append(backend, handle, 6, 1.0)
    cache = backend.get(handle, "a")
    history = np.array(cache.conv_states[(0, 0)].value())
    free = backend.free_page_count()

    reservation = backend.reserve_batch((cache,), (8,))
    _stage(reservation, 50.0)
    reservation.abort()
    reservation.abort()

    assert cache.length == 6
    assert backend.free_page_count() == free
    assert _stored(backend, handle, 0) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    np.testing.assert_array_equal(np.array(cache.conv_states[(0, 0)].value()), history)
    with pytest.raises(ValueError, match="already finished"):
        reservation.commit()


@pytest.mark.parametrize("accepted", [0, 1, 2, 5])
def test_commit_keeps_only_accepted_tokens(accepted: int) -> None:
    backend = _backend()
    handle = backend.create("a")
    _append(backend, handle, 6, 1.0)

    _append(backend, handle, 5, 50.0, accepted=accepted)

    cache = backend.get(handle, "a")
    tokens = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tokens += [50.0 + index for index in range(accepted)]
    assert cache.length == 6 + accepted
    assert _stored(backend, handle, 0) == tokens
    assert _stored(backend, handle, 1) == tokens
    assert len(cache.block_table) == -(-cache.length // _PAGE)
    history = np.array(cache.conv_states[(2, 0)].value())[:, 0]
    assert history.tolist() == tokens[-3:]
    _append(backend, handle, 2, 80.0)
    assert _stored(backend, handle, 1) == [*tokens, 80.0, 81.0]


def test_commit_rejects_counts_outside_the_append_and_missing_layers() -> None:
    backend = _backend()
    cache = backend.get(backend.create("a"), "a")
    reservation = backend.reserve_batch((cache,), (3,))
    with pytest.raises(ValueError, match="missing one or more layers"):
        reservation.commit()
    _stage(reservation, 1.0)
    with pytest.raises(ValueError, match="staged more than once"):
        reservation.stage_layer(0, mx.zeros((3, 2, _HEADS, _DIM)))
    for accepted in ((4,), (-1,), (1, 1)):
        with pytest.raises(ValueError, match="outside the staged append"):
            reservation.commit(accepted=accepted)
    assert reservation.commit() == (3,)

    long = backend.reserve_batch((cache,), (MAX_PARTIAL_TOKENS + 1,))
    _stage(long, 1.0)
    with pytest.raises(ValueError, match="only short steps"):
        long.commit(accepted=(4,))
    assert long.commit() == (3 + MAX_PARTIAL_TOKENS + 1,)


def test_batched_rows_keep_separate_pages_and_histories() -> None:
    backend = _backend()
    first, second = backend.create("a"), backend.create("b")
    _append(backend, first, 5, 1.0)
    caches = (backend.get(second, "b"), backend.get(first, "a"))

    reservation = backend.reserve_batch(caches, (4, 3))
    _stage(reservation, 10.0)
    assert reservation.commit() == (4, 8)

    assert _stored(backend, second, 0) == [10.0, 11.0, 12.0, 13.0]
    assert _stored(backend, first, 1) == [1.0, 2.0, 3.0, 4.0, 5.0, 14.0, 15.0, 16.0]
    assert np.array(caches[0].conv_states[(0, 0)].value())[:, 0].tolist() == [
        11.0,
        12.0,
        13.0,
    ]
    again = backend.reserve_batch(caches, (1, 1))
    assert again.conv_history(0, 0, _CHANNELS) is caches[0].conv_states[(0, 0)].batch
    again.abort()
    swapped = backend.reserve_batch((caches[1], caches[0]), (1, 1))
    assert np.array(swapped.conv_history(0, 0, _CHANNELS))[:, 2, 0].tolist() == [
        16.0,
        13.0,
    ]
    swapped.abort()


def test_preflight_reports_which_pool_is_exhausted() -> None:
    backend = _backend()
    caches = [backend.get(backend.create(name), name) for name in "abcdefgh"]
    window = backend.num_window_pages

    results = backend.preflight(tuple(caches), (8 * window,) * 8)

    assert results[0] is None
    assert "windowed KV capacity" in str(results[1])
    assert backend.metrics()["allocation_failures"] == 7
    too_long = backend.preflight((caches[0],), (_PAGE * (backend.num_pages + 1),))
    assert "paged KV capacity" in str(too_long[0])
    with pytest.raises(ValueError, match="capacity exhausted"):
        backend.reserve_batch((caches[0],), (_PAGE * (backend.num_pages + 1),))
    assert backend.free_page_count() == backend.num_pages


def test_reservations_validate_their_requests() -> None:
    backend = _backend()
    cache = backend.get(backend.create("a"), "a")
    for caches, lengths, message in (
        ((), (), "matching requests"),
        ((cache,), (1, 1), "matching requests"),
        ((cache, cache), (1, 1), "duplicate handles"),
        ((cache,), (0,), "must be positive"),
    ):
        with pytest.raises(ValueError, match=message):
            backend.reserve_batch(caches, lengths)
    with pytest.raises(ValueError, match="different request"):
        backend.get(cache.handle, "b")
    with pytest.raises(ValueError, match="cannot be forked"):
        backend.fork(cache.handle, "b")


def test_budget_must_hold_one_windowed_request() -> None:
    with pytest.raises(ValueError, match="cannot hold one windowed request"):
        _backend(pages=2)


def test_windowed_pool_never_exceeds_half_of_a_large_budget() -> None:
    backend = _backend(pages=40)

    window_bytes = backend.num_window_pages * backend.window_bytes_per_page
    assert window_bytes <= backend.budget_bytes // 2
    used = window_bytes + backend.num_pages * backend.bytes_per_page
    assert backend.budget_bytes - backend.bytes_per_page < used <= backend.budget_bytes
    assert backend.window_pages_for(3) == 2
    assert backend.window_pages_for(10_000) == 4


def test_reset_returns_every_page() -> None:
    backend = _backend()
    _append(backend, backend.create("a"), 8, 1.0)
    backend.preflight(
        (backend.get(backend.create("b"), "b"),), (_PAGE * (backend.num_pages + 1),)
    )

    backend.reset(reset_counters=True)

    assert backend.free_page_count() == backend.num_pages
    assert backend.metrics()["allocation_failures"] == 0


@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")
def test_windowed_geometry_selects_the_two_pool_backend() -> None:
    geometry = KVCacheGeometry(
        num_layers=_LAYERS,
        num_kv_heads=_HEADS,
        head_dim=_DIM,
        dtype=mx.float16,
        window_layer_indices=(0, 2),
        window_size=_WINDOW,
        max_append_tokens=8,
    )

    bundle = build_native_execution_backend(
        DEFAULT_NATIVE_EXECUTION_BACKEND,
        geometry,
        page_size=_PAGE,
        cache_budget_bytes=1 << 20,
        cache_family="windowed",
    )

    assert isinstance(bundle.cache_backend, WindowedPagedKVCacheBackend)
    assert isinstance(bundle.attention_backend, WindowedMetalAttentionBackend)
