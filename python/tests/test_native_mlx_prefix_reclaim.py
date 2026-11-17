"""Prefix-cache pages returned to the KV pool on demand."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_worker.native_mlx.cache import PagedKVCacheBackend
from mlx_worker.native_mlx.prefix_cache import BlockHashPrefixCache, RadixPrefixCache
from mlx_worker.native_mlx.scheduler import NativeContinuousScheduler

from .test_native_mlx import (
    _FakeExecutor,
    _fingerprint,
    _PagedFakeCacheCoordinator,
    _schedulable,
)

_PAGE = 8
_OLD = tuple(range(1, 17))
_NEW = tuple(range(101, 117))


def _backend(pages: int = 8) -> PagedKVCacheBackend:
    # One layer, one head, head_dim 2, float16: 64 bytes per 8-token page.
    return PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=_PAGE,
        budget_bytes=64 * pages,
        dtype=mx.float16,
    )


def _finish_request(
    backend: PagedKVCacheBackend,
    cache: RadixPrefixCache | BlockHashPrefixCache,
    request_id: str,
    tokens: tuple[int, ...],
) -> None:
    handle = backend.create(request_id)
    request_cache = backend.get(handle, request_id)
    reservation = backend.reserve_batch((request_cache,), (len(tokens),))
    shape = (1, 1, len(tokens), 2)
    reservation.stage_layer(
        0, mx.ones(shape, dtype=mx.float16), mx.ones(shape, dtype=mx.float16)
    )
    reservation.commit()
    cache.publish(handle, tokens, len(tokens))
    cache.release(handle)
    backend.release(handle)


def _radix(backend: PagedKVCacheBackend) -> RadixPrefixCache:
    return RadixPrefixCache(
        backend=backend,
        compatibility=_fingerprint(page_size=_PAGE),
        page_size=_PAGE,
        max_entries=backend.num_pages,
        max_bytes=backend.num_pages * backend.bytes_per_page,
    )


def test_radix_reclaim_frees_least_recently_used_pages_first() -> None:
    backend = _backend()
    cache = _radix(backend)
    _finish_request(backend, cache, "old", _OLD)
    _finish_request(backend, cache, "new", _NEW)
    assert backend.free_page_count() == 4

    # The old leaf's second page is shared with nothing else.
    assert cache.reclaim(1) == 1
    assert backend.free_page_count() == 5
    assert cache.probe((*_OLD, 9)).matched_tokens == 8
    assert cache.probe((*_NEW, 9)).matched_tokens == 16

    assert cache.reclaim(10) == 3
    assert backend.free_page_count() == 8
    assert cache.metrics()["prefix_entries"] == 0


def test_radix_reclaim_keeps_entries_a_request_is_using() -> None:
    backend = _backend()
    cache = _radix(backend)
    _finish_request(backend, cache, "old", _OLD)
    _finish_request(backend, cache, "new", _NEW)
    probe = cache.probe((*_OLD, 9))
    admission = cache.acquire("reader", (*_OLD, 9), probe)
    assert admission is not None and admission.cache_length == 16

    cache.reclaim(10)

    assert cache.probe((*_OLD, 9)).matched_tokens == 16
    assert cache.probe((*_NEW, 9)).matched_tokens == 0


def test_block_hash_reclaim_frees_least_recently_used_pages_first() -> None:
    backend = _backend()
    cache = BlockHashPrefixCache(
        backend=backend,
        compatibility=_fingerprint(page_size=_PAGE),
        page_size=_PAGE,
        max_entries=backend.num_pages,
        max_bytes=backend.num_pages * backend.bytes_per_page,
    )
    _finish_request(backend, cache, "old", _OLD)
    _finish_request(backend, cache, "new", _NEW)

    assert cache.reclaim(2) == 2
    assert cache.probe((*_OLD, 9)).matched_tokens == 0
    assert cache.probe((*_NEW, 9)).matched_tokens == 16


@dataclass
class _ReclaimingCoordinator(_PagedFakeCacheCoordinator):
    prefix_pages: int = 0

    def kv_admission_capacity(self) -> tuple[int, int]:
        free, page_size = super().kv_admission_capacity()
        return free - self.prefix_pages, page_size

    def reclaim_pages(self, pages: int) -> int:
        freed = min(pages, self.prefix_pages)
        self.prefix_pages -= freed
        return freed


def test_scheduler_reclaims_prefix_pages_before_holding_a_request() -> None:
    executor = _FakeExecutor()
    coordinator = _ReclaimingCoordinator(
        executor.lengths, total_pages=8, prefix_pages=4
    )
    scheduler = NativeContinuousScheduler(
        executor,
        coordinator,  # type: ignore[arg-type]
        prefill_batch_size=4,
        prefill_step_size=256,
    )
    # Each request needs ceil((4 + 8) / 4) + 1 = 4 pages through its last token.
    scheduler.submit(_schedulable("first", (1, 2, 3, 4), max_tokens=8))
    scheduler.submit(_schedulable("second", (5, 6, 7, 8), max_tokens=8))

    scheduler.tick()

    assert [request.request_id for request in executor.batches[0].requests] == [
        "first",
        "second",
    ]
    assert coordinator.prefix_pages == 0
    assert scheduler._kv_prefix_reclaimed_pages == 4
    assert scheduler._kv_admission_waits == 0
