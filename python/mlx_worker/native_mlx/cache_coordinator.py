"""Scheduler-facing cache lifecycle coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .cache import KVCacheBackend
from .interfaces import (
    CacheAdmission,
    CachePublication,
    PrefixProbe,
)


class PrefixCache(Protocol):
    """Token-prefix index independent of physical MLX storage."""

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe: ...

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: PrefixProbe,
    ) -> CacheAdmission | None: ...

    def publish(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication: ...

    def release(self, cache_handle: str | None) -> None: ...

    def reclaim(self, pages: int) -> int: ...

    def metrics(self) -> dict[str, Any]: ...

    def reset(self, *, clear_cache: bool, reset_counters: bool) -> None: ...


@dataclass
class NativeCacheCoordinator:
    """Compose prefix lookup with physical KV lifecycle."""

    backend: KVCacheBackend
    prefix_cache: PrefixCache

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe:
        return self.prefix_cache.probe(token_ids)

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: PrefixProbe | None = None,
    ) -> CacheAdmission:
        active_probe = probe or self.probe(token_ids)
        if active_probe.matched_tokens:
            admission = self.prefix_cache.acquire(request_id, token_ids, active_probe)
            if admission is not None:
                return admission
        handle = self.backend.create(request_id)
        return CacheAdmission(cache_handle=handle, cache_length=0)

    def publish_committed(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication:
        return self.prefix_cache.publish(
            cache_handle,
            token_ids,
            committed_length,
        )

    def length(self, cache_handle: str | None) -> int:
        return self.backend.length(cache_handle)

    def kv_admission_capacity(self) -> tuple[int, int] | None:
        """Return ``(free_pages, page_size)`` for admission control.

        Returns ``None`` when the backend has no fixed page pool.
        """

        free_page_count = getattr(self.backend, "free_page_count", None)
        page_size = getattr(self.backend, "page_size", None)
        if not callable(free_page_count) or not page_size:
            return None
        return int(free_page_count()), int(page_size)

    def kv_window_pages(self) -> int:
        """Return pages reserved for sliding-window layers, 0 without any."""

        return int(getattr(self.backend, "num_window_pages", 0))

    def kv_window_pages_for(self, tokens: int) -> int:
        """Return the most windowed pages a request of ``tokens`` tokens holds."""

        pages_for = getattr(self.backend, "window_pages_for", None)
        return int(pages_for(tokens)) if callable(pages_for) else 0

    def reclaim_pages(self, pages: int) -> int:
        """Evict unpinned prefix-cache entries until ``pages`` pages are free.

        Returns:
            Pages returned to the backend's free list, possibly fewer than
            requested.
        """

        return self.prefix_cache.reclaim(pages)

    def release(self, cache_handle: str | None) -> None:
        self.prefix_cache.release(cache_handle)
        self.backend.release(cache_handle)

    def metrics(self) -> dict[str, Any]:
        return {**self.backend.metrics(), **self.prefix_cache.metrics()}

    def reset(self, *, clear_cache: bool, reset_counters: bool) -> dict[str, Any]:
        """Reset idle cache state without rebuilding the model or executor."""

        self.prefix_cache.reset(
            clear_cache=clear_cache,
            reset_counters=reset_counters,
        )
        if clear_cache:
            self.backend.reset(reset_counters=reset_counters)
        return self.metrics()
