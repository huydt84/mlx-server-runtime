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

    def publish(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication: ...

    def metrics(self) -> dict[str, Any]: ...


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
            raise ValueError("Phase 9 no-prefix adapter cannot return a cache hit")
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

    def release(self, cache_handle: str | None) -> None:
        self.backend.release(cache_handle)

    def metrics(self) -> dict[str, Any]:
        return {**self.backend.metrics(), **self.prefix_cache.metrics()}
