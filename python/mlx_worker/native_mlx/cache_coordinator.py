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
