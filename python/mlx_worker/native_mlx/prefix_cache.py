"""Replaceable prefix-index adapters for native MLX cache coordination."""

from __future__ import annotations

from dataclasses import dataclass

from .interfaces import CachePublication, PrefixProbe


@dataclass(frozen=True)
class NoPrefixCache:
    """Phase 9 adapter that deliberately performs no prefix reuse."""

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe:
        del token_ids
        return PrefixProbe()

    def publish(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication:
        del cache_handle, token_ids, committed_length
        return CachePublication()

    def metrics(self) -> dict[str, int | str]:
        return {
            "prefix_strategy": "none",
            "prefix_queries": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
            "prefix_reused_tokens": 0,
            "prefix_reused_pages": 0,
            "prefix_entries": 0,
            "prefix_evictions": 0,
        }
