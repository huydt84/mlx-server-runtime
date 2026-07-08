"""Replaceable prefix-index adapters for native MLX cache coordination."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .cache import KVCacheBackend
from .interfaces import CachePublication, PrefixProbe
from .interfaces import CacheAdmission


@dataclass(frozen=True)
class NoPrefixCache:
    """Phase 9 adapter that deliberately performs no prefix reuse."""

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe:
        del token_ids
        return PrefixProbe()

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: PrefixProbe,
    ) -> CacheAdmission | None:
        del request_id, token_ids, probe
        return None

    def publish(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication:
        del cache_handle, token_ids, committed_length
        return CachePublication()

    def release(self, cache_handle: str | None) -> None:
        del cache_handle

    def metrics(self) -> dict[str, int | str]:
        return {
            "prefix_strategy": "none",
            "prefix_queries": 0,
            "prefix_hits": 0,
            "prefix_misses": 0,
            "prefix_reused_tokens": 0,
            "prefix_reused_pages": 0,
            "prefix_entries": 0,
            "prefix_bytes": 0,
            "prefix_pinned_pages": 0,
            "prefix_collisions_rejected": 0,
            "prefix_evictions": 0,
        }


@dataclass(frozen=True)
class PrefixCompatibilityFingerprint:
    """Immutable compatibility namespace for reusable native KV pages."""

    checkpoint: str
    architecture_class: str
    tokenizer_assets_hash: str
    model_dtype: str
    kv_dtype: str
    quantization: str
    page_size: int
    cache_schema_version: int = 1

    @property
    def digest(self) -> str:
        payload = json.dumps(
            {
                "architecture_class": self.architecture_class,
                "cache_schema_version": self.cache_schema_version,
                "checkpoint": self.checkpoint,
                "kv_dtype": self.kv_dtype,
                "model_dtype": self.model_dtype,
                "page_size": self.page_size,
                "quantization": self.quantization,
                "tokenizer_assets_hash": self.tokenizer_assets_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class _BlockEntry:
    block_hash: str
    parent_hash: str
    token_prefix: tuple[int, ...]
    compatibility_digest: str
    cache_handle: str
    cached_length: int
    pages: int
    bytes: int
    last_used: int
    pins: int = 0


@dataclass
class BlockHashPrefixCache:
    """vLLM-style full-page SHA-256 prefix index over native paged KV."""

    backend: KVCacheBackend
    compatibility: PrefixCompatibilityFingerprint
    page_size: int
    max_entries: int = 32
    max_bytes: int = 8 * 1024 * 1024
    _entries: dict[str, _BlockEntry] = field(default_factory=dict, init=False)
    _active_handles: dict[str, str] = field(default_factory=dict, init=False)
    _clock: int = field(default=0, init=False)
    _queries: int = field(default=0, init=False)
    _hits: int = field(default=0, init=False)
    _misses: int = field(default=0, init=False)
    _reused_tokens: int = field(default=0, init=False)
    _reused_pages: int = field(default=0, init=False)
    _collisions_rejected: int = field(default=0, init=False)
    _evictions: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("block-hash prefix cache page size must be positive")
        if self.max_entries <= 0:
            raise ValueError("block-hash prefix cache max entries must be positive")
        if self.max_bytes <= 0:
            raise ValueError("block-hash prefix cache max bytes must be positive")

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe:
        self._queries += 1
        match = self._lookup(token_ids, mutate=False)
        if match is None:
            self._misses += 1
            return PrefixProbe()
        self._hits += 1
        return PrefixProbe(
            matched_tokens=match.cached_length,
            matched_pages=match.pages,
            cache_handle=match.cache_handle,
        )

    def acquire(
        self,
        request_id: str,
        token_ids: tuple[int, ...],
        probe: PrefixProbe,
    ) -> CacheAdmission | None:
        if probe.cache_handle is None or probe.matched_tokens <= 0:
            return None
        entry = self._lookup(token_ids, mutate=True)
        if entry is None or entry.cache_handle != probe.cache_handle:
            return None
        entry.pins += 1
        child = self.backend.fork(
            entry.cache_handle,
            request_id,
            length=entry.cached_length,
        )
        self._active_handles[child] = entry.block_hash
        self._reused_tokens += entry.cached_length
        self._reused_pages += entry.pages
        return CacheAdmission(
            cache_handle=child,
            cache_length=entry.cached_length,
            reused_tokens=entry.cached_length,
            reused_pages=entry.pages,
        )

    def publish(
        self,
        cache_handle: str,
        token_ids: tuple[int, ...],
        committed_length: int,
    ) -> CachePublication:
        publish_length = (min(committed_length, len(token_ids)) // self.page_size) * (
            self.page_size
        )
        if publish_length <= 0:
            return CachePublication()
        chain = self._hash_chain(token_ids[:publish_length])
        chain = chain[: self._publishable_chain_pages()]
        published = 0
        last_hash = ""
        publish_clock = self._tick()
        for block_hash, parent_hash, length in chain:
            last_hash = block_hash
            if block_hash in self._entries:
                continue
            snapshot = self.backend.fork(
                cache_handle,
                f"prefix-cache-{block_hash[:16]}",
                length=length,
            )
            pages = length // self.page_size
            self._entries[block_hash] = _BlockEntry(
                block_hash=block_hash,
                parent_hash=parent_hash,
                token_prefix=token_ids[:length],
                compatibility_digest=self.compatibility.digest,
                cache_handle=snapshot,
                cached_length=length,
                pages=pages,
                bytes=self._bytes_for_pages(1),
                last_used=publish_clock,
            )
            published = length
        self._evict_until_within_limits(protected=last_hash)
        return CachePublication(
            published_tokens=published,
            published_pages=published // self.page_size,
        )

    def release(self, cache_handle: str | None) -> None:
        if cache_handle is None:
            return
        block_hash = self._active_handles.pop(cache_handle, None)
        if block_hash is None:
            return
        entry = self._entries.get(block_hash)
        if entry is not None:
            entry.pins = max(0, entry.pins - 1)
        self._evict_until_within_limits()

    def metrics(self) -> dict[str, int | str]:
        return {
            "prefix_strategy": "block-hash",
            "prefix_queries": self._queries,
            "prefix_hits": self._hits,
            "prefix_misses": self._misses,
            "prefix_reused_tokens": self._reused_tokens,
            "prefix_reused_pages": self._reused_pages,
            "prefix_entries": len(self._entries),
            "prefix_bytes": sum(entry.bytes for entry in self._entries.values()),
            "prefix_pinned_pages": sum(
                entry.pages for entry in self._entries.values() if entry.pins
            ),
            "prefix_collisions_rejected": self._collisions_rejected,
            "prefix_evictions": self._evictions,
        }

    def _lookup(
        self, token_ids: tuple[int, ...], *, mutate: bool
    ) -> _BlockEntry | None:
        max_reusable = ((max(0, len(token_ids) - 1)) // self.page_size) * self.page_size
        if max_reusable <= 0:
            return None
        best: _BlockEntry | None = None
        for block_hash, _, _ in self._hash_chain(token_ids[:max_reusable]):
            entry = self._entries.get(block_hash)
            if entry is None:
                break
            if (
                entry.compatibility_digest != self.compatibility.digest
                or entry.token_prefix != token_ids[: entry.cached_length]
            ):
                self._collisions_rejected += 1
                break
            best = entry
        if best is not None and mutate:
            best.last_used = self._tick()
        return best

    def _hash_chain(
        self,
        token_ids: tuple[int, ...],
    ) -> tuple[tuple[str, str, int], ...]:
        parent_hash = ""
        blocks: list[tuple[str, str, int]] = []
        for end in range(self.page_size, len(token_ids) + 1, self.page_size):
            page_tokens = token_ids[end - self.page_size : end]
            block_hash = self._block_hash(parent_hash, page_tokens)
            blocks.append((block_hash, parent_hash, end))
            parent_hash = block_hash
        return tuple(blocks)

    def _block_hash(self, parent_hash: str, page_tokens: tuple[int, ...]) -> str:
        payload = json.dumps(
            {
                "compatibility": self.compatibility.digest,
                "parent": parent_hash,
                "tokens": page_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _evict_until_within_limits(self, protected: str | None = None) -> None:
        while (
            len(self._entries) > self.max_entries
            or self._stored_bytes() > self.max_bytes
        ):
            candidates = [
                entry
                for entry in self._entries.values()
                if entry.pins == 0 and entry.block_hash != protected
            ]
            if not candidates:
                return
            victim = min(candidates, key=lambda entry: (entry.last_used, -entry.pages))
            self._entries.pop(victim.block_hash, None)
            self.backend.release(victim.cache_handle)
            self._evictions += 1

    def _bytes_for_pages(self, pages: int) -> int:
        value = getattr(self.backend, "bytes_per_page", self.page_size)
        return int(value) * pages

    def _publishable_chain_pages(self) -> int:
        bytes_per_page = max(1, self._bytes_for_pages(1))
        return max(1, min(self.max_entries, self.max_bytes // bytes_per_page))

    def _stored_bytes(self) -> int:
        return sum(entry.bytes for entry in self._entries.values())

    def _tick(self) -> int:
        self._clock = max(self._clock + 1, time.monotonic_ns())
        return self._clock
