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
class _RadixNode:
    edge_tokens: tuple[int, ...] = ()
    token_prefix: tuple[int, ...] = ()
    cache_handle: str | None = None
    cached_length: int = 0
    pages: int = 0
    bytes: int = 0
    last_used: int = 0
    pins: int = 0
    parent: "_RadixNode | None" = None
    children: dict[int, "_RadixNode"] = field(default_factory=dict)


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


@dataclass
class RadixPrefixCache:
    """SGLang-style compressed exact-token radix index over native KV pages."""

    backend: KVCacheBackend
    compatibility: PrefixCompatibilityFingerprint
    page_size: int
    max_entries: int = 32
    max_bytes: int = 8 * 1024 * 1024
    _root: _RadixNode = field(default_factory=_RadixNode, init=False)
    _active_handles: dict[str, _RadixNode] = field(default_factory=dict, init=False)
    _clock: int = field(default=0, init=False)
    _queries: int = field(default=0, init=False)
    _hits: int = field(default=0, init=False)
    _misses: int = field(default=0, init=False)
    _reused_tokens: int = field(default=0, init=False)
    _reused_pages: int = field(default=0, init=False)
    _splits: int = field(default=0, init=False)
    _evictions: int = field(default=0, init=False)
    _metrics_cache: dict[str, int | str] = field(default_factory=dict, init=False)
    _metrics_dirty: bool = field(default=True, init=False)
    _published_lengths_by_handle: dict[str, int] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if self.page_size <= 0:
            raise ValueError("radix prefix cache page size must be positive")
        if self.max_entries <= 0:
            raise ValueError("radix prefix cache max entries must be positive")
        if self.max_bytes <= 0:
            raise ValueError("radix prefix cache max bytes must be positive")

    def probe(self, token_ids: tuple[int, ...]) -> PrefixProbe:
        self._queries += 1
        match = self._lookup(token_ids, mutate=False)
        if match is None:
            self._misses += 1
            self._metrics_dirty = True
            return PrefixProbe()
        self._hits += 1
        self._metrics_dirty = True
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
        node = self._lookup(token_ids, mutate=True)
        if node is None or node.cache_handle != probe.cache_handle:
            return None
        node.pins += 1
        child = self.backend.fork(
            node.cache_handle,
            request_id,
            length=node.cached_length,
        )
        self._active_handles[child] = node
        self._published_lengths_by_handle[child] = node.cached_length
        self._reused_tokens += node.cached_length
        self._reused_pages += node.pages
        self._metrics_dirty = True
        return CacheAdmission(
            cache_handle=child,
            cache_length=node.cached_length,
            reused_tokens=node.cached_length,
            reused_pages=node.pages,
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
        previous_publish_length = self._published_lengths_by_handle.get(
            cache_handle,
            0,
        )
        if publish_length <= previous_publish_length:
            return CachePublication()
        published = 0
        start_length = (
            (previous_publish_length // self.page_size) + 1
        ) * self.page_size
        for length in range(start_length, publish_length + 1, self.page_size):
            if length // self.page_size > self._publishable_chain_pages():
                break
            node = self._insert_prefix(token_ids[:length])
            if node.cache_handle is None:
                node.cache_handle = self.backend.fork(
                    cache_handle,
                    f"prefix-radix-{self._prefix_digest(token_ids[:length])[:16]}",
                    length=length,
                )
                node.cached_length = length
                node.pages = length // self.page_size
                node.bytes = self._bytes_for_pages(1)
            node.last_used = self._tick()
            published = length
        self._evict_until_within_limits()
        if published:
            self._published_lengths_by_handle[cache_handle] = published
        self._metrics_dirty = True
        return CachePublication(
            published_tokens=published,
            published_pages=published // self.page_size,
        )

    def release(self, cache_handle: str | None) -> None:
        if cache_handle is None:
            return
        self._published_lengths_by_handle.pop(cache_handle, None)
        node = self._active_handles.pop(cache_handle, None)
        if node is None:
            return
        node.pins = max(0, node.pins - 1)
        self._evict_until_within_limits()
        self._metrics_dirty = True

    def metrics(self) -> dict[str, int | str]:
        if not self._metrics_dirty:
            return dict(self._metrics_cache)
        nodes = tuple(self._walk_nodes())
        entries = [node for node in nodes if node.cache_handle is not None]
        leaves = [node for node in entries if not node.children]
        self._metrics_cache = {
            "prefix_strategy": "radix",
            "prefix_queries": self._queries,
            "prefix_hits": self._hits,
            "prefix_misses": self._misses,
            "prefix_reused_tokens": self._reused_tokens,
            "prefix_reused_pages": self._reused_pages,
            "prefix_entries": len(entries),
            "prefix_bytes": sum(node.bytes for node in entries),
            "prefix_pinned_pages": sum(node.pages for node in entries if node.pins),
            "prefix_collisions_rejected": 0,
            "prefix_evictions": self._evictions,
            "radix_nodes": len(nodes),
            "radix_splits": self._splits,
            "radix_shared_pages": sum(node.pages for node in entries if node.children),
            "radix_protected_pages": sum(node.pages for node in entries if node.pins),
            "radix_evictable_pages": sum(
                node.pages for node in leaves if node.pins == 0
            ),
            "radix_tree_depth": max(
                (len(node.token_prefix) for node in nodes),
                default=0,
            ),
            "radix_leaf_evictions": self._evictions,
        }
        self._metrics_dirty = False
        return dict(self._metrics_cache)

    def _lookup(self, token_ids: tuple[int, ...], *, mutate: bool) -> _RadixNode | None:
        max_reusable = ((max(0, len(token_ids) - 1)) // self.page_size) * self.page_size
        if max_reusable <= 0:
            return None
        node = self._root
        offset = 0
        best: _RadixNode | None = None
        while offset < max_reusable:
            child = node.children.get(token_ids[offset])
            if child is None:
                break
            common = _common_prefix_len_at(
                child.edge_tokens,
                token_ids,
                offset,
                max_reusable,
            )
            if common < len(child.edge_tokens):
                break
            offset += common
            node = child
            if node.cache_handle is not None and node.cached_length <= max_reusable:
                best = node
        if best is not None and mutate:
            best.last_used = self._tick()
        return best

    def _insert_prefix(self, token_prefix: tuple[int, ...]) -> _RadixNode:
        node = self._root
        offset = 0
        while offset < len(token_prefix):
            child = node.children.get(token_prefix[offset])
            if child is None:
                new = _RadixNode(
                    edge_tokens=token_prefix[offset:],
                    token_prefix=token_prefix,
                    parent=node,
                )
                node.children[new.edge_tokens[0]] = new
                return new
            common = _common_prefix_len(child.edge_tokens, token_prefix[offset:])
            if common == len(child.edge_tokens):
                offset += common
                node = child
                continue
            split = _RadixNode(
                edge_tokens=child.edge_tokens[:common],
                token_prefix=token_prefix[: offset + common],
                parent=node,
            )
            node.children[split.edge_tokens[0]] = split
            child.edge_tokens = child.edge_tokens[common:]
            child.parent = split
            split.children[child.edge_tokens[0]] = child
            self._splits += 1
            if offset + common == len(token_prefix):
                return split
            new = _RadixNode(
                edge_tokens=token_prefix[offset + common :],
                token_prefix=token_prefix,
                parent=split,
            )
            split.children[new.edge_tokens[0]] = new
            return new
        return node

    def _evict_until_within_limits(self) -> None:
        while (
            self._entry_count() > self.max_entries
            or self._stored_bytes() > self.max_bytes
        ):
            candidates = [
                node
                for node in self._walk_nodes()
                if node.cache_handle is not None
                and node.pins == 0
                and not node.children
            ]
            if not candidates:
                return
            victim = min(candidates, key=lambda node: (node.last_used, -node.pages))
            self.backend.release(victim.cache_handle)
            victim.cache_handle = None
            victim.cached_length = 0
            victim.pages = 0
            victim.bytes = 0
            self._evictions += 1
            self._metrics_dirty = True
            self._prune_empty_ancestors(victim)

    def _prune_empty_ancestors(self, node: _RadixNode) -> None:
        while (
            node.parent is not None
            and node.cache_handle is None
            and not node.children
            and node.pins == 0
        ):
            parent = node.parent
            parent.children.pop(node.edge_tokens[0], None)
            node = parent

    def _walk_nodes(self) -> list[_RadixNode]:
        pending = list(self._root.children.values())
        nodes: list[_RadixNode] = []
        while pending:
            node = pending.pop()
            nodes.append(node)
            pending.extend(node.children.values())
        return nodes

    def _prefix_digest(self, token_prefix: tuple[int, ...]) -> str:
        payload = json.dumps(
            {
                "compatibility": self.compatibility.digest,
                "tokens": token_prefix,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _bytes_for_pages(self, pages: int) -> int:
        value = getattr(self.backend, "bytes_per_page", self.page_size)
        return int(value) * pages

    def _publishable_chain_pages(self) -> int:
        bytes_per_page = max(1, self._bytes_for_pages(1))
        return max(1, min(self.max_entries, self.max_bytes // bytes_per_page))

    def _entry_count(self) -> int:
        return sum(node.cache_handle is not None for node in self._walk_nodes())

    def _stored_bytes(self) -> int:
        return sum(node.bytes for node in self._walk_nodes())

    def _tick(self) -> int:
        self._clock = max(self._clock + 1, time.monotonic_ns())
        return self._clock


def _common_prefix_len(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    count = 0
    for left_item, right_item in zip(left, right, strict=False):
        if left_item != right_item:
            break
        count += 1
    return count


def _common_prefix_len_at(
    left: tuple[int, ...],
    right: tuple[int, ...],
    offset: int,
    limit: int,
) -> int:
    count = 0
    max_count = min(len(left), max(0, limit - offset))
    while count < max_count and left[count] == right[offset + count]:
        count += 1
    return count
