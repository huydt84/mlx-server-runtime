"""Shared KV-cache backends for native MLX execution."""

from __future__ import annotations

import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Protocol

import mlx.core as mx


@dataclass(frozen=True)
class KVCacheGeometry:
    """Architecture-declared dimensions required by a shared KV backend."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: Any


class RequestCache(Protocol):
    """Opaque request cache state resolved only inside the executor/backend."""

    handle: str
    request_id: str

    def size(self) -> int: ...


class BatchCacheReservation(Protocol):
    """Transactional cache append reserved for one physical model step."""

    request_caches: tuple[RequestCache, ...]
    append_lengths: tuple[int, ...]

    def commit(self) -> tuple[int, ...]: ...

    def abort(self) -> None: ...


class KVCacheBackend(Protocol):
    """Physical KV storage and transactional batch-view interface."""

    num_layers: int

    def create(self, request_id: str) -> str: ...

    def get(self, handle: str | None, request_id: str) -> RequestCache: ...

    def length(self, handle: str | None) -> int: ...

    def preflight(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[str | None, ...]: ...

    def reserve_batch(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> BatchCacheReservation: ...

    def release(self, handle: str | None) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


@dataclass
class DenseLayerCache:
    """Contiguous cache retained only for diagnostics and test parity."""

    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0

    @property
    def offsets(self) -> mx.array:
        """Expose a batch-compatible RoPE offset."""

        return mx.array([self.offset], dtype=mx.int32)

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        self.offset = int(self.keys.shape[2])
        return self.keys, self.values

    def size(self) -> int:
        return self.offset

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return int(self.keys.nbytes + self.values.nbytes)


@dataclass
class DenseBatchLayerCache:
    """Transactional dense layer view for diagnostics and executor tests."""

    request_layers: tuple[DenseLayerCache, ...]
    append_lengths: tuple[int, ...]
    keys: mx.array | None = None
    values: mx.array | None = None
    offsets: mx.array = field(init=False)
    _committed_keys: tuple[mx.array, ...] = field(init=False, default=())
    _committed_values: tuple[mx.array, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        self.offsets = mx.array(
            [layer.size() for layer in self.request_layers],
            dtype=mx.int32,
        )

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        max_total = max(
            layer.size() + append_length
            for layer, append_length in zip(
                self.request_layers,
                self.append_lengths,
                strict=True,
            )
        )
        key_rows: list[mx.array] = []
        value_rows: list[mx.array] = []
        committed_keys: list[mx.array] = []
        committed_values: list[mx.array] = []
        for index, (layer, append_length) in enumerate(
            zip(self.request_layers, self.append_lengths, strict=True)
        ):
            valid_keys = keys[index : index + 1, :, :append_length, :]
            valid_values = values[index : index + 1, :, :append_length, :]
            if layer.size() == 0:
                row_keys = valid_keys
                row_values = valid_values
            else:
                if layer.keys is None or layer.values is None:
                    raise ValueError("cache layer missing KV state")
                row_keys = mx.concatenate([layer.keys, valid_keys], axis=2)
                row_values = mx.concatenate([layer.values, valid_values], axis=2)
            committed_keys.append(row_keys)
            committed_values.append(row_values)
            pad = max_total - int(row_keys.shape[2])
            if pad > 0:
                row_keys = mx.pad(row_keys, [(0, 0), (0, 0), (0, pad), (0, 0)])
                row_values = mx.pad(
                    row_values,
                    [(0, 0), (0, 0), (0, pad), (0, 0)],
                )
            key_rows.append(row_keys)
            value_rows.append(row_values)
        self.keys = mx.concatenate(key_rows, axis=0)
        self.values = mx.concatenate(value_rows, axis=0)
        self._committed_keys = tuple(committed_keys)
        self._committed_values = tuple(committed_values)
        return self.keys, self.values

    def validate_commit(self) -> None:
        if len(self._committed_keys) != len(self.request_layers):
            raise ValueError("batched cache has incomplete committed keys")
        if len(self._committed_values) != len(self.request_layers):
            raise ValueError("batched cache has incomplete committed values")
        for layer, keys, values, append_length in zip(
            self.request_layers,
            self._committed_keys,
            self._committed_values,
            self.append_lengths,
            strict=True,
        ):
            expected = layer.size() + append_length
            if int(keys.shape[2]) != expected or int(values.shape[2]) != expected:
                raise ValueError("committed dense cache length mismatch")

    def commit(self) -> None:
        self.validate_commit()
        for layer, keys, values in zip(
            self.request_layers,
            self._committed_keys,
            self._committed_values,
            strict=True,
        ):
            layer.keys = keys
            layer.values = values
            layer.offset = int(keys.shape[2])


@dataclass
class DenseRequestCache:
    """Opaque dense request cache retained outside the production composition."""

    handle: str
    request_id: str
    layers: list[DenseLayerCache]

    def size(self) -> int:
        return self.layers[0].size() if self.layers else 0


@dataclass
class DenseBatchReservation:
    """Transactional dense reservation used by the reference attention path."""

    request_caches: tuple[DenseRequestCache, ...]
    append_lengths: tuple[int, ...]
    layer_views: tuple[DenseBatchLayerCache, ...]
    _finished: bool = False

    def commit(self) -> tuple[int, ...]:
        if self._finished:
            raise ValueError("cache reservation already finished")
        for layer in self.layer_views:
            layer.validate_commit()
        for layer in self.layer_views:
            layer.commit()
        self._finished = True
        return tuple(cache.size() for cache in self.request_caches)

    def abort(self) -> None:
        self._finished = True


@dataclass
class DenseKVCacheBackend:
    """Bounded dense reference backend; never selected by native production."""

    num_layers: int
    _caches: dict[str, DenseRequestCache] = field(default_factory=dict)

    def create(self, request_id: str) -> str:
        handle = f"dense-kv-{request_id}-{uuid.uuid4().hex}"
        self._caches[handle] = DenseRequestCache(
            handle=handle,
            request_id=request_id,
            layers=[DenseLayerCache() for _ in range(self.num_layers)],
        )
        return handle

    def get(self, handle: str | None, request_id: str) -> DenseRequestCache:
        if handle is None or handle not in self._caches:
            raise ValueError("invalid cache handle")
        cache = self._caches[handle]
        if cache.request_id != request_id:
            raise ValueError("cache handle belongs to different request")
        return cache

    def length(self, handle: str | None) -> int:
        if handle is None or handle not in self._caches:
            return 0
        return self._caches[handle].size()

    def reserve_batch(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> DenseBatchReservation:
        dense = tuple(_require_dense_cache(cache) for cache in caches)
        return DenseBatchReservation(
            request_caches=dense,
            append_lengths=append_lengths,
            layer_views=tuple(
                DenseBatchLayerCache(
                    request_layers=tuple(cache.layers[index] for cache in dense),
                    append_lengths=append_lengths,
                )
                for index in range(self.num_layers)
            ),
        )

    def preflight(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[str | None, ...]:
        if len(caches) != len(append_lengths):
            raise ValueError("dense preflight requires matching requests")
        return (None,) * len(caches)

    def release(self, handle: str | None) -> None:
        if handle is not None:
            self._caches.pop(handle, None)

    def metrics(self) -> dict[str, Any]:
        return {
            "cache_backend": "dense-reference",
            "active_kv_bytes": sum(
                layer.nbytes
                for cache in self._caches.values()
                for layer in cache.layers
            ),
        }


@dataclass
class PagedRequestCache:
    """Logical request-to-page mapping for the production paged backend."""

    handle: str
    request_id: str
    block_table: list[int] = field(default_factory=list)
    length: int = 0

    def size(self) -> int:
        return self.length


@dataclass
class PagedBatchReservation:
    """All-request, all-layer transactional paged append reservation."""

    backend: "PagedKVCacheBackend"
    request_caches: tuple[PagedRequestCache, ...]
    append_lengths: tuple[int, ...]
    candidate_tables: tuple[tuple[int, ...], ...]
    reserved_pages: tuple[int, ...]
    copy_pages: tuple[tuple[int, int], ...]
    pinned_pages: tuple[int, ...]
    staged_keys: list[mx.array | None]
    staged_values: list[mx.array | None]
    _finished: bool = False

    @property
    def token_lengths(self) -> tuple[int, ...]:
        return self.append_lengths

    @property
    def cache_lengths(self) -> tuple[int, ...]:
        return tuple(cache.size() for cache in self.request_caches)

    @property
    def block_tables(self) -> mx.array:
        width = max((len(table) for table in self.candidate_tables), default=0)
        rows = [
            list(table) + [0] * (width - len(table)) for table in self.candidate_tables
        ]
        return mx.array(rows, dtype=mx.int32)

    def stage_layer(
        self,
        layer_index: int,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self._finished:
            raise ValueError("cache reservation already finished")
        if self.staged_keys[layer_index] is not None:
            raise ValueError(f"layer {layer_index} staged more than once")
        self.backend.validate_kv_shape(keys, values, self.append_lengths)
        key_cache = self.backend.key_pages[layer_index]
        value_cache = self.backend.value_pages[layer_index]
        for old_page, new_page in self.copy_pages:
            key_cache[new_page] = key_cache[old_page]
            value_cache[new_page] = value_cache[old_page]
        packed_keys: list[mx.array] = []
        packed_values: list[mx.array] = []
        slots: list[int] = []
        for index, (cache, append_length, table) in enumerate(
            zip(
                self.request_caches,
                self.append_lengths,
                self.candidate_tables,
                strict=True,
            )
        ):
            packed_keys.append(keys[index, :, :append_length, :].transpose(1, 0, 2))
            packed_values.append(values[index, :, :append_length, :].transpose(1, 0, 2))
            for position in range(cache.size(), cache.size() + append_length):
                page = table[position // self.backend.page_size]
                slots.append(
                    page * self.backend.page_size + position % self.backend.page_size
                )
        slot_array = mx.array(slots, dtype=mx.int32)
        flat_shape = (
            self.backend.num_pages * self.backend.page_size,
            self.backend.num_kv_heads,
            self.backend.head_dim,
        )
        flat_keys = key_cache.reshape(flat_shape)
        flat_values = value_cache.reshape(flat_shape)
        flat_keys[slot_array] = mx.concatenate(packed_keys, axis=0)
        flat_values[slot_array] = mx.concatenate(packed_values, axis=0)
        key_cache = flat_keys.reshape(key_cache.shape)
        value_cache = flat_values.reshape(value_cache.shape)
        self.staged_keys[layer_index] = key_cache
        self.staged_values[layer_index] = value_cache
        return key_cache, value_cache

    def commit(self) -> tuple[int, ...]:
        if self._finished:
            raise ValueError("cache reservation already finished")
        if any(value is None for value in self.staged_keys + self.staged_values):
            raise ValueError("paged cache commit missing one or more layers")
        staged_keys = tuple(_require_array(value) for value in self.staged_keys)
        staged_values = tuple(_require_array(value) for value in self.staged_values)
        mx.eval(*staged_keys, *staged_values)
        lengths = self.backend._commit_reservation(self, staged_keys, staged_values)
        self._finished = True
        return lengths

    def abort(self) -> None:
        if not self._finished:
            self.backend._abort_reservation(self)
            self._finished = True


@dataclass
class PagedKVCacheBackend:
    """Fixed-size MLX page pool with transactional request block tables."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    budget_bytes: int
    dtype: Any = mx.float16
    _caches: dict[str, PagedRequestCache] = field(default_factory=dict, init=False)
    _reserved_pages: set[int] = field(default_factory=set, init=False)
    _allocation_failures: int = field(default=0, init=False)
    _attention_mode: str = field(default="uninitialized", init=False)
    _attention_time_ms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.page_size not in (8, 16, 32):
            raise ValueError("native KV page size must be one of 8, 16, or 32")
        if min(self.num_layers, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("native KV geometry must be positive")
        if self.dtype not in (mx.float16, mx.bfloat16):
            raise ValueError("native paged KV supports float16 or bfloat16")
        dtype_bytes = int(mx.array(0, dtype=self.dtype).itemsize)
        self.bytes_per_page = (
            2
            * self.num_layers
            * self.page_size
            * self.num_kv_heads
            * self.head_dim
            * dtype_bytes
        )
        self.num_pages = self.budget_bytes // self.bytes_per_page
        if self.num_pages <= 0:
            raise ValueError(
                "native text cache budget cannot allocate one complete KV page"
            )
        shape = (
            self.num_pages,
            self.page_size,
            self.num_kv_heads,
            self.head_dim,
        )
        self.key_pages = [
            mx.zeros(shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self.value_pages = [
            mx.zeros(shape, dtype=self.dtype) for _ in range(self.num_layers)
        ]
        self._free_pages = list(range(self.num_pages))
        self._ref_counts = [0] * self.num_pages
        self._pin_counts = [0] * self.num_pages

    def create(self, request_id: str) -> str:
        handle = f"paged-kv-{request_id}-{uuid.uuid4().hex}"
        self._caches[handle] = PagedRequestCache(
            handle=handle,
            request_id=request_id,
        )
        return handle

    def get(self, handle: str | None, request_id: str) -> PagedRequestCache:
        if handle is None or handle not in self._caches:
            raise ValueError("invalid cache handle")
        cache = self._caches[handle]
        if cache.request_id != request_id:
            raise ValueError("cache handle belongs to different request")
        return cache

    def length(self, handle: str | None) -> int:
        if handle is None or handle not in self._caches:
            return 0
        return self._caches[handle].size()

    def reserve_batch(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> PagedBatchReservation:
        paged = tuple(_require_paged_cache(cache) for cache in caches)
        if len(paged) != len(append_lengths) or not paged:
            raise ValueError("paged cache reservation requires matching requests")
        if len({cache.handle for cache in paged}) != len(paged):
            raise ValueError("paged cache reservation contains duplicate handles")
        if any(length <= 0 for length in append_lengths):
            raise ValueError("paged cache append lengths must be positive")

        available = [
            page for page in self._free_pages if page not in self._reserved_pages
        ]
        candidate_tables: list[tuple[int, ...]] = []
        reserved: list[int] = []
        copy_pages: list[tuple[int, int]] = []
        try:
            for cache, append_length in zip(paged, append_lengths, strict=True):
                table = list(cache.block_table)
                new_length = cache.size() + append_length
                if (
                    cache.size() % self.page_size
                    and table
                    and self._ref_counts[table[-1]] > 1
                ):
                    replacement = available.pop(0)
                    reserved.append(replacement)
                    copy_pages.append((table[-1], replacement))
                    table[-1] = replacement
                required_pages = math.ceil(new_length / self.page_size)
                while len(table) < required_pages:
                    page = available.pop(0)
                    reserved.append(page)
                    table.append(page)
                candidate_tables.append(tuple(table))
        except IndexError as exc:
            self._allocation_failures += 1
            raise ValueError(
                "native paged KV capacity exhausted before model execution"
            ) from exc

        pinned = tuple(sorted({page for cache in paged for page in cache.block_table}))
        for page in pinned:
            self._pin_counts[page] += 1
        self._reserved_pages.update(reserved)
        return PagedBatchReservation(
            backend=self,
            request_caches=paged,
            append_lengths=append_lengths,
            candidate_tables=tuple(candidate_tables),
            reserved_pages=tuple(reserved),
            copy_pages=tuple(copy_pages),
            pinned_pages=pinned,
            staged_keys=[None] * self.num_layers,
            staged_values=[None] * self.num_layers,
        )

    def preflight(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[str | None, ...]:
        paged = tuple(_require_paged_cache(cache) for cache in caches)
        if len(paged) != len(append_lengths):
            raise ValueError("paged preflight requires matching requests")
        available = len(
            [page for page in self._free_pages if page not in self._reserved_pages]
        )
        results: list[str | None] = []
        for cache, append_length in zip(paged, append_lengths, strict=True):
            required_pages = math.ceil((cache.size() + append_length) / self.page_size)
            cost = max(0, required_pages - len(cache.block_table))
            if (
                cache.size() % self.page_size
                and cache.block_table
                and self._ref_counts[cache.block_table[-1]] > 1
            ):
                cost += 1
            if cost > available:
                results.append(
                    "native paged KV capacity exhausted before model execution"
                )
                self._allocation_failures += 1
            else:
                results.append(None)
                available -= cost
        return tuple(results)

    def validate_kv_shape(
        self,
        keys: mx.array,
        values: mx.array,
        append_lengths: tuple[int, ...],
    ) -> None:
        if keys.shape != values.shape or len(keys.shape) != 4:
            raise ValueError("paged K/V tensors must have identical rank-4 shapes")
        batch, kv_heads, sequence, head_dim = (int(value) for value in keys.shape)
        if batch != len(append_lengths):
            raise ValueError("paged K/V batch does not match reservation")
        if kv_heads != self.num_kv_heads or head_dim != self.head_dim:
            raise ValueError("paged K/V head geometry is unsupported")
        if sequence < max(append_lengths):
            raise ValueError("paged K/V sequence is shorter than scheduled append")
        if keys.dtype != self.dtype or values.dtype != self.dtype:
            raise ValueError("paged K/V dtype does not match configured cache dtype")

    def fork(
        self,
        handle: str,
        request_id: str,
        *,
        length: int | None = None,
    ) -> str:
        source = self.get(handle, self._caches[handle].request_id)
        fork_length = source.size() if length is None else int(length)
        if fork_length < 0 or fork_length > source.size():
            raise ValueError("fork length is outside source cache")
        pages = list(source.block_table[: math.ceil(fork_length / self.page_size)])
        new_handle = self.create(request_id)
        target = self._caches[new_handle]
        target.block_table = pages
        target.length = fork_length
        for page in pages:
            self._ref_counts[page] += 1
        return new_handle

    def release(self, handle: str | None) -> None:
        if handle is None:
            return
        cache = self._caches.pop(handle, None)
        if cache is None:
            return
        for page in cache.block_table:
            self._ref_counts[page] -= 1
            if self._ref_counts[page] < 0:
                raise ValueError("paged KV page reference count became negative")
            if (
                self._ref_counts[page] == 0
                and self._pin_counts[page] == 0
                and page not in self._free_pages
            ):
                self._free_pages.append(page)
        self._free_pages.sort()

    def record_attention(self, mode: str, elapsed_ms: int) -> None:
        self._attention_mode = mode
        self._attention_time_ms = max(0, int(elapsed_ms))

    def metrics(self) -> dict[str, Any]:
        used_pages = sum(count > 0 for count in self._ref_counts)
        pinned_pages = sum(count > 0 for count in self._pin_counts)
        fragmentation = sum(
            math.ceil(cache.size() / self.page_size) * self.page_size - cache.size()
            for cache in self._caches.values()
            if cache.size()
        )
        return {
            "cache_backend": "paged-mlx",
            "attention_backend": "native-metal-paged",
            "attention_mode": self._attention_mode,
            "attention_time_ms": self._attention_time_ms,
            "total_pages": self.num_pages,
            "used_pages": used_pages,
            "free_pages": self.num_pages - used_pages,
            "pinned_pages": pinned_pages,
            "internal_fragmentation_tokens": fragmentation,
            "active_kv_bytes": used_pages * self.bytes_per_page,
            "allocation_failures": self._allocation_failures,
            "page_size": self.page_size,
        }

    def _commit_reservation(
        self,
        reservation: PagedBatchReservation,
        staged_keys: tuple[mx.array, ...],
        staged_values: tuple[mx.array, ...],
    ) -> tuple[int, ...]:
        old_counts = Counter(
            page for cache in reservation.request_caches for page in cache.block_table
        )
        new_counts = Counter(
            page for table in reservation.candidate_tables for page in table
        )
        candidate_ref_counts = list(self._ref_counts)
        for page, count in old_counts.items():
            candidate_ref_counts[page] -= count
        for page, count in new_counts.items():
            candidate_ref_counts[page] += count
        if any(count < 0 for count in candidate_ref_counts):
            raise ValueError("paged KV page reference count became negative")
        self._ref_counts = candidate_ref_counts
        self.key_pages = list(staged_keys)
        self.value_pages = list(staged_values)
        for cache, table, append_length in zip(
            reservation.request_caches,
            reservation.candidate_tables,
            reservation.append_lengths,
            strict=True,
        ):
            cache.block_table = list(table)
            cache.length += append_length
        for page in reservation.reserved_pages:
            if page in self._free_pages:
                self._free_pages.remove(page)
        for page, count in enumerate(self._ref_counts):
            if (
                count == 0
                and self._pin_counts[page] == 0
                and page not in self._free_pages
                and page not in reservation.reserved_pages
            ):
                self._free_pages.append(page)
        self._finish_reservation(reservation)
        self._free_pages.sort()
        return tuple(cache.size() for cache in reservation.request_caches)

    def _abort_reservation(self, reservation: PagedBatchReservation) -> None:
        self._finish_reservation(reservation)

    def _finish_reservation(self, reservation: PagedBatchReservation) -> None:
        self._reserved_pages.difference_update(reservation.reserved_pages)
        for page in reservation.pinned_pages:
            self._pin_counts[page] -= 1
            if self._pin_counts[page] < 0:
                raise ValueError("paged KV page pin count became negative")


def _require_array(value: mx.array | None) -> mx.array:
    if value is None:
        raise ValueError("missing staged MLX array")
    return value


def _require_dense_cache(cache: RequestCache) -> DenseRequestCache:
    if not isinstance(cache, DenseRequestCache):
        raise TypeError("dense backend received incompatible request cache")
    return cache


def _require_paged_cache(cache: RequestCache) -> PagedRequestCache:
    if not isinstance(cache, PagedRequestCache):
        raise TypeError("paged backend received incompatible request cache")
    return cache
