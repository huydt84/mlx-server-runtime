"""Paged KV backend for models that mix global and sliding-window attention.

A sliding-window layer reads only the last ``window_size`` keys, so keeping its
whole history would spend most of the KV budget on entries no query can reach.
This backend keeps two page pools with the same page size:

- the global pool stores the layers that attend to the whole context, with one
  block table per request covering ``[0, length)``;
- the windowed pool stores the sliding layers, with one block table per request
  covering ``[window_base, length)``. Committing a step returns the pages that
  fell behind the window, so a request never holds more than its window plus
  one step.

Every layer also keeps request-local convolution history. A committed step
stores the batched history tensor and the request's row, so an unchanged batch
reuses it without restacking.

A reservation may be committed with fewer tokens than it staged. Speculative
decoding uses this to keep only accepted tokens: lengths, page ownership, and
convolution history all stop at the accepted token, and later steps overwrite
the rest.
"""

from __future__ import annotations

import heapq
import math
import uuid
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import mlx.core as mx

from .cache import RequestCache
from .causal_conv import CONV_LAGS, TokenLayout

_CAPACITY_ERROR = "native paged KV capacity exhausted before model execution"
_WINDOW_CAPACITY_ERROR = "native windowed KV capacity exhausted before model execution"
_MAX_FULL_WINDOWS = 16
MAX_PARTIAL_TOKENS = 16
"""Most tokens of a step that may keep only some of them."""


@dataclass(frozen=True)
class ConvStateRow:
    """One request's convolution history inside a batched history tensor.

    Attributes:
        batch: ``(rows, lags - 1, channels)`` histories of one step.
        row: The request's row.
    """

    batch: mx.array
    row: int

    def value(self) -> mx.array:
        """Return the ``(lags - 1, channels)`` inputs before the next token."""

        return self.batch[self.row]


@dataclass(frozen=True)
class _StagedConv:
    """Convolution inputs and history of one uncommitted step.

    Steps that cannot keep a part of their tokens hold no inputs.
    """

    before: mx.array
    inputs: mx.array | None
    after: mx.array


@dataclass
class WindowedRequestCache:
    """Logical request-to-page mapping for both pools."""

    handle: str
    request_id: str
    length: int = 0
    block_table: list[int] = field(default_factory=list)
    window_table: list[int] = field(default_factory=list)
    window_base: int = 0
    conv_states: dict[tuple[int, int], ConvStateRow] = field(default_factory=dict)
    draft_context: Any = None
    """Drafter state of the request, when speculative decoding is enabled."""

    def size(self) -> int:
        return self.length


@dataclass
class _PagePool:
    """Fixed set of pages with single-owner allocation."""

    num_pages: int
    _free: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self._free = list(range(self.num_pages))

    @property
    def free(self) -> int:
        return len(self._free)

    def allocate(self, count: int) -> list[int]:
        if count > len(self._free):
            raise ValueError(_CAPACITY_ERROR)
        return [heapq.heappop(self._free) for _ in range(count)]

    def release(self, pages: list[int]) -> None:
        for page in pages:
            heapq.heappush(self._free, page)

    def reset(self) -> None:
        self._free = list(range(self.num_pages))


@dataclass
class _StepTable:
    """Block tables and append slots one pool uses in one step."""

    tables: tuple[tuple[int, ...], ...]
    bases: tuple[int, ...]
    reserved: tuple[tuple[int, ...], ...]
    block_tables: mx.array = field(init=False)
    base_array: mx.array = field(init=False)
    slot_array: mx.array = field(init=False)
    slot_start: int | None = field(init=False, default=None)

    def build(
        self,
        page_size: int,
        lengths: tuple[int, ...],
        append_lengths: tuple[int, ...],
    ) -> None:
        width = max(len(table) for table in self.tables)
        self.block_tables = mx.array(
            [list(table) + [0] * (width - len(table)) for table in self.tables],
            dtype=mx.int32,
        )
        self.base_array = mx.array(self.bases, dtype=mx.int32)
        slots: list[int] = []
        for table, base, length, append in zip(
            self.tables, self.bases, lengths, append_lengths, strict=True
        ):
            for position in range(length - base, length - base + append):
                page = table[position // page_size]
                slots.append(page * page_size + position % page_size)
        self.slot_array = mx.array(slots, dtype=mx.int32)
        if all(b == a + 1 for a, b in pairwise(slots)):
            self.slot_start = slots[0]


@dataclass
class WindowedBatchReservation:
    """All-request, all-layer transactional append to both pools.

    Keys and values are written as each layer is staged. Only page ownership,
    lengths, and convolution history are transactional: staged rows land in
    reserved pages or past the committed length of pages the request owns, so
    an aborted reservation leaves no visible change.
    """

    backend: WindowedPagedKVCacheBackend
    request_caches: tuple[WindowedRequestCache, ...]
    append_lengths: tuple[int, ...]
    global_step: _StepTable
    window_step: _StepTable
    layout: TokenLayout = field(init=False)
    max_total_length: int = field(init=False)
    _staged_layers: set[int] = field(default_factory=set, init=False)
    _staged_conv: dict[tuple[int, int], _StagedConv] = field(
        default_factory=dict, init=False
    )
    _finished: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        lengths = self.cache_lengths
        self.global_step.build(self.backend.page_size, lengths, self.append_lengths)
        self.window_step.build(self.backend.page_size, lengths, self.append_lengths)
        self.layout = TokenLayout(lengths, self.append_lengths)
        self.max_total_length = max(
            length + append
            for length, append in zip(lengths, self.append_lengths, strict=True)
        )

    @property
    def cache_lengths(self) -> tuple[int, ...]:
        return tuple(cache.size() for cache in self.request_caches)

    @property
    def token_lengths(self) -> tuple[int, ...]:
        return self.append_lengths

    def step_table(self, layer_index: int) -> _StepTable:
        """Return the tables of the pool that stores ``layer_index``."""

        if layer_index in self.backend.window_layers:
            return self.window_step
        return self.global_step

    def stage_layer(self, layer_index: int, kv: mx.array) -> mx.array:
        """Write one layer's keys and values and return its pool.

        Args:
            layer_index: KV layer.
            kv: ``(tokens, 2, kv_heads, head_dim)`` keys then values of every
                row's tokens, concatenated.

        Returns:
            The layer's page pool including this append.

        Raises:
            ValueError: If the layer was already staged or the shape is wrong.
        """

        backend = self.backend
        if self._finished:
            raise ValueError("cache reservation already finished")
        if layer_index in self._staged_layers:
            raise ValueError(f"layer {layer_index} staged more than once")
        element = (2, backend.num_kv_heads, backend.head_dim)
        if tuple(kv.shape) != (self.layout.tokens, *element):
            raise ValueError("windowed K/V geometry does not match the reservation")
        if kv.dtype != backend.dtype:
            kv = kv.astype(backend.dtype)
        table = self.step_table(layer_index)
        pool = backend.kv_pages[layer_index]
        flat = pool.reshape(-1, *element)
        if table.slot_start is not None:
            flat[table.slot_start : table.slot_start + self.layout.tokens] = kv
        else:
            flat[table.slot_array] = kv
        pool = flat.reshape(pool.shape)
        backend.kv_pages[layer_index] = pool
        self._staged_layers.add(layer_index)
        return pool

    def conv_history(
        self, layer_index: int, conv_index: int, channels: int
    ) -> mx.array:
        """Return ``(rows, lags - 1, channels)`` inputs preceding this step."""

        key = (layer_index, conv_index)
        rows = [cache.conv_states.get(key) for cache in self.request_caches]
        first = rows[0]
        if (
            first is not None
            and int(first.batch.shape[0]) == len(rows)
            and all(
                row is not None and row.batch is first.batch and row.row == index
                for index, row in enumerate(rows)
            )
        ):
            return first.batch
        return mx.stack(
            [
                row.value()
                if row is not None
                else mx.zeros((CONV_LAGS - 1, channels), dtype=mx.float32)
                for row in rows
            ]
        )

    def stage_conv(
        self,
        layer_index: int,
        conv_index: int,
        before: mx.array,
        inputs: mx.array,
        after: mx.array,
    ) -> None:
        """Stage one convolution's history.

        Args:
            layer_index: KV layer.
            conv_index: Convolution of that layer.
            before: Result of :meth:`conv_history`.
            inputs: ``(tokens, channels)`` inputs of this step.
            after: History when every staged token is kept.
        """

        if self._finished:
            raise ValueError("cache reservation already finished")
        partial = self.layout.tokens <= MAX_PARTIAL_TOKENS
        self._staged_conv[(layer_index, conv_index)] = _StagedConv(
            before, inputs if partial else None, after
        )

    def commit(
        self,
        *,
        synchronize: bool = True,
        accepted: tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Publish the append, keeping ``accepted`` tokens of each row.

        Args:
            synchronize: Wait for the staged writes before publishing.
            accepted: Tokens to keep per row; every staged token by default.

        Returns:
            Committed length per request.

        Raises:
            ValueError: If a layer was not staged, a count is out of range, or
                a step of more than ``MAX_PARTIAL_TOKENS`` tokens keeps a part.
        """

        if self._finished:
            raise ValueError("cache reservation already finished")
        if len(self._staged_layers) != self.backend.num_layers:
            raise ValueError("paged cache commit missing one or more layers")
        kept = self.append_lengths if accepted is None else tuple(accepted)
        if len(kept) != len(self.append_lengths) or any(
            not 0 <= count <= append
            for count, append in zip(kept, self.append_lengths, strict=True)
        ):
            raise ValueError("accepted tokens are outside the staged append")
        if kept != self.append_lengths and self.layout.tokens > MAX_PARTIAL_TOKENS:
            raise ValueError("only short steps can keep a part of their tokens")
        if synchronize:
            mx.eval(
                *(self.backend.kv_pages[index] for index in self._staged_layers),
                *(staged.after for staged in self._staged_conv.values()),
            )
        lengths = self.backend._commit_reservation(self, kept)
        self._finished = True
        return lengths

    def abort(self) -> None:
        if not self._finished:
            self.backend._abort_reservation(self)
            self._finished = True


@dataclass
class WindowedPagedKVCacheBackend:
    """Two page pools with transactional request block tables.

    Attributes:
        num_layers: KV layers across both pools.
        num_kv_heads: Key/value heads per layer.
        head_dim: Elements per head.
        page_size: Tokens per page.
        budget_bytes: Bytes shared by both pools.
        window_layers: Layers stored in the windowed pool.
        window_size: Keys a windowed query can attend to.
        max_append_tokens: Largest append one request makes in a step.
        dtype: Stored K/V dtype.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    budget_bytes: int
    window_layers: frozenset[int]
    window_size: int
    max_append_tokens: int = 1024
    dtype: Any = mx.float16
    _caches: dict[str, WindowedRequestCache] = field(default_factory=dict, init=False)
    _allocation_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.page_size not in (8, 16, 32):
            raise ValueError("native KV page size must be one of 8, 16, or 32")
        if (
            min(self.num_layers, self.num_kv_heads, self.head_dim, self.window_size)
            <= 0
        ):
            raise ValueError("native KV geometry must be positive")
        if self.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("native paged KV supports float16, bfloat16, or float32")
        layers = frozenset(range(self.num_layers))
        if not self.window_layers or not self.window_layers < layers:
            raise ValueError("windowed KV needs both windowed and global layers")
        layer_page = (
            2
            * self.page_size
            * self.num_kv_heads
            * self.head_dim
            * int(mx.array(0, dtype=self.dtype).itemsize)
        )
        self.window_bytes_per_page = layer_page * len(self.window_layers)
        self.bytes_per_page = layer_page * (self.num_layers - len(self.window_layers))
        self.num_window_pages = self._window_pages()
        self.num_pages = (
            self.budget_bytes - self.num_window_pages * self.window_bytes_per_page
        ) // self.bytes_per_page
        self._global = _PagePool(self.num_pages)
        self._window = _PagePool(self.num_window_pages)
        element = (self.page_size, 2, self.num_kv_heads, self.head_dim)
        self.kv_pages = [
            mx.zeros(
                (
                    self.num_window_pages
                    if index in self.window_layers
                    else self.num_pages,
                    *element,
                ),
                dtype=self.dtype,
            )
            for index in range(self.num_layers)
        ]

    def window_pages_for(self, tokens: int) -> int:
        """Return the most windowed pages a request of ``tokens`` tokens holds.

        Args:
            tokens: Final context length of the request.

        Returns:
            Pages that cover the window plus one step, or the whole request
            when it is shorter.
        """

        held = min(max(1, tokens), self.window_size + self.max_append_tokens)
        return math.ceil(held / self.page_size) + 1

    def _window_pages(self) -> int:
        """Split the budget: how many pages the windowed pool receives.

        One request needs a full window. Beyond that the windowed pool takes
        up to half of the budget, but never more than the active requests can
        use, so long contexts keep the rest.
        """

        single = self.window_pages_for(self.window_size + self.max_append_tokens)
        half = self.budget_bytes // 2 // self.window_bytes_per_page
        pages = max(single, min(half, single * _MAX_FULL_WINDOWS))
        needed = pages * self.window_bytes_per_page + self.bytes_per_page
        if needed > self.budget_bytes:
            raise ValueError(
                "native text cache budget cannot hold one windowed request: "
                f"{needed} bytes needed, {self.budget_bytes} available"
            )
        return pages

    def create(self, request_id: str) -> str:
        handle = f"paged-windowed-kv-{request_id}-{uuid.uuid4().hex}"
        self._caches[handle] = WindowedRequestCache(handle, request_id)
        return handle

    def get(self, handle: str | None, request_id: str) -> WindowedRequestCache:
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

    def free_page_count(self) -> int:
        """Return global pages not owned by any request."""

        return self._global.free

    def preflight(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[str | None, ...]:
        windowed = tuple(_require_windowed_cache(cache) for cache in caches)
        if len(windowed) != len(append_lengths):
            raise ValueError("paged preflight requires matching requests")
        free_global, free_window = self._global.free, self._window.free
        results: list[str | None] = []
        for cache, append in zip(windowed, append_lengths, strict=True):
            global_cost, window_cost = self._page_costs(cache, append)
            error = None
            if global_cost > free_global:
                error = _CAPACITY_ERROR
            elif window_cost > free_window:
                error = _WINDOW_CAPACITY_ERROR
            if error is None:
                free_global -= global_cost
                free_window -= window_cost
            else:
                self._allocation_failures += 1
            results.append(error)
        return tuple(results)

    def reserve_batch(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> WindowedBatchReservation:
        windowed = tuple(_require_windowed_cache(cache) for cache in caches)
        if len(windowed) != len(append_lengths) or not windowed:
            raise ValueError("paged cache reservation requires matching requests")
        if len({cache.handle for cache in windowed}) != len(windowed):
            raise ValueError("paged cache reservation contains duplicate handles")
        if any(length <= 0 for length in append_lengths):
            raise ValueError("paged cache append lengths must be positive")
        costs = [
            self._page_costs(cache, append)
            for cache, append in zip(windowed, append_lengths, strict=True)
        ]
        if (
            sum(cost for cost, _ in costs) > self._global.free
            or sum(cost for _, cost in costs) > self._window.free
        ):
            self._allocation_failures += 1
            raise ValueError(_CAPACITY_ERROR)
        global_pages = [self._global.allocate(cost) for cost, _ in costs]
        window_pages = [self._window.allocate(cost) for _, cost in costs]
        return WindowedBatchReservation(
            backend=self,
            request_caches=windowed,
            append_lengths=tuple(append_lengths),
            global_step=_StepTable(
                tables=tuple(
                    (*cache.block_table, *pages)
                    for cache, pages in zip(windowed, global_pages, strict=True)
                ),
                bases=(0,) * len(windowed),
                reserved=tuple(tuple(pages) for pages in global_pages),
            ),
            window_step=_StepTable(
                tables=tuple(
                    (*cache.window_table, *pages)
                    for cache, pages in zip(windowed, window_pages, strict=True)
                ),
                bases=tuple(cache.window_base for cache in windowed),
                reserved=tuple(tuple(pages) for pages in window_pages),
            ),
        )

    def release(self, handle: str | None) -> None:
        if handle is None:
            return
        cache = self._caches.pop(handle, None)
        if cache is None:
            return
        self._global.release(cache.block_table)
        self._window.release(cache.window_table)

    def fork(
        self,
        handle: str,
        request_id: str,
        *,
        length: int | None = None,
    ) -> str:
        """Reject forking: released window pages cannot be shared."""

        raise ValueError("windowed KV caches cannot be forked")

    def metrics(self) -> dict[str, Any]:
        used = self.num_pages - self._global.free
        window_used = self.num_window_pages - self._window.free
        return {
            "cache_backend": "paged-mlx-windowed",
            "total_pages": self.num_pages,
            "used_pages": used,
            "free_pages": self._global.free,
            "pinned_pages": 0,
            "internal_fragmentation_tokens": sum(
                len(cache.block_table) * self.page_size - cache.size()
                for cache in self._caches.values()
                if cache.size()
            ),
            "active_kv_bytes": used * self.bytes_per_page
            + window_used * self.window_bytes_per_page,
            "allocation_failures": self._allocation_failures,
            "page_size": self.page_size,
            "window_total_pages": self.num_window_pages,
            "window_used_pages": window_used,
        }

    def reset(self, *, reset_counters: bool) -> None:
        """Release every page mapping and optionally reset allocation failures."""

        self._caches.clear()
        self._global.reset()
        self._window.reset()
        if reset_counters:
            self._allocation_failures = 0

    def _page_costs(self, cache: WindowedRequestCache, append: int) -> tuple[int, int]:
        total = cache.size() + append
        return (
            max(0, math.ceil(total / self.page_size) - len(cache.block_table)),
            max(
                0,
                math.ceil((total - cache.window_base) / self.page_size)
                - len(cache.window_table),
            ),
        )

    def _commit_reservation(
        self,
        reservation: WindowedBatchReservation,
        kept: tuple[int, ...],
    ) -> tuple[int, ...]:
        page_size = self.page_size
        start = 0
        for index, (cache, count) in enumerate(
            zip(reservation.request_caches, kept, strict=True)
        ):
            cache.length += count
            cache.block_table = self._trim(
                self._global,
                reservation.global_step.tables[index],
                0,
                math.ceil(cache.length / page_size),
            )
            base = cache.window_base
            keep_from = max(0, cache.length - self.window_size + 1) // page_size
            first = max(base // page_size, keep_from)
            cache.window_table = self._trim(
                self._window,
                reservation.window_step.tables[index],
                first - base // page_size,
                math.ceil(cache.length / page_size) - base // page_size,
            )
            cache.window_base = first * page_size
            full = count == reservation.append_lengths[index]
            for key, staged in reservation._staged_conv.items():
                cache.conv_states[key] = (
                    ConvStateRow(staged.after, index)
                    if full
                    else ConvStateRow(
                        _partial_history(staged, start, index, count)[None], 0
                    )
                )
            start += reservation.append_lengths[index]
        reservation._staged_conv.clear()
        return tuple(cache.size() for cache in reservation.request_caches)

    @staticmethod
    def _trim(
        pool: _PagePool, table: tuple[int, ...], first: int, last: int
    ) -> list[int]:
        """Keep ``table[first:last]`` and return the other pages to the pool."""

        pool.release([*table[:first], *table[last:]])
        return list(table[first:last])

    def _abort_reservation(self, reservation: WindowedBatchReservation) -> None:
        for pages in reservation.global_step.reserved:
            self._global.release(list(pages))
        for pages in reservation.window_step.reserved:
            self._window.release(list(pages))
        reservation._staged_conv.clear()


def _partial_history(staged: _StagedConv, start: int, row: int, count: int) -> mx.array:
    """Return a row's history when only its first ``count`` tokens are kept."""

    assert staged.inputs is not None
    seen = mx.concatenate([staged.before[row], staged.inputs[start : start + count]])
    return seen[count : count + CONV_LAGS - 1]


def _require_windowed_cache(cache: RequestCache) -> WindowedRequestCache:
    if not isinstance(cache, WindowedRequestCache):
        raise TypeError("windowed backend received incompatible request cache")
    return cache
