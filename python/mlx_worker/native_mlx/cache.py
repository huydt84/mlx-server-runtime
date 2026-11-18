"""Shared KV-cache backends for native MLX execution."""

from __future__ import annotations

import bisect
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Protocol

import mlx.core as mx


@dataclass(frozen=True)
class RecurrentStateRow:
    """One request's row of a batch-stacked recurrent state tensor.

    Committing a step keeps a reference to the whole batched state instead of
    slicing it per request. When the next step runs the same requests in the
    same order, the batched tensor is reused as-is, so a large recurrent state
    (Gated DeltaNet keeps ~3 MiB per layer per request) is not copied by a
    re-stack on every decode step.
    """

    batch: mx.array
    row: int

    def value(self) -> mx.array:
        """Return this request's state without the batch axis."""

        return self.batch[self.row]


def gather_recurrent_rows(
    rows: list[RecurrentStateRow | None],
    shape: tuple[int, ...],
    dtype: Any,
) -> mx.array:
    """Batch per-request recurrent states, reusing an unchanged batch tensor.

    Args:
        rows: Committed state row per request, ``None`` for a fresh request.
        shape: Per-request state shape without the batch axis.
        dtype: State dtype used for fresh requests.

    Returns:
        The states stacked on a new leading batch axis.
    """

    first = rows[0] if rows else None
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
        [row.value() if row is not None else mx.zeros(shape, dtype) for row in rows],
        axis=0,
    )


def valid_token_mask(append_lengths: tuple[int, ...], seq_len: int) -> mx.array | None:
    """Return a ``(batch, seq_len)`` mask of real tokens, or ``None`` if unpadded."""

    if all(length == seq_len for length in append_lengths):
        return None
    return (
        mx.arange(seq_len)[None, :] < mx.array(append_lengths, dtype=mx.int32)[:, None]
    )


@dataclass(frozen=True)
class KVCacheGeometry:
    """Architecture-declared dimensions required by a shared KV backend."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype: Any
    kv_layer_indices: tuple[int, ...] | None = None
    """Layers that store paged KV; ``None`` means every layer.

    Hybrid models whose other layers keep only request-local state list their
    attention layers here so the KV budget is not spent on unused pages.
    """


class RequestCache(Protocol):
    """Opaque request cache state resolved only inside the executor/backend."""

    handle: str
    request_id: str

    def size(self) -> int: ...


class BatchCacheReservation(Protocol):
    """Transactional cache append reserved for one physical model step."""

    request_caches: tuple[RequestCache, ...]
    append_lengths: tuple[int, ...]

    def commit(self, *, synchronize: bool = True) -> tuple[int, ...]:
        """Publish the staged append.

        ``synchronize=False`` publishes lengths and page ownership while the
        staged MLX writes may still be running; later steps that read the
        cache depend on those writes through the MLX graph.
        """

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

    def fork(
        self,
        handle: str,
        request_id: str,
        *,
        length: int | None = None,
    ) -> str: ...

    def metrics(self) -> dict[str, Any]: ...

    def reset(self, *, reset_counters: bool) -> None: ...


@dataclass
class DenseLayerCache:
    """Contiguous cache retained only for diagnostics and test parity."""

    keys: mx.array | None = None
    values: mx.array | None = None
    offset: int = 0
    conv_state: mx.array | None = None
    recurrent_state: mx.array | None = None

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
    _committed_conv_states: tuple[mx.array, ...] = field(init=False, default=())
    _committed_recurrent_states: tuple[mx.array, ...] = field(init=False, default=())

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
        if self._committed_conv_states:
            if len(self._committed_conv_states) != len(self.request_layers):
                raise ValueError("batched cache has incomplete convolution state")
            return
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
        if self._committed_conv_states:
            for layer, state in zip(
                self.request_layers,
                self._committed_conv_states,
                strict=True,
            ):
                layer.conv_state = state
            for layer, state in zip(
                self.request_layers,
                self._committed_recurrent_states,
                strict=False,
            ):
                layer.recurrent_state = state
            return
        for layer, keys, values in zip(
            self.request_layers,
            self._committed_keys,
            self._committed_values,
            strict=True,
        ):
            layer.keys = keys
            layer.values = values
            layer.offset = int(keys.shape[2])

    def stage_conv_states(self, states: tuple[mx.array, ...]) -> None:
        if self._committed_keys or self._committed_values:
            raise ValueError("dense layer cannot stage both KV and convolution state")
        self._committed_conv_states = states

    def stage_recurrent_states(self, states: tuple[mx.array, ...]) -> None:
        if self._committed_keys or self._committed_values:
            raise ValueError("dense layer cannot stage both KV and recurrent state")
        self._committed_recurrent_states = states


@dataclass
class DenseRequestCache:
    """Opaque dense request cache retained outside the production composition."""

    handle: str
    request_id: str
    layers: list[DenseLayerCache]
    length: int = 0

    def size(self) -> int:
        return self.length


@dataclass
class DenseBatchReservation:
    """Transactional dense reservation used by the reference attention path."""

    request_caches: tuple[DenseRequestCache, ...]
    append_lengths: tuple[int, ...]
    layer_views: tuple[DenseBatchLayerCache, ...]
    _finished: bool = False

    def commit(self, *, synchronize: bool = True) -> tuple[int, ...]:
        del synchronize
        if self._finished:
            raise ValueError("cache reservation already finished")
        for layer in self.layer_views:
            layer.validate_commit()
        for layer in self.layer_views:
            layer.commit()
        for cache, append_length in zip(
            self.request_caches, self.append_lengths, strict=True
        ):
            cache.length += append_length
        self._finished = True
        return tuple(cache.size() for cache in self.request_caches)

    def prepare_conv_state(
        self,
        layer_index: int,
        values: mx.array,
        cache_size: int,
    ) -> mx.array:
        """Return padded convolution input prefixed with each request state."""

        states = []
        for cache in self.request_caches:
            state = cache.layers[layer_index].conv_state
            if state is None:
                state = mx.zeros(
                    (cache_size - 1, int(values.shape[-1])), dtype=values.dtype
                )
            states.append(state)
        state_batch = mx.stack(states, axis=0)
        valid = (
            mx.arange(int(values.shape[1]))[None, :]
            < mx.array(self.append_lengths, dtype=mx.int32)[:, None]
        )
        values = mx.where(valid[..., None], values, 0)
        return mx.concatenate([state_batch, values], axis=1)

    def stage_conv_state(
        self,
        layer_index: int,
        combined: mx.array,
        cache_size: int,
    ) -> None:
        """Stage per-request convolution state for a transactional commit."""

        states = []
        for index, append_length in enumerate(self.append_lengths):
            states.append(
                combined[index, append_length : append_length + cache_size - 1]
            )
        self.layer_views[layer_index].stage_conv_states(tuple(states))

    def recurrent_state(
        self,
        layer_index: int,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> mx.array:
        """Return each request's recurrent state stacked on the batch axis."""

        states = []
        for cache in self.request_caches:
            state = cache.layers[layer_index].recurrent_state
            states.append(mx.zeros(shape, dtype) if state is None else state)
        return mx.stack(states, axis=0)

    def stage_recurrent_state(self, layer_index: int, state: mx.array) -> None:
        """Stage per-request recurrent state for a transactional commit."""

        self.layer_views[layer_index].stage_recurrent_states(
            tuple(state[index] for index in range(len(self.request_caches)))
        )

    def token_mask(self, seq_len: int) -> mx.array | None:
        """Return the real-token mask for this step's padded rows."""

        return valid_token_mask(self.append_lengths, seq_len)

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
        new_handle = self.create(request_id)
        target = self._caches[new_handle]
        target.length = fork_length
        for source_layer, target_layer in zip(
            source.layers, target.layers, strict=True
        ):
            if source_layer.keys is not None:
                target_layer.keys = source_layer.keys[:, :, :fork_length, :]
            if source_layer.values is not None:
                target_layer.values = source_layer.values[:, :, :fork_length, :]
            target_layer.offset = fork_length
            target_layer.conv_state = source_layer.conv_state
            target_layer.recurrent_state = source_layer.recurrent_state
        return new_handle

    def metrics(self) -> dict[str, Any]:
        return {
            "cache_backend": "dense-reference",
            "active_kv_bytes": sum(
                layer.nbytes
                for cache in self._caches.values()
                for layer in cache.layers
            ),
        }

    def reset(self, *, reset_counters: bool) -> None:
        """Release every dense request cache."""

        del reset_counters
        self._caches.clear()


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
    """All-request, all-layer transactional paged append reservation.

    K/V rows are written into the backend page pool as each layer is staged, so
    MLX can update the pool buffer in place instead of copying it every step.
    Only page ownership and request lengths are transactional: staged rows land
    in reserved pages or past the committed length of pages the request owns
    exclusively, so an aborted reservation leaves no visible change.
    """

    backend: PagedKVCacheBackend
    request_caches: tuple[PagedRequestCache, ...]
    append_lengths: tuple[int, ...]
    candidate_tables: tuple[tuple[int, ...], ...]
    reserved_pages: tuple[int, ...]
    copy_pages: tuple[tuple[int, int], ...]
    pinned_pages: tuple[int, ...]
    staged_keys: list[mx.array | None]
    staged_values: list[mx.array | None]
    _finished: bool = False
    _slot_array: mx.array = field(init=False, repr=False)
    _slot_start: int | None = field(init=False, default=None, repr=False)
    """First slot when the step's slots are consecutive, else ``None``."""
    _page_id_arrays: tuple[mx.array, ...] = field(init=False, repr=False)
    _total_lengths: tuple[int, ...] = field(init=False, repr=False)
    _max_total_length: int = field(init=False, repr=False)
    _page_table_inputs: tuple[mx.array, mx.array] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        self._page_id_arrays = tuple(
            mx.array(table, dtype=mx.int32) for table in self.candidate_tables
        )
        self._total_lengths = tuple(
            cache.size() + append_length
            for cache, append_length in zip(
                self.request_caches, self.append_lengths, strict=True
            )
        )
        self._max_total_length = max(self._total_lengths, default=0)
        slots: list[int] = []
        for cache, append_length, table in zip(
            self.request_caches,
            self.append_lengths,
            self.candidate_tables,
            strict=True,
        ):
            for position in range(cache.size(), cache.size() + append_length):
                page = table[position // self.backend.page_size]
                slots.append(
                    page * self.backend.page_size + position % self.backend.page_size
                )
        self._slot_array = mx.array(slots, dtype=mx.int32)
        if slots and all(b == a + 1 for a, b in pairwise(slots)):
            self._slot_start = slots[0]

    @property
    def token_lengths(self) -> tuple[int, ...]:
        return self.append_lengths

    def page_table_inputs(self) -> tuple[mx.array, mx.array]:
        """Return padded int32 block tables and post-append lengths per row.

        Built once per step and shared by every layer's attention.
        """

        if self._page_table_inputs is None:
            width = max(len(table) for table in self.candidate_tables)
            tables = [
                list(table) + [0] * (width - len(table))
                for table in self.candidate_tables
            ]
            self._page_table_inputs = (
                mx.array(tables, dtype=mx.int32),
                mx.array(self._total_lengths, dtype=mx.int32),
            )
        return self._page_table_inputs

    @property
    def cache_lengths(self) -> tuple[int, ...]:
        return tuple(cache.size() for cache in self.request_caches)

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
        slot_array = self._slot_array
        sequence = int(keys.shape[2])
        if all(length == sequence for length in self.append_lengths):
            # No row is padded (every decode step, for one): the slot order is
            # the row-major (row, position) order, so a single reshape packs
            # the step without per-row slices and a concatenation.
            packed_key_rows = keys.swapaxes(1, 2).reshape(
                -1, self.backend.num_kv_heads, self.backend.head_dim
            )
            packed_value_rows = values.swapaxes(1, 2).reshape(
                -1, self.backend.num_kv_heads, self.backend.head_dim
            )
        else:
            packed_key_rows = mx.concatenate(
                [
                    keys[index, :, :length, :].transpose(1, 0, 2)
                    for index, length in enumerate(self.append_lengths)
                ],
                axis=0,
            )
            packed_value_rows = mx.concatenate(
                [
                    values[index, :, :length, :].transpose(1, 0, 2)
                    for index, length in enumerate(self.append_lengths)
                ],
                axis=0,
            )
        flat_shape = (
            self.backend.num_pages * self.backend.page_size,
            self.backend.num_kv_heads,
            self.backend.head_dim,
        )
        flat_keys = key_cache.reshape(flat_shape)
        flat_values = value_cache.reshape(flat_shape)
        if self._slot_start is not None:
            # Consecutive slots (one decode row, for one) take a plain slice
            # copy, which is cheaper on the GPU than an indexed scatter.
            end = self._slot_start + int(packed_key_rows.shape[0])
            flat_keys[self._slot_start : end] = packed_key_rows
            flat_values[self._slot_start : end] = packed_value_rows
        else:
            flat_keys[slot_array] = packed_key_rows
            flat_values[slot_array] = packed_value_rows
        key_cache = flat_keys.reshape(key_cache.shape)
        value_cache = flat_values.reshape(value_cache.shape)
        # Drop the pool's reference to the previous buffer so the scatter above
        # can donate it rather than allocate and copy a whole new pool layer.
        self.backend.key_pages[layer_index] = key_cache
        self.backend.value_pages[layer_index] = value_cache
        self.staged_keys[layer_index] = key_cache
        self.staged_values[layer_index] = value_cache
        return key_cache, value_cache

    def commit(self, *, synchronize: bool = True) -> tuple[int, ...]:
        if self._finished:
            raise ValueError("cache reservation already finished")
        for keys, values in zip(self.staged_keys, self.staged_values, strict=True):
            if (keys is None) != (values is None) or keys is None:
                raise ValueError("paged cache commit missing one or more layers")
        staged_keys = tuple(
            _require_array(value) if value is not None else None
            for value in self.staged_keys
        )
        staged_values = tuple(
            _require_array(value) if value is not None else None
            for value in self.staged_values
        )
        if synchronize:
            mx.eval(
                *(value for value in staged_keys if value is not None),
                *(value for value in staged_values if value is not None),
            )
        lengths = self.backend._commit_reservation(self, staged_keys, staged_values)
        self._finished = True
        return lengths

    def abort(self) -> None:
        if not self._finished:
            self.backend._abort_reservation(self)
            self._finished = True


@dataclass
class HybridRequestCache(PagedRequestCache):
    """Paged request cache with request-local state for hybrid models.

    ``conv_states`` holds short-convolution history (LFM2 and Gated DeltaNet
    layers); ``recurrent_states`` holds linear-attention state (Gated DeltaNet
    layers only).
    """

    conv_states: dict[int, mx.array] = field(default_factory=dict)
    recurrent_states: dict[int, RecurrentStateRow] = field(default_factory=dict)


@dataclass
class HybridBatchReservation(PagedBatchReservation):
    """Transactional paged KV plus convolution-state reservation."""

    staged_states: list[mx.array | None] = field(default_factory=list)
    staged_recurrent_states: list[mx.array | None] = field(default_factory=list)

    def stage_layer(
        self,
        layer_index: int,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        if self.staged_states[layer_index] is not None:
            raise ValueError(f"layer {layer_index} staged both state and KV")
        return super().stage_layer(layer_index, keys, values)

    def prepare_conv_state(
        self,
        layer_index: int,
        values: mx.array,
        cache_size: int,
    ) -> mx.array:
        """Return values prefixed with the request-local convolution state."""

        states = []
        for cache in self.request_caches:
            hybrid = _require_hybrid_cache(cache)
            state = hybrid.conv_states.get(layer_index)
            if state is None:
                state = mx.zeros(
                    (cache_size - 1, int(values.shape[-1])), dtype=values.dtype
                )
            states.append(state)
        state_batch = mx.stack(states, axis=0)
        valid = (
            mx.arange(int(values.shape[1]))[None, :]
            < mx.array(self.append_lengths, dtype=mx.int32)[:, None]
        )
        values = mx.where(valid[..., None], values, 0)
        return mx.concatenate([state_batch, values], axis=1)

    def stage_conv_state(
        self,
        layer_index: int,
        combined: mx.array,
        cache_size: int,
    ) -> None:
        """Stage one convolution state layer for transactional commit."""

        if self._finished:
            raise ValueError("cache reservation already finished")
        if self.staged_keys[layer_index] is not None:
            raise ValueError(f"layer {layer_index} staged both state and KV")
        states = tuple(
            combined[index, append_length : append_length + cache_size - 1]
            for index, append_length in enumerate(self.append_lengths)
        )
        self.staged_states[layer_index] = mx.stack(states, axis=0)

    def recurrent_state(
        self,
        layer_index: int,
        shape: tuple[int, ...],
        dtype: Any,
    ) -> mx.array:
        """Return each request's recurrent state stacked on the batch axis."""

        return gather_recurrent_rows(
            [
                _require_hybrid_cache(cache).recurrent_states.get(layer_index)
                for cache in self.request_caches
            ],
            shape,
            dtype,
        )

    def stage_recurrent_state(self, layer_index: int, state: mx.array) -> None:
        """Stage one layer's batch-stacked recurrent state for commit."""

        if self._finished:
            raise ValueError("cache reservation already finished")
        if self.staged_keys[layer_index] is not None:
            raise ValueError(f"layer {layer_index} staged both state and KV")
        if int(state.shape[0]) != len(self.request_caches):
            raise ValueError("recurrent state batch does not match the reservation")
        self.staged_recurrent_states[layer_index] = state

    def token_mask(self, seq_len: int) -> mx.array | None:
        """Return the real-token mask for this step's padded rows."""

        return valid_token_mask(self.append_lengths, seq_len)

    def commit(self, *, synchronize: bool = True) -> tuple[int, ...]:
        if self._finished:
            raise ValueError("cache reservation already finished")
        for keys, values, state, recurrent in zip(
            self.staged_keys,
            self.staged_values,
            self.staged_states,
            self.staged_recurrent_states,
            strict=True,
        ):
            kv_complete = keys is not None and values is not None
            state_complete = state is not None
            if kv_complete == state_complete:
                raise ValueError("hybrid cache commit requires one operation per layer")
            if recurrent is not None and not state_complete:
                raise ValueError("recurrent state requires a convolution state")
        staged_keys = tuple(
            _require_array(value) if value is not None else None
            for value in self.staged_keys
        )
        staged_values = tuple(
            _require_array(value) if value is not None else None
            for value in self.staged_values
        )
        staged_states = tuple(
            _require_array(value) if value is not None else None
            for value in self.staged_states
        )
        staged_recurrent = tuple(self.staged_recurrent_states)
        if synchronize:
            mx.eval(
                *(value for value in staged_keys if value is not None),
                *(value for value in staged_values if value is not None),
                *(value for value in staged_states if value is not None),
                *(value for value in staged_recurrent if value is not None),
            )
        lengths = self.backend._commit_hybrid_reservation(
            self, staged_keys, staged_values, staged_states, staged_recurrent
        )
        self._finished = True
        return lengths


@dataclass
class PagedKVCacheBackend:
    """Fixed-size MLX page pool with transactional request block tables."""

    num_layers: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    budget_bytes: int
    dtype: Any = mx.float16
    kv_layer_indices: tuple[int, ...] | None = None
    _caches: dict[str, PagedRequestCache] = field(default_factory=dict, init=False)
    _reserved_pages: set[int] = field(default_factory=set, init=False)
    _allocation_failures: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.page_size not in (8, 16, 32):
            raise ValueError("native KV page size must be one of 8, 16, or 32")
        if min(self.num_layers, self.num_kv_heads, self.head_dim) <= 0:
            raise ValueError("native KV geometry must be positive")
        if self.dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError("native paged KV supports float16, bfloat16, or float32")
        kv_layers = (
            frozenset(range(self.num_layers))
            if self.kv_layer_indices is None
            else frozenset(self.kv_layer_indices)
        )
        if not kv_layers or not kv_layers <= frozenset(range(self.num_layers)):
            raise ValueError("native KV layer indices must name existing layers")
        dtype_bytes = int(mx.array(0, dtype=self.dtype).itemsize)
        self.bytes_per_page = (
            2
            * len(kv_layers)
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
        # State-only layers keep an empty placeholder so per-layer indexing
        # stays uniform; staging KV into one fails on its zero-page shape.
        empty = (0, *shape[1:])
        self.key_pages = [
            mx.zeros(shape if index in kv_layers else empty, dtype=self.dtype)
            for index in range(self.num_layers)
        ]
        self.value_pages = [
            mx.zeros(shape if index in kv_layers else empty, dtype=self.dtype)
            for index in range(self.num_layers)
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

        available = (
            page for page in self._free_pages if page not in self._reserved_pages
        )
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
                    replacement = next(available)
                    reserved.append(replacement)
                    copy_pages.append((table[-1], replacement))
                    table[-1] = replacement
                required_pages = math.ceil(new_length / self.page_size)
                while len(table) < required_pages:
                    page = next(available)
                    reserved.append(page)
                    table.append(page)
                candidate_tables.append(tuple(table))
        except StopIteration as exc:
            self._allocation_failures += 1
            raise ValueError(
                "native paged KV capacity exhausted before model execution"
            ) from exc

        pinned = tuple(sorted({page for cache in paged for page in cache.block_table}))
        for page in pinned:
            self._pin_counts[page] += 1
        self._reserved_pages.update(reserved)
        return self._build_reservation(
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

    def _build_reservation(
        self,
        *,
        backend: PagedKVCacheBackend,
        request_caches: tuple[PagedRequestCache, ...],
        append_lengths: tuple[int, ...],
        candidate_tables: tuple[tuple[int, ...], ...],
        reserved_pages: tuple[int, ...],
        copy_pages: tuple[tuple[int, int], ...],
        pinned_pages: tuple[int, ...],
        staged_keys: list[mx.array | None],
        staged_values: list[mx.array | None],
    ) -> PagedBatchReservation:
        """Construct the backend-specific reservation type once per step."""

        return PagedBatchReservation(
            backend=backend,
            request_caches=request_caches,
            append_lengths=append_lengths,
            candidate_tables=candidate_tables,
            reserved_pages=reserved_pages,
            copy_pages=copy_pages,
            pinned_pages=pinned_pages,
            staged_keys=staged_keys,
            staged_values=staged_values,
        )

    def free_page_count(self) -> int:
        """Return pages neither owned by a request nor reserved by a step."""

        return len(self._free_pages) - sum(
            1 for page in self._reserved_pages if self._is_free(page)
        )

    def preflight(
        self,
        caches: tuple[RequestCache, ...],
        append_lengths: tuple[int, ...],
    ) -> tuple[str | None, ...]:
        paged = tuple(_require_paged_cache(cache) for cache in caches)
        if len(paged) != len(append_lengths):
            raise ValueError("paged preflight requires matching requests")
        available = self.free_page_count()
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
            self._free_if_unused(page)

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
            "total_pages": self.num_pages,
            "used_pages": used_pages,
            "free_pages": self.num_pages - used_pages,
            "pinned_pages": pinned_pages,
            "internal_fragmentation_tokens": fragmentation,
            "active_kv_bytes": used_pages * self.bytes_per_page,
            "allocation_failures": self._allocation_failures,
            "page_size": self.page_size,
        }

    def reset(self, *, reset_counters: bool) -> None:
        """Release every page mapping and optionally reset allocation failures."""

        if self._reserved_pages or any(self._pin_counts):
            raise RuntimeError(
                "cannot reset KV cache while pages are reserved or pinned"
            )
        self._caches.clear()
        self._free_pages = list(range(self.num_pages))
        self._ref_counts = [0] * self.num_pages
        if reset_counters:
            self._allocation_failures = 0

    def _commit_reservation(
        self,
        reservation: PagedBatchReservation,
        staged_keys: tuple[mx.array | None, ...],
        staged_values: tuple[mx.array | None, ...],
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
        if any(candidate_ref_counts[page] < 0 for page in old_counts):
            raise ValueError("paged KV page reference count became negative")
        self._ref_counts = candidate_ref_counts
        self.key_pages = [
            value if value is not None else current
            for current, value in zip(self.key_pages, staged_keys, strict=True)
        ]
        self.value_pages = [
            value if value is not None else current
            for current, value in zip(self.value_pages, staged_values, strict=True)
        ]
        for cache, table, append_length in zip(
            reservation.request_caches,
            reservation.candidate_tables,
            reservation.append_lengths,
            strict=True,
        ):
            cache.block_table = list(table)
            cache.length += append_length
        for page in reservation.reserved_pages:
            self._remove_free(page)
        # Only pages this step stopped referencing can have become unused;
        # pages whose last pin drops are freed by ``_finish_reservation``.
        for page in old_counts:
            self._free_if_unused(page)
        self._finish_reservation(reservation)
        return tuple(cache.size() for cache in reservation.request_caches)

    def _abort_reservation(self, reservation: PagedBatchReservation) -> None:
        self._finish_reservation(reservation)

    def _finish_reservation(self, reservation: PagedBatchReservation) -> None:
        # A finished reservation must not keep pool buffers alive, or the next
        # step's in-place page write would fall back to a full copy.
        reservation.staged_keys[:] = [None] * len(reservation.staged_keys)
        reservation.staged_values[:] = [None] * len(reservation.staged_values)
        self._reserved_pages.difference_update(reservation.reserved_pages)
        for page in reservation.pinned_pages:
            self._pin_counts[page] -= 1
            if self._pin_counts[page] < 0:
                raise ValueError("paged KV page pin count became negative")
            self._free_if_unused(page)

    def _is_free(self, page: int) -> bool:
        index = bisect.bisect_left(self._free_pages, page)
        return index < len(self._free_pages) and self._free_pages[index] == page

    def _remove_free(self, page: int) -> None:
        index = bisect.bisect_left(self._free_pages, page)
        if index < len(self._free_pages) and self._free_pages[index] == page:
            del self._free_pages[index]

    def _free_if_unused(self, page: int) -> None:
        """Return an unreferenced, unpinned page to the sorted free list."""

        if self._ref_counts[page] or self._pin_counts[page]:
            return
        index = bisect.bisect_left(self._free_pages, page)
        if index == len(self._free_pages) or self._free_pages[index] != page:
            self._free_pages.insert(index, page)


@dataclass
class HybridPagedKVCacheBackend(PagedKVCacheBackend):
    """Paged KV backend with convolution state isolated to hybrid models."""

    _caches: dict[str, HybridRequestCache] = field(default_factory=dict, init=False)

    def create(self, request_id: str) -> str:
        handle = f"paged-hybrid-kv-{request_id}-{uuid.uuid4().hex}"
        self._caches[handle] = HybridRequestCache(
            handle=handle,
            request_id=request_id,
        )
        return handle

    def _build_reservation(
        self,
        *,
        backend: PagedKVCacheBackend,
        request_caches: tuple[PagedRequestCache, ...],
        append_lengths: tuple[int, ...],
        candidate_tables: tuple[tuple[int, ...], ...],
        reserved_pages: tuple[int, ...],
        copy_pages: tuple[tuple[int, int], ...],
        pinned_pages: tuple[int, ...],
        staged_keys: list[mx.array | None],
        staged_values: list[mx.array | None],
    ) -> HybridBatchReservation:
        hybrid = tuple(_require_hybrid_cache(cache) for cache in request_caches)
        return HybridBatchReservation(
            backend=backend,
            request_caches=hybrid,
            append_lengths=append_lengths,
            candidate_tables=candidate_tables,
            reserved_pages=reserved_pages,
            copy_pages=copy_pages,
            pinned_pages=pinned_pages,
            staged_keys=staged_keys,
            staged_values=staged_values,
            staged_states=[None] * self.num_layers,
            staged_recurrent_states=[None] * self.num_layers,
        )

    def fork(
        self,
        handle: str,
        request_id: str,
        *,
        length: int | None = None,
    ) -> str:
        source = _require_hybrid_cache(self._caches[handle])
        new_handle = super().fork(handle, request_id, length=length)
        target = _require_hybrid_cache(self.get(new_handle, request_id))
        target.conv_states = dict(source.conv_states)
        target.recurrent_states = dict(source.recurrent_states)
        return new_handle

    def _commit_hybrid_reservation(
        self,
        reservation: HybridBatchReservation,
        staged_keys: tuple[mx.array | None, ...],
        staged_values: tuple[mx.array | None, ...],
        staged_states: tuple[mx.array | None, ...],
        staged_recurrent: tuple[mx.array | None, ...] = (),
    ) -> tuple[int, ...]:
        lengths = super()._commit_reservation(reservation, staged_keys, staged_values)
        for index, state in enumerate(staged_states):
            if state is not None:
                for cache, row in zip(reservation.request_caches, state, strict=True):
                    _require_hybrid_cache(cache).conv_states[index] = row
        for index, state in enumerate(staged_recurrent):
            if state is not None:
                for row, cache in enumerate(reservation.request_caches):
                    _require_hybrid_cache(cache).recurrent_states[index] = (
                        RecurrentStateRow(state, row)
                    )
        return lengths

    def metrics(self) -> dict[str, Any]:
        metrics = super().metrics()
        metrics["cache_backend"] = "paged-mlx-hybrid"
        return metrics


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


def _require_hybrid_cache(cache: RequestCache) -> HybridRequestCache:
    if not isinstance(cache, HybridRequestCache):
        raise TypeError("hybrid backend received incompatible request cache")
    return cache
