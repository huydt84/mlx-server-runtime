"""Shared attention adapters for dense diagnostics and paged production."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

import mlx.core as mx

from .cache import (
    BatchCacheReservation,
    DenseBatchReservation,
    DenseLayerCache,
    DenseKVCacheBackend,
    PagedBatchReservation,
    PagedKVCacheBackend,
    HybridBatchReservation,
    HybridPagedKVCacheBackend,
)
from .interfaces import ForwardMode, HybridLayerAttentionContext, LayerAttentionContext
from .paged_decode import paged_decode_attention, simdgroups_for


@dataclass(frozen=True)
class AttentionBackendCapabilities:
    """Static compatibility contract for one attention implementation."""

    backend_id: str
    cache_backend_types: tuple[type[Any], ...]
    reservation_types: tuple[type[Any], ...]
    supported_masks: frozenset[str | None]
    supported_forward_modes: frozenset[ForwardMode]
    requires_metal: bool
    supports_attention_sinks: bool = False
    supports_sliding_window: bool = False
    consumes_page_tables_directly: bool = False

    def validate_cache_backend(self, cache_backend: Any) -> None:
        """Fail before serving when cache storage is incompatible."""

        if not isinstance(cache_backend, self.cache_backend_types):
            expected = ", ".join(item.__name__ for item in self.cache_backend_types)
            raise TypeError(
                f"{self.backend_id} requires cache backend type: {expected}"
            )

    def validate_context(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> None:
        """Validate per-step reservation and forward-mode compatibility."""

        if not isinstance(reservation, self.reservation_types):
            expected = ", ".join(item.__name__ for item in self.reservation_types)
            raise TypeError(f"{self.backend_id} requires reservation type: {expected}")
        if forward_mode not in self.supported_forward_modes:
            raise ValueError(
                f"{self.backend_id} does not support {forward_mode.value} forwards"
            )


class AttentionBackend(Protocol):
    """Build model-facing layer contexts for one cache reservation."""

    capabilities: AttentionBackendCapabilities

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]: ...

    def add_metrics(self, metrics: dict[str, Any]) -> None: ...


@dataclass
class DenseLayerAttentionContext:
    """Built-in SDPA adapter retained only for diagnostics and unit tests."""

    reservation: DenseBatchReservation
    layer_index: int

    def append_and_attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: str | None = None,
        window_size: int | None = None,
    ) -> mx.array:
        window_size = window_size if window_size is not None else _window_size(mask)
        if mask not in (None, "causal") and window_size is None:
            raise ValueError("dense reference attention received unsupported mask")
        dense_keys, dense_values = self.reservation.layer_views[
            self.layer_index
        ].update_and_fetch(keys, values)
        attention_mask = _dense_causal_mask(
            self.reservation.layer_views[self.layer_index].offsets,
            self.reservation.append_lengths,
            int(queries.shape[2]),
            int(dense_keys.shape[2]),
            window_size=window_size,
        )
        return mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=attention_mask.astype(queries.dtype),
        )

    def prepare_conv_state(self, values: mx.array, cache_size: int) -> mx.array:
        return self.reservation.prepare_conv_state(self.layer_index, values, cache_size)

    def stage_conv_state(self, combined: mx.array, cache_size: int) -> None:
        self.reservation.stage_conv_state(self.layer_index, combined, cache_size)


@dataclass
class DenseTraceAttentionContext:
    """Single-request dense context used only by semantic tracing."""

    cache: DenseLayerCache

    def append_and_attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: str | None = None,
        window_size: int | None = None,
    ) -> mx.array:
        window_size = window_size if window_size is not None else _window_size(mask)
        old_length = self.cache.size()
        dense_keys, dense_values = self.cache.update_and_fetch(keys, values)
        attention_mask = _dense_causal_mask(
            mx.array([old_length], dtype=mx.int32),
            (int(queries.shape[2]),),
            int(queries.shape[2]),
            int(dense_keys.shape[2]),
            window_size=window_size,
        )
        return mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=attention_mask.astype(queries.dtype),
        )

    def prepare_conv_state(self, values: mx.array, cache_size: int) -> mx.array:
        state = self.cache.conv_state
        if state is None:
            state = mx.zeros(
                (cache_size - 1, int(values.shape[-1])), dtype=values.dtype
            )
        return mx.concatenate([state[None, ...], values], axis=1)

    def stage_conv_state(self, combined: mx.array, cache_size: int) -> None:
        self.cache.conv_state = combined[0, -cache_size + 1 :]


@dataclass(frozen=True)
class DenseReferenceAttentionBackend:
    """Explicit non-production attention adapter for parity and tests."""

    capabilities: ClassVar[AttentionBackendCapabilities] = AttentionBackendCapabilities(
        backend_id="dense-reference-sdpa",
        cache_backend_types=(DenseKVCacheBackend,),
        reservation_types=(DenseBatchReservation,),
        supported_masks=frozenset(("causal", None)),
        supported_forward_modes=frozenset(ForwardMode),
        requires_metal=False,
    )

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]:
        self.capabilities.validate_context(reservation, forward_mode)
        assert isinstance(reservation, DenseBatchReservation)
        return tuple(
            DenseLayerAttentionContext(reservation, index)
            for index in range(len(reservation.layer_views))
        )

    def metrics(self) -> dict[str, Any]:
        """Return diagnostic backend identity without claiming kernel timing."""

        metrics: dict[str, Any] = {}
        self.add_metrics(metrics)
        return metrics

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """Add backend-owned telemetry without a hot-path temporary mapping."""

        metrics["attention_backend"] = self.capabilities.backend_id


@dataclass
class PagedLayerAttentionContext:
    """Layer-local paged append plus optimized MLX attention dispatch."""

    reservation: PagedBatchReservation
    layer_index: int
    forward_mode: ForwardMode
    attention_backend: "PagedMetalAttentionBackend"
    step_masks: dict[int | None, mx.array | str | None] = field(default_factory=dict)
    """Masks shared by every layer of one step, keyed by sliding-window size."""

    def append_and_attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: str | None = None,
        window_size: int | None = None,
    ) -> mx.array:
        window_size = window_size if window_size is not None else _window_size(mask)
        if mask not in (None, "causal") and window_size is None:
            raise ValueError("native paged attention received unsupported mask")
        if int(queries.shape[1]) % self.reservation.backend.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if int(queries.shape[3]) != self.reservation.backend.head_dim:
            raise ValueError("paged attention query head dimension is unsupported")
        if keys.dtype != self.reservation.backend.dtype:
            keys = keys.astype(self.reservation.backend.dtype)
        if values.dtype != self.reservation.backend.dtype:
            values = values.astype(self.reservation.backend.dtype)
        key_cache, value_cache = self.reservation.stage_layer(
            self.layer_index,
            keys,
            values,
        )
        started = time.perf_counter()
        if self.forward_mode is ForwardMode.DECODE and simdgroups_for(
            int(queries.shape[1]),
            self.reservation.backend.num_kv_heads,
            int(queries.shape[3]),
        ):
            # Decode reads keys straight from the pages instead of gathering
            # every row's cache into dense K/V first.
            block_tables, lengths = self.reservation.page_table_inputs()
            output = paged_decode_attention(
                queries,
                key_cache,
                value_cache,
                block_tables,
                lengths,
                max_length=self.reservation._max_total_length,
                scale=scale,
                window_size=window_size,
            )
            self.attention_backend.record_attention(
                self.forward_mode.value,
                max(0, int((time.perf_counter() - started) * 1000)),
            )
            return output
        if self.reservation.cache_lengths == (0,):
            # A single row with no cached prefix attends only to its own new
            # K/V, which are already dense: skip the page gather.
            dense_keys, dense_values = keys, values
        else:
            dense_keys, dense_values = self._dense_kv_rows(key_cache, value_cache)
        if window_size not in self.step_masks:
            self.step_masks[window_size] = _step_attention_mask(
                self.reservation.cache_lengths,
                self.reservation.token_lengths,
                int(queries.shape[2]),
                int(dense_keys.shape[2]),
                window_size=window_size,
            )
        output = mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=self.step_masks[window_size],
        )
        self.attention_backend.record_attention(
            self.forward_mode.value,
            max(0, int((time.perf_counter() - started) * 1000)),
        )
        return output

    def _dense_kv_rows(
        self,
        key_cache: mx.array,
        value_cache: mx.array,
    ) -> tuple[mx.array, mx.array]:
        max_total = self.reservation._max_total_length
        key_rows: list[mx.array] = []
        value_rows: list[mx.array] = []
        for total_length, page_ids in zip(
            self.reservation._total_lengths,
            self.reservation._page_id_arrays,
            strict=True,
        ):
            row_keys = key_cache[page_ids].reshape(
                (
                    -1,
                    self.reservation.backend.num_kv_heads,
                    self.reservation.backend.head_dim,
                )
            )[:total_length]
            row_values = value_cache[page_ids].reshape(
                (
                    -1,
                    self.reservation.backend.num_kv_heads,
                    self.reservation.backend.head_dim,
                )
            )[:total_length]
            row_keys = row_keys.transpose(1, 0, 2)
            row_values = row_values.transpose(1, 0, 2)
            pad = max_total - total_length
            if pad > 0:
                row_keys = mx.pad(row_keys, [(0, 0), (0, pad), (0, 0)])
                row_values = mx.pad(row_values, [(0, 0), (0, pad), (0, 0)])
            key_rows.append(row_keys)
            value_rows.append(row_values)
        return mx.stack(key_rows, axis=0), mx.stack(value_rows, axis=0)


@dataclass
class PagedMetalAttentionBackend:
    """Paged-KV attention: a page-table decode kernel, SDPA over gathered rows.

    Decode steps read K/V directly from the page pool. Prefill and mixed steps
    gather each row's pages into dense K/V for MLX SDPA.
    """

    capabilities: ClassVar[AttentionBackendCapabilities] = AttentionBackendCapabilities(
        backend_id="native-metal-paged-sdpa",
        cache_backend_types=(PagedKVCacheBackend,),
        reservation_types=(PagedBatchReservation,),
        supported_masks=frozenset(("causal", None)),
        supported_forward_modes=frozenset(ForwardMode),
        requires_metal=True,
        consumes_page_tables_directly=True,
    )
    _attention_mode: str = field(default="uninitialized", init=False)
    _attention_time_ms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not mx.metal.is_available():
            raise ValueError("native paged attention requires MLX Metal")

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]:
        self.capabilities.validate_context(reservation, forward_mode)
        assert isinstance(reservation, PagedBatchReservation)
        step_masks: dict[int | None, mx.array | str | None] = {}
        return tuple(
            PagedLayerAttentionContext(
                reservation=reservation,
                layer_index=index,
                forward_mode=forward_mode,
                attention_backend=self,
                step_masks=step_masks,
            )
            for index in range(reservation.backend.num_layers)
        )

    def record_attention(self, mode: str, elapsed_ms: int) -> None:
        """Record the latest lazy MLX attention dispatch observation."""

        self._attention_mode = mode
        self._attention_time_ms = max(0, int(elapsed_ms))

    def metrics(self) -> dict[str, Any]:
        """Return backend-owned dispatch telemetry."""

        metrics: dict[str, Any] = {}
        self.add_metrics(metrics)
        return metrics

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """Add backend-owned telemetry without a hot-path temporary mapping."""

        metrics["attention_backend"] = self.capabilities.backend_id
        metrics["attention_mode"] = self._attention_mode
        metrics["attention_time_ms"] = self._attention_time_ms


@dataclass
class HybridPagedLayerAttentionContext(PagedLayerAttentionContext):
    """Paged attention context with convolution state for hybrid layers."""

    reservation: HybridBatchReservation

    def prepare_conv_state(self, values: mx.array, cache_size: int) -> mx.array:
        return self.reservation.prepare_conv_state(self.layer_index, values, cache_size)

    def stage_conv_state(self, combined: mx.array, cache_size: int) -> None:
        self.reservation.stage_conv_state(self.layer_index, combined, cache_size)


@dataclass
class HybridPagedMetalAttentionBackend(PagedMetalAttentionBackend):
    """Paged SDPA backend paired with the LFM2 hybrid cache."""

    capabilities: ClassVar[AttentionBackendCapabilities] = AttentionBackendCapabilities(
        backend_id="native-metal-paged-sdpa",
        cache_backend_types=(HybridPagedKVCacheBackend,),
        reservation_types=(HybridBatchReservation,),
        supported_masks=frozenset(("causal", None)),
        supported_forward_modes=frozenset(ForwardMode),
        requires_metal=True,
        consumes_page_tables_directly=True,
    )

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[HybridLayerAttentionContext, ...]:
        self.capabilities.validate_context(reservation, forward_mode)
        assert isinstance(reservation, HybridBatchReservation)
        step_masks: dict[int | None, mx.array | str | None] = {}
        return tuple(
            HybridPagedLayerAttentionContext(
                reservation=reservation,
                layer_index=index,
                forward_mode=forward_mode,
                attention_backend=self,
                step_masks=step_masks,
            )
            for index in range(reservation.backend.num_layers)
        )


def _step_attention_mask(
    cache_lengths: tuple[int, ...],
    token_lengths: tuple[int, ...],
    max_tokens: int,
    max_total: int,
    *,
    window_size: int | None = None,
) -> mx.array | str | None:
    """Return the cheapest SDPA mask that is exact for one paged step.

    A single row needs only MLX's built-in causal mask, which aligns the
    queries with the end of the keys. Equal-length decode rows need no mask.
    Everything else gets one boolean mask that every layer of the step shares.
    Padding query rows attend to every key so that no row is fully masked.
    """

    if window_size is None and len(token_lengths) == 1:
        return "causal" if max_tokens > 1 else None
    totals = {
        cache + tokens
        for cache, tokens in zip(cache_lengths, token_lengths, strict=True)
    }
    if window_size is None and max_tokens == 1 and len(totals) == 1:
        return None
    offsets = mx.array(cache_lengths, dtype=mx.int32)[:, None, None]
    query_indices = mx.arange(max_tokens, dtype=mx.int32)[None, :, None]
    query_positions = offsets + query_indices
    key_positions = mx.arange(max_total, dtype=mx.int32)[None, None, :]
    valid = key_positions <= query_positions
    if window_size is not None:
        valid = valid & (key_positions >= query_positions - window_size + 1)
    padding = query_indices >= mx.array(token_lengths, dtype=mx.int32)[:, None, None]
    return (valid | padding)[:, None, :, :]


def _dense_causal_mask(
    cache_offsets: mx.array,
    token_lengths: tuple[int, ...],
    max_tokens: int,
    max_total: int,
    *,
    window_size: int | None = None,
) -> mx.array:
    # Keep mask construction lazy and device-side.  The previous Python loop
    # called ``tolist`` once per layer, synchronizing MLX and rebuilding a
    # nested Python list for every request row.
    offsets = cache_offsets.astype(mx.int32)[:, None, None]
    query_indices = mx.arange(max_tokens, dtype=mx.int32)[None, :, None]
    query_positions = offsets + query_indices
    key_positions = mx.arange(max_total, dtype=mx.int32)[None, None, :]
    valid_tokens = (
        query_indices < mx.array(token_lengths, dtype=mx.int32)[:, None, None]
    )
    valid = valid_tokens & (key_positions <= query_positions)
    if window_size is not None:
        valid = valid & (key_positions >= query_positions - window_size + 1)
    return mx.where(valid[:, None, :, :], mx.array(0.0), mx.array(-1e9)).astype(
        mx.float32
    )


def _window_size(mask: str | None) -> int | None:
    """Decode the compact sliding-window mask used by native model adapters."""

    if mask is None or mask == "causal":
        return None
    prefix = "sliding_window:"
    if mask.startswith(prefix):
        value = int(mask[len(prefix) :])
        if value > 0:
            return value
    return None
