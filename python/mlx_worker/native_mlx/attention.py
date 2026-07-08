"""Shared attention adapters for dense diagnostics and paged production."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx

from .cache import (
    BatchCacheReservation,
    DenseBatchReservation,
    DenseLayerCache,
    PagedBatchReservation,
)
from .interfaces import ForwardMode, LayerAttentionContext


class AttentionBackend(Protocol):
    """Build model-facing layer contexts for one cache reservation."""

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]: ...


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
        mask: str | None,
    ) -> mx.array:
        dense_keys, dense_values = self.reservation.layer_views[
            self.layer_index
        ].update_and_fetch(keys, values)
        attention_mask = _dense_causal_mask(
            self.reservation.layer_views[self.layer_index].offsets,
            self.reservation.append_lengths,
            int(queries.shape[2]),
            int(dense_keys.shape[2]),
        )
        return mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=attention_mask.astype(queries.dtype),
        )


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
        mask: str | None,
    ) -> mx.array:
        del mask
        old_length = self.cache.size()
        dense_keys, dense_values = self.cache.update_and_fetch(keys, values)
        attention_mask = _dense_causal_mask(
            mx.array([old_length], dtype=mx.int32),
            (int(queries.shape[2]),),
            int(queries.shape[2]),
            int(dense_keys.shape[2]),
        )
        return mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=attention_mask.astype(queries.dtype),
        )


@dataclass(frozen=True)
class DenseReferenceAttentionBackend:
    """Explicit non-production attention adapter for parity and tests."""

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]:
        del forward_mode
        if not isinstance(reservation, DenseBatchReservation):
            raise TypeError("dense attention requires a dense cache reservation")
        return tuple(
            DenseLayerAttentionContext(reservation, index)
            for index in range(len(reservation.layer_views))
        )


@dataclass
class PagedLayerAttentionContext:
    """Layer-local paged append plus optimized MLX attention dispatch."""

    reservation: PagedBatchReservation
    layer_index: int
    forward_mode: ForwardMode

    def append_and_attend(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        *,
        scale: float,
        mask: str | None,
    ) -> mx.array:
        if mask not in (None, "causal"):
            raise ValueError("native paged attention supports causal masking only")
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
        dense_keys, dense_values = self._dense_kv_rows(key_cache, value_cache)
        attention_mask = _dense_causal_mask(
            mx.array(self.reservation.cache_lengths, dtype=mx.int32),
            self.reservation.token_lengths,
            int(queries.shape[2]),
            int(dense_keys.shape[2]),
        )
        output = mx.fast.scaled_dot_product_attention(
            queries,
            dense_keys,
            dense_values,
            scale=scale,
            mask=attention_mask.astype(queries.dtype),
        )
        self.reservation.backend.record_attention(
            self.forward_mode.value,
            max(0, int((time.perf_counter() - started) * 1000)),
        )
        return output

    def _dense_kv_rows(
        self,
        key_cache: mx.array,
        value_cache: mx.array,
    ) -> tuple[mx.array, mx.array]:
        max_total = max(
            cache_length + append_length
            for cache_length, append_length in zip(
                self.reservation.cache_lengths,
                self.reservation.append_lengths,
                strict=True,
            )
        )
        key_rows: list[mx.array] = []
        value_rows: list[mx.array] = []
        for cache, total_length, table in zip(
            self.reservation.request_caches,
            (
                cache_length + append_length
                for cache_length, append_length in zip(
                    self.reservation.cache_lengths,
                    self.reservation.append_lengths,
                    strict=True,
                )
            ),
            self.reservation.candidate_tables,
            strict=True,
        ):
            del cache
            page_ids = mx.array(list(table), dtype=mx.int32)
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
    """MLX-native attention over fixed-size paged K/V."""

    def __post_init__(self) -> None:
        if not mx.metal.is_available():
            raise ValueError("native paged attention requires MLX Metal")

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]:
        if not isinstance(reservation, PagedBatchReservation):
            raise TypeError("paged attention requires a paged cache reservation")
        return tuple(
            PagedLayerAttentionContext(
                reservation=reservation,
                layer_index=index,
                forward_mode=forward_mode,
            )
            for index in range(reservation.backend.num_layers)
        )


def _dense_causal_mask(
    cache_offsets: mx.array,
    token_lengths: tuple[int, ...],
    max_tokens: int,
    max_total: int,
) -> mx.array:
    offsets = [int(value) for value in cache_offsets.tolist()]
    rows: list[list[list[list[float]]]] = []
    for cache_length, token_length in zip(offsets, token_lengths, strict=True):
        query_rows = []
        for query_index in range(max_tokens):
            query_rows.append(
                [
                    0.0
                    if query_index < token_length
                    and key_index <= cache_length + query_index
                    else -1e9
                    for key_index in range(max_total)
                ]
            )
        rows.append([query_rows])
    return mx.array(rows, dtype=mx.float32)
