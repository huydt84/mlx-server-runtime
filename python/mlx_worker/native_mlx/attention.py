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


# Phase 9 provenance:
# This MLX Metal kernel is a project-local paged-attention implementation. Its
# required capability follows the vLLM/SGLang paged-KV practice recorded in
# plan/references/06-upstream-sources.md and the phase references, and it can be
# compared against Hugging Face's community Metal paged-attention kernel for
# operator expectations. The source below is intentionally not vendored from
# those projects; keep behavior/metrics compatible while preserving this local
# ownership boundary.
_PAGED_ATTENTION_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    uint batch_size = queries_shape[0];
    uint query_heads = queries_shape[1];
    uint max_tokens = queries_shape[2];
    uint head_dim = queries_shape[3];
    uint kv_heads = key_cache_shape[2];
    uint page_size = key_cache_shape[1];
    uint max_blocks = block_tables_shape[1];
    uint total_rows = batch_size * query_heads * max_tokens;
    if (tid >= total_rows) {
        return;
    }

    uint query_index = tid % max_tokens;
    uint head_index = (tid / max_tokens) % query_heads;
    uint batch_index = tid / (max_tokens * query_heads);
    uint query_base =
        ((batch_index * query_heads + head_index) * max_tokens + query_index)
        * head_dim;

    if (query_index >= uint(token_lengths[batch_index])) {
        for (uint dim = 0; dim < head_dim; ++dim) {
            output[query_base + dim] = 0;
        }
        return;
    }

    uint group_size = query_heads / kv_heads;
    uint kv_head = head_index / group_size;
    uint context_limit =
        uint(cache_lengths[batch_index]) + query_index + 1;
    float max_score = -INFINITY;

    for (uint position = 0; position < context_limit; ++position) {
        uint block = uint(
            block_tables[batch_index * max_blocks + position / page_size]
        );
        uint cache_base =
            ((block * page_size + position % page_size) * kv_heads + kv_head)
            * head_dim;
        float score = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            score += float(queries[query_base + dim])
                * float(key_cache[cache_base + dim]);
        }
        score *= scale[0];
        max_score = metal::max(max_score, score);
    }

    float denominator = 0.0f;
    for (uint position = 0; position < context_limit; ++position) {
        uint block = uint(
            block_tables[batch_index * max_blocks + position / page_size]
        );
        uint cache_base =
            ((block * page_size + position % page_size) * kv_heads + kv_head)
            * head_dim;
        float score = 0.0f;
        for (uint dim = 0; dim < head_dim; ++dim) {
            score += float(queries[query_base + dim])
                * float(key_cache[cache_base + dim]);
        }
        denominator += metal::exp(score * scale[0] - max_score);
    }

    for (uint dim = 0; dim < head_dim; ++dim) {
        float accumulator = 0.0f;
        for (uint position = 0; position < context_limit; ++position) {
            uint block = uint(
                block_tables[batch_index * max_blocks + position / page_size]
            );
            uint cache_base =
                ((block * page_size + position % page_size) * kv_heads + kv_head)
                * head_dim;
            float score = 0.0f;
            for (uint inner = 0; inner < head_dim; ++inner) {
                score += float(queries[query_base + inner])
                    * float(key_cache[cache_base + inner]);
            }
            float weight = metal::exp(score * scale[0] - max_score);
            accumulator += weight * float(value_cache[cache_base + dim]);
        }
        output[query_base + dim] = accumulator / denominator;
    }
"""


@dataclass
class PagedLayerAttentionContext:
    """Layer-local paged append plus native Metal attention dispatch."""

    reservation: PagedBatchReservation
    layer_index: int
    forward_mode: ForwardMode
    kernel: object       

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
        token_lengths = mx.array(self.reservation.token_lengths, dtype=mx.int32)
        cache_lengths = mx.array(self.reservation.cache_lengths, dtype=mx.int32)
        scale_value = mx.array([scale], dtype=mx.float32)
        total_rows = (
            int(queries.shape[0]) * int(queries.shape[1]) * int(queries.shape[2])
        )
        started = time.perf_counter()
        output = self.kernel(  # type: ignore[operator]
            inputs=[
                mx.contiguous(queries),
                key_cache,
                value_cache,
                self.reservation.block_tables,
                token_lengths,
                cache_lengths,
                scale_value,
            ],
            output_shapes=[queries.shape],
            output_dtypes=[queries.dtype],
            grid=(total_rows, 1, 1),
            threadgroup=(min(256, total_rows), 1, 1),
        )[0]
        self.reservation.backend.record_attention(
            self.forward_mode.value,
            max(0, int((time.perf_counter() - started) * 1000)),
        )
        return output


@dataclass
class PagedMetalAttentionBackend:
    """MLX-native Metal attention over fixed-size paged K/V."""

    _kernel: object | None = None

    def __post_init__(self) -> None:
        if not mx.metal.is_available():
            raise ValueError("native paged attention requires MLX Metal")
        self._kernel = mx.fast.metal_kernel(
            name="native_mlx_paged_attention",
            input_names=[
                "queries",
                "key_cache",
                "value_cache",
                "block_tables",
                "token_lengths",
                "cache_lengths",
                "scale",
            ],
            output_names=["output"],
            source=_PAGED_ATTENTION_SOURCE,
        )

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[LayerAttentionContext, ...]:
        if not isinstance(reservation, PagedBatchReservation):
            raise TypeError("paged attention requires a paged cache reservation")
        if self._kernel is None:
            raise RuntimeError("paged attention kernel is not initialized")
        return tuple(
            PagedLayerAttentionContext(
                reservation=reservation,
                layer_index=index,
                forward_mode=forward_mode,
                kernel=self._kernel,
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
