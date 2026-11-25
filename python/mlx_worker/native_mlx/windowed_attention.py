"""Attention over the two-pool cache of windowed models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import mlx.core as mx

from .attention import AttentionBackendCapabilities
from .cache import BatchCacheReservation
from .causal_conv import TokenLayout
from .interfaces import ForwardMode
from .relative_attention import relative_paged_attention
from .windowed_cache import WindowedBatchReservation, WindowedPagedKVCacheBackend


@dataclass(frozen=True)
class WindowedLayerContext:
    """One KV layer's cache append, convolution history, and attention.

    Attributes:
        reservation: The step's cache reservation.
        layer_index: KV layer.
    """

    reservation: WindowedBatchReservation
    layer_index: int

    @property
    def layout(self) -> TokenLayout:
        """Return the layout of the step's concatenated tokens."""

        return self.reservation.layout

    def conv_history(self, conv_index: int, channels: int) -> mx.array:
        """Return the inputs a convolution reads from before this step."""

        return self.reservation.conv_history(self.layer_index, conv_index, channels)

    def stage_conv(
        self,
        conv_index: int,
        before: mx.array,
        inputs: mx.array,
        after: mx.array,
    ) -> None:
        """Stage one convolution's history for transactional commit."""

        self.reservation.stage_conv(self.layer_index, conv_index, before, inputs, after)

    def append_and_attend(
        self,
        queries: mx.array,
        kv: mx.array,
        *,
        rel: mx.array,
        rel_proj: mx.array,
        sinks: mx.array,
        scale: float,
    ) -> mx.array:
        """Append keys and values and attend with a distance bias and sinks.

        Args:
            queries: ``(tokens, query_heads, head_dim)`` queries.
            kv: ``(tokens, 2, kv_heads, head_dim)`` keys then values.
            rel: ``(tokens, query_heads, rel_dim)`` distance states.
            rel_proj: ``(extent, rel_dim)`` distance embeddings.
            sinks: ``(tokens, query_heads)`` sink logits.
            scale: Factor applied to the query-key product.

        Returns:
            ``(tokens, query_heads * head_dim)`` attention output.
        """

        reservation = self.reservation
        backend = reservation.backend
        pages = reservation.stage_layer(self.layer_index, kv)
        table = reservation.step_table(self.layer_index)
        return relative_paged_attention(
            queries,
            pages,
            table.block_tables,
            table.base_array,
            reservation.layout.rows,
            reservation.layout.positions,
            rel,
            rel_proj,
            sinks,
            max_length=reservation.max_total_length,
            scale=scale,
            window_size=(
                backend.window_size
                if self.layer_index in backend.window_layers
                else None
            ),
        )


@dataclass
class WindowedMetalAttentionBackend:
    """Relative-bias attention that reads both page pools directly."""

    capabilities: ClassVar[AttentionBackendCapabilities] = AttentionBackendCapabilities(
        backend_id="native-metal-paged-sdpa",
        cache_backend_types=(WindowedPagedKVCacheBackend,),
        reservation_types=(WindowedBatchReservation,),
        supported_masks=frozenset(("causal", None)),
        supported_forward_modes=frozenset(ForwardMode),
        requires_metal=True,
        supports_attention_sinks=True,
        supports_sliding_window=True,
        consumes_page_tables_directly=True,
    )

    def __post_init__(self) -> None:
        if not mx.metal.is_available():
            raise ValueError("native paged attention requires MLX Metal")

    def contexts(
        self,
        reservation: BatchCacheReservation,
        forward_mode: ForwardMode,
    ) -> tuple[WindowedLayerContext, ...]:
        """Return one context per KV layer of the reservation."""

        self.capabilities.validate_context(reservation, forward_mode)
        assert isinstance(reservation, WindowedBatchReservation)
        return tuple(
            WindowedLayerContext(reservation, index)
            for index in range(reservation.backend.num_layers)
        )

    def add_metrics(self, metrics: dict[str, Any]) -> None:
        """Add backend-owned telemetry without a hot-path temporary mapping."""

        metrics["attention_backend"] = self.capabilities.backend_id
