"""KV pool sizing from a memory-utilization fraction.

The native worker sizes its paged KV pool the way vLLM sizes its GPU KV cache
from ``--gpu-memory-utilization``: the configured fraction of the device's
recommended working set covers the weights, an activation reserve, and the KV
pool together, and the pool receives what remains once the weights are loaded.
An absolute ``kv_cache_bytes`` overrides the fraction for the pool.

The per-request token limit then follows from the pool: an implicit limit is
lowered to what one request can hold, and an explicit limit that does not fit
refuses startup instead of failing requests later.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

_GIB = 1024**3
_ACTIVATION_SLACK_BYTES = 256 * 1024 * 1024
_ACTIVATION_DTYPE_BYTES = 2
_LOGITS_DTYPE_BYTES = 4


class KVMemoryError(ValueError):
    """The configured memory cannot hold the model and its KV pool."""


@dataclass(frozen=True)
class KVMemorySettings:
    """Operator memory settings consumed by native bootstrap.

    Attributes:
        utilization: Fraction in (0, 1] of the recommended working set.
        kv_cache_bytes: Absolute pool size; overrides ``utilization``.
        max_total_tokens_per_request: Configured per-request token limit.
        max_total_tokens_explicit: Whether the operator set that limit.
        prefill_tokens: Tokens in the largest prefill step.
        prefill_rows: Requests in the largest prefill step.
        buffer_cache_limit_bytes: Freed buffers MLX may retain.
    """

    utilization: float = 0.5
    kv_cache_bytes: int | None = None
    max_total_tokens_per_request: int = 65_536
    max_total_tokens_explicit: bool = False
    prefill_tokens: int = 1024
    prefill_rows: int = 4
    buffer_cache_limit_bytes: int = _GIB


@dataclass(frozen=True)
class KVMemoryPlan:
    """Resolved KV pool size and the inputs that produced it.

    Attributes:
        pool_bytes: Bytes allocated to the paged KV pool.
        source: ``kv_cache_bytes`` when pinned, else ``memory_utilization``.
        utilization: Configured fraction of the recommended working set.
        working_set_bytes: Device recommended working set, when known.
        weights_bytes: Bytes held by the loaded weights.
        reserve_bytes: Activation and MLX buffer-cache reserve.
    """

    pool_bytes: int
    source: str
    utilization: float
    working_set_bytes: int | None
    weights_bytes: int
    reserve_bytes: int

    def describe(self) -> str:
        """Return a one-line description for startup logs."""

        def gib(value: int | None) -> str:
            return "unknown" if value is None else f"{value / _GIB:.2f}GiB"

        return (
            f"kv_pool={gib(self.pool_bytes)} source={self.source} "
            f"utilization={self.utilization:g} "
            f"working_set={gib(self.working_set_bytes)} "
            f"weights={gib(self.weights_bytes)} reserve={gib(self.reserve_bytes)}"
        )


def recommended_working_set_bytes() -> int | None:
    """Return the Metal recommended working set, or ``None`` when unavailable."""

    import mlx.core as mx

    device_info = getattr(mx, "device_info", None)
    if device_info is None:
        metal = getattr(mx, "metal", None)
        device_info = getattr(metal, "device_info", None)
    if device_info is None:
        return None
    try:
        info = device_info()
    except Exception:
        return None
    value = info.get("max_recommended_working_set_size")
    return int(value) if value else None


def weights_nbytes(weights: Iterable[tuple[str, Any]]) -> int:
    """Return the stored size of loaded weight tensors.

    Args:
        weights: ``(name, array)`` pairs as returned by the weight loader.

    Returns:
        Sum of ``nbytes`` over the arrays; quantized tensors count their
        packed storage.
    """

    return sum(int(getattr(array, "nbytes", 0)) for _, array in weights)


def activation_reserve_bytes(
    raw_config: Mapping[str, Any],
    *,
    prefill_tokens: int,
    prefill_rows: int,
    buffer_cache_limit_bytes: int,
) -> int:
    """Estimate memory outside the KV pool needed while serving.

    The estimate covers the MLX buffer cache, the hidden and MLP activations
    of one full prefill step, and the last-position logits of every row, plus
    a fixed slack for the attention workspace and Python-side buffers.

    Args:
        raw_config: The checkpoint ``config.json`` contents.
        prefill_tokens: Tokens in the largest prefill step.
        prefill_rows: Requests in the largest step.
        buffer_cache_limit_bytes: Freed buffers MLX may retain.

    Returns:
        Reserved bytes.
    """

    text_config = raw_config.get("text_config")
    source: Mapping[str, Any] = (
        text_config if isinstance(text_config, Mapping) else raw_config
    )
    hidden = int(source.get("hidden_size") or 0)
    intermediate = int(
        source.get("intermediate_size") or source.get("moe_intermediate_size") or 0
    )
    vocab = int(source.get("vocab_size") or raw_config.get("vocab_size") or 0)
    per_token = _ACTIVATION_DTYPE_BYTES * (4 * hidden + 3 * intermediate)
    logits = max(1, prefill_rows) * vocab * _LOGITS_DTYPE_BYTES
    return (
        max(0, buffer_cache_limit_bytes)
        + max(0, prefill_tokens) * per_token
        + logits
        + _ACTIVATION_SLACK_BYTES
    )


def plan_kv_pool(
    *,
    kv_cache_bytes: int | None,
    utilization: float,
    working_set_bytes: int | None,
    weights_bytes: int,
    reserve_bytes: int,
) -> KVMemoryPlan:
    """Resolve the KV pool size.

    Args:
        kv_cache_bytes: Absolute pool size; overrides ``utilization``.
        utilization: Fraction in (0, 1] of the recommended working set.
        working_set_bytes: Device recommended working set.
        weights_bytes: Bytes held by the loaded weights.
        reserve_bytes: Activation and buffer-cache reserve.

    Returns:
        The resolved plan.

    Raises:
        KVMemoryError: When the inputs leave no room for a KV pool.
    """

    if not (0.0 < utilization <= 1.0):
        raise KVMemoryError(
            f"memory utilization must be greater than 0 and at most 1, got {utilization}"
        )
    if kv_cache_bytes is not None:
        if kv_cache_bytes <= 0:
            raise KVMemoryError("kv_cache_bytes must be positive")
        return KVMemoryPlan(
            pool_bytes=int(kv_cache_bytes),
            source="kv_cache_bytes",
            utilization=utilization,
            working_set_bytes=working_set_bytes,
            weights_bytes=weights_bytes,
            reserve_bytes=reserve_bytes,
        )
    if working_set_bytes is None:
        raise KVMemoryError(
            "the Metal recommended working set is unavailable; set an absolute "
            "KV pool with --kv-cache-memory or memory.kv_cache_bytes"
        )
    budget = int(utilization * working_set_bytes)
    pool = budget - weights_bytes - reserve_bytes
    if pool <= 0:
        needed = (weights_bytes + reserve_bytes) / working_set_bytes
        raise KVMemoryError(
            f"memory utilization {utilization:g} gives {budget / _GIB:.2f}GiB of the "
            f"{working_set_bytes / _GIB:.2f}GiB working set, but the weights need "
            f"{weights_bytes / _GIB:.2f}GiB and the activation reserve "
            f"{reserve_bytes / _GIB:.2f}GiB; raise --memory-utilization above "
            f"{needed:.2f} or set --kv-cache-memory"
        )
    return KVMemoryPlan(
        pool_bytes=pool,
        source="memory_utilization",
        utilization=utilization,
        working_set_bytes=working_set_bytes,
        weights_bytes=weights_bytes,
        reserve_bytes=reserve_bytes,
    )


def resolve_max_total_tokens(
    *,
    capacity_tokens: int,
    configured: int,
    explicit: bool,
    bytes_per_token: int,
) -> int:
    """Return the per-request token limit the KV pool can honor.

    Args:
        capacity_tokens: Tokens the whole pool can hold.
        configured: ``limits.max_total_tokens_per_request``.
        explicit: Whether ``configured`` was set by the operator.
        bytes_per_token: KV bytes one token occupies.

    Returns:
        ``configured`` when it fits, otherwise the pool capacity when the
        limit was implicit.

    Raises:
        KVMemoryError: When an explicit limit does not fit one request.
    """

    if configured <= capacity_tokens:
        return configured
    if explicit:
        needed = configured * bytes_per_token
        held = capacity_tokens * bytes_per_token
        raise KVMemoryError(
            f"limits.max_total_tokens_per_request={configured} needs "
            f"{needed / _GIB:.2f}GiB of KV cache, but the pool holds "
            f"{held / _GIB:.2f}GiB ({capacity_tokens} tokens); raise "
            "--memory-utilization or --kv-cache-memory, or lower "
            "limits.max_total_tokens_per_request"
        )
    return capacity_tokens


def pages_for(tokens: int, page_size: int) -> int:
    """Return the pages needed to hold ``tokens`` tokens."""

    return math.ceil(max(0, tokens) / page_size)
