"""Compatible cache-and-attention backend construction for native MLX."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..native_backend_ids import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    NATIVE_EXECUTION_BACKEND_IDS,
)
from .attention import AttentionBackend, PagedMetalAttentionBackend
from .cache import KVCacheGeometry, PagedKVCacheBackend


@dataclass(frozen=True)
class NativeExecutionBackendBundle:
    """A validated cache and attention pairing selected at startup."""

    backend_id: str
    cache_backend: PagedKVCacheBackend
    attention_backend: AttentionBackend

    def validate(self) -> None:
        """Reject incompatible cache and attention implementations."""

        capabilities = self.attention_backend.capabilities
        if capabilities.backend_id != self.backend_id:
            raise ValueError(
                "execution backend identifier does not match attention capabilities"
            )
        capabilities.validate_cache_backend(self.cache_backend)


ExecutionBackendFactory = Callable[
    [KVCacheGeometry, int, int], NativeExecutionBackendBundle
]


def _build_native_metal_paged_sdpa(
    geometry: KVCacheGeometry,
    page_size: int,
    cache_budget_bytes: int,
) -> NativeExecutionBackendBundle:
    cache_backend = PagedKVCacheBackend(
        num_layers=geometry.num_layers,
        num_kv_heads=geometry.num_kv_heads,
        head_dim=geometry.head_dim,
        page_size=page_size,
        budget_bytes=cache_budget_bytes,
        dtype=geometry.dtype,
    )
    bundle = NativeExecutionBackendBundle(
        backend_id=DEFAULT_NATIVE_EXECUTION_BACKEND,
        cache_backend=cache_backend,
        attention_backend=PagedMetalAttentionBackend(),
    )
    bundle.validate()
    return bundle


_EXECUTION_BACKEND_FACTORIES: dict[str, ExecutionBackendFactory] = {
    DEFAULT_NATIVE_EXECUTION_BACKEND: _build_native_metal_paged_sdpa,
}

if frozenset(_EXECUTION_BACKEND_FACTORIES) != frozenset(
    NATIVE_EXECUTION_BACKEND_IDS
):  # pragma: no cover - import-time maintenance guard
    raise RuntimeError("native execution backend identifiers and factories disagree")


def available_native_execution_backends() -> tuple[str, ...]:
    """Return stable native execution backend identifiers."""

    return NATIVE_EXECUTION_BACKEND_IDS


def validate_native_execution_backend_id(backend_id: str) -> None:
    """Reject unknown backend identifiers without constructing MLX state."""

    if backend_id not in _EXECUTION_BACKEND_FACTORIES:
        choices = ", ".join(available_native_execution_backends())
        raise ValueError(
            f"unsupported native execution backend {backend_id!r}; expected one of: {choices}"
        )


def build_native_execution_backend(
    backend_id: str,
    geometry: KVCacheGeometry,
    *,
    page_size: int,
    cache_budget_bytes: int,
) -> NativeExecutionBackendBundle:
    """Construct and validate one registered execution backend bundle."""

    validate_native_execution_backend_id(backend_id)
    factory = _EXECUTION_BACKEND_FACTORIES[backend_id]
    bundle = factory(geometry, page_size, cache_budget_bytes)
    bundle.validate()
    return bundle
