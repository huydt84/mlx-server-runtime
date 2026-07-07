"""Worker bootstrap configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration needed for the worker runtime.

    Attributes:
        socket_path: Unix domain socket path for IPC with the Rust gateway.
        backend: Backend selected by Rust startup. ``v1`` remains default.
        model: Text-only model identifier (e.g. an ``mlx-community`` hub path).
        vlm_model: Optional VLM model identifier. When ``None`` or empty,
            no VLM engine is constructed and the worker runs text-only.
    """

    socket_path: str
    model: str
    backend: str = "v1"
    vlm_model: str | None = None
    max_vlm_images: int = 5
    continuous_batching: bool = False
    prompt_concurrency: int = 4
    decode_concurrency: int = 4
    prefill_chunk_size: int = 256
    text_prompt_concurrency: int = 4
    text_decode_concurrency: int = 4
    text_prefill_chunk_size: int = 256
    vlm_prompt_concurrency: int = 4
    vlm_decode_concurrency: int = 4
    vlm_prefill_chunk_size: int = 256
    text_cache_budget_bytes: int = 8 * 1024 * 1024
    text_cache_max_entries: int = 32
    native_kv_page_size: int = 16
    vlm_apc_cache_budget_bytes: int = 8 * 1024 * 1024
    vlm_apc_cache_max_entries: int = 32
    vision_feature_cache_budget_bytes: int = 8 * 1024 * 1024
    vision_feature_cache_max_entries: int = 20


def load_config() -> WorkerConfig:
    """Load the worker bootstrap configuration from environment variables.

    Reads ``MLX_RUNTIME_SOCKET``, ``MLX_RUNTIME_BACKEND``,
    ``MLX_RUNTIME_MODEL``, and the optional ``MLX_RUNTIME_VLM_MODEL``.
    An empty VLM model variable is treated the same as unset (no VLM engine).
    """

    raw_vlm = os.environ.get("MLX_RUNTIME_VLM_MODEL", "")
    vlm_model: str | None = raw_vlm if raw_vlm.strip() else None
    backend = os.environ.get("MLX_RUNTIME_BACKEND", "v1").strip() or "v1"
    max_vlm_images = int(os.environ.get("MLX_RUNTIME_MAX_VLM_IMAGES", "5"))
    continuous_batching = _load_bool("MLX_RUNTIME_CONTINUOUS_BATCHING", "0")
    prompt_concurrency = int(os.environ.get("MLX_RUNTIME_PROMPT_CONCURRENCY", "4"))
    decode_concurrency = int(os.environ.get("MLX_RUNTIME_DECODE_CONCURRENCY", "4"))
    prefill_chunk_size = int(os.environ.get("MLX_RUNTIME_PREFILL_CHUNK_SIZE", "256"))
    text_prompt_concurrency = _load_int_with_alias(
        "MLX_RUNTIME_TEXT_PROMPT_CONCURRENCY",
        "MLX_RUNTIME_PROMPT_CONCURRENCY",
        4,
    )
    text_decode_concurrency = _load_int_with_alias(
        "MLX_RUNTIME_TEXT_DECODE_CONCURRENCY",
        "MLX_RUNTIME_DECODE_CONCURRENCY",
        4,
    )
    text_prefill_chunk_size = _load_int_with_alias(
        "MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE",
        "MLX_RUNTIME_PREFILL_CHUNK_SIZE",
        256,
    )
    vlm_prompt_concurrency = _load_int_with_alias(
        "MLX_RUNTIME_VLM_PROMPT_CONCURRENCY",
        "MLX_RUNTIME_PROMPT_CONCURRENCY",
        4,
    )
    vlm_decode_concurrency = _load_int_with_alias(
        "MLX_RUNTIME_VLM_DECODE_CONCURRENCY",
        "MLX_RUNTIME_DECODE_CONCURRENCY",
        4,
    )
    vlm_prefill_chunk_size = _load_int_with_alias(
        "MLX_RUNTIME_VLM_PREFILL_CHUNK_SIZE",
        "MLX_RUNTIME_PREFILL_CHUNK_SIZE",
        256,
    )
    text_cache_budget_bytes = _load_int_with_alias(
        "MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES",
        None,
        8 * 1024 * 1024,
    )
    text_cache_max_entries = _load_int_with_alias(
        "MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES",
        None,
        32,
    )
    native_kv_page_size = int(os.environ.get("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE", "16"))
    if native_kv_page_size not in (8, 16, 32):
        raise ValueError("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE must be one of 8, 16, or 32")
    vlm_apc_cache_budget_bytes = _load_int_with_alias(
        "MLX_RUNTIME_VLM_APC_CACHE_BUDGET_BYTES",
        None,
        8 * 1024 * 1024,
    )
    vlm_apc_cache_max_entries = _load_int_with_alias(
        "MLX_RUNTIME_VLM_APC_CACHE_MAX_ENTRIES",
        None,
        32,
    )
    vision_feature_cache_budget_bytes = _load_int_with_alias(
        "MLX_RUNTIME_VISION_FEATURE_CACHE_BUDGET_BYTES",
        None,
        8 * 1024 * 1024,
    )
    vision_feature_cache_max_entries = _load_int_with_alias(
        "MLX_RUNTIME_VISION_FEATURE_CACHE_MAX_ENTRIES",
        None,
        20,
    )

    return WorkerConfig(
        socket_path=os.environ.get("MLX_RUNTIME_SOCKET", "/tmp/mlx-runtime.sock"),
        backend=backend,
        model=os.environ.get(
            "MLX_RUNTIME_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
        ),
        vlm_model=vlm_model,
        max_vlm_images=max_vlm_images,
        continuous_batching=continuous_batching,
        prompt_concurrency=prompt_concurrency,
        decode_concurrency=decode_concurrency,
        prefill_chunk_size=prefill_chunk_size,
        text_prompt_concurrency=text_prompt_concurrency,
        text_decode_concurrency=text_decode_concurrency,
        text_prefill_chunk_size=text_prefill_chunk_size,
        vlm_prompt_concurrency=vlm_prompt_concurrency,
        vlm_decode_concurrency=vlm_decode_concurrency,
        vlm_prefill_chunk_size=vlm_prefill_chunk_size,
        text_cache_budget_bytes=text_cache_budget_bytes,
        text_cache_max_entries=text_cache_max_entries,
        native_kv_page_size=native_kv_page_size,
        vlm_apc_cache_budget_bytes=vlm_apc_cache_budget_bytes,
        vlm_apc_cache_max_entries=vlm_apc_cache_max_entries,
        vision_feature_cache_budget_bytes=vision_feature_cache_budget_bytes,
        vision_feature_cache_max_entries=vision_feature_cache_max_entries,
    )


def _load_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _load_int_with_alias(name: str, alias: str | None, default: int) -> int:
    if name in os.environ:
        return int(os.environ[name])
    if alias is not None and alias in os.environ:
        return int(os.environ[alias])
    return default
