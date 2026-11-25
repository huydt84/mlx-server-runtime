"""Worker bootstrap configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

from .native_backend_ids import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    NATIVE_EXECUTION_BACKEND_IDS,
)
from .tool_call_parsers import AUTO_TOOL_CALL_PARSER, TOOL_CALL_PARSER_SETTINGS


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration needed for the worker runtime.

    Attributes:
        socket_path: Unix domain socket path for IPC with the Rust gateway.
        backend: Backend selected by Rust startup. ``native-mlx`` is the
            default; ``v1`` serves through ``mlx-lm``.
        model: Text-only model identifier (e.g. an ``mlx-community`` hub path).
        drafter: Optional drafter checkpoint that enables speculative decoding.
        vlm_model: Optional VLM model identifier. When ``None`` or empty,
            no VLM engine is constructed and the worker runs text-only.
        transcription_model: Optional Whisper speech-to-text model served
            through ``mlx-audio``. When ``None``, transcription requests are
            rejected.
        memory_utilization: Fraction in (0, 1] of the Metal recommended
            working set for weights, activations, and the native KV pool.
        kv_cache_bytes: Absolute native KV pool size; overrides
            ``memory_utilization`` when set.
        max_total_tokens_per_request: Configured per-request token limit.
        max_total_tokens_explicit: Whether the operator set that limit. When
            false, native-mlx lowers it to the KV pool capacity.
        text_cache_max_entries_explicit: Whether the operator set
            ``text_cache_max_entries``. When false, the native prefix cache
            may hold every page of the KV pool and gives pages back on demand.
        tool_call_parser: Tool-call format of the text model: ``auto`` infers
            it from the checkpoint, ``none`` rejects requests with tools, and a
            parser name overrides the inference.
    """

    socket_path: str
    model: str
    backend: str = "native-mlx"
    drafter: str | None = None
    vlm_model: str | None = None
    transcription_model: str | None = None
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
    text_cache_max_entries_explicit: bool = False
    memory_utilization: float = 0.5
    kv_cache_bytes: int | None = None
    max_total_tokens_per_request: int = 65_536
    max_total_tokens_explicit: bool = False
    native_kv_page_size: int = 16
    native_mlx_cache_limit_bytes: int = 1024 * 1024 * 1024
    native_execution_backend: str = DEFAULT_NATIVE_EXECUTION_BACKEND
    native_execution_mode: str = "overlap"
    native_prefix_cache_strategy: str = "radix"
    native_scheduling_policy: str = "fcfs"
    native_graph_profile: bool = False
    native_pipeline_profile: bool = False
    native_pipeline_profile_dir: str | None = None
    vlm_apc_cache_budget_bytes: int = 8 * 1024 * 1024
    vlm_apc_cache_max_entries: int = 32
    vision_feature_cache_budget_bytes: int = 8 * 1024 * 1024
    vision_feature_cache_max_entries: int = 20
    tool_call_parser: str = AUTO_TOOL_CALL_PARSER


def load_config() -> WorkerConfig:
    """Load the worker bootstrap configuration from environment variables.

    Reads ``MLX_RUNTIME_SOCKET``, ``MLX_RUNTIME_BACKEND``,
    ``MLX_RUNTIME_MODEL``, and the optional ``MLX_RUNTIME_DRAFTER``,
    ``MLX_RUNTIME_VLM_MODEL`` and ``MLX_RUNTIME_TRANSCRIPTION_MODEL``. An empty
    optional model variable is treated the same as unset.
    """

    raw_vlm = os.environ.get("MLX_RUNTIME_VLM_MODEL", "")
    vlm_model: str | None = raw_vlm if raw_vlm.strip() else None
    raw_transcription = os.environ.get("MLX_RUNTIME_TRANSCRIPTION_MODEL", "").strip()
    transcription_model: str | None = raw_transcription or None
    backend = (
        os.environ.get("MLX_RUNTIME_BACKEND", "native-mlx").strip() or "native-mlx"
    )
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
    memory_utilization = float(os.environ.get("MLX_RUNTIME_MEMORY_UTILIZATION", "0.5"))
    if not 0.0 < memory_utilization <= 1.0:
        raise ValueError(
            "MLX_RUNTIME_MEMORY_UTILIZATION must be greater than 0 and at most 1"
        )
    kv_cache_bytes: int | None = None
    if os.environ.get("MLX_RUNTIME_KV_CACHE_BYTES", "").strip():
        kv_cache_bytes = int(os.environ["MLX_RUNTIME_KV_CACHE_BYTES"])
        if kv_cache_bytes <= 0:
            raise ValueError("MLX_RUNTIME_KV_CACHE_BYTES must be positive")
    elif (
        text_cache_budget_bytes > 0
        and os.environ.get("MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES", "").strip()
    ):
        kv_cache_bytes = text_cache_budget_bytes
    max_total_tokens_per_request = int(
        os.environ.get("MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST", "65536")
    )
    if max_total_tokens_per_request <= 0:
        raise ValueError("MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST must be positive")
    max_total_tokens_explicit = _load_bool("MLX_RUNTIME_MAX_TOTAL_TOKENS_EXPLICIT", "0")
    native_kv_page_size = int(os.environ.get("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE", "16"))
    if native_kv_page_size not in (8, 16, 32):
        raise ValueError("MLX_RUNTIME_NATIVE_KV_PAGE_SIZE must be one of 8, 16, or 32")
    native_mlx_cache_limit_bytes = int(
        os.environ.get("MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES", str(1024**3))
    )
    if native_mlx_cache_limit_bytes < 0:
        raise ValueError(
            "MLX_RUNTIME_NATIVE_MLX_CACHE_LIMIT_BYTES must be non-negative"
        )
    native_execution_backend = os.environ.get(
        "MLX_RUNTIME_NATIVE_EXECUTION_BACKEND",
        DEFAULT_NATIVE_EXECUTION_BACKEND,
    ).strip()
    if native_execution_backend not in NATIVE_EXECUTION_BACKEND_IDS:
        choices = ", ".join(NATIVE_EXECUTION_BACKEND_IDS)
        raise ValueError(
            "MLX_RUNTIME_NATIVE_EXECUTION_BACKEND must be one of: " + choices
        )
    native_execution_mode = os.environ.get(
        "MLX_RUNTIME_NATIVE_EXECUTION_MODE", "overlap"
    ).strip()
    if native_execution_mode not in {"serial", "overlap"}:
        raise ValueError("MLX_RUNTIME_NATIVE_EXECUTION_MODE must be serial or overlap")
    native_prefix_cache_strategy = os.environ.get(
        "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY",
        "radix",
    ).strip()
    if native_prefix_cache_strategy not in {"block-hash", "radix"}:
        raise ValueError(
            "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY must be block-hash or radix"
        )
    native_scheduling_policy = os.environ.get(
        "MLX_RUNTIME_NATIVE_SCHEDULING_POLICY",
        "fcfs",
    ).strip()
    if native_scheduling_policy not in {"fcfs", "lpm", "lof", "priority"}:
        raise ValueError(
            "MLX_RUNTIME_NATIVE_SCHEDULING_POLICY must be fcfs, lpm, lof, or priority"
        )
    native_graph_profile = _load_bool("MLX_RUNTIME_NATIVE_GRAPH_PROFILE", "0")
    native_pipeline_profile = _load_bool("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "0")
    native_pipeline_profile_dir = os.environ.get(
        "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR"
    )
    if native_pipeline_profile and not native_pipeline_profile_dir:
        raise ValueError(
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR is required when pipeline profiling is enabled"
        )
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
    tool_call_parser = (
        os.environ.get("MLX_RUNTIME_TOOL_CALL_PARSER", "").strip()
        or AUTO_TOOL_CALL_PARSER
    )
    if tool_call_parser not in TOOL_CALL_PARSER_SETTINGS:
        raise ValueError(
            "MLX_RUNTIME_TOOL_CALL_PARSER must be one of: "
            + ", ".join(TOOL_CALL_PARSER_SETTINGS)
        )

    return WorkerConfig(
        socket_path=os.environ.get("MLX_RUNTIME_SOCKET", "/tmp/mlx-runtime.sock"),
        backend=backend,
        model=os.environ.get(
            "MLX_RUNTIME_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
        ),
        drafter=os.environ.get("MLX_RUNTIME_DRAFTER", "").strip() or None,
        vlm_model=vlm_model,
        transcription_model=transcription_model,
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
        text_cache_max_entries_explicit=bool(
            os.environ.get("MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES", "").strip()
        ),
        memory_utilization=memory_utilization,
        kv_cache_bytes=kv_cache_bytes,
        max_total_tokens_per_request=max_total_tokens_per_request,
        max_total_tokens_explicit=max_total_tokens_explicit,
        native_kv_page_size=native_kv_page_size,
        native_mlx_cache_limit_bytes=native_mlx_cache_limit_bytes,
        native_execution_backend=native_execution_backend,
        native_execution_mode=native_execution_mode,
        native_prefix_cache_strategy=native_prefix_cache_strategy,
        native_scheduling_policy=native_scheduling_policy,
        native_graph_profile=native_graph_profile,
        native_pipeline_profile=native_pipeline_profile,
        native_pipeline_profile_dir=native_pipeline_profile_dir,
        vlm_apc_cache_budget_bytes=vlm_apc_cache_budget_bytes,
        vlm_apc_cache_max_entries=vlm_apc_cache_max_entries,
        vision_feature_cache_budget_bytes=vision_feature_cache_budget_bytes,
        vision_feature_cache_max_entries=vision_feature_cache_max_entries,
        tool_call_parser=tool_call_parser,
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
