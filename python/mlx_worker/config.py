"""Worker bootstrap configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration needed for the worker runtime.

    Attributes:
        socket_path: Unix domain socket path for IPC with the Rust gateway.
        model: Text-only model identifier (e.g. an ``mlx-community`` hub path).
        vlm_model: Optional VLM model identifier. When ``None`` or empty,
            no VLM engine is constructed and the worker runs text-only.
    """

    socket_path: str
    model: str
    vlm_model: str | None = None
    max_vlm_images: int = 5
    continuous_batching: bool = False
    prompt_concurrency: int = 4
    decode_concurrency: int = 4
    prefill_chunk_size: int = 256


def load_config() -> WorkerConfig:
    """Load the worker bootstrap configuration from environment variables.

    Reads ``MLX_RUNTIME_SOCKET``, ``MLX_RUNTIME_MODEL``, and the optional
    ``MLX_RUNTIME_VLM_MODEL``. An empty VLM model variable is treated the
    same as unset (no VLM engine).
    """

    raw_vlm = os.environ.get("MLX_RUNTIME_VLM_MODEL", "")
    vlm_model: str | None = raw_vlm if raw_vlm.strip() else None
    max_vlm_images = int(os.environ.get("MLX_RUNTIME_MAX_VLM_IMAGES", "5"))
    continuous_batching = _load_bool("MLX_RUNTIME_CONTINUOUS_BATCHING", "0")
    prompt_concurrency = int(os.environ.get("MLX_RUNTIME_PROMPT_CONCURRENCY", "4"))
    decode_concurrency = int(os.environ.get("MLX_RUNTIME_DECODE_CONCURRENCY", "4"))
    prefill_chunk_size = int(os.environ.get("MLX_RUNTIME_PREFILL_CHUNK_SIZE", "256"))

    return WorkerConfig(
        socket_path=os.environ.get("MLX_RUNTIME_SOCKET", "/tmp/mlx-runtime.sock"),
        model=os.environ.get(
            "MLX_RUNTIME_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
        ),
        vlm_model=vlm_model,
        max_vlm_images=max_vlm_images,
        continuous_batching=continuous_batching,
        prompt_concurrency=prompt_concurrency,
        decode_concurrency=decode_concurrency,
        prefill_chunk_size=prefill_chunk_size,
    )


def _load_bool(name: str, default: str) -> bool:
    value = os.environ.get(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}
