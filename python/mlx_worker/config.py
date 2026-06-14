"""Worker bootstrap configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration needed for the Phase 0 readiness handshake."""

    socket_path: str
    model: str


def load_config() -> WorkerConfig:
    """Load the worker bootstrap configuration from environment variables."""

    return WorkerConfig(
        socket_path=os.environ.get("MLX_RUNTIME_SOCKET", "/tmp/mlx-runtime.sock"),
        model=os.environ.get(
            "MLX_RUNTIME_MODEL", "mlx-community/Qwen2.5-7B-Instruct-4bit"
        ),
    )

