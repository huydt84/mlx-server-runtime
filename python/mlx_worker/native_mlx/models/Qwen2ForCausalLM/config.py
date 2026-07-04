"""Qwen2 config parser for native MLX startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Qwen2ModelConfig:
    """Subset of Hugging Face config required for Qwen2 skeleton startup."""

    architecture_class: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    rope_traditional: bool
    rope_scaling: dict[str, Any] | None
    tie_word_embeddings: bool
    quantization: dict[str, Any] | None


def parse_qwen2_config(payload: dict[str, Any]) -> Qwen2ModelConfig:
    """Parse and validate Qwen2 config payload.

    Raises:
        ValueError: If required config fields are missing or invalid.
    """

    architecture_class = _require_str(
        payload, "architectures[0]", _architecture_name(payload)
    )
    model_type = _require_str(payload, "model_type")
    if model_type != "qwen2":
        raise ValueError(f"expected model_type='qwen2', got {model_type!r}")

    hidden_size = _require_int(payload, "hidden_size")
    intermediate_size = _require_int(payload, "intermediate_size")
    num_attention_heads = _require_int(payload, "num_attention_heads")
    num_hidden_layers = _require_int(payload, "num_hidden_layers")
    num_key_value_heads = int(payload.get("num_key_value_heads", num_attention_heads))
    vocab_size = _require_int(payload, "vocab_size")
    max_position_embeddings = _require_int(payload, "max_position_embeddings")
    rms_norm_eps = _require_float(payload, "rms_norm_eps")
    rope_theta = float(payload.get("rope_theta", 1000000.0))
    rope_traditional = bool(payload.get("rope_traditional", False))
    rope_scaling = payload.get("rope_scaling")
    if rope_scaling is not None and not isinstance(rope_scaling, dict):
        raise ValueError("missing or invalid rope_scaling")
    tie_word_embeddings = bool(payload.get("tie_word_embeddings", True))
    quantization = payload.get("quantization")
    if quantization is not None and not isinstance(quantization, dict):
        raise ValueError("missing or invalid quantization")
    if hidden_size % num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

    return Qwen2ModelConfig(
        architecture_class=architecture_class,
        model_type=model_type,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_traditional=rope_traditional,
        rope_scaling=rope_scaling,
        tie_word_embeddings=tie_word_embeddings,
        quantization=quantization,
    )


def _architecture_name(payload: dict[str, Any]) -> Any:
    architectures = payload.get("architectures")
    if isinstance(architectures, list) and architectures:
        return architectures[0]
    return None


def _require_str(payload: dict[str, Any], key: str, value: Any | None = None) -> str:
    candidate = payload.get(key) if value is None else value
    if not isinstance(candidate, str) or not candidate.strip():
        raise ValueError(f"missing or invalid {key}")
    return candidate


def _require_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"missing or invalid {key}")
    return value


def _require_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"missing or invalid {key}")
    return float(value)
