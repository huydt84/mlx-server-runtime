"""Architecture registry for native MLX startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .cache import DenseKVCacheBackend, KVCacheBackend
from .interfaces import NativeModel
from .mapping import WeightMappingAdapter
from .models.qwen2 import (
    Qwen2ModelConfig,
    Qwen2WeightAdapter,
    build_qwen2_model,
    parse_qwen2_config,
)


@dataclass(frozen=True)
class CompatibilityProbe:
    """Named compatibility probe for one architecture class."""

    name: str
    model_ref: str
    expected_category: str
    expected_stage: str


@dataclass(frozen=True)
class ArchitectureSpec:
    """Architecture-specific construction plugged into shared execution."""

    architecture_class: str
    known_good_checkpoint: str
    compatibility_probes: tuple[CompatibilityProbe, ...]
    parse_config: Callable[[dict[str, Any]], Any]
    create_weight_adapter: Callable[[], WeightMappingAdapter]
    create_model: Callable[[Any, list[tuple[str, Any]]], NativeModel]
    create_cache_backend: Callable[[Any], KVCacheBackend]


_REGISTRY: dict[str, ArchitectureSpec] = {
    "Qwen2ForCausalLM": ArchitectureSpec(
        architecture_class="Qwen2ForCausalLM",
        known_good_checkpoint="mlx-community/Qwen2.5-7B-Instruct-4bit",
        compatibility_probes=(
            CompatibilityProbe(
                name="unsupported-llama-class",
                model_ref="local-probe/LlamaForCausalLM",
                expected_category="unsupported_class",
                expected_stage="architecture_detection",
            ),
            CompatibilityProbe(
                name="missing-tokenizer-assets",
                model_ref="local-probe/Qwen2ForCausalLM-missing-tokenizer",
                expected_category="malformed_checkpoint",
                expected_stage="prompt_tokenizer_readiness",
            ),
        ),
        parse_config=parse_qwen2_config,
        create_weight_adapter=Qwen2WeightAdapter,
        create_model=build_qwen2_model,
        create_cache_backend=lambda config: DenseKVCacheBackend(
            num_layers=int(config.num_hidden_layers)
        ),
    )
}


def get_architecture_spec(architecture_class: str) -> ArchitectureSpec | None:
    """Return the registered architecture specification."""

    return _REGISTRY.get(architecture_class)


def qwen2_spec() -> ArchitectureSpec:
    """Return the Qwen2 specification."""

    return _REGISTRY["Qwen2ForCausalLM"]


__all__ = [
    "ArchitectureSpec",
    "CompatibilityProbe",
    "Qwen2ModelConfig",
    "get_architecture_spec",
    "qwen2_spec",
]
