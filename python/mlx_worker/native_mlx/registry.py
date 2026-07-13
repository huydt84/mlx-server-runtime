"""Architecture registry for native MLX startup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx

from .cache import KVCacheGeometry
from .interfaces import NativeModel
from .mapping import WeightMappingAdapter
from .models.qwen2 import (
    Qwen2ModelConfig,
    Qwen2WeightAdapter,
    build_qwen2_model,
    parse_qwen2_config,
)
from .models.qwen3 import (
    Qwen3WeightAdapter,
    build_qwen3_model,
    parse_qwen3_config,
)
from .models.gemma3 import (
    Gemma3WeightAdapter,
    build_gemma3_model,
    parse_gemma3_config,
)
from .models.lfm2 import (
    Lfm2WeightAdapter,
    build_lfm2_model,
    parse_lfm2_config,
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
    cache_geometry: Callable[[Any], KVCacheGeometry]
    supports_prefix_cache: bool = True


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
        cache_geometry=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.hidden_size // config.num_attention_heads),
            dtype=(mx.bfloat16 if config.kv_cache_dtype == "bfloat16" else mx.float16),
        ),
    ),
    "Qwen3ForCausalLM": ArchitectureSpec(
        architecture_class="Qwen3ForCausalLM",
        known_good_checkpoint="mlx-community/Qwen3-4B-Instruct-2507-4bit",
        compatibility_probes=(),
        parse_config=parse_qwen3_config,
        create_weight_adapter=Qwen3WeightAdapter,
        create_model=build_qwen3_model,
        cache_geometry=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            dtype=(mx.bfloat16 if config.kv_cache_dtype == "bfloat16" else mx.float16),
        ),
    ),
    "Gemma3ForCausalLM": ArchitectureSpec(
        architecture_class="Gemma3ForCausalLM",
        known_good_checkpoint="mlx-community/gemma-3-270m-it-qat-8bit",
        compatibility_probes=(),
        parse_config=parse_gemma3_config,
        create_weight_adapter=Gemma3WeightAdapter,
        create_model=build_gemma3_model,
        cache_geometry=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            dtype=(mx.bfloat16 if config.kv_cache_dtype == "bfloat16" else mx.float16),
        ),
    ),
    "Lfm2MoeForCausalLM": ArchitectureSpec(
        architecture_class="Lfm2MoeForCausalLM",
        known_good_checkpoint="mlx-community/LFM2.5-8B-A1B-MLX-4bit",
        compatibility_probes=(),
        parse_config=parse_lfm2_config,
        create_weight_adapter=Lfm2WeightAdapter,
        create_model=build_lfm2_model,
        cache_geometry=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.hidden_size // config.num_attention_heads),
            dtype=(mx.bfloat16 if config.kv_cache_dtype == "bfloat16" else mx.float16),
        ),
        supports_prefix_cache=False,
    ),
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
