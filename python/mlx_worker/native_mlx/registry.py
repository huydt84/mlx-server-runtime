"""Architecture registry for native MLX startup."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib
from typing import Any, Callable

import mlx.core as mx

from .cache import KVCacheGeometry
from .interfaces import NativeModel
from .mapping import WeightMappingAdapter


@dataclass(frozen=True)
class CompatibilityProbe:
    """Named compatibility probe for one architecture class."""

    name: str
    model_ref: str
    expected_category: str
    expected_stage: str


@dataclass(frozen=True)
class ArchitectureSpec:
    """Lazy architecture metadata selected during native bootstrap.

    The registry is imported by every native worker, while a model module is
    needed by only the selected worker.  Keeping the import target in the
    manifest prevents adding another family from adding model imports to the
    request process or to an unrelated family's execution path.
    """

    architecture_class: str
    known_good_checkpoint: str
    compatibility_probes: tuple[CompatibilityProbe, ...]
    module_name: str
    parse_config_name: str
    weight_adapter_name: str
    model_factory_name: str
    cache_geometry_factory: Callable[[Any], KVCacheGeometry]
    supports_prefix_cache: bool = True
    cache_family: str = "kv"

    @lru_cache(maxsize=None)
    def resolve(self) -> "ArchitectureExecutionPlan":
        """Freeze selected family callables once during bootstrap."""

        module = _load_module(self.module_name)
        return ArchitectureExecutionPlan(
            architecture_class=self.architecture_class,
            parse_config=getattr(module, self.parse_config_name),
            create_weight_adapter=getattr(module, self.weight_adapter_name),
            create_model=getattr(module, self.model_factory_name),
            cache_geometry=self.cache_geometry_factory,
            supports_prefix_cache=self.supports_prefix_cache,
            cache_family=self.cache_family,
        )

    @property
    def parse_config(self) -> Callable[[dict[str, Any]], Any]:
        """Load the selected family's config parser during startup only."""

        return self.resolve().parse_config

    def create_weight_adapter(self) -> WeightMappingAdapter:
        """Construct the selected family's weight adapter during startup."""

        return self.resolve().create_weight_adapter()

    def create_model(self, config: Any, weights: list[tuple[str, Any]]) -> NativeModel:
        """Construct the selected family's model during startup."""

        return self.resolve().create_model(config, weights)

    def cache_geometry(self, config: Any) -> KVCacheGeometry:
        """Return the startup-frozen cache geometry for this family."""

        return self.resolve().cache_geometry(config)


@dataclass(frozen=True)
class ArchitectureExecutionPlan:
    """Concrete startup plan used after registry selection."""

    architecture_class: str
    parse_config: Callable[[dict[str, Any]], Any]
    create_weight_adapter: Callable[[], WeightMappingAdapter]
    create_model: Callable[[Any, list[tuple[str, Any]]], NativeModel]
    cache_geometry: Callable[[Any], KVCacheGeometry]
    supports_prefix_cache: bool
    cache_family: str


@lru_cache(maxsize=None)
def _load_module(module_name: str) -> Any:
    """Import one model module once, after architecture selection."""

    return importlib.import_module(module_name, __package__)


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
        module_name=".models.qwen2",
        parse_config_name="parse_qwen2_config",
        weight_adapter_name="Qwen2WeightAdapter",
        model_factory_name="build_qwen2_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
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
        module_name=".models.qwen3",
        parse_config_name="parse_qwen3_config",
        weight_adapter_name="Qwen3WeightAdapter",
        model_factory_name="build_qwen3_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
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
        module_name=".models.gemma3",
        parse_config_name="parse_gemma3_config",
        weight_adapter_name="Gemma3WeightAdapter",
        model_factory_name="build_gemma3_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
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
        module_name=".models.lfm2",
        parse_config_name="parse_lfm2_config",
        weight_adapter_name="Lfm2WeightAdapter",
        model_factory_name="build_lfm2_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.hidden_size // config.num_attention_heads),
            dtype=(mx.bfloat16 if config.kv_cache_dtype == "bfloat16" else mx.float16),
        ),
        supports_prefix_cache=False,
        cache_family="hybrid",
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
    "get_architecture_spec",
    "qwen2_spec",
]
