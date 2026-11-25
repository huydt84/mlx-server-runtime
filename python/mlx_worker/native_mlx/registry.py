"""Architecture registry for native MLX startup."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache, cached_property
from pathlib import Path
from typing import Any

import mlx.core as mx

from .cache import KVCacheGeometry
from .interfaces import NativeModel
from .mapping import WeightIndex, WeightMappingAdapter, load_weight_index
from .speculative import DrafterCheckpoint


@dataclass(frozen=True)
class CompatibilityProbe:
    """Named compatibility probe for one architecture class."""

    name: str
    model_ref: str
    expected_category: str
    expected_stage: str


@dataclass(frozen=True)
class StepMemory:
    """Measured memory of a family's steps beside its weights and KV.

    Attributes:
        base_bytes: Activations of a step of few tokens.
        bytes_per_token: Further activations per prefill token.
        buffer_cache_bytes: Freed buffers worth keeping for reuse.
    """

    base_bytes: int
    bytes_per_token: int
    buffer_cache_bytes: int


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
    weight_index_loader_name: str | None = None
    """Loader in the model module for checkpoints without safetensors indexes."""
    drafter_module_name: str | None = None
    """Module whose ``load_drafter`` opens the family's drafter checkpoints."""
    step_memory: StepMemory | None = None
    """Replaces the estimate of activation memory where it does not fit."""

    @cached_property
    def resolve(self) -> ArchitectureExecutionPlan:
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
            load_weight_index=(
                load_weight_index
                if self.weight_index_loader_name is None
                else getattr(module, self.weight_index_loader_name)
            ),
        )

    def load_drafter(self, drafter_path: Path, model_path: Path) -> DrafterCheckpoint:
        """Open a drafter checkpoint for this family.

        Args:
            drafter_path: Local drafter checkpoint directory.
            model_path: Local directory of the served checkpoint.

        Raises:
            ValueError: If the family has no drafter or cannot use this one.
        """

        if self.drafter_module_name is None:
            raise ValueError(
                f"native-mlx has no speculative decoding for {self.architecture_class}"
            )
        module = _load_module(self.drafter_module_name)
        return module.load_drafter(drafter_path, model_path)

    @property
    def parse_config(self) -> Callable[[dict[str, Any]], Any]:
        """Load the selected family's config parser during startup only."""

        return self.resolve.parse_config

    def create_weight_adapter(self) -> WeightMappingAdapter:
        """Construct the selected family's weight adapter during startup."""

        return self.resolve.create_weight_adapter()

    def create_model(self, config: Any, weights: list[tuple[str, Any]]) -> NativeModel:
        """Construct the selected family's model during startup."""

        return self.resolve.create_model(config, weights)

    def cache_geometry(self, config: Any) -> KVCacheGeometry:
        """Return the startup-frozen cache geometry for this family."""

        return self.resolve.cache_geometry(config)


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
    load_weight_index: Callable[[Path], WeightIndex] = load_weight_index


_MIB = 1024 * 1024
_KV_DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}


@cache
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
    "Qwen3_5ForConditionalGeneration": ArchitectureSpec(
        architecture_class="Qwen3_5ForConditionalGeneration",
        known_good_checkpoint="prism-ml/Bonsai-27B-mlx-1bit",
        compatibility_probes=(),
        module_name=".models.qwen3_5",
        parse_config_name="parse_qwen3_5_config",
        weight_adapter_name="Qwen3_5WeightAdapter",
        model_factory_name="build_qwen3_5_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
            num_layers=int(config.num_hidden_layers),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            dtype=_KV_DTYPES[config.kv_cache_dtype],
            kv_layer_indices=tuple(config.full_attention_layers),
        ),
        supports_prefix_cache=False,
        cache_family="hybrid",
    ),
    "Inkling2LoopForConditionalGeneration": ArchitectureSpec(
        architecture_class="Inkling2LoopForConditionalGeneration",
        known_good_checkpoint=("shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-5.4GB"),
        compatibility_probes=(),
        module_name=".models.inkling2",
        parse_config_name="parse_inkling2_config",
        weight_adapter_name="Inkling2PackAdapter",
        model_factory_name="build_inkling2_model",
        cache_geometry_factory=lambda config: KVCacheGeometry(
            num_layers=int(config.num_visits),
            num_kv_heads=int(config.num_key_value_heads),
            head_dim=int(config.head_dim),
            dtype=_KV_DTYPES[config.kv_cache_dtype],
            window_layer_indices=tuple(config.window_visits),
            window_size=int(config.window_size),
        ),
        supports_prefix_cache=False,
        cache_family="windowed",
        weight_index_loader_name="load_packed_weight_index",
        drafter_module_name=".models.inkling2_drafter",
        step_memory=StepMemory(
            base_bytes=1024 * _MIB,
            bytes_per_token=_MIB // 2,
            buffer_cache_bytes=256 * _MIB,
        ),
    ),
}

_MODEL_TYPE_ARCHITECTURES: dict[str, str] = {
    "prism_hadamard_qwen35": "Qwen3_5ForConditionalGeneration",
}


def get_architecture_spec(architecture_class: str) -> ArchitectureSpec | None:
    """Return the registered architecture specification."""

    return _REGISTRY.get(architecture_class)


def architecture_for_model_type(model_type: object) -> str | None:
    """Return the architecture class implementing an architecture-less model type."""

    return (
        _MODEL_TYPE_ARCHITECTURES.get(model_type)
        if isinstance(model_type, str)
        else None
    )


def qwen2_spec() -> ArchitectureSpec:
    """Return the Qwen2 specification."""

    return _REGISTRY["Qwen2ForCausalLM"]


__all__ = [
    "ArchitectureSpec",
    "CompatibilityProbe",
    "architecture_for_model_type",
    "get_architecture_spec",
    "qwen2_spec",
]
