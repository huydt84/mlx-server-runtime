"""Architecture registry for native MLX startup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .interfaces import NativeMlxExecutor
from .mapping import WeightIndex, WeightMappingAdapter, WeightMappingPlan
from .models.Qwen2ForCausalLM.config import Qwen2ModelConfig, parse_qwen2_config
from .models.Qwen2ForCausalLM.model import Qwen2NativeMlxExecutor
from .models.Qwen2ForCausalLM.weights import Qwen2WeightAdapter


@dataclass(frozen=True)
class CompatibilityProbe:
    """Named compatibility probe for one architecture class."""

    name: str
    model_ref: str
    expected_category: str
    expected_stage: str


@dataclass(frozen=True)
class ArchitectureSpec:
    """Registry entry for one explicitly implemented architecture class."""

    architecture_class: str
    known_good_checkpoint: str
    compatibility_probes: tuple[CompatibilityProbe, ...]
    parse_config: Callable[[dict[str, Any]], Any]
    create_weight_adapter: Callable[[], WeightMappingAdapter]
    create_executor: Callable[
        [Path, Any, WeightMappingPlan, WeightIndex], NativeMlxExecutor
    ]


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
        create_executor=lambda model_path, model_config, weight_plan, weight_index: (
            Qwen2NativeMlxExecutor(
                model_path=model_path,
                model_config=model_config,
                weight_plan=weight_plan,
                weight_index=weight_index,
            )
        ),
    )
}


def get_architecture_spec(architecture_class: str) -> ArchitectureSpec | None:
    """Return registry entry for architecture class, if supported."""

    return _REGISTRY.get(architecture_class)


def qwen2_spec() -> ArchitectureSpec:
    """Return typed spec for ``Qwen2ForCausalLM``."""

    return _REGISTRY["Qwen2ForCausalLM"]


__all__ = [
    "ArchitectureSpec",
    "CompatibilityProbe",
    "Qwen2ModelConfig",
    "get_architecture_spec",
    "qwen2_spec",
]
