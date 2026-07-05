"""Native MLX architecture detection and construction."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ipc import ModelError
from .diagnostics import ModelDiagnostics
from .executor import MlxGenerationExecutor
from .interfaces import NativeBackendOptions
from .mapping import (
    WeightArtifactValidationError,
    WeightMappingBug,
    load_mapped_weights,
    load_weight_index,
)
from .registry import ArchitectureSpec, get_architecture_spec


@dataclass(frozen=True)
class NativeArchitecture:
    """Detected architecture metadata."""

    model: str
    model_path: Path
    architecture_class: str
    raw_config: dict[str, Any]
    spec: ArchitectureSpec


@dataclass(frozen=True)
class BootstrapArtifacts:
    """Fully constructed model-independent serving dependencies."""

    architecture: NativeArchitecture
    options: NativeBackendOptions
    executor: MlxGenerationExecutor
    diagnostics: ModelDiagnostics
    tokenizer: Any
    decode_target: Any
    eos_token_ids: tuple[int, ...]


class NativeBootstrapFailure(RuntimeError):
    """Structured native startup failure."""

    def __init__(self, error: ModelError) -> None:
        super().__init__(error.message)
        self.error = error


def build_native_artifacts(
    model: str,
    stage_callback: Any | None = None,
) -> BootstrapArtifacts:
    """Validate and construct the registered architecture and shared executor."""

    architecture = detect_native_architecture(model)
    if stage_callback:
        stage_callback("artifact_validation", "verifying")
    try:
        config = architecture.spec.parse_config(architecture.raw_config)
    except ValueError as exc:
        raise _failure(
            "INVALID_NATIVE_CONFIG",
            str(exc),
            "artifact_validation",
            "malformed_checkpoint",
            str(architecture.model_path / "config.json"),
        ) from exc
    try:
        index = load_weight_index(architecture.model_path)
    except WeightArtifactValidationError as exc:
        raise _failure(
            "INVALID_WEIGHT_ARTIFACTS",
            str(exc),
            "artifact_validation",
            "malformed_checkpoint",
            str(architecture.model_path),
        ) from exc
    try:
        plan = architecture.spec.create_weight_adapter().build_plan(index)
    except WeightMappingBug as exc:
        raise _failure(
            "WEIGHT_MAPPING_UNSUPPORTED",
            str(exc),
            "weight_mapping",
            "supported_class_bug",
            model,
        ) from exc
    try:
        weights = load_mapped_weights(index, plan)
    except WeightArtifactValidationError as exc:
        raise _failure(
            "INVALID_WEIGHT_ARTIFACTS",
            str(exc),
            "artifact_validation",
            "malformed_checkpoint",
            str(architecture.model_path),
        ) from exc
    if stage_callback:
        stage_callback("weight_mapping", "loading_weights")
    try:
        native_model = architecture.spec.create_model(config, weights)
        executor = MlxGenerationExecutor(
            architecture_class=architecture.architecture_class,
            model=native_model,
            cache_backend=architecture.spec.create_cache_backend(config),
        )
    except Exception as exc:
        raise _failure(
            "NATIVE_EXECUTOR_CONSTRUCTION_FAILED",
            str(exc),
            "native_executor_construction",
            "supported_class_bug",
            model,
        ) from exc
    if stage_callback:
        stage_callback("native_executor_construction", "initializing_runtime")
    if stage_callback:
        stage_callback("prompt_tokenizer_readiness", "initializing_runtime")
    tokenizer, decode_target, eos = _load_tokenizer(architecture.model_path)
    return BootstrapArtifacts(
        architecture=architecture,
        options=NativeBackendOptions(
            model=model,
            architecture_class=architecture.architecture_class,
        ),
        executor=executor,
        diagnostics=ModelDiagnostics(
            model=native_model,
            executor=executor,
            model_config=config,
        ),
        tokenizer=tokenizer,
        decode_target=decode_target,
        eos_token_ids=eos,
    )


def detect_native_architecture(model: str) -> NativeArchitecture:
    """Resolve config.json and select one explicitly registered model."""

    path = resolve_model_path(model)
    config_path = path / "config.json"
    try:
        payload = json.loads(config_path.read_text())
    except Exception as exc:
        raise _failure(
            "INVALID_MODEL_CONFIG",
            f"could not parse {config_path}: {exc}",
            "artifact_validation",
            "malformed_checkpoint",
            str(config_path),
        ) from exc
    architectures = payload.get("architectures")
    name = (
        architectures[0] if isinstance(architectures, list) and architectures else None
    )
    if not isinstance(name, str) or not name:
        raise _failure(
            "MISSING_ARCHITECTURES",
            "config.json is missing architectures[0]",
            "architecture_detection",
            "malformed_checkpoint",
            str(config_path),
        )
    spec = get_architecture_spec(name)
    if spec is None:
        raise _failure(
            "UNSUPPORTED_ARCHITECTURE_CLASS",
            f"native-mlx does not implement {name}",
            "architecture_detection",
            "unsupported_class",
            model,
        )
    return NativeArchitecture(model, path, name, payload, spec)


def _load_tokenizer(model_path: Path) -> tuple[Any, Any, tuple[int, ...]]:
    required = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
    missing = [name for name in required if not (model_path / name).exists()]
    if missing:
        raise _failure(
            "MISSING_TOKENIZER_ASSET",
            "missing tokenizer assets: " + ", ".join(missing),
            "prompt_tokenizer_readiness",
            "malformed_checkpoint",
            str(model_path),
        )
    try:
        from mlx_lm.utils import load_tokenizer

        wrapper = load_tokenizer(model_path)
    except Exception as exc:
        raise _failure(
            "TOKENIZER_LOAD_FAILED",
            str(exc),
            "prompt_tokenizer_readiness",
            "malformed_checkpoint",
            str(model_path),
        ) from exc
    raw = getattr(wrapper, "_tokenizer", None) or getattr(wrapper, "tokenizer", None)
    decode_target = raw or wrapper
    eos: Any = getattr(raw, "eos_token_ids", None)
    if eos is None:
        value = getattr(raw, "eos_token_id", None)
        eos = [] if value is None else [value]
    elif isinstance(eos, int):
        eos = [eos]
    return wrapper, decode_target, tuple(int(value) for value in eos)


def resolve_model_path(model: str) -> Path:
    """Resolve a local path or cached Hugging Face checkpoint."""

    path = Path(model)
    if path.is_file():
        return path.parent
    if path.is_dir():
        return path
    try:
        from mlx_lm.utils import hf_repo_to_path

        return Path(hf_repo_to_path(model))
    except Exception as exc:
        raise _failure(
            "MODEL_RESOLUTION_FAILED",
            str(exc),
            "artifact_validation",
            "malformed_checkpoint",
            model,
        ) from exc


def build_finalized_token_ids(
    model_path: Path,
    messages: list[dict[str, str]],
) -> list[int]:
    """Build runtime-equivalent finalized prompt tokens for diagnostics."""

    tokenizer, _, _ = _load_tokenizer(Path(model_path))
    values = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return [int(value) for value in values]


def _failure(
    code: str,
    message: str,
    stage: str,
    category: str,
    detail: str,
) -> NativeBootstrapFailure:
    return NativeBootstrapFailure(
        ModelError(
            code=code,
            message=f"{message}. Default v1 backend remains available.",
            at=int(time.time()),
            backend="native-mlx",
            stage=stage,
            category=category,
            detail=detail,
        )
    )
