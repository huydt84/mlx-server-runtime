"""Native MLX architecture detection and construction."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..ipc import ModelError
from .cache import PagedKVCacheBackend
from .cache_coordinator import NativeCacheCoordinator
from .diagnostics import ModelDiagnostics
from .executor import MlxGenerationExecutor
from .execution_backends import (
    DEFAULT_NATIVE_EXECUTION_BACKEND,
    build_native_execution_backend,
    validate_native_execution_backend_id,
)
from .graph_profile import GraphProfiledModel
from .interfaces import NativeBackendOptions
from .mapping import (
    WeightArtifactValidationError,
    WeightMappingBug,
    load_mapped_weights,
    load_weight_index,
)
from .registry import ArchitectureSpec, get_architecture_spec
from .prefix_cache import (
    BlockHashPrefixCache,
    NoPrefixCache,
    PrefixCompatibilityFingerprint,
    RadixPrefixCache,
)


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
    execution_backend_id: str
    options: NativeBackendOptions
    executor: MlxGenerationExecutor
    cache_coordinator: NativeCacheCoordinator
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
    *,
    cache_budget_bytes: int = 8 * 1024 * 1024,
    cache_max_entries: int = 32,
    kv_page_size: int = 16,
    execution_backend_id: str = DEFAULT_NATIVE_EXECUTION_BACKEND,
    prefix_cache_strategy: str = "radix",
    graph_profile: bool = False,
) -> BootstrapArtifacts:
    """Validate and construct the registered architecture and shared executor."""

    try:
        validate_native_execution_backend_id(execution_backend_id)
    except ValueError as exc:
        raise _failure(
            "UNSUPPORTED_NATIVE_EXECUTION_BACKEND",
            str(exc),
            "backend_selection",
            "invalid_configuration",
            execution_backend_id,
        ) from exc
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
        if graph_profile:
            native_model = GraphProfiledModel(native_model)
        geometry = architecture.spec.cache_geometry(config)
        execution_backend = build_native_execution_backend(
            execution_backend_id,
            geometry,
            page_size=kv_page_size,
            cache_budget_bytes=cache_budget_bytes,
        )
        cache_backend = execution_backend.cache_backend
        attention_backend = execution_backend.attention_backend
        prefix_cache = _build_prefix_cache(
            strategy=prefix_cache_strategy,
            supports_prefix_cache=architecture.spec.supports_prefix_cache,
            backend=cache_backend,
            model=model,
            model_path=architecture.model_path,
            architecture_class=architecture.architecture_class,
            model_config=config,
            page_size=kv_page_size,
            cache_budget_bytes=cache_budget_bytes,
            cache_max_entries=cache_max_entries,
        )
        cache_coordinator = NativeCacheCoordinator(
            backend=cache_backend,
            prefix_cache=prefix_cache,
        )
        executor = MlxGenerationExecutor(
            architecture_class=architecture.architecture_class,
            model=native_model,
            cache_backend=cache_backend,
            attention_backend=attention_backend,
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
        execution_backend_id=execution_backend.backend_id,
        options=NativeBackendOptions(
            model=model,
            architecture_class=architecture.architecture_class,
        ),
        executor=executor,
        cache_coordinator=cache_coordinator,
        diagnostics=ModelDiagnostics(
            model=native_model,
            executor=executor,
            cache_coordinator=cache_coordinator,
            model_config=config,
        ),
        tokenizer=tokenizer,
        decode_target=decode_target,
        eos_token_ids=eos,
    )


def _build_prefix_cache(
    *,
    strategy: str,
    supports_prefix_cache: bool,
    backend: PagedKVCacheBackend,
    model: str,
    model_path: Path,
    architecture_class: str,
    model_config: Any,
    page_size: int,
    cache_budget_bytes: int,
    cache_max_entries: int,
) -> NoPrefixCache | BlockHashPrefixCache | RadixPrefixCache:
    if not supports_prefix_cache:
        return NoPrefixCache()
    compatibility = PrefixCompatibilityFingerprint(
        checkpoint=_checkpoint_identity(model, model_path),
        architecture_class=architecture_class,
        tokenizer_assets_hash=_tokenizer_assets_hash(model_path),
        model_dtype=str(getattr(model_config, "kv_cache_dtype", "float16")),
        kv_dtype=str(getattr(model_config, "kv_cache_dtype", "float16")),
        quantization=str(getattr(model_config, "quantization", None)),
        page_size=page_size,
    )
    if strategy == "block-hash":
        return BlockHashPrefixCache(
            backend=backend,
            compatibility=compatibility,
            page_size=page_size,
            max_entries=cache_max_entries,
            max_bytes=cache_budget_bytes,
        )
    if strategy == "radix":
        return RadixPrefixCache(
            backend=backend,
            compatibility=compatibility,
            page_size=page_size,
            max_entries=cache_max_entries,
            max_bytes=cache_budget_bytes,
        )
    raise ValueError(
        "MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY must be block-hash or radix"
    )


def _checkpoint_identity(model: str, model_path: Path) -> str:
    config_path = model_path / "config.json"
    stat = config_path.stat()
    return f"{model}|{model_path.resolve()}|config:{stat.st_size}:{stat.st_mtime_ns}"


def _tokenizer_assets_hash(model_path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = model_path / name
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        if path.exists():
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    # Some valid MLX exports (including LFM2) store special-token metadata in
    # tokenizer.json/tokenizer_config.json and omit the optional sidecar.
    required = ("tokenizer.json", "tokenizer_config.json")
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
