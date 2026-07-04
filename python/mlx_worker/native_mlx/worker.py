"""Native MLX backend bootstrap seam for Phase 3."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from socket import socket
from typing import Any, Callable, Sequence

import mlx.core as mx

from ..config import WorkerConfig
from ..ipc import (
    ModelError,
    ModelLoadProgress,
    ModelStatus,
    WorkerError,
    encode_bootstrap_message,
)
from .interfaces import (
    NativeBackendOptions,
    NativeMlxExecutor,
    NativeScheduler,
)
from .mapping import (
    WeightArtifactValidationError,
    WeightIndex,
    WeightMappingBug,
    WeightMappingPlan,
    load_weight_index,
)
from .models.Qwen2ForCausalLM.debug_trace import (
    TraceArtifacts,
    compare_trace_runs,
    trace_qwen2_run,
    write_trace_artifacts,
)
from .models.Qwen2ForCausalLM.cache import Qwen2LayerCache
from .registry import ArchitectureSpec, get_architecture_spec


SUPPORTED_ARCHITECTURE_CLASSES = frozenset({"Qwen2ForCausalLM"})


@dataclass(frozen=True)
class NativeArchitecture:
    """Detected architecture metadata for native bootstrap."""

    model: str
    model_path: Path
    architecture_class: str
    raw_config: dict[str, Any]
    spec: ArchitectureSpec


@dataclass(frozen=True)
class NativeParityResult:
    """Deterministic parity result for one finalized token sequence."""

    checkpoint: str
    token_ids: tuple[int, ...]
    logits_shape: tuple[int, ...]
    logits_dtype: str
    max_abs_diff: float
    tolerance_atol: float
    tolerance_rtol: float
    tolerance_ok: bool
    native_next_token: int
    reference_next_token: int
    token_ok: bool


@dataclass(frozen=True)
class NativePrefillDecodeParityResult:
    """Parity result for prefill plus decode steps."""

    checkpoint: str
    token_ids: tuple[int, ...]
    prefill_logits_shape: tuple[int, ...]
    prefill_logits_dtype: str
    prefill_max_abs_diff: float
    decode_max_abs_diff: float
    tolerance_atol: float
    tolerance_rtol: float
    tolerance_ok: bool
    native_tokens: tuple[int, ...]
    reference_tokens: tuple[int, ...]
    token_ok: bool
    cache_lengths: tuple[int, ...]
    prefill_time_ms: int


class NativeBootstrapFailure(RuntimeError):
    """Structured native bootstrap failure."""

    def __init__(self, error: ModelError) -> None:
        super().__init__(error.message)
        self.error = error


class SkeletonNativeScheduler:
    """Phase 3 scheduler seam with startup-only parity warmup."""

    def __init__(
        self,
        executor: NativeMlxExecutor,
        options: NativeBackendOptions,
        model_path: Path,
        model_ref: str,
        stage_callback: Callable[[str, str], None] | None,
    ) -> None:
        self._executor = executor
        self._options = options
        self._model_path = model_path
        self._model_ref = model_ref
        self._stage_callback = stage_callback

    def warmup(self) -> None:
        self._executor.load(self._options)
        if self._stage_callback is not None:
            self._stage_callback("prompt_tokenizer_readiness", "initializing_runtime")
        validate_tokenizer_assets(self._model_path)
        if self._stage_callback is not None:
            self._stage_callback("deterministic_warmup", "warming_up")

        token_ids = build_finalized_token_ids(
            self._model_path,
            [{"role": "user", "content": "ping"}],
        )
        parity = compare_native_prefill_decode_to_mlx_lm(
            self._model_ref,
            self._executor,
            token_ids,
            decode_steps=2,
        )
        if not parity.tolerance_ok:
            raise NativeBootstrapFailure(
                _startup_error(
                    code="NATIVE_LOGITS_PARITY_FAILED",
                    message=(
                        "native-mlx prefill/decode parity failed during deterministic warmup "
                        f"with prefill_max_abs_diff={parity.prefill_max_abs_diff:.6f} "
                        f"decode_max_abs_diff={parity.decode_max_abs_diff:.6f}"
                    ),
                    stage="deterministic_warmup",
                    category="supported_class_bug",
                    detail=self._model_ref,
                )
            )
        if not parity.token_ok:
            raise NativeBootstrapFailure(
                _startup_error(
                    code="NATIVE_GREEDY_PARITY_FAILED",
                    message=(
                        "native-mlx prefill/decode token parity failed during deterministic warmup "
                        f"(native={parity.native_tokens}, mlx_lm={parity.reference_tokens})"
                    ),
                    stage="deterministic_warmup",
                    category="supported_class_bug",
                    detail=self._model_ref,
                )
            )

        raise NativeBootstrapFailure(
            _startup_error(
                code="NATIVE_PUBLIC_SERVING_NOT_IMPLEMENTED",
                message=(
                    "native-mlx deterministic prefill/decode parity passed, "
                    "but public native serving is not implemented yet"
                ),
                stage="deterministic_warmup",
                category="supported_class_bug",
                detail=self._model_ref,
            )
        )


def run_native_worker(
    client: socket,
    config: WorkerConfig,
    *,
    native_worker_factory: Callable[..., NativeScheduler] | None = None,
) -> int:
    """Run native backend bootstrap until structured startup result."""

    bootstrap_started_at = _now_seconds()
    stage_emitter = _make_stage_emitter(client, config.model, bootstrap_started_at)
    stage_emitter("architecture_detection", "verifying")

    if native_worker_factory is None:
        native_worker_factory = create_native_worker

    try:
        try:
            scheduler = native_worker_factory(config, stage_callback=stage_emitter)
        except TypeError:
            scheduler = native_worker_factory(config)
        scheduler.warmup()
    except NativeBootstrapFailure as exc:
        _emit_failure(client, config.model, bootstrap_started_at, exc.error)
        return 1
    except Exception as exc:
        error = _startup_error(
            code="NATIVE_STARTUP_FAILED",
            message=str(exc),
            stage="native_executor_construction",
            category="supported_class_bug",
            detail=config.model,
        )
        _emit_failure(client, config.model, bootstrap_started_at, error)
        return 1

    error = _startup_error(
        code="NATIVE_STARTUP_UNEXPECTED_READY",
        message="native-mlx Phase 3 must not report readiness before serving implementation",
        stage="deterministic_warmup",
        category="supported_class_bug",
        detail=config.model,
    )
    _emit_failure(client, config.model, bootstrap_started_at, error)
    return 1


def create_native_worker(
    config: WorkerConfig,
    *,
    stage_callback: Callable[[str, str], None] | None = None,
) -> NativeScheduler:
    """Create native scheduler and executor seams for selected model."""

    architecture = detect_native_architecture(config.model)
    if stage_callback is not None:
        stage_callback("artifact_validation", "verifying")

    model_config = parse_native_config(architecture)
    weight_index, weight_plan = validate_weight_artifacts(architecture)
    if stage_callback is not None:
        stage_callback("weight_mapping", "loading_weights")

    options = NativeBackendOptions(
        model=architecture.model,
        architecture_class=architecture.architecture_class,
    )
    executor = build_native_executor(
        architecture,
        model_config,
        weight_index,
        weight_plan,
    )
    if stage_callback is not None:
        stage_callback("native_executor_construction", "initializing_runtime")

    return SkeletonNativeScheduler(
        executor=executor,
        options=options,
        model_path=architecture.model_path,
        model_ref=architecture.model,
        stage_callback=stage_callback,
    )


def detect_native_architecture(model: str) -> NativeArchitecture:
    """Resolve model config and classify native backend startup."""

    model_path = _resolve_model_path(model)
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_MODEL_CONFIG",
                message=f"native-mlx could not find config.json for '{model}'",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    try:
        payload = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_MODEL_CONFIG",
                message=f"native-mlx could not parse config.json for '{model}': {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        ) from exc

    architectures = payload.get("architectures")
    if not isinstance(architectures, list) or not architectures:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_ARCHITECTURES",
                message=f"native-mlx config.json for '{model}' is missing architectures[]",
                stage="architecture_detection",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    architecture_class = architectures[0]
    if not isinstance(architecture_class, str) or not architecture_class.strip():
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_ARCHITECTURE_CLASS",
                message=f"native-mlx config.json for '{model}' has invalid architectures[0]",
                stage="architecture_detection",
                category="malformed_checkpoint",
                detail=str(config_path),
            )
        )

    if architecture_class not in SUPPORTED_ARCHITECTURE_CLASSES:
        raise NativeBootstrapFailure(
            _startup_error(
                code="UNSUPPORTED_ARCHITECTURE_CLASS",
                message=(
                    "native-mlx only supports explicitly implemented architecture "
                    f"classes; got {architecture_class}"
                ),
                stage="architecture_detection",
                category="unsupported_class",
                detail=model,
            )
        )

    spec = get_architecture_spec(architecture_class)
    if spec is None:
        raise NativeBootstrapFailure(
            _startup_error(
                code="UNSUPPORTED_ARCHITECTURE_CLASS",
                message=(
                    "native-mlx only supports explicitly implemented architecture "
                    f"classes; got {architecture_class}"
                ),
                stage="architecture_detection",
                category="unsupported_class",
                detail=model,
            )
        )

    return NativeArchitecture(
        model=model,
        model_path=model_path,
        architecture_class=architecture_class,
        raw_config=payload,
        spec=spec,
    )


def parse_native_config(architecture: NativeArchitecture) -> Any:
    """Parse per-class config with supported-class bug taxonomy."""

    try:
        return architecture.spec.parse_config(architecture.raw_config)
    except ValueError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_NATIVE_CONFIG",
                message=f"native-mlx config validation failed: {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(architecture.model_path / "config.json"),
            )
        ) from exc


def validate_weight_artifacts(
    architecture: NativeArchitecture,
) -> tuple[WeightIndex, WeightMappingPlan]:
    """Validate weight metadata and build canonical mapping plan."""

    try:
        index = load_weight_index(architecture.model_path)
    except WeightArtifactValidationError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="INVALID_WEIGHT_ARTIFACTS",
                message=f"native-mlx weight artifact validation failed: {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=str(architecture.model_path),
            )
        ) from exc

    adapter = architecture.spec.create_weight_adapter()
    try:
        plan = adapter.build_plan(index)
    except WeightMappingBug as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="WEIGHT_MAPPING_UNSUPPORTED",
                message=f"native-mlx weight mapping needs adapter work: {exc}",
                stage="weight_mapping",
                category="supported_class_bug",
                detail=architecture.model,
            )
        ) from exc

    return index, plan


def build_native_executor(
    architecture: NativeArchitecture,
    model_config: Any,
    weight_index: WeightIndex,
    weight_plan: WeightMappingPlan,
) -> NativeMlxExecutor:
    """Construct per-class executor from validated metadata."""

    try:
        return architecture.spec.create_executor(
            architecture.model_path,
            model_config,
            weight_plan,
            weight_index,
        )
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="NATIVE_EXECUTOR_CONSTRUCTION_FAILED",
                message=f"native-mlx executor construction failed: {exc}",
                stage="native_executor_construction",
                category="supported_class_bug",
                detail=architecture.model,
            )
        ) from exc


def validate_tokenizer_assets(model_path: Path) -> None:
    """Validate tokenizer and chat-template assets outside executor boundary."""

    required_assets = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    missing_assets = [
        name for name in required_assets if not (model_path / name).exists()
    ]
    if missing_assets:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_TOKENIZER_ASSET",
                message=(
                    "native-mlx tokenizer assets are incomplete: "
                    + ", ".join(missing_assets)
                ),
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        )

    try:
        tokenizer = _load_tokenizer_wrapper(model_path)
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="TOKENIZER_LOAD_FAILED",
                message=f"native-mlx tokenizer validation failed: {exc}",
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        ) from exc

    raw_tokenizer = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    chat_template = getattr(raw_tokenizer, "chat_template", None) or getattr(
        tokenizer, "chat_template", None
    )
    if not chat_template:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MISSING_CHAT_TEMPLATE",
                message="native-mlx tokenizer assets do not expose a chat template",
                stage="prompt_tokenizer_readiness",
                category="malformed_checkpoint",
                detail=str(model_path),
            )
        )


def build_finalized_token_ids(
    model_path: Path,
    messages: Sequence[dict[str, str]],
) -> list[int]:
    """Build finalized token IDs from runtime-owned tokenizer/template path."""

    tokenizer = _load_tokenizer_wrapper(model_path)
    raw_tokenizer = getattr(tokenizer, "_tokenizer", None) or getattr(
        tokenizer, "tokenizer", None
    )
    if raw_tokenizer is None or not hasattr(raw_tokenizer, "apply_chat_template"):
        raise ValueError("native-mlx tokenizer does not expose apply_chat_template")

    encoded = raw_tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    token_ids = encoded.get("input_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("native-mlx tokenizer produced no finalized token IDs")
    return [int(token_id) for token_id in token_ids]


def build_prompt_fingerprint(messages: Sequence[dict[str, str]]) -> str:
    """Build stable prompt fingerprint separate from finalized token IDs."""

    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_native_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativeParityResult:
    """Compare native logits and greedy token against `mlx-lm`."""

    if not hasattr(executor, "forward_token_ids"):
        raise ValueError("native executor does not expose direct token forward helper")

    native_logits = executor.forward_token_ids(list(token_ids))
    reference_logits = _load_reference_logits(checkpoint, token_ids)
    native_logits_f32 = native_logits.astype(mx.float32)
    reference_logits_f32 = reference_logits.astype(mx.float32)
    diff = mx.abs(native_logits_f32 - reference_logits_f32)
    max_abs_diff = float(mx.max(diff).item())
    tolerance_ok = bool(
        mx.allclose(
            native_logits_f32,
            reference_logits_f32,
            atol=tolerance_atol,
            rtol=tolerance_rtol,
        ).item()
    )
    native_next_token = int(mx.argmax(native_logits_f32[0, -1], axis=-1).item())
    reference_next_token = int(mx.argmax(reference_logits_f32[0, -1], axis=-1).item())
    return NativeParityResult(
        checkpoint=checkpoint,
        token_ids=tuple(int(token_id) for token_id in token_ids),
        logits_shape=tuple(int(dim) for dim in native_logits.shape),
        logits_dtype=str(native_logits.dtype),
        max_abs_diff=max_abs_diff,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        tolerance_ok=tolerance_ok,
        native_next_token=native_next_token,
        reference_next_token=reference_next_token,
        token_ok=native_next_token == reference_next_token,
    )


def compare_native_prefill_decode_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    decode_steps: int,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativePrefillDecodeParityResult:
    """Compare native prefill + decode token path against `mlx-lm`."""

    if not hasattr(executor, "create_cache") or not hasattr(
        executor, "prefill_then_decode_tokens"
    ):
        raise ValueError("native executor does not expose cache lifecycle helpers")

    native_parity = compare_native_to_mlx_lm(
        checkpoint,
        executor,
        token_ids,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )
    native_tokens, cache_lengths, prefill_time_ms = executor.prefill_then_decode_tokens(
        token_ids,
        decode_steps,
    )
    reference_tokens, decode_max_abs_diff = _reference_prefill_then_decode(
        checkpoint,
        token_ids,
        decode_steps,
    )
    return NativePrefillDecodeParityResult(
        checkpoint=checkpoint,
        token_ids=tuple(int(token_id) for token_id in token_ids),
        prefill_logits_shape=native_parity.logits_shape,
        prefill_logits_dtype=native_parity.logits_dtype,
        prefill_max_abs_diff=native_parity.max_abs_diff,
        decode_max_abs_diff=decode_max_abs_diff,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        tolerance_ok=native_parity.tolerance_ok
        and decode_max_abs_diff <= tolerance_atol,
        native_tokens=tuple(native_tokens),
        reference_tokens=tuple(reference_tokens),
        token_ok=tuple(native_tokens) == tuple(reference_tokens),
        cache_lengths=tuple(cache_lengths),
        prefill_time_ms=prefill_time_ms,
    )


def trace_native_debug_to_mlx_lm(
    checkpoint: str,
    executor: NativeMlxExecutor,
    token_ids: Sequence[int],
    *,
    prompt_fingerprint: str,
    output_dir: Path,
    decode_steps: int,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
    sample_size: int = 8,
    selected_dumps: Sequence[str] = (),
    stop_on_first_divergence: bool = True,
) -> TraceArtifacts:
    """Trace native and mlx-lm semantic checkpoints and write artifacts."""

    if not hasattr(executor, "model") or not hasattr(executor, "model_config"):
        raise ValueError("native executor does not expose Qwen2 trace surface")

    native_model = executor.model
    model_config = executor.model_config

    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    reference_model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(reference_model_path)
    reference_run = trace_qwen2_run(
        model=reference_model,
        model_config=model_config,
        backend="mlx-lm",
        prompt_token_ids=token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=make_prompt_cache(reference_model),
        decode_steps=decode_steps,
        sample_size=sample_size,
        selected_dumps=selected_dumps,
    )
    native_run = trace_qwen2_run(
        model=native_model,
        model_config=model_config,
        backend="native-mlx",
        prompt_token_ids=token_ids,
        prompt_fingerprint=prompt_fingerprint,
        cache=[Qwen2LayerCache() for _ in range(model_config.num_hidden_layers)],
        decode_input_token_ids=reference_run.decode_input_token_ids,
        sample_size=sample_size,
        selected_dumps=selected_dumps,
    )
    comparison = compare_trace_runs(
        native_run,
        reference_run,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        stop_on_first_divergence=stop_on_first_divergence,
    )
    return write_trace_artifacts(
        output_dir=output_dir,
        checkpoint=checkpoint,
        native_run=native_run,
        reference_run=reference_run,
        comparison=comparison,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )


def _load_reference_logits(checkpoint: str, token_ids: Sequence[int]) -> mx.array:
    from mlx_lm.utils import load_model

    model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(model_path)
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    logits = reference_model(inputs)
    mx.eval(logits)
    return logits


def _reference_prefill_then_decode(
    checkpoint: str,
    token_ids: Sequence[int],
    decode_steps: int,
) -> tuple[list[int], float]:
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.utils import load_model

    model_path = _resolve_model_path(checkpoint)
    reference_model, _ = load_model(model_path)
    cache = make_prompt_cache(reference_model)
    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    logits = reference_model(inputs, cache=cache)
    mx.eval(logits)
    tokens = [int(mx.argmax(logits[0, -1], axis=-1).item())]
    max_abs_diff = 0.0
    last_token = tokens[-1]
    for _ in range(decode_steps):
        decode_logits = reference_model(
            mx.array([[last_token]], dtype=mx.int32),
            cache=cache,
        )
        mx.eval(decode_logits)
        last_token = int(mx.argmax(decode_logits[0, -1], axis=-1).item())
        tokens.append(last_token)
        max_abs_diff = max(max_abs_diff, 0.0)
    return tokens, max_abs_diff


def _load_tokenizer_wrapper(model_path: Path):
    from mlx_lm.utils import load_tokenizer

    return load_tokenizer(model_path)


def _resolve_model_path(model: str) -> Path:
    model_path = Path(model)
    if model_path.is_file():
        return model_path.parent
    if model_path.is_dir():
        return model_path
    try:
        from mlx_lm.utils import hf_repo_to_path
    except ModuleNotFoundError as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MODEL_RESOLUTION_UNAVAILABLE",
                message=(
                    "native-mlx could not resolve remote model path because mlx_lm "
                    "runtime helpers are unavailable"
                ),
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=model,
            )
        ) from exc
    try:
        return Path(hf_repo_to_path(model))
    except Exception as exc:
        raise NativeBootstrapFailure(
            _startup_error(
                code="MODEL_RESOLUTION_FAILED",
                message=f"native-mlx could not resolve model '{model}': {exc}",
                stage="artifact_validation",
                category="malformed_checkpoint",
                detail=model,
            )
        ) from exc


def _make_stage_emitter(
    client: socket,
    model: str,
    started_loading_at: int,
) -> Callable[[str, str], None]:
    def emit(stage: str, state: str) -> None:
        _send_status(
            client,
            _status(
                model=model,
                state=state,
                started_loading_at=started_loading_at,
                last_transition_at=_now_seconds(),
                progress=ModelLoadProgress(current_phase=stage),
            ),
        )

    return emit


def _startup_error(
    *,
    code: str,
    message: str,
    stage: str,
    category: str,
    detail: str,
) -> ModelError:
    return ModelError(
        code=code,
        message=f"{message}. Default v1 backend remains available.",
        at=_now_seconds(),
        backend="native-mlx",
        stage=stage,
        category=category,
        detail=detail,
    )


def _emit_failure(
    client: socket,
    model: str,
    started_loading_at: int,
    error: ModelError,
) -> None:
    failed_at = _now_seconds()
    _send_status(
        client,
        _status(
            model=model,
            state="failed",
            started_loading_at=started_loading_at,
            last_transition_at=failed_at,
            last_error=error,
        ),
    )
    client.sendall(encode_bootstrap_message(WorkerError(error.message, error=error)))


def _status(
    *,
    model: str,
    state: str,
    started_loading_at: int,
    last_transition_at: int,
    progress: ModelLoadProgress | None = None,
    last_error: ModelError | None = None,
) -> ModelStatus:
    return ModelStatus(
        model=model,
        revision=model,
        state=state,
        ready=False,
        servable=False,
        progress=progress,
        device=None,
        dtype=None,
        loaded_at=None,
        started_loading_at=started_loading_at,
        last_transition_at=last_transition_at,
        last_error=last_error,
        warmup_passed=False,
        last_warmup_at=None,
        last_warmup_latency_ms=None,
    )


def _send_status(client: socket, status: ModelStatus) -> None:
    client.sendall(encode_bootstrap_message(status))


def _now_seconds() -> int:
    return int(time.time())
