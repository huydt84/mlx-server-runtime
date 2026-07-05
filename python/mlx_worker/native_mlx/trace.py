"""Semantic model tracing primitives for native MLX diagnostics."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx
import numpy as np

from .cache import DenseLayerCache

REQUIRED_TRACE_OPERATIONS = (
    "embedding",
    "attention",
    "mlp",
    "residual",
    "final_norm",
    "logits",
    "kv_append",
)


@dataclass(frozen=True)
class TraceRecord:
    """JSONL-safe semantic checkpoint summary."""

    backend: str
    architecture_class: str
    phase: str
    step_index: int
    checkpoint_id: str
    module: str
    operation: str
    tensor_name: str
    layer_index: int | None
    dtype: str
    shape: tuple[int, ...]
    finite_count: int
    nan_count: int
    inf_count: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None
    stddev_value: float | None
    stable_hash: str
    sample_values: tuple[float | int, ...]
    cache_length: int | None = None
    full_values: Any | None = None


@dataclass(frozen=True)
class TraceCheckpoint:
    """In-memory checkpoint with comparable tensor values."""

    record: TraceRecord
    values: np.ndarray


@dataclass(frozen=True)
class ModelTraceRun:
    """Trace output for one backend run."""

    backend: str
    architecture_class: str
    prompt_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    decode_input_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    prefill: tuple[TraceCheckpoint, ...]
    decode: tuple[TraceCheckpoint, ...]


@dataclass(frozen=True)
class TraceMismatch:
    """First or later divergence between two checkpoint streams."""

    kind: str
    checkpoint_id: str
    phase: str
    detail: str
    max_abs_diff: float | None = None
    max_rel_diff: float | None = None


@dataclass(frozen=True)
class TraceComparison:
    """Comparison result across native and reference traces."""

    aligned: bool
    first_mismatch: TraceMismatch | None
    mismatches: tuple[TraceMismatch, ...]


@dataclass(frozen=True)
class TraceArtifacts:
    """Written trace artifact paths and comparison metadata."""

    checkpoint: str
    architecture_class: str
    prompt_fingerprint: str
    prompt_token_ids: tuple[int, ...]
    decode_input_token_ids: tuple[int, ...]
    native_generated_token_ids: tuple[int, ...]
    reference_generated_token_ids: tuple[int, ...]
    tolerance_atol: float
    tolerance_rtol: float
    prefill_jsonl_path: Path
    decode_jsonl_path: Path
    summary_markdown_path: Path
    comparison: TraceComparison


def trace_model_run(
    *,
    model: Any,
    model_config: Any,
    backend: str,
    prompt_token_ids: Sequence[int],
    prompt_fingerprint: str,
    cache: list[Any],
    decode_steps: int = 0,
    decode_input_token_ids: Sequence[int] | None = None,
    sample_size: int = 8,
    selected_dumps: Sequence[str] = (),
) -> ModelTraceRun:
    """Trace semantic decoder-model checkpoints for one backend.

    Args:
        model: Native or reference decoder model object.
        model_config: Parsed architecture config.
        backend: Backend label written to records.
        prompt_token_ids: Finalized prompt token IDs.
        prompt_fingerprint: Stable prompt fingerprint from runtime-owned input.
        cache: Mutable per-layer KV cache list.
        decode_steps: Number of greedy decode steps when decode inputs are omitted.
        decode_input_token_ids: Forced decode input tokens for aligned backend runs.
        sample_size: Maximum flattened sample values per checkpoint.
        selected_dumps: Checkpoint IDs that should carry full tensor dumps.

    Returns:
        ModelTraceRun: Bounded checkpoint stream and generated tokens.
    """

    if cache and len(cache) != model_config.num_hidden_layers:
        raise ValueError("trace cache layer count does not match model config")

    collector = _TraceCollector(
        backend=backend,
        architecture_class=model_config.architecture_class,
        sample_size=sample_size,
        selected_dumps=frozenset(selected_dumps),
    )
    prefill_logits, prefill_records = trace_model_step(
        model=model,
        model_config=model_config,
        token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
        cache=cache,
        backend=backend,
        phase="prefill",
        step_index=0,
        collector=collector,
    )
    generated_token_ids = [int(np.asarray(prefill_logits)[0, -1].argmax())]

    if decode_input_token_ids is None:
        forced_decode_inputs: list[int] = []
        next_decode_input = generated_token_ids[0]
        for step_index in range(decode_steps):
            forced_decode_inputs.append(next_decode_input)
            decode_logits, decode_records = trace_model_step(
                model=model,
                model_config=model_config,
                token_ids=(next_decode_input,),
                cache=cache,
                backend=backend,
                phase="decode",
                step_index=step_index,
                collector=collector,
            )
            collector.decode_records.extend(decode_records)
            next_decode_input = int(np.asarray(decode_logits)[0, -1].argmax())
            generated_token_ids.append(next_decode_input)
        decode_input_token_ids = tuple(forced_decode_inputs)
    else:
        decode_input_token_ids = tuple(
            int(token_id) for token_id in decode_input_token_ids
        )
        if decode_steps and len(decode_input_token_ids) != decode_steps:
            raise ValueError(
                "decode_steps does not match forced decode_input_token_ids"
            )
        for step_index, decode_input_token_id in enumerate(decode_input_token_ids):
            decode_logits, decode_records = trace_model_step(
                model=model,
                model_config=model_config,
                token_ids=(decode_input_token_id,),
                cache=cache,
                backend=backend,
                phase="decode",
                step_index=step_index,
                collector=collector,
            )
            collector.decode_records.extend(decode_records)
            generated_token_ids.append(int(np.asarray(decode_logits)[0, -1].argmax()))

    run = ModelTraceRun(
        backend=backend,
        architecture_class=model_config.architecture_class,
        prompt_fingerprint=prompt_fingerprint,
        prompt_token_ids=tuple(int(token_id) for token_id in prompt_token_ids),
        decode_input_token_ids=tuple(
            int(token_id) for token_id in decode_input_token_ids
        ),
        generated_token_ids=tuple(generated_token_ids),
        prefill=tuple(prefill_records),
        decode=tuple(collector.decode_records),
    )
    _validate_trace_coverage(run, model_config.num_hidden_layers)
    return run


def trace_model_step(
    *,
    model: Any,
    model_config: Any,
    token_ids: Sequence[int],
    cache: list[Any],
    backend: str,
    phase: str,
    step_index: int,
    collector: "_TraceCollector",
) -> tuple[np.ndarray, list[TraceCheckpoint]]:
    """Trace one prefill or decode semantic step."""

    inputs = mx.array([list(token_ids)], dtype=mx.int32)
    hidden = model.model.embed_tokens(inputs)
    mask = _attention_mask_for_trace(hidden, cache)
    records = [
        collector.record(
            phase=phase,
            step_index=step_index,
            module="model.embed_tokens",
            operation="embedding",
            tensor_name="hidden_states",
            tensor=hidden,
        )
    ]

    for layer_index, (layer, layer_cache) in enumerate(zip(model.model.layers, cache)):
        attn_input = layer.input_layernorm(hidden)
        cache_offset = _cache_length(layer_cache)
        positions = mx.array(
            [list(range(cache_offset, cache_offset + len(token_ids)))],
            dtype=mx.int32,
        )
        attn_output = _call_attention_for_trace(
            layer.self_attn,
            attn_input,
            positions=positions,
            mask=mask,
            cache=layer_cache,
        )
        records.append(
            collector.record(
                phase=phase,
                step_index=step_index,
                module=f"model.layers.{layer_index}.self_attn",
                operation="attention",
                tensor_name="hidden_states",
                tensor=attn_output,
                layer_index=layer_index,
            )
        )
        records.extend(
            _record_kv_append(
                collector=collector,
                phase=phase,
                step_index=step_index,
                layer_index=layer_index,
                layer_cache=layer_cache,
            )
        )
        residual = hidden + attn_output
        records.append(
            collector.record(
                phase=phase,
                step_index=step_index,
                module=f"model.layers.{layer_index}",
                operation="residual",
                tensor_name="hidden_states",
                tensor=residual,
                layer_index=layer_index,
            )
        )
        mlp_input = layer.post_attention_layernorm(residual)
        mlp_output = layer.mlp(mlp_input)
        records.append(
            collector.record(
                phase=phase,
                step_index=step_index,
                module=f"model.layers.{layer_index}.mlp",
                operation="mlp",
                tensor_name="hidden_states",
                tensor=mlp_output,
                layer_index=layer_index,
            )
        )
        hidden = residual + mlp_output

    hidden = model.model.norm(hidden)
    records.append(
        collector.record(
            phase=phase,
            step_index=step_index,
            module="model.norm",
            operation="final_norm",
            tensor_name="hidden_states",
            tensor=hidden,
        )
    )
    if model_config.tie_word_embeddings:
        logits = model.model.embed_tokens.as_linear(hidden)
        logits_module = "model.embed_tokens"
    else:
        logits = model.lm_head(hidden)
        logits_module = "lm_head"
    records.append(
        collector.record(
            phase=phase,
            step_index=step_index,
            module=logits_module,
            operation="logits",
            tensor_name="logits",
            tensor=logits,
        )
    )

    mx.eval(logits)
    return np.asarray(logits), records


def compare_trace_runs(
    native_run: ModelTraceRun,
    reference_run: ModelTraceRun,
    *,
    tolerance_atol: float,
    tolerance_rtol: float,
    stop_on_first_divergence: bool = True,
) -> TraceComparison:
    """Compare two semantic trace streams in checkpoint order."""

    native_records = list(native_run.prefill + native_run.decode)
    reference_records = list(reference_run.prefill + reference_run.decode)
    mismatches: list[TraceMismatch] = []
    native_by_id = {
        checkpoint.record.checkpoint_id: checkpoint for checkpoint in native_records
    }
    reference_by_id = {
        checkpoint.record.checkpoint_id: checkpoint for checkpoint in reference_records
    }

    ordered_checkpoint_ids = [
        checkpoint.record.checkpoint_id for checkpoint in reference_records
    ]
    ordered_checkpoint_ids.extend(
        checkpoint.record.checkpoint_id
        for checkpoint in native_records
        if checkpoint.record.checkpoint_id not in reference_by_id
    )

    for checkpoint_id in ordered_checkpoint_ids:
        native_checkpoint = native_by_id.get(checkpoint_id)
        reference_checkpoint = reference_by_id.get(checkpoint_id)
        if native_checkpoint is None or reference_checkpoint is None:
            present_backend = (
                "native-mlx" if native_checkpoint is not None else "mlx-lm"
            )
            mismatches.append(
                TraceMismatch(
                    kind="missing_checkpoint",
                    checkpoint_id=checkpoint_id,
                    phase=(native_checkpoint or reference_checkpoint).record.phase,
                    detail=f"checkpoint present only in {present_backend}",
                )
            )
        else:
            record_mismatch = _compare_checkpoint_values(
                native_checkpoint,
                reference_checkpoint,
                tolerance_atol=tolerance_atol,
                tolerance_rtol=tolerance_rtol,
            )
            if record_mismatch is not None:
                mismatches.append(record_mismatch)
        if mismatches and stop_on_first_divergence:
            break

    if (
        not mismatches
        and native_run.generated_token_ids != reference_run.generated_token_ids
    ):
        mismatches.append(
            TraceMismatch(
                kind="token_mismatch",
                checkpoint_id="generated_token_ids",
                phase="decode" if reference_run.decode_input_token_ids else "prefill",
                detail=(
                    f"native generated {native_run.generated_token_ids} but mlx-lm generated "
                    f"{reference_run.generated_token_ids}"
                ),
            )
        )

    return TraceComparison(
        aligned=not mismatches,
        first_mismatch=mismatches[0] if mismatches else None,
        mismatches=tuple(mismatches),
    )


def write_trace_artifacts(
    *,
    output_dir: Path,
    checkpoint: str,
    native_run: ModelTraceRun,
    reference_run: ModelTraceRun,
    comparison: TraceComparison,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> TraceArtifacts:
    """Write bounded JSONL checkpoints and markdown summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefill_jsonl_path = output_dir / "prefill.jsonl"
    decode_jsonl_path = output_dir / "decode.jsonl"
    summary_markdown_path = output_dir / "first_divergence.md"

    _write_jsonl(prefill_jsonl_path, native_run.prefill + reference_run.prefill)
    _write_jsonl(decode_jsonl_path, native_run.decode + reference_run.decode)
    _write_markdown_summary(
        summary_markdown_path,
        checkpoint=checkpoint,
        native_run=native_run,
        reference_run=reference_run,
        comparison=comparison,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )

    return TraceArtifacts(
        checkpoint=checkpoint,
        architecture_class=native_run.architecture_class,
        prompt_fingerprint=native_run.prompt_fingerprint,
        prompt_token_ids=native_run.prompt_token_ids,
        decode_input_token_ids=reference_run.decode_input_token_ids,
        native_generated_token_ids=native_run.generated_token_ids,
        reference_generated_token_ids=reference_run.generated_token_ids,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        prefill_jsonl_path=prefill_jsonl_path,
        decode_jsonl_path=decode_jsonl_path,
        summary_markdown_path=summary_markdown_path,
        comparison=comparison,
    )


class _TraceCollector:
    def __init__(
        self,
        *,
        backend: str,
        architecture_class: str,
        sample_size: int,
        selected_dumps: frozenset[str],
    ) -> None:
        self.backend = backend
        self.architecture_class = architecture_class
        self.sample_size = sample_size
        self.selected_dumps = selected_dumps
        self.decode_records: list[TraceCheckpoint] = []

    def record(
        self,
        *,
        phase: str,
        step_index: int,
        module: str,
        operation: str,
        tensor_name: str,
        tensor: Any,
        layer_index: int | None = None,
        cache_length: int | None = None,
    ) -> TraceCheckpoint:
        mx.eval(tensor)
        values = np.asarray(tensor)
        checkpoint_id = _checkpoint_id(
            phase=phase,
            step_index=step_index,
            layer_index=layer_index,
            operation=operation,
            tensor_name=tensor_name,
        )
        record = TraceRecord(
            backend=self.backend,
            architecture_class=self.architecture_class,
            phase=phase,
            step_index=step_index,
            checkpoint_id=checkpoint_id,
            module=module,
            operation=operation,
            tensor_name=tensor_name,
            layer_index=layer_index,
            dtype=str(values.dtype),
            shape=tuple(int(dim) for dim in values.shape),
            finite_count=_finite_count(values),
            nan_count=_nan_count(values),
            inf_count=_inf_count(values),
            min_value=_summary_stat(values, np.nanmin),
            max_value=_summary_stat(values, np.nanmax),
            mean_value=_summary_stat(values, np.nanmean),
            stddev_value=_summary_stat(values, np.nanstd),
            stable_hash=_stable_hash(values),
            sample_values=_sample_values(values, self.sample_size),
            cache_length=cache_length,
            full_values=values.tolist()
            if checkpoint_id in self.selected_dumps
            else None,
        )
        return TraceCheckpoint(record=record, values=values.copy())


def _record_kv_append(
    *,
    collector: _TraceCollector,
    phase: str,
    step_index: int,
    layer_index: int,
    layer_cache: Any,
) -> list[TraceCheckpoint]:
    if layer_cache is None:
        raise ValueError("kv_append checkpoint requires layer cache")
    keys, values, cache_length = _normalized_cache_state(layer_cache)
    return [
        collector.record(
            phase=phase,
            step_index=step_index,
            module=f"model.layers.{layer_index}.self_attn.cache",
            operation="kv_append",
            tensor_name="keys",
            tensor=keys,
            layer_index=layer_index,
            cache_length=cache_length,
        ),
        collector.record(
            phase=phase,
            step_index=step_index,
            module=f"model.layers.{layer_index}.self_attn.cache",
            operation="kv_append",
            tensor_name="values",
            tensor=values,
            layer_index=layer_index,
            cache_length=cache_length,
        ),
    ]


def _attention_mask_for_trace(hidden: Any, cache: list[Any]) -> Any | None:
    if not cache:
        return None
    first_cache = cache[0]
    if first_cache is None or not hasattr(first_cache, "make_mask"):
        return None
    from mlx_lm.models.base import create_attention_mask

    return create_attention_mask(hidden, first_cache)


def _call_attention_for_trace(
    attention: Any,
    hidden: Any,
    *,
    positions: Any,
    mask: Any | None,
    cache: Any,
) -> Any:
    parameters = inspect.signature(attention.__call__).parameters
    if "mask" in parameters:
        return attention(hidden, mask=mask, cache=cache)
    if "positions" in parameters:
        return attention(hidden, positions=positions, cache=cache)
    return attention(hidden, cache=cache)


def _normalized_cache_state(layer_cache: Any) -> tuple[Any, Any, int]:
    keys = getattr(layer_cache, "keys", None)
    values = getattr(layer_cache, "values", None)
    if keys is None or values is None:
        raise ValueError("layer cache missing KV tensors for trace")
    cache_length = int(getattr(layer_cache, "offset", 0) or 0)
    if not cache_length and hasattr(layer_cache, "size"):
        size_value = layer_cache.size
        cache_length = int(size_value() if callable(size_value) else size_value)
    if cache_length <= 0:
        raise ValueError("layer cache offset must be positive after KV append")
    return keys[:, :, :cache_length, :], values[:, :, :cache_length, :], cache_length


def _cache_length(layer_cache: Any) -> int:
    offset = getattr(layer_cache, "offset", 0) or 0
    if offset:
        return int(offset)
    size = getattr(layer_cache, "size", 0)
    return int(size() if callable(size) else size or 0)


def _validate_trace_coverage(run: ModelTraceRun, num_layers: int) -> None:
    for phase_name, checkpoints in (("prefill", run.prefill), ("decode", run.decode)):
        if phase_name == "decode" and not checkpoints:
            continue
        operations = {
            (record.record.operation, record.record.layer_index)
            for record in checkpoints
        }
        if ("embedding", None) not in operations:
            raise ValueError(
                f"traceability bug: missing {phase_name} embedding checkpoint"
            )
        if ("final_norm", None) not in operations:
            raise ValueError(
                f"traceability bug: missing {phase_name} final_norm checkpoint"
            )
        if ("logits", None) not in operations:
            raise ValueError(
                f"traceability bug: missing {phase_name} logits checkpoint"
            )
        for layer_index in range(num_layers):
            for operation in ("attention", "mlp", "residual", "kv_append"):
                if (operation, layer_index) not in operations:
                    raise ValueError(
                        "traceability bug: missing "
                        f"{phase_name} layer {layer_index} {operation} checkpoint"
                    )


def _compare_checkpoint_values(
    native_checkpoint: TraceCheckpoint,
    reference_checkpoint: TraceCheckpoint,
    *,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> TraceMismatch | None:
    native_record = native_checkpoint.record
    reference_record = reference_checkpoint.record
    if (
        native_record.shape != reference_record.shape
        or native_record.dtype != reference_record.dtype
    ):
        return TraceMismatch(
            kind="metadata_mismatch",
            checkpoint_id=native_record.checkpoint_id,
            phase=native_record.phase,
            detail=(
                f"native dtype/shape {native_record.dtype}{native_record.shape} != "
                f"mlx-lm {reference_record.dtype}{reference_record.shape}"
            ),
        )

    native_values = native_checkpoint.values.astype(np.float64, copy=False)
    reference_values = reference_checkpoint.values.astype(np.float64, copy=False)
    if np.allclose(
        native_values,
        reference_values,
        atol=tolerance_atol,
        rtol=tolerance_rtol,
        equal_nan=True,
    ):
        return None

    diff = np.abs(native_values - reference_values)
    denom = np.maximum(np.abs(reference_values), 1e-12)
    rel_diff = diff / denom
    return TraceMismatch(
        kind="numeric_mismatch",
        checkpoint_id=native_record.checkpoint_id,
        phase=native_record.phase,
        detail=(
            f"numeric mismatch at {native_record.checkpoint_id} "
            f"(native_hash={native_record.stable_hash}, mlx_lm_hash={reference_record.stable_hash})"
        ),
        max_abs_diff=float(np.max(diff)),
        max_rel_diff=float(np.max(rel_diff)),
    )


def _write_jsonl(path: Path, checkpoints: Sequence[TraceCheckpoint]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for checkpoint in checkpoints:
            handle.write(json.dumps(asdict(checkpoint.record), sort_keys=True))
            handle.write("\n")


def _write_markdown_summary(
    path: Path,
    *,
    checkpoint: str,
    native_run: ModelTraceRun,
    reference_run: ModelTraceRun,
    comparison: TraceComparison,
    tolerance_atol: float,
    tolerance_rtol: float,
) -> None:
    mismatch = comparison.first_mismatch
    lines = [
        "# First Divergence Summary",
        "",
        f"- checkpoint: `{checkpoint}`",
        f"- architecture_class: `{native_run.architecture_class}`",
        f"- prompt_fingerprint: `{native_run.prompt_fingerprint}`",
        f"- prompt_token_ids: `{list(native_run.prompt_token_ids)}`",
        f"- decode_input_token_ids: `{list(reference_run.decode_input_token_ids)}`",
        f"- native_generated_token_ids: `{list(native_run.generated_token_ids)}`",
        f"- reference_generated_token_ids: `{list(reference_run.generated_token_ids)}`",
        f"- tolerance: `atol={tolerance_atol}` `rtol={tolerance_rtol}`",
        f"- status: `{'aligned' if comparison.aligned else mismatch.kind}`",
    ]
    if mismatch is None:
        lines.append("- first_divergence: `none`")
        lines.append(
            "- detail: all required semantic checkpoints aligned within tolerance"
        )
    else:
        lines.append(f"- first_divergence: `{mismatch.checkpoint_id}`")
        lines.append(f"- detail: {mismatch.detail}")
        if mismatch.max_abs_diff is not None:
            lines.append(f"- max_abs_diff: `{mismatch.max_abs_diff:.6g}`")
        if mismatch.max_rel_diff is not None:
            lines.append(f"- max_rel_diff: `{mismatch.max_rel_diff:.6g}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _checkpoint_id(
    *,
    phase: str,
    step_index: int,
    layer_index: int | None,
    operation: str,
    tensor_name: str,
) -> str:
    layer_key = f"layer{layer_index}" if layer_index is not None else "global"
    return f"{phase}.step{step_index}.{layer_key}.{operation}.{tensor_name}"


def _stable_hash(values: np.ndarray) -> str:
    payload = hashlib.sha256()
    payload.update(str(values.dtype).encode("utf-8"))
    payload.update(str(tuple(values.shape)).encode("utf-8"))
    payload.update(values.tobytes(order="C"))
    return payload.hexdigest()


def _sample_values(values: np.ndarray, sample_size: int) -> tuple[float | int, ...]:
    flat = values.reshape(-1)
    sample: list[float | int] = []
    for value in flat[:sample_size]:
        scalar = value.item()
        sample.append(float(scalar) if isinstance(scalar, float) else int(scalar))
    return tuple(sample)


def _finite_count(values: np.ndarray) -> int:
    if not np.issubdtype(values.dtype, np.number):
        return int(values.size)
    if np.issubdtype(values.dtype, np.integer):
        return int(values.size)
    return int(np.isfinite(values).sum())


def _nan_count(values: np.ndarray) -> int:
    if not np.issubdtype(values.dtype, np.floating):
        return 0
    return int(np.isnan(values).sum())


def _inf_count(values: np.ndarray) -> int:
    if not np.issubdtype(values.dtype, np.floating):
        return 0
    return int(np.isinf(values).sum())


def _summary_stat(values: np.ndarray, fn: Any) -> float | None:
    if not np.issubdtype(values.dtype, np.number):
        return None
    numeric = values.astype(np.float64, copy=False)
    if numeric.size == 0:
        return None
    if np.issubdtype(values.dtype, np.floating):
        finite = numeric[np.isfinite(numeric)]
        if finite.size == 0:
            return None
        numeric = finite
    return float(fn(numeric))


__all__ = [
    "DenseLayerCache",
    "ModelTraceRun",
    "TraceArtifacts",
    "TraceComparison",
    "TraceMismatch",
    "TraceRecord",
    "compare_trace_runs",
    "trace_model_run",
    "trace_model_step",
    "write_trace_artifacts",
]
