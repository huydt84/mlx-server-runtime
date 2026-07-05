"""Native MLX parity and trace orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx

from .cache import DenseLayerCache
from .interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    ForwardBatch,
    NativeMlxDiagnostics,
    NativeMlxExecutor,
    NativeModel,
    SamplingParams,
)


@dataclass(frozen=True)
class ModelDiagnostics:
    """Explicit diagnostics over injected model and executor interfaces."""

    model: NativeModel
    executor: NativeMlxExecutor
    model_config: object

    def forward_token_ids(self, token_ids: Sequence[int]) -> mx.array:
        inputs = mx.array([list(token_ids)], dtype=mx.int32)
        positions = mx.array([list(range(len(token_ids)))], dtype=mx.int32)
        batch = ForwardBatch(
            token_lengths=(len(token_ids),),
            cache_lengths=(0,),
            attention_mask="causal",
            layer_caches=(),
        )
        logits = self.model(inputs, positions, batch)
        mx.eval(logits)
        return logits

    def prefill_then_decode_tokens(
        self,
        prompt_token_ids: Sequence[int],
        decode_steps: int,
    ) -> tuple[list[int], list[int], int]:
        request_id = "diagnostic"
        handle = self.executor.create_cache(request_id)
        try:
            started = time.perf_counter()
            prefill = self.executor.prefill_batch(
                ExecutionBatch(
                    phase="prefill",
                    requests=(
                        ExecutionRequest(
                            request_id=request_id,
                            token_ids=tuple(int(token) for token in prompt_token_ids),
                            positions=tuple(range(len(prompt_token_ids))),
                            cache_handle=handle,
                            sampling=SamplingParams(),
                        ),
                    ),
                )
            )
            tokens = [int(prefill.results[0].next_token_id)]
            lengths = [prefill.results[0].cache_length]
            prefill_ms = max(1, int((time.perf_counter() - started) * 1000))
            for _ in range(decode_steps):
                result = self.executor.decode_batch(
                    ExecutionBatch(
                        phase="decode",
                        requests=(
                            ExecutionRequest(
                                request_id=request_id,
                                token_ids=(tokens[-1],),
                                positions=(self.executor.cache_len(handle),),
                                cache_handle=handle,
                                sampling=SamplingParams(),
                            ),
                        ),
                    )
                )
                tokens.append(int(result.results[0].next_token_id))
                lengths.append(result.results[0].cache_length)
            return tokens, lengths, prefill_ms
        finally:
            self.executor.release(handle)

    def trace_to_mlx_lm(
        self,
        checkpoint: str,
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
    ):
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.utils import load_model

        from .trace import compare_trace_runs, trace_model_run, write_trace_artifacts

        reference_model, _ = load_model(_reference_model_path(checkpoint))
        reference = trace_model_run(
            model=reference_model,
            model_config=self.model_config,
            backend="mlx-lm",
            prompt_token_ids=token_ids,
            prompt_fingerprint=prompt_fingerprint,
            cache=make_prompt_cache(reference_model),
            decode_steps=decode_steps,
            sample_size=sample_size,
            selected_dumps=selected_dumps,
        )
        native = trace_model_run(
            model=self.model,
            model_config=self.model_config,
            backend="native-mlx",
            prompt_token_ids=token_ids,
            prompt_fingerprint=prompt_fingerprint,
            cache=[
                DenseLayerCache()
                for _ in range(int(getattr(self.model_config, "num_hidden_layers")))
            ],
            decode_input_token_ids=reference.decode_input_token_ids,
            sample_size=sample_size,
            selected_dumps=selected_dumps,
        )
        comparison = compare_trace_runs(
            native,
            reference,
            tolerance_atol=tolerance_atol,
            tolerance_rtol=tolerance_rtol,
            stop_on_first_divergence=stop_on_first_divergence,
        )
        return write_trace_artifacts(
            output_dir=output_dir,
            checkpoint=checkpoint,
            native_run=native,
            reference_run=reference,
            comparison=comparison,
            tolerance_atol=tolerance_atol,
            tolerance_rtol=tolerance_rtol,
        )


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


def build_prompt_fingerprint(messages: Sequence[dict[str, str]]) -> str:
    """Build stable prompt fingerprint separate from finalized token IDs."""

    payload = json.dumps(list(messages), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_native_to_mlx_lm(
    checkpoint: str,
    diagnostics: NativeMlxDiagnostics,
    token_ids: Sequence[int],
    *,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativeParityResult:
    """Compare native logits and greedy token against `mlx-lm`."""

    native_logits = diagnostics.forward_token_ids(token_ids)
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
    diagnostics: NativeMlxDiagnostics,
    token_ids: Sequence[int],
    *,
    decode_steps: int,
    tolerance_atol: float = 2e-2,
    tolerance_rtol: float = 2e-2,
) -> NativePrefillDecodeParityResult:
    """Compare native prefill + decode token path against `mlx-lm`."""

    native_parity = compare_native_to_mlx_lm(
        checkpoint,
        diagnostics,
        token_ids,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
    )
    native_tokens, cache_lengths, prefill_time_ms = (
        diagnostics.prefill_then_decode_tokens(
            token_ids,
            decode_steps,
        )
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
    diagnostics: NativeMlxDiagnostics,
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
):
    """Trace native and mlx-lm semantic checkpoints through diagnostics seam."""

    return diagnostics.trace_to_mlx_lm(
        checkpoint,
        token_ids,
        prompt_fingerprint=prompt_fingerprint,
        output_dir=output_dir,
        decode_steps=decode_steps,
        tolerance_atol=tolerance_atol,
        tolerance_rtol=tolerance_rtol,
        sample_size=sample_size,
        selected_dumps=selected_dumps,
        stop_on_first_divergence=stop_on_first_divergence,
    )


def _load_reference_logits(checkpoint: str, token_ids: Sequence[int]) -> mx.array:
    from mlx_lm.utils import load_model

    reference_model, _ = load_model(_reference_model_path(checkpoint))
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

    reference_model, _ = load_model(_reference_model_path(checkpoint))
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


def _reference_model_path(checkpoint: str) -> Path:
    path = Path(checkpoint)
    if path.exists():
        return path
    from mlx_lm.utils import hf_repo_to_path

    return Path(hf_repo_to_path(checkpoint))
