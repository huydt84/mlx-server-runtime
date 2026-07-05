#!/usr/bin/env bash
#
# native-v2 Phase 4 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_4.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - `local-probe/LlamaForCausalLM` unsupported-class reference from earlier phases
#   - `local-probe/Qwen2ForCausalLM-missing-tokenizer` malformed-artifact reference from earlier phases
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_lm_import_ok=1`
#   - `checkpoint=mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - `token_ids=` with finalized prompt token IDs
#   - `native_prefill_physical_batch_size=2`
#   - `native_prefill_model_forward_calls=1`
#   - `native_decode_physical_batch_size=2`
#   - `native_decode_model_forward_calls=1`
#   - `native_batched_single_parity_ok=1`
#   - `native_unequal_cache_lengths_ok=1`
#   - `native_cache_isolation_ok=1`
#   - `prefill_logits_shape=`
#   - `prefill_time_ms=` and `decode_time_ms=`
#   - `cache_lengths=` showing growth across prefill/decode
#   - `tolerance_ok=1`
#   - `token_ok=1`
#   - `release_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - printed native construction/mapping/forward/cache error
#   - missing `tolerance_ok=1`, `token_ok=1`, or `release_ok=1`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
CHECKPOINT="mlx-community/Qwen2.5-7B-Instruct-4bit"

echo "[1/3] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/3] Verify Apple Silicon, mlx, and mlx_lm imports"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

import mlx.core as mx
print("mlx_import_ok=1")

from mlx_lm import load  # noqa: F401
print("mlx_lm_import_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/3] Run real native prefill/decode parity, physical batching, and cache lifecycle"
uv run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_worker.native_mlx.bootstrap import (
    build_native_artifacts,
    build_finalized_token_ids,
)
from mlx_worker.native_mlx.diagnostics import (
    compare_native_prefill_decode_to_mlx_lm,
)
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    SamplingParams,
)

checkpoint = sys.argv[1]
artifacts = build_native_artifacts(checkpoint)
executor = artifacts.executor
model_path = Path(artifacts.architecture.model_path)
token_ids = build_finalized_token_ids(
    model_path,
    [{"role": "user", "content": "ping"}],
)
parity = compare_native_prefill_decode_to_mlx_lm(
    checkpoint,
    artifacts.diagnostics,
    token_ids,
    decode_steps=2,
)


class RecordingModel:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, inputs, *args, **kwargs):
        self.calls += 1
        self.batch_sizes.append(int(inputs.shape[0]))
        return self.inner(inputs, *args, **kwargs)


def prompt_ids(min_tokens: int) -> tuple[int, ...]:
    words: list[str] = []
    while True:
        start = len(words)
        words.extend(f"phase4_batch_{index:04d}" for index in range(start, start + 32))
        candidate = tuple(
            build_finalized_token_ids(
                model_path,
                [{"role": "user", "content": " ".join(words)}],
            )
        )
        if len(candidate) >= min_tokens:
            return candidate[:min_tokens]


prompt_a = prompt_ids(5)
prompt_b = prompt_ids(2)


def prefill_request(request_id: str, tokens: tuple[int, ...], handle: str) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        token_ids=tokens,
        positions=tuple(range(len(tokens))),
        cache_handle=handle,
        sampling=SamplingParams(),
    )


def decode_request(request_id: str, token_id: int, position: int, handle: str) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        token_ids=(token_id,),
        positions=(position,),
        cache_handle=handle,
        sampling=SamplingParams(),
    )


def result_pair_ok(batched, single, atol: float) -> bool:
    return (
        batched.next_token_id == single.next_token_id
        and batched.cache_length == single.cache_length
    )

handle_a = executor.create_cache("req-a")
handle_b = executor.create_cache("req-b")
ind_a = executor.create_cache("ind-a")
ind_b = executor.create_cache("ind-b")
iso_a = executor.create_cache("iso-a")
iso_shared_b = executor.create_cache("iso-shared-b")
iso_b = executor.create_cache("iso-b")
try:
    recorder = RecordingModel(executor.model)
    executor.model = recorder
    prefill_started = time.perf_counter()
    prefill = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                prefill_request("req-a", prompt_a, handle_a),
                prefill_request("req-b", prompt_b, handle_b),
            ),
        )
    )
    prefill_calls = recorder.calls
    prefill_batch_size = max(recorder.batch_sizes)

    independent_prefill_a = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                prefill_request("ind-a", prompt_a, ind_a),
            ),
        )
    )
    independent_prefill_b = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                prefill_request("ind-b", prompt_b, ind_b),
            ),
        )
    )
    isolated_prefill_b = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                prefill_request("iso-b", prompt_b, iso_b),
            ),
        )
    )
    shared_isolation_prefill = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                prefill_request("iso-a", prompt_a, iso_a),
                prefill_request("iso-shared-b", prompt_b, iso_shared_b),
            ),
        )
    )
    prefill_elapsed_ms = max(1, int((time.perf_counter() - prefill_started) * 1000))

    decode_input_a = int(prefill.results[0].next_token_id)
    decode_input_b = int(prefill.results[1].next_token_id)
    executor.model = RecordingModel(recorder.inner)
    decode_started = time.perf_counter()
    decode = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                decode_request("req-a", decode_input_a, executor.cache_len(handle_a), handle_a),
                decode_request("req-b", decode_input_b, executor.cache_len(handle_b), handle_b),
            ),
        )
    )
    decode_calls = executor.model.calls
    decode_batch_size = max(executor.model.batch_sizes)
    decode_elapsed_ms = max(1, int((time.perf_counter() - decode_started) * 1000))

    independent_decode_a = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                decode_request(
                    "ind-a",
                    int(independent_prefill_a.results[0].next_token_id),
                    executor.cache_len(ind_a),
                    ind_a,
                ),
            ),
        )
    )
    independent_decode_b = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                decode_request(
                    "ind-b",
                    int(independent_prefill_b.results[0].next_token_id),
                    executor.cache_len(ind_b),
                    ind_b,
                ),
            ),
        )
    )

    parity_ok = all(
        result_pair_ok(batched, single, atol=0.5)
        for batched, single in zip(
            prefill.results,
            (independent_prefill_a.results[0], independent_prefill_b.results[0]),
            strict=True,
        )
    ) and all(
        result_pair_ok(batched, single, atol=0.5)
        for batched, single in zip(
            decode.results,
            (independent_decode_a.results[0], independent_decode_b.results[0]),
            strict=True,
        )
    )

    unequal_cache_lengths_ok = (
        prefill.results[0].cache_length != prefill.results[1].cache_length
        and decode.results[0].cache_length != decode.results[1].cache_length
    )

    executor.release(iso_a)
    isolated_decode_b = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                decode_request(
                    "iso-shared-b",
                    int(shared_isolation_prefill.results[1].next_token_id),
                    executor.cache_len(iso_shared_b),
                    iso_shared_b,
                ),
            ),
        )
    )
    isolated_reference_b = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                decode_request(
                    "iso-b",
                    int(isolated_prefill_b.results[0].next_token_id),
                    executor.cache_len(iso_b),
                    iso_b,
                ),
            ),
        )
    )

    isolation_ok = (
        result_pair_ok(
            isolated_decode_b.results[0],
            isolated_reference_b.results[0],
            atol=0.5,
        )
    )

    print(f"checkpoint={parity.checkpoint}")
    print(f"token_ids={list(parity.token_ids)}")
    print(f"native_prefill_physical_batch_size={prefill_batch_size}")
    print(f"native_prefill_model_forward_calls={prefill_calls}")
    print(f"native_decode_physical_batch_size={decode_batch_size}")
    print(f"native_decode_model_forward_calls={decode_calls}")
    print(f"native_batched_single_parity_ok={1 if parity_ok else 0}")
    print(f"native_unequal_cache_lengths_ok={1 if unequal_cache_lengths_ok else 0}")
    print(f"native_cache_isolation_ok={1 if isolation_ok else 0}")
    print(f"prefill_logits_shape={parity.prefill_logits_shape}")
    print(f"prefill_logits_dtype={parity.prefill_logits_dtype}")
    print(f"prefill_max_abs_diff={parity.prefill_max_abs_diff:.6f}")
    print(f"decode_max_abs_diff={parity.decode_max_abs_diff:.6f}")
    print(f"prefill_time_ms={prefill_elapsed_ms}")
    print(f"decode_time_ms={decode_elapsed_ms}")
    print(f"cache_lengths={parity.cache_lengths}")
    print(f"native_tokens={parity.native_tokens}")
    print(f"reference_tokens={parity.reference_tokens}")
    print(f"tolerance_ok={1 if parity.tolerance_ok else 0}")
    print(f"token_ok={1 if parity.token_ok else 0}")
    print(f"request_isolation_ok={1 if isolation_ok else 0}")

    if prefill_batch_size != 2 or prefill_calls != 1:
        raise SystemExit("prefill physical batching proof failed")
    if decode_batch_size != 2 or decode_calls != 1:
        raise SystemExit("decode physical batching proof failed")
    if not parity_ok:
        raise SystemExit("batched and single execution parity failed")
    if not unequal_cache_lengths_ok:
        raise SystemExit("unequal cache lengths were not exercised")
    if not isolation_ok:
        raise SystemExit("cache isolation probe failed")
    if not parity.tolerance_ok:
        raise SystemExit("native prefill/decode parity failed tolerance check")
    if not parity.token_ok:
        raise SystemExit("native prefill/decode token parity failed")
finally:
    executor.release(handle_a)
    executor.release(handle_b)
    executor.release(ind_a)
    executor.release(ind_b)
    executor.release(iso_a)
    executor.release(iso_shared_b)
    executor.release(iso_b)

release_ok = all(
    executor.cache_len(handle) == 0
    for handle in (handle_a, handle_b, ind_a, ind_b, iso_a, iso_shared_b, iso_b)
)
print(f"release_ok={1 if release_ok else 0}")
if not release_ok:
    raise SystemExit("native cache release failed")
PY
