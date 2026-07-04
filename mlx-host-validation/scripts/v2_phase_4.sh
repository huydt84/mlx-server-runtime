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

echo "[3/3] Run real native prefill/decode parity and cache lifecycle"
uv run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import hf_repo_to_path

from mlx_worker.native_mlx.interfaces import ExecutionBatch, ExecutionRequest
from mlx_worker.native_mlx.worker import (
    build_finalized_token_ids,
    compare_native_prefill_decode_to_mlx_lm,
    create_native_worker,
)

checkpoint = sys.argv[1]
scheduler = create_native_worker(type("Cfg", (), {"model": checkpoint})())
executor = scheduler._executor
model_path = Path(hf_repo_to_path(checkpoint))
token_ids = build_finalized_token_ids(
    model_path,
    [{"role": "user", "content": "ping"}],
)
parity = compare_native_prefill_decode_to_mlx_lm(
    checkpoint,
    executor,
    token_ids,
    decode_steps=2,
)

handle_a = executor.create_cache("req-a")
handle_b = executor.create_cache("req-b")
try:
    prefill_started = time.perf_counter()
    prefill = executor.prefill_batch(
        ExecutionBatch(
            phase="prefill",
            requests=(
                ExecutionRequest(
                    request_id="req-a",
                    token_ids=tuple(token_ids),
                    positions=tuple(range(len(token_ids))),
                    cache_handle=handle_a,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
                ExecutionRequest(
                    request_id="req-b",
                    token_ids=tuple(token_ids),
                    positions=tuple(range(len(token_ids))),
                    cache_handle=handle_b,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
            ),
        )
    )
    prefill_elapsed_ms = max(1, int((time.perf_counter() - prefill_started) * 1000))

    decode_input_a = int(prefill.results[0].next_token_id)
    decode_input_b = int(prefill.results[1].next_token_id)
    decode_started = time.perf_counter()
    decode = executor.decode_batch(
        ExecutionBatch(
            phase="decode",
            requests=(
                ExecutionRequest(
                    request_id="req-a",
                    token_ids=(decode_input_a,),
                    positions=(executor.cache_len(handle_a),),
                    cache_handle=handle_a,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
                ExecutionRequest(
                    request_id="req-b",
                    token_ids=(decode_input_b,),
                    positions=(executor.cache_len(handle_b),),
                    cache_handle=handle_b,
                    max_new_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                ),
            ),
        )
    )
    decode_elapsed_ms = max(1, int((time.perf_counter() - decode_started) * 1000))

    isolation_ok = executor.cache_len(handle_a) == executor.cache_len(handle_b)
    if not isolation_ok:
        raise SystemExit("cache isolation length mismatch")

    print(f"checkpoint={parity.checkpoint}")
    print(f"token_ids={list(parity.token_ids)}")
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

    if not parity.tolerance_ok:
        raise SystemExit("native prefill/decode parity failed tolerance check")
    if not parity.token_ok:
        raise SystemExit("native prefill/decode token parity failed")
finally:
    executor.release(handle_a)
    executor.release(handle_b)

release_ok = executor.cache_len(handle_a) == 0 and executor.cache_len(handle_b) == 0
print(f"release_ok={1 if release_ok else 0}")
if not release_ok:
    raise SystemExit("native cache release failed")
PY
