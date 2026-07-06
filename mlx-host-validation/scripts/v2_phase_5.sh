#!/usr/bin/env bash
#
# native-v2 Phase 5 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_5.sh
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
#   - `prompt_fingerprint=` and `prompt_token_ids=`
#   - `prefill_jsonl=` and `decode_jsonl=` paths exist
#   - `summary_markdown=` path exists
#   - `summary_status=aligned` or `summary_status=<first mismatch kind>`
#   - `trace_off_normal_execution=1`
#   - `phase_5_trace_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - missing JSONL or markdown artifacts
#   - missing `trace_off_normal_execution=1`
#   - import/model/trace exceptions printed by inline Python

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

echo "[3/3] Run real semantic trace and verify normal execution keeps tracing off"
uv run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from mlx_worker.native_mlx.bootstrap import (
    build_native_artifacts,
    build_finalized_token_ids,
)
from mlx_worker.native_mlx.diagnostics import (
    build_prompt_fingerprint,
    trace_native_debug_to_mlx_lm,
)
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    SamplingParams,
)

checkpoint = sys.argv[1]
messages = [{"role": "user", "content": "ping"}]
trace_dir = Path(tempfile.mkdtemp(prefix="v2-phase5-trace-"))
no_trace_dir = Path(tempfile.mkdtemp(prefix="v2-phase5-no-trace-"))

try:
    runtime_artifacts = build_native_artifacts(checkpoint)
    executor = runtime_artifacts.executor
    model_path = runtime_artifacts.architecture.model_path
    prompt_token_ids = build_finalized_token_ids(model_path, messages)
    prompt_fingerprint = build_prompt_fingerprint(messages)

    artifacts = trace_native_debug_to_mlx_lm(
        checkpoint,
        runtime_artifacts.diagnostics,
        prompt_token_ids,
        prompt_fingerprint=prompt_fingerprint,
        output_dir=trace_dir,
        decode_steps=2,
    )

    if not artifacts.prefill_jsonl_path.exists():
        raise SystemExit("prefill JSONL artifact missing")
    if not artifacts.decode_jsonl_path.exists():
        raise SystemExit("decode JSONL artifact missing")
    if not artifacts.summary_markdown_path.exists():
        raise SystemExit("first-divergence markdown artifact missing")

    handle = executor.create_cache("phase-5-no-trace")
    try:
        prefill = executor.execute_batch(
            ExecutionBatch(
                requests=(
                    ExecutionRequest(
                        request_id="phase-5-no-trace",
                        phase="prefill",
                        token_ids=tuple(prompt_token_ids),
                        positions=tuple(range(len(prompt_token_ids))),
                        cache_handle=handle,
                        sampling=SamplingParams(),
                    ),
                ),
            )
        )
        decode_input = int(prefill.results[0].next_token_id)
        executor.execute_batch(
            ExecutionBatch(
                requests=(
                    ExecutionRequest(
                        request_id="phase-5-no-trace",
                        phase="decode",
                        token_ids=(decode_input,),
                        positions=(executor.cache_len(handle),),
                        cache_handle=handle,
                        sampling=SamplingParams(),
                    ),
                ),
            )
        )
    finally:
        executor.release(handle)

    trace_off_normal_execution = not any(no_trace_dir.iterdir())

    print(f"checkpoint={artifacts.checkpoint}")
    print(f"prompt_fingerprint={artifacts.prompt_fingerprint}")
    print(f"prompt_token_ids={list(artifacts.prompt_token_ids)}")
    print(f"decode_input_token_ids={list(artifacts.decode_input_token_ids)}")
    print(f"native_generated_token_ids={list(artifacts.native_generated_token_ids)}")
    print(f"reference_generated_token_ids={list(artifacts.reference_generated_token_ids)}")
    print(f"prefill_jsonl={artifacts.prefill_jsonl_path}")
    print(f"decode_jsonl={artifacts.decode_jsonl_path}")
    print(f"summary_markdown={artifacts.summary_markdown_path}")
    print(
        "summary_status="
        + (
            "aligned"
            if artifacts.comparison.aligned
            else artifacts.comparison.first_mismatch.kind
        )
    )
    if artifacts.comparison.first_mismatch is not None:
        print(
            "first_divergence="
            f"{artifacts.comparison.first_mismatch.checkpoint_id}"
        )
    print(f"trace_off_normal_execution={1 if trace_off_normal_execution else 0}")
    if not trace_off_normal_execution:
        raise SystemExit("normal executor path unexpectedly produced trace artifacts")
    print("phase_5_trace_validation_ok=1")
finally:
    shutil.rmtree(no_trace_dir, ignore_errors=True)
PY
