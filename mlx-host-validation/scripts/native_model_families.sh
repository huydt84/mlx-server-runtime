#!/usr/bin/env bash
#
# native-v2 model-family host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_model_families.sh [options]
#   bash mlx-host-validation/scripts/native_model_families.sh --help
#
# Known-good checkpoints:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - `mlx-community/LFM2.5-8B-A1B-MLX-4bit`
#   - `mlx-community/Qwen3-4B-Instruct-2507-4bit`
#   - `mlx-community/gemma-3-270m-it-qat-8bit`
#
# Probe checkpoints:
#   - `local-probe/LlamaForCausalLM` unsupported-class reference from model-validation
#   - `local-probe/Qwen2ForCausalLM-missing-tokenizer` malformed-artifact reference from model-validation
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
#   - one `checkpoint=...` line for every checkpoint above
#   - `token_ids=` with finalized prompt token IDs
#   - `logits_shape=` with native logits shape
#   - `tolerance_ok=1`
#   - `token_ok=1`
#   - `prefill_decode_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - printed error from native construction, mapping, or forward/parity stage
#   - missing `tolerance_ok=1` or `token_ok=1`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate native forward-pass and greedy-token parity against mlx_lm for each registered model family."
cli_option CHECKPOINT_LIST --checkpoints MLX_MODEL_FAMILIES_CHECKPOINTS list \
    "mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/LFM2.5-8B-A1B-MLX-4bit,mlx-community/Qwen3-4B-Instruct-2507-4bit,mlx-community/gemma-3-270m-it-qat-8bit" \
    "Comma-separated checkpoints to check, in order."
cli_option DECODE_STEPS --decode-steps MLX_MODEL_FAMILIES_DECODE_STEPS int 2 \
    "Greedy decode steps compared after prefill."
cli_option SKIP_SYNC --skip-sync MLX_MODEL_FAMILIES_SKIP_SYNC bool 0 \
    "Skip uv sync; reuse the existing Python environment."
cli_example "bash mlx-host-validation/scripts/native_model_families.sh --checkpoints mlx-community/Qwen3-4B-Instruct-2507-4bit"
cli_parse "$@"
cli_split_list CHECKPOINTS "$CHECKPOINT_LIST"
if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
    echo "error: --checkpoints needs at least one checkpoint" >&2
    exit 2
fi

echo "[1/3] Sync Python dev environment"
cd "$PYTHON_DIR"
if [[ "$SKIP_SYNC" == "1" ]]; then
    echo "sync_skipped=1"
else
    uv sync --group dev
fi

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

echo "[3/3] Run real native forward pass and greedy-token parity"
uv run python - <<'PY' "$DECODE_STEPS" "${CHECKPOINTS[@]}"
from __future__ import annotations

import sys

from mlx_worker.native_mlx.bootstrap import (
    build_native_artifacts,
    build_finalized_token_ids,
)
from mlx_worker.native_mlx.diagnostics import (
    compare_native_to_mlx_lm,
    compare_native_prefill_decode_to_mlx_lm,
)

decode_steps = int(sys.argv[1])
for checkpoint in sys.argv[2:]:
    artifacts = build_native_artifacts(checkpoint)
    model_path = artifacts.architecture.model_path
    token_ids = build_finalized_token_ids(
        model_path,
        [{"role": "user", "content": "ping"}],
    )
    parity = compare_native_to_mlx_lm(checkpoint, artifacts.diagnostics, token_ids)

    print(f"checkpoint={parity.checkpoint}")
    print(f"token_ids={list(parity.token_ids)}")
    print(f"logits_shape={parity.logits_shape}")
    print(f"logits_dtype={parity.logits_dtype}")
    print(f"max_abs_diff={parity.max_abs_diff:.6f}")
    print(f"tolerance_atol={parity.tolerance_atol:.6f}")
    print(f"tolerance_rtol={parity.tolerance_rtol:.6f}")
    print(f"native_next_token={parity.native_next_token}")
    print(f"reference_next_token={parity.reference_next_token}")
    print(f"tolerance_ok={1 if parity.tolerance_ok else 0}")
    print(f"token_ok={1 if parity.token_ok else 0}")

    if not parity.tolerance_ok:
        raise SystemExit(f"native logits parity failed for {checkpoint}")
    if not parity.token_ok:
        raise SystemExit(f"native greedy-token parity failed for {checkpoint}")
    prefill_decode = compare_native_prefill_decode_to_mlx_lm(
        checkpoint,
        artifacts.diagnostics,
        token_ids,
        decode_steps=decode_steps,
    )
    print(
        "prefill_decode_ok="
        f"{int(prefill_decode.tolerance_ok and prefill_decode.token_ok)}"
    )
    if not prefill_decode.tolerance_ok:
        raise SystemExit(f"native prefill/decode parity failed for {checkpoint}")
    if not prefill_decode.token_ok:
        raise SystemExit(f"native prefill/decode token parity failed for {checkpoint}")
PY
