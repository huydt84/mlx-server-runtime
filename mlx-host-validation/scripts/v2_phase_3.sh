#!/usr/bin/env bash
#
# native-v2 Phase 3 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_3.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - `local-probe/LlamaForCausalLM` unsupported-class reference from Phase 2
#   - `local-probe/Qwen2ForCausalLM-missing-tokenizer` malformed-artifact reference from Phase 2
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
#   - `logits_shape=` with native logits shape
#   - `tolerance_ok=1`
#   - `token_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - printed error from native construction, mapping, or forward/parity stage
#   - missing `tolerance_ok=1` or `token_ok=1`

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

echo "[3/3] Run real native forward pass and greedy-token parity"
uv run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import sys

from mlx_lm.utils import hf_repo_to_path

from mlx_worker.native_mlx.worker import (
    build_finalized_token_ids,
    compare_native_to_mlx_lm,
    create_native_worker,
)

checkpoint = sys.argv[1]
scheduler = create_native_worker(type("Cfg", (), {"model": checkpoint})())
executor = scheduler._executor
model_path = hf_repo_to_path(checkpoint)
token_ids = build_finalized_token_ids(
    model_path,
    [{"role": "user", "content": "ping"}],
)
parity = compare_native_to_mlx_lm(checkpoint, executor, token_ids)

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
    raise SystemExit("native logits parity failed tolerance check")
if not parity.token_ok:
    raise SystemExit("native greedy-token parity failed")
PY
