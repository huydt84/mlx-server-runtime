#!/usr/bin/env bash
#
# native-v2 Phase 14 completion gate for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_14.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - required phase scripts `v2_phase_1.sh` through `v2_phase_12.sh`
#   - Rust formatting, clippy, and workspace tests
#   - Python sync, ruff format/check, and pytest
#   - full native-v2 public-gateway workstreams delegated to phase-owned
#     scripts for startup, serving, parity, streaming, batching, paged KV,
#     prefix strategies, policies, cancellation, metrics, unsupported class
#     behavior, and v1 non-regression
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#   - `cargo` toolchain for `mlx_runtime_gateway`
#
# Expected success signals:
#   - `phase14_required_scripts_ok=1`
#   - `phase14_python_validation_ok=1`
#   - `phase14_rust_validation_ok=1`
#   - each delegated `phase_<N>_validation_ok=1`
#   - `phase14_completion_report=<path>`
#   - `phase_14_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - missing required phase script
#   - local Rust/Python validation failure
#   - any delegated phase script failure
#   - blocked report with `phase_14_validation_blocked=1`

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
export PYTHONPATH="$ROOT/mlx-host-validation/scripts/python:$PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${MLX_PHASE14_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-14"
SCRIPT_STATUS="$TMP_ROOT/required-scripts.json"
COMMAND_LOG="$TMP_ROOT/commands.json"
REPORT="${MLX_PHASE14_REPORT:-$ROOT/benchmarks/results/v2_phase_14_completion.md}"
HELPER="$ROOT/mlx-host-validation/scripts/python/phase14_completion.py"

mkdir -p "$TMP_ROOT" "$(dirname "$REPORT")"
printf '[]\n' >"$COMMAND_LOG"

run_logged() {
    local label="$1"
    shift
    local started
    started="$(date +%s)"
    set +e
    "$@"
    local rc=$?
    set -e
    COMMAND_LOG_PATH="$COMMAND_LOG" COMMAND_LABEL="$label" COMMAND_RC="$rc" COMMAND_STARTED="$started" uv --directory "$PYTHON_DIR" run python - <<'PY'
from __future__ import annotations

import json
import os
import pathlib
import time

path = pathlib.Path(os.environ["COMMAND_LOG_PATH"])
items = json.loads(path.read_text())
items.append(
    {
        "cmd": os.environ["COMMAND_LABEL"],
        "returncode": int(os.environ["COMMAND_RC"]),
        "elapsed_s": time.time() - float(os.environ["COMMAND_STARTED"]),
    }
)
path.write_text(json.dumps(items, indent=2) + "\n")
PY
    return "$rc"
}

write_blocked_report() {
    uv --directory "$PYTHON_DIR" run python "$HELPER" report \
        --root "$ROOT" \
        --script-status "$SCRIPT_STATUS" \
        --command-log "$COMMAND_LOG" \
        --output "$REPORT" \
        --checkpoint "$CHECKPOINT"
}

run_or_blocked() {
    if ! run_logged "$@"; then
        write_blocked_report
        exit 1
    fi
}

echo "[1/5] Check required phase scripts and Bash syntax"
if ! uv --directory "$PYTHON_DIR" run python "$HELPER" check-scripts \
    --root "$ROOT" \
    --output "$SCRIPT_STATUS"; then
    write_blocked_report
    exit 1
fi

echo "[2/5] Run Python validation"
run_or_blocked "uv --directory python sync" uv --directory "$PYTHON_DIR" sync
run_or_blocked "uv --directory python run ruff format --check ." uv --directory "$PYTHON_DIR" run ruff format --check .
run_or_blocked "uv --directory python run ruff check ." uv --directory "$PYTHON_DIR" run ruff check .
run_or_blocked "uv --directory python run pytest" uv --directory "$PYTHON_DIR" run pytest
echo "phase14_python_validation_ok=1"

echo "[3/5] Run Rust validation"
run_or_blocked "cargo fmt --check" cargo fmt --check
run_or_blocked "cargo clippy --workspace --all-targets --all-features -- -D warnings" cargo clippy --workspace --all-targets --all-features -- -D warnings
run_or_blocked "cargo test --workspace --all-features" cargo test --workspace --all-features
echo "phase14_rust_validation_ok=1"

echo "[4/5] Run required host phase scripts"
for phase in $(seq 1 12); do
    script="$ROOT/mlx-host-validation/scripts/v2_phase_${phase}.sh"
    run_or_blocked "bash mlx-host-validation/scripts/v2_phase_${phase}.sh" bash "$script"
done

echo "[5/5] Write completion report"
uv --directory "$PYTHON_DIR" run python "$HELPER" report \
    --root "$ROOT" \
    --script-status "$SCRIPT_STATUS" \
    --command-log "$COMMAND_LOG" \
    --output "$REPORT" \
    --checkpoint "$CHECKPOINT" \
    --host-ran
