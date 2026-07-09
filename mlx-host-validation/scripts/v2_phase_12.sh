#!/usr/bin/env bash
#
# native-v2 Phase 12 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_12.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - public native gateway with `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY=fcfs`
#   - public native gateway with `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY=lpm`
#   - public native gateway with `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY=lof`
#   - public native gateway with `MLX_RUNTIME_NATIVE_SCHEDULING_POLICY=priority`
#   - concurrent policy fixtures that distinguish policy labels and tradeoffs
#   - anti-starvation/fairness evidence through policy wait-spread reporting
#   - separate gateway and scheduler queue-wait metrics
#   - streaming disconnect cancellation with cleanup/health verification
#   - default v1 public request through the same gateway surface
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint available to the local Hugging Face cache
#   - `cargo` toolchain for `mlx_runtime_gateway`
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_metal_available=1`
#   - `phase12_policy_fcfs_probe_ok=1`
#   - `phase12_policy_lpm_probe_ok=1`
#   - `phase12_policy_lof_probe_ok=1`
#   - `phase12_policy_priority_probe_ok=1`
#   - `phase12_cancellation_cleanup_ok=1`
#   - `v1_non_regression_ok=1`
#   - `phase12_benchmark_report=<path>`
#   - `phase_12_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - missing `gateway_queue` or `scheduler_queue` metrics
#   - missing scheduler policy label
#   - missing cancellation latency/cancellation counter
#   - gateway fails readiness or health after cancellation
#   - v1 public request fails

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
export PYTHONPATH="$ROOT/mlx-host-validation/scripts/python:$PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${MLX_PHASE12_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
NATIVE_PORT="${MLX_PHASE12_NATIVE_PORT:-18122}"
V1_PORT="${MLX_PHASE12_V1_PORT:-18123}"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-12"
REQUEST_DIR="$TMP_ROOT/requests"
NATIVE_CONFIG="$TMP_ROOT/runtime-native.toml"
V1_CONFIG="$TMP_ROOT/runtime-v1.toml"
HEALTH_CAPTURE="$TMP_ROOT/health.txt"
NATIVE_LOG="$TMP_ROOT/native.log"
V1_LOG="$TMP_ROOT/v1.log"
METRICS_CAPTURE="$TMP_ROOT/metrics.txt"
REPORT="${MLX_PHASE12_REPORT:-$ROOT/benchmarks/results/v2_phase_12_policies.md}"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
HELPER="$ROOT/mlx-host-validation/scripts/python/phase12_policies.py"

mkdir -p "$REQUEST_DIR" "$(dirname "$REPORT")"

GATEWAY_PID=""
POLICY_OUTPUTS=()

cleanup() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_healthy() {
    local log_path="$1"
    local port="$2"
    rm -f "$HEALTH_CAPTURE"
    for _ in $(seq 1 360); do
        if [[ -n "$GATEWAY_PID" ]] && ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
            echo "gateway exited unexpectedly; inspect $log_path" >&2
            return 1
        fi
        if curl -fsS "http://127.0.0.1:${port}/health" >"$HEALTH_CAPTURE"; then
            if grep -qx 'healthy' "$HEALTH_CAPTURE"; then
                return 0
            fi
        fi
        sleep 1
    done
    echo "gateway did not become healthy; inspect $log_path" >&2
    return 1
}

start_gateway() {
    local log_path="$1"
    local port="$2"
    local config_path="$3"
    shift 3
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port $port is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env \
            MLX_RUNTIME_CONFIG="$config_path" \
            "$@" \
            "$GATEWAY_BIN"
    ) >"$log_path" 2>&1 &
    GATEWAY_PID=$!
    wait_healthy "$log_path" "$port"
}

stop_gateway() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
    GATEWAY_PID=""
}

echo "[1/8] Sync Python environment and build gateway"
uv --directory "$PYTHON_DIR" sync --group dev
cargo build -p mlx_runtime_gateway

echo "[2/8] Verify Apple Silicon and MLX Metal"
uv --directory "$PYTHON_DIR" run python - <<'PY'
from __future__ import annotations

import platform

import mlx.core as mx

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")
print("mlx_import_ok=1")
if not mx.metal.is_available():
    raise SystemExit("MLX Metal is not available")
print("mlx_metal_available=1")
PY

echo "[3/8] Build runtime configs and request fixtures"
uv --directory "$PYTHON_DIR" run python "$HELPER" fixtures \
    --runtime-template "$ROOT/config/runtime.toml" \
    --native-config "$NATIVE_CONFIG" \
    --v1-config "$V1_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --native-port "$NATIVE_PORT" \
    --v1-port "$V1_PORT" \
    --request-dir "$REQUEST_DIR"

echo "[4/8] Run native policy probes"
for policy in fcfs lpm lof priority; do
    output="$TMP_ROOT/policy-${policy}.json"
    POLICY_OUTPUTS+=("$output")
    start_gateway \
        "$NATIVE_LOG" \
        "$NATIVE_PORT" \
        "$NATIVE_CONFIG" \
        MLX_RUNTIME_NATIVE_SCHEDULING_POLICY="$policy" \
        MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="${MLX_PHASE12_PREFILL_CHUNK_SIZE:-16}" \
        MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="${MLX_PHASE12_TEXT_CACHE_MAX_ENTRIES:-64}" \
        MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE12_TEXT_CACHE_BUDGET_BYTES:-268435456}"
    uv --directory "$PYTHON_DIR" run python "$HELPER" policy-probe \
        --request-dir "$REQUEST_DIR" \
        --port "$NATIVE_PORT" \
        --policy "$policy" \
        --metrics-capture "$METRICS_CAPTURE" \
        --output "$output"
    stop_gateway
done

echo "[5/8] Run native lifecycle-wide cancellation probe"
CANCEL_OUTPUT="$TMP_ROOT/cancel.json"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_SCHEDULING_POLICY=priority \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="${MLX_PHASE12_PREFILL_CHUNK_SIZE:-16}" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="${MLX_PHASE12_TEXT_CACHE_MAX_ENTRIES:-64}" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE12_TEXT_CACHE_BUDGET_BYTES:-268435456}"
uv --directory "$PYTHON_DIR" run python "$HELPER" cancel-probe \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --metrics-capture "$METRICS_CAPTURE" \
    --output "$CANCEL_OUTPUT"
stop_gateway

echo "[6/8] Run v1 public non-regression request"
start_gateway "$V1_LOG" "$V1_PORT" "$V1_CONFIG"
uv --directory "$PYTHON_DIR" run python "$HELPER" v1-probe \
    --request-dir "$REQUEST_DIR" \
    --port "$V1_PORT"
stop_gateway

echo "[7/8] Write Phase 12 policy/fairness/SLO report"
REPORT_ARGS=()
for output in "${POLICY_OUTPUTS[@]}"; do
    REPORT_ARGS+=(--policy-json "$output")
done
uv --directory "$PYTHON_DIR" run python "$HELPER" report \
    "${REPORT_ARGS[@]}" \
    --cancel-json "$CANCEL_OUTPUT" \
    --output "$REPORT"

echo "[8/8] Validate expected report and metrics signals"
grep -q "gateway_queue" "$METRICS_CAPTURE"
grep -q "scheduler_queue" "$METRICS_CAPTURE"
grep -q "cancellation" "$METRICS_CAPTURE"
test -s "$REPORT"
echo "phase_12_validation_ok=1"
