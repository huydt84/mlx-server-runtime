#!/usr/bin/env bash
#
# native-v2 scheduling-policy host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_scheduling_policies.sh [options]
#   bash mlx-host-validation/scripts/native_scheduling_policies.sh --help
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
#   - `policy_policy_fcfs_probe_ok=1`
#   - `policy_policy_lpm_probe_ok=1`
#   - `policy_policy_lof_probe_ok=1`
#   - `policy_policy_priority_probe_ok=1`
#   - `policy_cancellation_cleanup_ok=1`
#   - `v1_non_regression_ok=1`
#   - `policy_benchmark_report=<path>`
#   - `policy_validation_ok=1`
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
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate native-mlx scheduling policies through the public gateway and write a policy comparison report."
cli_option CHECKPOINT --checkpoint MLX_POLICY_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served by both gateways."
cli_option NATIVE_PORT --native-port MLX_POLICY_NATIVE_PORT port 18122 "native-mlx gateway port."
cli_option V1_PORT --v1-port MLX_POLICY_V1_PORT port 18123 "v1 gateway port."
cli_option PREFILL_CHUNK_SIZE --prefill-chunk-size MLX_POLICY_PREFILL_CHUNK_SIZE int 16 \
    "Native prefill chunk size."
cli_option TEXT_CACHE_MAX_ENTRIES --text-cache-max-entries MLX_POLICY_TEXT_CACHE_MAX_ENTRIES int 64 \
    "Native prefix-cache entry limit."
cli_option TEXT_CACHE_BUDGET_BYTES --text-cache-budget-bytes MLX_POLICY_TEXT_CACHE_BUDGET_BYTES int \
    268435456 "Native paged KV budget in bytes."
cli_option HEALTH_TIMEOUT --health-timeout MLX_POLICY_HEALTH_TIMEOUT int 360 \
    "Seconds to wait for each gateway to report healthy."
cli_option REPORT --report MLX_POLICY_REPORT path \
    "$ROOT/benchmarks/results/native_scheduling_policies_policies.md" "Markdown report written at the end."
cli_option TMP_ROOT --artifacts-dir MLX_POLICY_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-scheduling-policy" "Directory for logs, configs, requests, and captures."
cli_option SKIP_BUILD --skip-build MLX_POLICY_SKIP_BUILD bool 0 \
    "Skip uv sync and cargo build; reuse the existing environment and gateway binary."
cli_parse "$@"
cli_require_free_port "$NATIVE_PORT" --native-port
cli_require_free_port "$V1_PORT" --v1-port

REQUEST_DIR="$TMP_ROOT/requests"
NATIVE_CONFIG="$TMP_ROOT/runtime-native.toml"
V1_CONFIG="$TMP_ROOT/runtime-v1.toml"
HEALTH_CAPTURE="$TMP_ROOT/health.txt"
NATIVE_LOG="$TMP_ROOT/native.log"
V1_LOG="$TMP_ROOT/v1.log"
METRICS_CAPTURE="$TMP_ROOT/metrics.txt"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
HELPER="$ROOT/mlx-host-validation/scripts/python/scheduling_policies.py"

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
    for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
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
if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "build_skipped=1"
else
    uv --directory "$PYTHON_DIR" sync --group dev
    (cd "$ROOT" && cargo build -p mlx_runtime_gateway)
fi

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
        MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
        MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
        MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
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
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
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

echo "[7/8] Write scheduling policy/fairness/SLO report"
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
echo "policy_validation_ok=1"
