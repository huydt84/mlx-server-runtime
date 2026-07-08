#!/usr/bin/env bash
#
# native-v2 Phase 10 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_10.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - public native gateway with `MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash`
#   - repeated full-page shared-prefix requests for miss, exact hit, and partial hit
#   - incompatible-key miss by restarting with a different native KV page size
#   - partial-tail miss with overlap shorter than one complete page
#   - concurrent sharing through two overlapping public requests
#   - cancellation cleanup through public streaming request cancellation
#   - failure non-publication through an invalid public request
#   - eviction through a deliberately tiny prefix-cache entry limit
#   - default v1 public gateway request against the same checkpoint
#   - benchmark comparison of native block-hash and v1 on streaming and
#     non-streaming single requests, shared-prefix miss/exact/partial/mixed
#     ratios, concurrent few-long/many-short, and concurrent few-short/many-long
#     scenarios
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#   - `cargo` toolchain for `mlx_runtime_gateway`
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_metal_available=1`
#   - `phase10_public_miss_ok=1`
#   - `phase10_public_exact_hit_ok=1`
#   - `phase10_public_partial_hit_ok=1`
#   - `phase10_incompatible_key_miss_ok=1`
#   - `phase10_partial_tail_miss_ok=1`
#   - `phase10_concurrent_sharing_ok=1`
#   - `phase10_cancellation_cleanup_ok=1`
#   - `phase10_failure_non_publication_ok=1`
#   - `phase10_eviction_ok=1`
#   - `phase10_metrics_labels_ok=1`
#   - `phase10_benchmark_report=<path>`
#   - `v1_non_regression_ok=1`
#   - `phase_10_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly
#   - block-hash metrics are absent or not labeled `native-mlx`, `text`, `block-hash`
#   - repeated shared-prefix requests do not increase reused tokens/pages
#   - cancellation/failure publishes leaked or incompatible reusable pages
#   - v1 public request fails

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
export PYTHONPATH="$PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="${MLX_PHASE10_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
NATIVE_PORT="${MLX_PHASE10_NATIVE_PORT:-18102}"
V1_PORT="${MLX_PHASE10_V1_PORT:-18103}"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-10"
REQUEST_DIR="$TMP_ROOT/requests"
NATIVE_CONFIG="$TMP_ROOT/runtime-native.toml"
V1_CONFIG="$TMP_ROOT/runtime-v1.toml"
HEALTH_CAPTURE="$TMP_ROOT/health.txt"
NATIVE_LOG="$TMP_ROOT/native.log"
V1_LOG="$TMP_ROOT/v1.log"
CAPTURE="$TMP_ROOT/capture.json"
METRICS_CAPTURE="$TMP_ROOT/metrics.txt"
NATIVE_BENCHMARK="$TMP_ROOT/benchmark-native.json"
V1_BENCHMARK="$TMP_ROOT/benchmark-v1.json"
BENCHMARK_REPORT="${MLX_PHASE10_BENCHMARK_REPORT:-$ROOT/benchmarks/results/v2_phase_10_benchmark.md}"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
PHASE10_HELPER="$ROOT/mlx-host-validation/scripts/python/phase10_benchmark.py"

mkdir -p "$REQUEST_DIR" "$(dirname "$BENCHMARK_REPORT")"

GATEWAY_PID=""

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

echo "[1/7] Sync Python environment and build gateway"
uv --directory "$PYTHON_DIR" sync --group dev
cargo build -p mlx_runtime_gateway

echo "[2/7] Verify Apple Silicon and MLX Metal"
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

echo "[3/7] Build runtime configs and request fixtures"
uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" fixtures \
    --runtime-template "$ROOT/config/runtime.toml" \
    --native-config "$NATIVE_CONFIG" \
    --v1-config "$V1_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --native-port "$NATIVE_PORT" \
    --v1-port "$V1_PORT" \
    --request-dir "$REQUEST_DIR"

echo "[4/7] Run native block-hash public gateway probes"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="${MLX_PHASE10_PREFILL_CHUNK_SIZE:-16}" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="${MLX_PHASE10_TEXT_CACHE_MAX_ENTRIES:-64}" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE10_TEXT_CACHE_BUDGET_BYTES:-268435456}"

uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" native-probes \
    --request-dir "$REQUEST_DIR" \
    --capture "$CAPTURE" \
    --metrics-capture "$METRICS_CAPTURE" \
    --port "$NATIVE_PORT"

stop_gateway

echo "[5/7] Run incompatible-key miss probe with different page size"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
    MLX_RUNTIME_NATIVE_KV_PAGE_SIZE=32 \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="${MLX_PHASE10_PREFILL_CHUNK_SIZE:-16}" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="${MLX_PHASE10_TEXT_CACHE_MAX_ENTRIES:-4}" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE10_TEXT_CACHE_BUDGET_BYTES:-268435456}"
uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" incompatible-miss \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT"
stop_gateway

echo "[6/7] Run v1 public non-regression request"
start_gateway "$V1_LOG" "$V1_PORT" "$V1_CONFIG"
V1_STATUS=$(curl -sS -o "$TMP_ROOT/v1.json" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    --data-binary @"$REQUEST_DIR/v1.json" \
    "http://127.0.0.1:${V1_PORT}/v1/chat/completions")
if [[ "$V1_STATUS" != "200" ]]; then
    echo "v1 request failed with status $V1_STATUS; inspect $V1_LOG" >&2
    exit 1
fi
echo "v1_non_regression_ok=1"
stop_gateway

echo "[7/7] Run native-v2 vs v1 public benchmark"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="${MLX_PHASE10_PREFILL_CHUNK_SIZE:-16}" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="${MLX_PHASE10_TEXT_CACHE_MAX_ENTRIES:-64}" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE10_TEXT_CACHE_BUDGET_BYTES:-268435456}"
uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" benchmark \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --backend "native-mlx" \
    --output "$NATIVE_BENCHMARK" \
    --metrics-capture "$METRICS_CAPTURE"
stop_gateway

start_gateway "$V1_LOG" "$V1_PORT" "$V1_CONFIG"
uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" benchmark \
    --request-dir "$REQUEST_DIR" \
    --port "$V1_PORT" \
    --backend "v1" \
    --output "$V1_BENCHMARK"
stop_gateway

uv --directory "$PYTHON_DIR" run python "$PHASE10_HELPER" report \
    --native-json "$NATIVE_BENCHMARK" \
    --v1-json "$V1_BENCHMARK" \
    --output "$BENCHMARK_REPORT"

echo "phase_10_validation_ok=1"
