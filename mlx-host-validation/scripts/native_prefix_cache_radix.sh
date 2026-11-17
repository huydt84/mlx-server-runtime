#!/usr/bin/env bash
#
# native-v2 radix prefix-cache host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_prefix_cache_radix.sh [options]
#   bash mlx-host-validation/scripts/native_prefix_cache_radix.sh --help
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - public native gateway with default `radix`
#   - public native gateway with explicit `block-hash`
#   - exact, partial, branching, failed-step, cancellation cleanup,
#     next-admission reuse, concurrent sharing, deterministic eviction signal,
#     and strategy-labeled prefix metrics
#   - default v1 public gateway request against the same checkpoint
#   - benchmark comparison of default `radix`, explicit `block-hash`, and v1
#     on streaming and non-streaming single requests, shared-prefix
#     miss/exact/partial/mixed ratios, concurrent few-long/many-short, and
#     concurrent few-short/many-long scenarios
#   - native model graph profiling with `MLX_RUNTIME_NATIVE_GRAPH_PROFILE=1`
#     for radix mixed concurrent workloads; graph-profile timings are
#     diagnostic and separate from fair benchmark latency/throughput rows
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
#   - `radix_radix_strategy_probe_ok=1`
#   - `radix_radix_metrics_ok=1`
#   - `radix_block_hash_strategy_probe_ok=1`
#   - `block_hash_incompatible_key_miss_ok=1` for both strategy contract runs
#   - `block_hash_cancellation_cleanup_ok=1` for both strategy contract runs
#   - `block_hash_failure_non_publication_ok=1` for both strategy contract runs
#   - `radix_cache_benchmark_report=<path>`
#   - graph profile JSON under the host-validation temp directory
#   - `v1_non_regression_ok=1`
#   - `radix_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly
#   - default radix startup fails or silently falls back to another strategy
#   - radix or block-hash strategy metrics are absent or mislabeled
#   - shared-prefix requests do not increase reused tokens/pages
#   - v1 public request fails

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
export PYTHONPATH="$ROOT/mlx-host-validation/scripts/python:$PYTHON_DIR${PYTHONPATH:+:$PYTHONPATH}"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate the native-mlx radix prefix cache contract and benchmark it against block-hash and v1."
cli_option CHECKPOINT --checkpoint MLX_RADIX_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served by both gateways."
cli_option NATIVE_PORT --native-port MLX_RADIX_NATIVE_PORT port 18112 "native-mlx gateway port."
cli_option V1_PORT --v1-port MLX_RADIX_V1_PORT port 18113 "v1 gateway port."
cli_option PREFILL_CHUNK_SIZE --prefill-chunk-size MLX_RADIX_PREFILL_CHUNK_SIZE int 16 \
    "Native prefill chunk size."
cli_option TEXT_CACHE_MAX_ENTRIES --text-cache-max-entries MLX_RADIX_TEXT_CACHE_MAX_ENTRIES int 64 \
    "Native prefix-cache entry limit."
cli_option TEXT_CACHE_BUDGET_BYTES --text-cache-budget-bytes MLX_RADIX_TEXT_CACHE_BUDGET_BYTES int \
    268435456 "Native paged KV budget in bytes."
cli_option HEALTH_TIMEOUT --health-timeout MLX_RADIX_HEALTH_TIMEOUT int 360 \
    "Seconds to wait for each gateway to report healthy."
cli_option BENCHMARK_REPORT --report MLX_RADIX_BENCHMARK_REPORT path \
    "$ROOT/benchmarks/results/native_prefix_cache_radix_benchmark.md" "Markdown report written at the end."
cli_option TMP_ROOT --artifacts-dir MLX_RADIX_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-radix prefix-cache" "Directory for logs, configs, requests, and captures."
cli_option SKIP_BUILD --skip-build MLX_RADIX_SKIP_BUILD bool 0 \
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
RADIX_CONTRACT_CAPTURE="$TMP_ROOT/radix-contract.json"
BLOCK_HASH_CONTRACT_CAPTURE="$TMP_ROOT/block-hash-contract.json"
RADIX_BENCHMARK="$TMP_ROOT/benchmark-radix.json"
BLOCK_HASH_BENCHMARK="$TMP_ROOT/benchmark-block-hash.json"
V1_BENCHMARK="$TMP_ROOT/benchmark-v1.json"
GRAPH_PROFILE_JSON="$TMP_ROOT/graph-profile-radix.json"
GRAPH_PROFILE_METRICS="$TMP_ROOT/graph-profile-metrics.txt"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
RADIX_HELPER="$ROOT/mlx-host-validation/scripts/python/radix_cache_benchmark.py"

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
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" fixtures \
    --runtime-template "$ROOT/config/runtime.toml" \
    --native-config "$NATIVE_CONFIG" \
    --v1-config "$V1_CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --native-port "$NATIVE_PORT" \
    --v1-port "$V1_PORT" \
    --request-dir "$REQUEST_DIR"

echo "[4/8] Run default radix public gateway probes"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" strategy-probes \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --strategy radix \
    --metrics-capture "$METRICS_CAPTURE"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" cache-contract \
    --request-dir "$REQUEST_DIR" \
    --capture "$RADIX_CONTRACT_CAPTURE" \
    --port "$NATIVE_PORT" \
    --strategy radix \
    --metrics-capture "$METRICS_CAPTURE"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" incompatible-miss \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT"
stop_gateway

echo "[5/8] Run explicit block-hash public gateway probes"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" strategy-probes \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --strategy block-hash \
    --metrics-capture "$METRICS_CAPTURE"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" cache-contract \
    --request-dir "$REQUEST_DIR" \
    --capture "$BLOCK_HASH_CONTRACT_CAPTURE" \
    --port "$NATIVE_PORT" \
    --strategy block-hash \
    --metrics-capture "$METRICS_CAPTURE"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" incompatible-miss \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT"
stop_gateway

echo "[6/8] Run v1 public non-regression request"
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

echo "[7/8] Run radix, block-hash, and v1 public benchmark"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" benchmark \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --backend "radix" \
    --output "$RADIX_BENCHMARK" \
    --metrics-capture "$METRICS_CAPTURE"
stop_gateway

start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" benchmark \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --backend "block-hash" \
    --output "$BLOCK_HASH_BENCHMARK" \
    --metrics-capture "$METRICS_CAPTURE"
stop_gateway

start_gateway "$V1_LOG" "$V1_PORT" "$V1_CONFIG"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" benchmark \
    --request-dir "$REQUEST_DIR" \
    --port "$V1_PORT" \
    --backend "v1" \
    --output "$V1_BENCHMARK"
stop_gateway

echo "[8/8] Run native radix graph profile and write report"
start_gateway \
    "$NATIVE_LOG" \
    "$NATIVE_PORT" \
    "$NATIVE_CONFIG" \
    MLX_RUNTIME_NATIVE_GRAPH_PROFILE=1 \
    MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
    MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES="$TEXT_CACHE_MAX_ENTRIES" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES"
uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" graph-profile \
    --request-dir "$REQUEST_DIR" \
    --port "$NATIVE_PORT" \
    --output "$GRAPH_PROFILE_JSON" \
    --metrics-capture "$GRAPH_PROFILE_METRICS"
stop_gateway

uv --directory "$PYTHON_DIR" run python "$RADIX_HELPER" report \
    --radix-json "$RADIX_BENCHMARK" \
    --block-hash-json "$BLOCK_HASH_BENCHMARK" \
    --v1-json "$V1_BENCHMARK" \
    --graph-profile-json "$GRAPH_PROFILE_JSON" \
    --output "$BENCHMARK_REPORT"

echo "radix_validation_ok=1"
