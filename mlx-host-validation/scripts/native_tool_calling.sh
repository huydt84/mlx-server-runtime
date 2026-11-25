#!/usr/bin/env bash
#
# Tool-calling host-only validation through the public gateway.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_tool_calling.sh [options]
#   bash mlx-host-validation/scripts/native_tool_calling.sh --help
#
# Every checkpoint in --checkpoints is served by native-mlx and asked for a
# call without and with streaming, a reply to the tool result, a single call
# with parallel_tool_calls false, and the requests that must not call tools.
# Then --v1-checkpoint is served by v1 without and with continuous batching,
# and --unsupported-checkpoint must reject requests with tools. Pass an empty
# --checkpoints, --v1-checkpoint or --unsupported-checkpoint to skip a part.
#
# Host requirements:
#   - Apple Silicon (`arm64`) with a Metal-capable MLX environment
#   - `uv` environment for `python/` and the `cargo` toolchain
#   - the checkpoints in the local Hugging Face cache
#
# Expected success signals:
#   - `native_tool_calling_ok=<checkpoint>` per checkpoint
#   - `v1_tool_calling_ok=1`
#   - `v1_continuous_tool_calling_ok=1`
#   - `unsupported_model_ok=1`
#   - `tool_calling_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit with `tool_calling_failed=<reason>`
#   - a gateway that fails readiness or exits unexpectedly

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
CHECK="$ROOT/mlx-host-validation/scripts/python/tool_calling.py"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
# shellcheck source=lib/gateway.sh
source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
cli_describe "Validate tool calling through the public gateway on native-mlx and v1."
cli_option CHECKPOINTS --checkpoints MLX_TOOL_CALLING_CHECKPOINTS list \
    "mlx-community/Qwen2.5-7B-Instruct-4bit,mlx-community/Qwen3-4B-Instruct-2507-4bit,mlx-community/Qwen3.5-2B-4bit,mlx-community/LFM2.5-8B-A1B-MLX-4bit" \
    "Checkpoints served by native-mlx, one after the other."
cli_option V1_CHECKPOINT --v1-checkpoint MLX_TOOL_CALLING_V1_CHECKPOINT string \
    "mlx-community/Qwen3-4B-Instruct-2507-4bit" "Checkpoint served by v1."
cli_option UNSUPPORTED_CHECKPOINT --unsupported-checkpoint MLX_TOOL_CALLING_UNSUPPORTED_CHECKPOINT string \
    "mlx-community/gemma-3-270m-it-qat-8bit" "native-mlx checkpoint without a tool-call parser."
cli_option PORT --port MLX_TOOL_CALLING_PORT port 18194 \
    "Gateway port; the gateways run one after the other on it."
cli_option HEALTH_TIMEOUT --health-timeout MLX_TOOL_CALLING_HEALTH_TIMEOUT int 300 \
    "Seconds to wait for each gateway to report ready."
cli_option REQUEST_TIMEOUT --request-timeout MLX_TOOL_CALLING_REQUEST_TIMEOUT int 300 \
    "Seconds to wait for each request."
cli_option ARTIFACTS_DIR --artifacts-dir MLX_TOOL_CALLING_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-tool-calling" "Directory for logs, configs, and captured responses."
cli_option SKIP_BUILD --skip-build MLX_TOOL_CALLING_SKIP_BUILD bool 0 \
    "Skip uv sync and cargo build; reuse the existing environment and gateway binary."
cli_example "bash mlx-host-validation/scripts/native_tool_calling.sh --port 18294"
cli_example "bash mlx-host-validation/scripts/native_tool_calling.sh --checkpoints mlx-community/Qwen3.5-2B-4bit --v1-checkpoint ''"
cli_parse "$@"
cli_require_free_port "$PORT" --port
cli_split_list CHECKPOINT_ARRAY "$CHECKPOINTS"

GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
GATEWAY_PID=""
mkdir -p "$ARTIFACTS_DIR"
SOCKET_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-tools.XXXXXX")"

stop_gateway() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
    GATEWAY_PID=""
}
cleanup() {
    stop_gateway
    rm -rf "$SOCKET_DIR"
}
trap cleanup EXIT

start_gateway() {
    local name="$1" backend="$2" checkpoint="$3"
    shift 3
    local config="$ARTIFACTS_DIR/$name.toml"
    local log="$ARTIFACTS_DIR/$name.log"
    cat >"$config" <<TOML
[server]
port = $PORT

[worker]
backend = "$backend"
model = "$checkpoint"
ipc_path = "$SOCKET_DIR/worker.sock"
TOML
    (
        cd "$ROOT"
        exec env MLX_RUNTIME_CONFIG="$config" "$@" "$GATEWAY_BIN"
    ) >"$log" 2>&1 &
    GATEWAY_PID=$!
    gateway_wait_ready "$PORT" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$log"
}

run_check() {
    local command="$1" name="$2" checkpoint="$3"
    uv --directory "$PYTHON_DIR" run python "$CHECK" "$command" \
        --port "$PORT" \
        --model "$checkpoint" \
        --timeout "$REQUEST_TIMEOUT" \
        --output "$ARTIFACTS_DIR/$name.json"
}

echo "[1/5] Prepare the environment"
if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "build_skipped=1"
else
    uv --directory "$PYTHON_DIR" sync --group dev
    cargo build --manifest-path "$ROOT/Cargo.toml" -p mlx_runtime_gateway
fi

echo "[2/5] Verify Apple Silicon, mlx, and mlx_lm imports"
uv --directory "$PYTHON_DIR" run python - <<'PY'
import platform

machine = platform.machine()
if machine != "arm64":
    raise SystemExit(f"tool_calling_failed=expected arm64, got {machine}")

import mlx.core as mx
import mlx_lm

print(f"mlx_compute_ok={(mx.array([1.0, 2.0]) * 2).tolist()}")
print(f"mlx_lm_version={mlx_lm.__version__}")
PY

echo "[3/5] Check native-mlx"
for index in "${!CHECKPOINT_ARRAY[@]}"; do
    checkpoint="${CHECKPOINT_ARRAY[$index]}"
    start_gateway "native-$index" native-mlx "$checkpoint"
    run_check supported "native-$index" "$checkpoint"
    stop_gateway
    echo "native_tool_calling_ok=$checkpoint"
done

echo "[4/5] Check v1"
if [[ -n "$V1_CHECKPOINT" ]]; then
    start_gateway v1 v1 "$V1_CHECKPOINT"
    run_check supported v1 "$V1_CHECKPOINT"
    stop_gateway
    echo "v1_tool_calling_ok=1"
    start_gateway v1-continuous v1 "$V1_CHECKPOINT" MLX_RUNTIME_CONTINUOUS_BATCHING=true
    run_check supported v1-continuous "$V1_CHECKPOINT"
    stop_gateway
    echo "v1_continuous_tool_calling_ok=1"
fi

echo "[5/5] Check a model without a tool-call parser"
if [[ -n "$UNSUPPORTED_CHECKPOINT" ]]; then
    start_gateway unsupported native-mlx "$UNSUPPORTED_CHECKPOINT"
    run_check unsupported unsupported "$UNSUPPORTED_CHECKPOINT"
    stop_gateway
    echo "unsupported_model_ok=1"
fi

echo "artifacts_dir=$ARTIFACTS_DIR"
echo "tool_calling_validation_ok=1"
