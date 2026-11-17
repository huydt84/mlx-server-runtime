#!/usr/bin/env bash
#
# Native-v2 whole inference-pipeline profiling validation.
#
# Usage:
#   bash mlx-host-validation/scripts/native_pipeline_profile.sh [options]
#   bash mlx-host-validation/scripts/native_pipeline_profile.sh --help
#   MTL_CAPTURE_ENABLED=1 bash mlx-host-validation/scripts/native_pipeline_profile.sh --metal
#
# Known-good checkpoint: mlx-community/Qwen2.5-7B-Instruct-4bit
# Probe checkpoint override: --checkpoint /path/or/hub-id
#
# Host requirements:
#   - Apple Silicon arm64 with Metal-capable MLX; full Xcode for `.gputrace`
#   - `uv`, `cargo`, the checkpoint in the local Hugging Face cache
#   - at least 2 GiB free in the --artifacts-dir directory (default under /tmp)
#   - MTL_CAPTURE_ENABLED=1 must be present before startup when heavy capture is requested
#
# Low-overhead validation always runs a bounded public HTTP request and writes
# pipeline-events.jsonl, pipeline-trace.json, and pipeline-report.md. Optional
# heavy Metal capture and optional `xctrace` Metal System Trace are diagnostic;
# captured wall-clock is never reported as fair benchmark latency. For a full
# Instruments trace, wrap this script with an Xcode Metal System Trace template.
#
# Expected success: profile_* metric/artifact lines and profile_validation_ok=1.
# Expected failure: non-arm64 host, missing tool/checkpoint, gateway readiness
# timeout, missing pipeline component, invalid capture environment, or non-zero exit.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Profile the whole native-mlx inference pipeline for one bounded public-gateway request."
cli_option CHECKPOINT --checkpoint MLX_PROFILE_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint or local path to serve."
cli_option PORT --port MLX_PROFILE_PORT port 18016 "Gateway port used during capture."
cli_option METAL --metal MLX_PROFILE_METAL bool 0 \
    "Capture a bounded Metal .gputrace; requires MTL_CAPTURE_ENABLED=1."
cli_option RUN_ID --run-id MLX_PROFILE_RUN_ID string "" "Profile run id; defaults to profile-<unix time>."
cli_option HEALTH_TIMEOUT --health-timeout MLX_PROFILE_HEALTH_TIMEOUT int 300 \
    "Seconds to wait for the gateway to report healthy."
cli_option TRACE_DIR --artifacts-dir MLX_PROFILE_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-pipeline-profile" "Directory for the config, gateway log, and profile artifacts."
cli_parse "$@"
RUN_ID="${RUN_ID:-profile-$(date +%s)}"
cli_require_free_port "$PORT" --port
CONFIG="$TRACE_DIR/runtime.toml"
HELPER="$ROOT/mlx-host-validation/scripts/python/pipeline_profile_check.py"
GATEWAY_LOG="$TRACE_DIR/gateway.log"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "profile_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi
if [[ "$METAL" == "1" && "${MTL_CAPTURE_ENABLED:-0}" != "1" ]]; then
    echo "profile_capture_error=MTL_CAPTURE_ENABLED=1 must be set before process startup" >&2
    exit 1
fi

mkdir -p "$TRACE_DIR"
sed -e "s/port = 8000/port = $PORT/" \
    -e 's/backend = "v1"/backend = "native-mlx"/' \
    -e "s|model = \".*\"|model = \"$CHECKPOINT\"|" \
    "$ROOT/config/runtime.toml" >"$CONFIG"

export MLX_RUNTIME_CONFIG="$CONFIG"
export MLX_RUNTIME_NATIVE_PIPELINE_PROFILE=1
export MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR="$TRACE_DIR"
export MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID="$RUN_ID"
export MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD=profile-public-gateway
export MLX_RUNTIME_NATIVE_METAL_CAPTURE="$METAL"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[1/3] Start native-mlx gateway with bounded pipeline profiling"
cargo run -p mlx_runtime_gateway --bin mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!
for _ in $(seq 1 "$HEALTH_TIMEOUT"); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
        tail -100 "$GATEWAY_LOG" >&2
        exit 1
    fi
    sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null

echo "[2/3] Run bounded public-gateway workload and join timeline"
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$HELPER" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$CHECKPOINT" \
    --output-dir "$TRACE_DIR" \
    --run-id "$RUN_ID"

echo "[3/3] Verify durable artifacts"
test -s "$TRACE_DIR/pipeline-events.jsonl"
test -s "$TRACE_DIR/pipeline-trace.json"
test -s "$TRACE_DIR/pipeline-report.md"
echo "profile_metal_capture=$METAL"
if [[ "$METAL" == "1" ]]; then
    test -s "$TRACE_DIR/pipeline.gputrace"
    echo "profile_gputrace=$TRACE_DIR/pipeline.gputrace"
fi
echo "profile_xctrace=optional_operator_capture"
