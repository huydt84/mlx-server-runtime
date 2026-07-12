#!/usr/bin/env bash
#
# Native-v2 Phase 16 whole inference-pipeline profiling validation.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_16.sh
#   MLX_PHASE16_METAL_CAPTURE=1 MTL_CAPTURE_ENABLED=1 bash mlx-host-validation/scripts/v2_phase_16.sh
#
# Known-good checkpoint: mlx-community/Qwen2.5-7B-Instruct-4bit
# Probe checkpoint override: MLX_PHASE16_CHECKPOINT=/path/or/hub-id
#
# Host requirements:
#   - Apple Silicon arm64 with Metal-capable MLX; full Xcode for `.gputrace`
#   - `uv`, `cargo`, the checkpoint in the local Hugging Face cache
#   - at least 2 GiB free in MLX_PHASE16_TRACE_DIR (default under /tmp)
#   - MTL_CAPTURE_ENABLED=1 must be present before startup when heavy capture is requested
#
# Low-overhead validation always runs a bounded public HTTP request and writes
# pipeline-events.jsonl, pipeline-trace.json, and pipeline-report.md. Optional
# heavy Metal capture and optional `xctrace` Metal System Trace are diagnostic;
# captured wall-clock is never reported as fair benchmark latency. For a full
# Instruments trace, wrap this script with an Xcode Metal System Trace template.
#
# Expected success: phase16_* metric/artifact lines and phase_16_validation_ok=1.
# Expected failure: non-arm64 host, missing tool/checkpoint, gateway readiness
# timeout, missing pipeline component, invalid capture environment, or non-zero exit.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
TRACE_DIR="${MLX_PHASE16_TRACE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-v2-phase-16}"
CHECKPOINT="${MLX_PHASE16_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
RUN_ID="phase16-$(date +%s)"
PORT="${MLX_PHASE16_PORT:-18016}"
CONFIG="$TRACE_DIR/runtime.toml"
HELPER="$ROOT/mlx-host-validation/scripts/python/phase16_pipeline.py"
GATEWAY_LOG="$TRACE_DIR/gateway.log"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "phase16_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi
if [[ "${MLX_PHASE16_METAL_CAPTURE:-0}" == "1" && "${MTL_CAPTURE_ENABLED:-0}" != "1" ]]; then
    echo "phase16_capture_error=MTL_CAPTURE_ENABLED=1 must be set before process startup" >&2
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
export MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD=phase16-public-gateway
export MLX_RUNTIME_NATIVE_METAL_CAPTURE="${MLX_PHASE16_METAL_CAPTURE:-0}"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "[1/3] Start native-mlx gateway with bounded pipeline profiling"
cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!
for _ in $(seq 1 300); do
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
echo "phase16_metal_capture=${MLX_PHASE16_METAL_CAPTURE:-0}"
if [[ "${MLX_PHASE16_METAL_CAPTURE:-0}" == "1" ]]; then
    test -s "$TRACE_DIR/pipeline.gputrace"
    echo "phase16_gputrace=$TRACE_DIR/pipeline.gputrace"
fi
echo "phase16_xctrace=optional_operator_capture"
