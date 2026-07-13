#!/usr/bin/env bash
#
# Native-v2 Phase 17 same-thread MLX overlap validation.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_17.sh
#   MTL_CAPTURE_ENABLED=1 MLX_PHASE17_METAL_CAPTURE=1 \
#       bash mlx-host-validation/scripts/v2_phase_17.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
TRACE_DIR="${MLX_PHASE17_TRACE_DIR:-${TMPDIR:-/tmp}/mlx-runtime-v2-phase-17}"
CHECKPOINT="${MLX_PHASE17_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
PORT="${MLX_PHASE17_PORT:-18017}"
WARMUPS="${MLX_PHASE17_WARMUPS:-5}"
SAMPLES="${MLX_PHASE17_SAMPLES:-10}"
MAX_TOKENS="${MLX_PHASE17_MAX_TOKENS:-8}"
MAX_REGRESSION="${MLX_PHASE17_MAX_REGRESSION:-0.02}"
METAL_CAPTURE="${MLX_PHASE17_METAL_CAPTURE:-0}"
XCTRACE_CAPTURE="${MLX_PHASE17_XCTRACE:-$METAL_CAPTURE}"
XCTRACE_SECONDS="${MLX_PHASE17_XCTRACE_SECONDS:-15}"
XCTRACE_REQUESTS="${MLX_PHASE17_XCTRACE_REQUESTS:-4}"
XCTRACE_MAX_TOKENS="${MLX_PHASE17_XCTRACE_MAX_TOKENS:-128}"
PROBE_HELPER="$ROOT/mlx-host-validation/scripts/python/phase17_mlx_probe.py"
BENCH_HELPER="$ROOT/mlx-host-validation/scripts/python/phase17_overlap_benchmark.py"
PIPELINE_HELPER="$ROOT/mlx-host-validation/scripts/python/phase16_pipeline.py"
SERIAL_CONFIG="$TRACE_DIR/serial.toml"
OVERLAP_CONFIG="$TRACE_DIR/overlap.toml"
PROFILE_DIR="$TRACE_DIR/overlap-profile"
SERIAL_XCTRACE_PROFILE_DIR="$TRACE_DIR/serial-xctrace-profile"
OVERLAP_XCTRACE_PROFILE_DIR="$TRACE_DIR/overlap-xctrace-profile"
TRACE_STAMP="$(date +%s)"
SERIAL_METAL_SYSTEM_TRACE="$TRACE_DIR/serial-metal-system-$TRACE_STAMP.trace"
OVERLAP_METAL_SYSTEM_TRACE="$TRACE_DIR/overlap-metal-system-$TRACE_STAMP.trace"
SERIAL_METAL_SYSTEM_XML="$TRACE_DIR/serial-metal-gpu-intervals.xml"
OVERLAP_METAL_SYSTEM_XML="$TRACE_DIR/overlap-metal-gpu-intervals.xml"
SERIAL_METAL_ANALYSIS="$TRACE_DIR/serial-metal-analysis.json"
OVERLAP_METAL_ANALYSIS="$TRACE_DIR/metal-analysis.json"
METAL_COMPARISON="$TRACE_DIR/metal-comparison.json"
ROUND_SAMPLES="$((SAMPLES / 2))"

if [[ "$(uname -m)" != "arm64" ]]; then
    echo "phase17_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi
if (( SAMPLES < 2 || SAMPLES % 2 != 0 )); then
    echo "phase17_samples_error=MLX_PHASE17_SAMPLES must be an even integer >= 2" >&2
    exit 1
fi
if [[ "$METAL_CAPTURE" == "1" && "${MTL_CAPTURE_ENABLED:-0}" != "1" ]]; then
    echo "phase17_capture_error=MTL_CAPTURE_ENABLED=1 is required" >&2
    exit 1
fi
if [[ "$XCTRACE_CAPTURE" == "1" ]] && ! xcrun xctrace list templates \
    | grep 'Metal System Trace' >/dev/null; then
    echo "phase17_xctrace_error=Metal System Trace template is unavailable" >&2
    exit 1
fi

mkdir -p "$TRACE_DIR" "$PROFILE_DIR"
sed -e "s/port = 8000/port = $PORT/" \
    -e 's/backend = "v1"/backend = "native-mlx"/' \
    -e "s|model = \".*\"|model = \"$CHECKPOINT\"|" \
    -e 's|ipc_path = "/tmp/mlx-runtime.sock"|ipc_path = "/tmp/mlx-runtime-phase17.sock"|' \
    "$ROOT/config/runtime.toml" >"$SERIAL_CONFIG"
cp "$SERIAL_CONFIG" "$OVERLAP_CONFIG"

GATEWAY_PID=""
XCTRACE_PID=""
NOTIFY_PID=""
stop_gateway() {
    if [[ -n "$NOTIFY_PID" ]]; then
        kill "$NOTIFY_PID" 2>/dev/null || true
        wait "$NOTIFY_PID" 2>/dev/null || true
        NOTIFY_PID=""
    fi
    if [[ -n "$XCTRACE_PID" ]]; then
        kill "$XCTRACE_PID" 2>/dev/null || true
        wait "$XCTRACE_PID" 2>/dev/null || true
        XCTRACE_PID=""
    fi
    if [[ -n "$GATEWAY_PID" ]]; then
        kill "$GATEWAY_PID" 2>/dev/null || true
        wait "$GATEWAY_PID" 2>/dev/null || true
        GATEWAY_PID=""
    fi
}
trap stop_gateway EXIT

start_gateway() {
    local mode="$1"
    local config="$2"
    local log="$3"
    local profile="$4"
    local metal_capture="${5:-$METAL_CAPTURE}"
    local profile_dir="${6:-$PROFILE_DIR}"
    local run_id="phase17-${mode}-$(date +%s)"
    env \
        MLX_RUNTIME_CONFIG="$config" \
        MLX_RUNTIME_NATIVE_EXECUTION_MODE="$mode" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE="$profile" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR="$profile_dir" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID="$run_id" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD="phase17-$mode" \
        MLX_RUNTIME_NATIVE_METAL_CAPTURE="$metal_capture" \
        MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=536870912 \
        MTL_CAPTURE_ENABLED="${MTL_CAPTURE_ENABLED:-0}" \
        cargo run -p mlx_runtime_gateway >"$log" 2>&1 &
    GATEWAY_PID=$!
    for _ in $(seq 1 300); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return
        fi
        if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
            tail -100 "$log" >&2
            exit 1
        fi
        sleep 1
    done
    echo "phase17_gateway_error=readiness timeout for $mode" >&2
    exit 1
}

capture_metal_timeline() {
    local mode="$1"
    local config="$2"
    local trace="$3"
    local xml="$4"
    local analysis="$5"
    local profile_dir="$6"
    local log="$7"
    local notification="com.mlx-runtime.phase17.$$.${mode}.tracing-started"

    mkdir -p "$profile_dir"
    start_gateway "$mode" "$config" "$log" 0 0 "$profile_dir"
    notifyutil -1 "$notification" >"$TRACE_DIR/${mode}-xctrace-ready.log" 2>&1 &
    NOTIFY_PID=$!
    xcrun xctrace record \
        --template 'Metal System Trace' \
        --all-processes \
        --time-limit "${XCTRACE_SECONDS}s" \
        --output "$trace" \
        --notify-tracing-started "$notification" \
        --no-prompt >"$TRACE_DIR/${mode}-xctrace.log" 2>&1 &
    XCTRACE_PID=$!
    wait "$NOTIFY_PID"
    NOTIFY_PID=""
    PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" metal-workload \
        --port "$PORT" \
        --model "$CHECKPOINT" \
        --mode "$mode" \
        --output "$profile_dir/metal-workload.json" \
        --requests "$XCTRACE_REQUESTS" \
        --max-tokens "$XCTRACE_MAX_TOKENS"
    wait "$XCTRACE_PID"
    XCTRACE_PID=""
    stop_gateway
    xcrun xctrace export \
        --input "$trace" \
        --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]' \
        --output "$xml"
    PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" metal \
        --xml "$xml" \
        --output "$analysis"
}

run_mode() {
    local mode="$1"
    local config="$2"
    local output="$3"
    local log="$4"
    start_gateway "$mode" "$config" "$log" 0
    PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" run \
        --port "$PORT" \
        --model "$CHECKPOINT" \
        --mode "$mode" \
        --output "$output" \
        --warmups "$WARMUPS" \
        --samples "$SAMPLES" \
        --max-tokens "$MAX_TOKENS"
    stop_gateway
}

echo "[1/7] Probe MLX stream dependency ordering"
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$PROBE_HELPER" \
    --output "$TRACE_DIR/stream-probe.json"

echo "[2/7] Benchmark AB rounds: serial then overlap"
TOTAL_SAMPLES="$SAMPLES"
SAMPLES="$ROUND_SAMPLES"
run_mode serial "$SERIAL_CONFIG" "$TRACE_DIR/serial-round-1.json" "$TRACE_DIR/serial-round-1.log"
run_mode overlap "$OVERLAP_CONFIG" "$TRACE_DIR/overlap-round-1.json" "$TRACE_DIR/overlap-round-1.log"

echo "[3/7] Benchmark BA rounds: overlap then serial"
run_mode overlap "$OVERLAP_CONFIG" "$TRACE_DIR/overlap-round-2.json" "$TRACE_DIR/overlap-round-2.log"
run_mode serial "$SERIAL_CONFIG" "$TRACE_DIR/serial-round-2.json" "$TRACE_DIR/serial-round-2.log"
SAMPLES="$TOTAL_SAMPLES"
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" merge \
    --mode serial \
    --output "$TRACE_DIR/serial.json" \
    "$TRACE_DIR/serial-round-1.json" "$TRACE_DIR/serial-round-2.json"
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" merge \
    --mode overlap \
    --output "$TRACE_DIR/overlap.json" \
    "$TRACE_DIR/overlap-round-1.json" "$TRACE_DIR/overlap-round-2.json"

echo "[4/7] Enforce output parity and non-regression"
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" compare \
    --serial "$TRACE_DIR/serial.json" \
    --overlap "$TRACE_DIR/overlap.json" \
    --output-json "$TRACE_DIR/comparison.json" \
    --output-markdown "$TRACE_DIR/phase17-report.md" \
    --max-regression "$MAX_REGRESSION"

echo "[5/7] Capture a separate overlap diagnostic timeline"
start_gateway overlap "$OVERLAP_CONFIG" "$TRACE_DIR/overlap-profile.log" 1
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$PIPELINE_HELPER" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$CHECKPOINT" \
    --output-dir "$PROFILE_DIR" \
    --run-id "phase17-overlap-profile"
stop_gateway
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" timeline \
    --events "$PROFILE_DIR/pipeline-events.jsonl" \
    --output "$TRACE_DIR/timeline-analysis.json"

echo "[6/7] Capture serial and overlap Metal System Trace intervals"
if [[ "$XCTRACE_CAPTURE" == "1" ]]; then
    capture_metal_timeline \
        serial "$SERIAL_CONFIG" "$SERIAL_METAL_SYSTEM_TRACE" \
        "$SERIAL_METAL_SYSTEM_XML" "$SERIAL_METAL_ANALYSIS" \
        "$SERIAL_XCTRACE_PROFILE_DIR" "$TRACE_DIR/serial-xctrace-gateway.log"
    capture_metal_timeline \
        overlap "$OVERLAP_CONFIG" "$OVERLAP_METAL_SYSTEM_TRACE" \
        "$OVERLAP_METAL_SYSTEM_XML" "$OVERLAP_METAL_ANALYSIS" \
        "$OVERLAP_XCTRACE_PROFILE_DIR" "$TRACE_DIR/overlap-xctrace-gateway.log"
    PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$BENCH_HELPER" metal-compare \
        --serial "$SERIAL_METAL_ANALYSIS" \
        --overlap "$OVERLAP_METAL_ANALYSIS" \
        --output "$METAL_COMPARISON" \
        --max-regression "$MAX_REGRESSION"
else
    echo "phase17_xctrace=disabled"
fi

echo "[7/7] Verify durable evidence"
test -s "$TRACE_DIR/stream-probe.json"
test -s "$TRACE_DIR/serial.json"
test -s "$TRACE_DIR/overlap.json"
test -s "$TRACE_DIR/comparison.json"
test -s "$TRACE_DIR/phase17-report.md"
test -s "$TRACE_DIR/timeline-analysis.json"
test -s "$PROFILE_DIR/pipeline-trace.json"
if [[ "$METAL_CAPTURE" == "1" ]]; then
    test -s "$PROFILE_DIR/pipeline.gputrace"
    echo "phase17_metal_trace=$PROFILE_DIR/pipeline.gputrace"
fi
if [[ "$XCTRACE_CAPTURE" == "1" ]]; then
    test -d "$SERIAL_METAL_SYSTEM_TRACE"
    test -d "$OVERLAP_METAL_SYSTEM_TRACE"
    test -s "$SERIAL_METAL_SYSTEM_XML"
    test -s "$OVERLAP_METAL_SYSTEM_XML"
    test -s "$SERIAL_METAL_ANALYSIS"
    test -s "$OVERLAP_METAL_ANALYSIS"
    test -s "$METAL_COMPARISON"
    echo "phase17_serial_metal_system_trace=$SERIAL_METAL_SYSTEM_TRACE"
    echo "phase17_overlap_metal_system_trace=$OVERLAP_METAL_SYSTEM_TRACE"
fi
echo "phase17_validation_artifacts=$TRACE_DIR"
echo "phase17_validation_ok=1"
