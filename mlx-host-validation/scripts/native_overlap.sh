#!/usr/bin/env bash
#
# Native-v2 same-thread MLX overlap validation.
#
# Usage:
#   bash mlx-host-validation/scripts/native_overlap.sh [options]
#   bash mlx-host-validation/scripts/native_overlap.sh --help
#   MTL_CAPTURE_ENABLED=1 bash mlx-host-validation/scripts/native_overlap.sh --metal-capture

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
# shellcheck source=lib/gateway.sh
source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
cli_describe "Validate serial versus overlap execution: token parity, latency regression, MLX overlap probe, and optional Metal captures."
cli_option CHECKPOINT --checkpoint MLX_OVERLAP_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served in both execution modes."
cli_option PORT --port MLX_OVERLAP_PORT port 18017 "Gateway port, reused by the serial and overlap runs in turn."
cli_option WARMUPS --warmups MLX_OVERLAP_WARMUPS int 5 "Warmup requests per mode before timing."
cli_option SAMPLES --samples MLX_OVERLAP_SAMPLES int 10 "Timed requests per mode; an even number >= 2."
cli_option MAX_TOKENS --max-tokens MLX_OVERLAP_MAX_TOKENS int 8 "Completion tokens per timed request."
cli_option MAX_REGRESSION --max-regression MLX_OVERLAP_MAX_REGRESSION number 0.02 \
    "Largest allowed overlap latency regression, as a fraction of serial."
cli_option HEALTH_TIMEOUT --health-timeout MLX_OVERLAP_HEALTH_TIMEOUT int 300 \
    "Seconds to wait for each gateway to report healthy."
cli_option METAL_CAPTURE --metal-capture MLX_OVERLAP_METAL_CAPTURE bool 0 \
    "Record an MLX .gputrace capture; requires MTL_CAPTURE_ENABLED=1."
cli_option XCTRACE_CAPTURE --xctrace MLX_OVERLAP_XCTRACE bool "" \
    "Record Metal System Traces with xctrace; defaults to the --metal-capture value."
cli_option XCTRACE_SECONDS --xctrace-seconds MLX_OVERLAP_XCTRACE_SECONDS int 15 "xctrace recording length in seconds."
cli_option XCTRACE_REQUESTS --xctrace-requests MLX_OVERLAP_XCTRACE_REQUESTS int 4 "Requests sent during each xctrace recording."
cli_option XCTRACE_MAX_TOKENS --xctrace-max-tokens MLX_OVERLAP_XCTRACE_MAX_TOKENS int 128 \
    "Completion tokens per xctrace request."
cli_option TRACE_DIR --artifacts-dir MLX_OVERLAP_TRACE_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-overlap" "Directory for configs, logs, reports, and traces."
cli_example "bash mlx-host-validation/scripts/native_overlap.sh --samples 20 --max-tokens 32"
cli_example "MTL_CAPTURE_ENABLED=1 bash mlx-host-validation/scripts/native_overlap.sh --metal-capture"
cli_parse "$@"
XCTRACE_CAPTURE="${XCTRACE_CAPTURE:-$METAL_CAPTURE}"
cli_require_free_port "$PORT" --port
PROBE_HELPER="$ROOT/mlx-host-validation/scripts/python/overlap_mlx_probe.py"
BENCH_HELPER="$ROOT/mlx-host-validation/scripts/python/overlap_benchmark.py"
PIPELINE_HELPER="$ROOT/mlx-host-validation/scripts/python/pipeline_profile_check.py"
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
    echo "overlap_host_error=Apple Silicon arm64 is required" >&2
    exit 1
fi
if (( SAMPLES < 2 || SAMPLES % 2 != 0 )); then
    echo "overlap_samples_error=--samples must be an even integer >= 2" >&2
    exit 1
fi
if [[ "$METAL_CAPTURE" == "1" && "${MTL_CAPTURE_ENABLED:-0}" != "1" ]]; then
    echo "overlap_capture_error=MTL_CAPTURE_ENABLED=1 is required" >&2
    exit 1
fi
if [[ "$XCTRACE_CAPTURE" == "1" ]] && ! xcrun xctrace list templates \
    | grep 'Metal System Trace' >/dev/null; then
    echo "overlap_xctrace_error=Metal System Trace template is unavailable" >&2
    exit 1
fi

# Metal refuses to overwrite an existing capture, so drop the last run's.
rm -rf "$PROFILE_DIR" "$SERIAL_XCTRACE_PROFILE_DIR" "$OVERLAP_XCTRACE_PROFILE_DIR"
mkdir -p "$TRACE_DIR" "$PROFILE_DIR"
sed -e "s/port = 8000/port = $PORT/" \
    -e 's/backend = "v1"/backend = "native-mlx"/' \
    -e "s|model = \".*\"|model = \"$CHECKPOINT\"|" \
    -e 's|ipc_path = "/tmp/mlx-runtime.sock"|ipc_path = "/tmp/mlx-runtime-overlap.sock"|' \
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
    local run_id="overlap-${mode}-$(date +%s)"
    env \
        MLX_RUNTIME_CONFIG="$config" \
        MLX_RUNTIME_NATIVE_EXECUTION_MODE="$mode" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE="$profile" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR="$profile_dir" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID="$run_id" \
        MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD="overlap-$mode" \
        MLX_RUNTIME_NATIVE_METAL_CAPTURE="$metal_capture" \
        MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=536870912 \
        MTL_CAPTURE_ENABLED="${MTL_CAPTURE_ENABLED:-0}" \
        cargo run -p mlx_runtime_gateway --bin mlx_runtime_gateway >"$log" 2>&1 &
    GATEWAY_PID=$!
    if ! gateway_wait_ready "$PORT" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$log"; then
        echo "overlap_gateway_error=gateway not ready for $mode" >&2
        exit 1
    fi
}

capture_metal_timeline() {
    local mode="$1"
    local config="$2"
    local trace="$3"
    local xml="$4"
    local analysis="$5"
    local profile_dir="$6"
    local log="$7"
    local notification="com.mlx-runtime.overlap.$$.${mode}.tracing-started"

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
    --output-markdown "$TRACE_DIR/overlap-report.md" \
    --max-regression "$MAX_REGRESSION"

echo "[5/7] Capture a separate overlap diagnostic timeline"
start_gateway overlap "$OVERLAP_CONFIG" "$TRACE_DIR/overlap-profile.log" 1
PYTHONPATH="$PYTHON_DIR" uv --directory "$PYTHON_DIR" run python "$PIPELINE_HELPER" \
    --url "http://127.0.0.1:$PORT/v1/chat/completions" \
    --model "$CHECKPOINT" \
    --output-dir "$PROFILE_DIR" \
    --run-id "overlap-overlap-profile"
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
    echo "overlap_xctrace=disabled"
fi

echo "[7/7] Verify durable evidence"
test -s "$TRACE_DIR/stream-probe.json"
test -s "$TRACE_DIR/serial.json"
test -s "$TRACE_DIR/overlap.json"
test -s "$TRACE_DIR/comparison.json"
test -s "$TRACE_DIR/overlap-report.md"
test -s "$TRACE_DIR/timeline-analysis.json"
test -s "$PROFILE_DIR/pipeline-trace.json"
if [[ "$METAL_CAPTURE" == "1" ]]; then
    test -s "$PROFILE_DIR/pipeline.gputrace"
    echo "overlap_metal_trace=$PROFILE_DIR/pipeline.gputrace"
fi
if [[ "$XCTRACE_CAPTURE" == "1" ]]; then
    test -d "$SERIAL_METAL_SYSTEM_TRACE"
    test -d "$OVERLAP_METAL_SYSTEM_TRACE"
    test -s "$SERIAL_METAL_SYSTEM_XML"
    test -s "$OVERLAP_METAL_SYSTEM_XML"
    test -s "$SERIAL_METAL_ANALYSIS"
    test -s "$OVERLAP_METAL_ANALYSIS"
    test -s "$METAL_COMPARISON"
    echo "overlap_serial_metal_system_trace=$SERIAL_METAL_SYSTEM_TRACE"
    echo "overlap_overlap_metal_system_trace=$OVERLAP_METAL_SYSTEM_TRACE"
fi
echo "overlap_validation_artifacts=$TRACE_DIR"
echo "overlap_validation_ok=1"
