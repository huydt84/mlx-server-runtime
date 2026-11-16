#!/usr/bin/env bash
# Host-only validation of `mlx-air bench`: load modes, interrupt cleanup, cache
# states and reset authorization, diagnostics, and calibration statistics.
# Run on Apple Silicon with Metal, uv, and the fixture models available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-cli-bench.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
COMMAND_LOG="$WORK_DIR/command.log"
MANUAL_LOG="$WORK_DIR/manual-gateway.log"
FIXTURE="$ROOT/mlx-host-validation/fixtures/bench.toml"
DELAY_PROXY="$ROOT/mlx-host-validation/scripts/python/bench_delay_gateway.py"
UV_CACHE="${MLX_AIR_HOST_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    for pid in "${COMMAND_PID:-}" "${GATEWAY_PID:-}" "${REQUEST_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ -d "$STAGE_DIR" ]]; then
        pkill -TERM -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "command log after validation failure:" >&2
        sed -n '1,400p' "$COMMAND_LOG" >&2
        echo "manual gateway log after validation failure:" >&2
        sed -n '1,300p' "$MANUAL_LOG" >&2
        find "$WORK_DIR" -path '*/logs/*.log' -type f -print -exec sed -n '1,200p' {} \; >&2
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

bench() {
    HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
        "$STAGE_DIR/bin/mlx-air" bench "$@" >>"$COMMAND_LOG" 2>&1 &
    COMMAND_PID=$!
}

wait_for_gateway() {
    local gateway_pid=""
    for _ in $(seq 1 600); do
        if ! kill -0 "$COMMAND_PID" >/dev/null 2>&1; then
            return 1
        fi
        gateway_pid="$(pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" | head -n 1 || true)"
        if [[ -n "$gateway_pid" ]]; then
            echo "$gateway_pid"
            return 0
        fi
        sleep 0.1
    done
    return 1
}

wait_command() {
    local label="$1"
    local pid="$COMMAND_PID"
    unset COMMAND_PID
    for _ in $(seq 1 36000); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "$label exceeded the 3600 second bound" >&2
    kill -TERM "$pid" >/dev/null 2>&1 || true
    return 1
}

assert_reaped() {
    local pid="$1"
    local label="$2"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "$label remains alive: $pid" >&2
        exit 1
    fi
}

assert_no_staged_gateway() {
    if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
        echo "a staged gateway remains alive" >&2
        exit 1
    fi
}

reserve_port() {
    python3 - <<'PY'
import socket
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
}

wait_ready() {
    python3 - "$1" <<'PY'
import json
import sys
import time
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/ready"
deadline = time.monotonic() + 1800
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if response.status == 200 and json.load(response).get("ready") is True:
                raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.2)
raise SystemExit("manual gateway did not become ready within 1800 seconds")
PY
}

# Start a regular gateway from the cache-state run's runtime config.
start_manual_gateway() {
    local name="$1"
    shift
    local port config
    port="$(reserve_port)"
    config="$WORK_DIR/$name.toml"
    python3 - "$CACHE_DIR/runtime-00.toml" "$config" "$port" "$WORK_DIR/$name.sock" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
text = re.sub(r"(?m)^port = \d+$", f"port = {sys.argv[3]}", text)
text = re.sub(r'(?m)^ipc_path = ".*"$', f'ipc_path = "{sys.argv[4]}"', text)
text = re.sub(r"(?m)^max_completion_tokens = \d+$", "max_completion_tokens = 128", text)
pathlib.Path(sys.argv[2]).write_text(text)
PY
    env "$@" \
        MLX_RUNTIME_CONFIG="$config" \
        MLX_RUNTIME_CONTINUOUS_BATCHING=1 \
        MLX_RUNTIME_NATIVE_EXECUTION_MODE=serial \
        MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
        MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=268435456 \
        MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES=2 \
        "$STAGE_DIR/bin/mlx_runtime_gateway" >>"$MANUAL_LOG" 2>&1 &
    GATEWAY_PID=$!
    PORT="$port"
    wait_ready "$PORT"
}

stop_manual_gateway() {
    if [[ -n "${GATEWAY_PID:-}" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill -TERM "$GATEWAY_PID"
        wait "$GATEWAY_PID"
    fi
    unset GATEWAY_PID
}

echo "[1/7] Stage the benchmark-capable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/7] Load modes: one server start and exactly 11 measured requests"
LOAD_DIR="$WORK_DIR/load-modes"
bench run --suite load_modes --benchmark-config "$FIXTURE" --output-dir "$LOAD_DIR"
LOAD_GATEWAY_PID="$(wait_for_gateway)"
LOAD_WORKER_PID="$(pgrep -P "$LOAD_GATEWAY_PID" | head -n 1 || true)"
wait_command "load-mode benchmark"
python3 - "$LOAD_DIR/results.json" "$FIXTURE" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["status"] == "succeeded", data
assert data["failure_stage"] is None, data
assert not pathlib.Path(data["server"]["runtime_configuration"]["worker"]["ipc_path"]).exists(), data
configuration = data["configuration"]
assert configuration["benchmark_config"] == str(pathlib.Path(sys.argv[2]).resolve()), configuration
assert configuration["suite"] == "load_modes", configuration
assert configuration["sampling"]["seed"] == 7007, configuration
assert [workload["name"] for workload in configuration["workloads"]] == [
    "sequential_stream",
    "burst_nonstream",
    "closed_stream",
], configuration
assert [warmup["workload"] for warmup in data["warmups"]] == [
    "sequential_stream",
    "burst_nonstream",
    "closed_stream",
], data["warmups"]

trials = data["trials"]
assert [trial["round"] for trial in trials] == [1, 1, 1], trials
assert [trial["request_count"] for trial in trials] == [2, 3, 6], trials
assert [trial["load_mode"] for trial in trials] == ["sequential", "burst", "closed-loop"], trials
assert [trial["streaming"] for trial in trials] == [True, False, True], trials
assert [trial["maximum_observed_in_flight"] for trial in trials] == [1, 3, 2], trials
assert trials[2]["submission_policy"] == "replace-each-completed-request", trials[2]

requests = [request for trial in trials for request in trial["requests"]]
assert [request["request_order"] for request in requests] == list(range(11)), requests
assert [request["prompt_index"] for request in trials[0]["requests"]] == [0, 1]
assert [request["prompt_index"] for request in trials[1]["requests"]] == [0, 1, 2]
assert [request["prompt_index"] for request in trials[2]["requests"]] == list(range(6))
for trial in trials:
    for request in trial["requests"]:
        assert request["status"] == "succeeded", request
        assert request["prompt_tokens"] > 0, request
        assert request["completion_tokens"] > 0, request
        assert request["total_tokens"] == request["prompt_tokens"] + request["completion_tokens"], request
        assert request["output_sha256"], request
        if trial["streaming"]:
            for field in (
                "submitted_monotonic_ns",
                "first_byte_monotonic_ns",
                "first_token_monotonic_ns",
                "final_token_monotonic_ns",
                "completed_monotonic_ns",
            ):
                assert isinstance(request[field], int), request

metrics = data["analysis"]["metrics"]
primary = {summary["workload"]: summary["metric"] for summary in metrics if summary["primary"]}
assert primary == {
    "sequential_stream": "ttft",
    "burst_nonstream": "end_to_end_latency",
    "closed_stream": "aggregate_output_tokens_per_second",
}, primary
assert not any(
    summary["metric"] in {"ttft", "tpot"}
    for summary in metrics
    if summary["workload"] == "burst_nonstream"
), metrics
PY
python3 - "$COMMAND_LOG" <<'PY'
import pathlib
import sys

progress = [
    line
    for line in pathlib.Path(sys.argv[1]).read_text().splitlines()
    if line.startswith("[") and " round " in line
]
assert progress == [
    "[1/3] round 1 primary/v1 sequential_stream trial 0: 2 ok, 0 errors",
    "[2/3] round 1 primary/v1 burst_nonstream trial 0: 3 ok, 0 errors",
    "[3/3] round 1 primary/v1 closed_stream trial 0: 6 ok, 0 errors",
], progress
PY
test -s "$LOAD_DIR/report.md"
test -s "$LOAD_DIR/logs/gateway.log"
test -s "$LOAD_DIR/logs/worker.log"
assert_reaped "$LOAD_GATEWAY_PID" "load-mode gateway"
assert_reaped "$LOAD_WORKER_PID" "load-mode worker"
assert_no_staged_gateway
echo "load_modes_ok=1"

echo "[3/7] Interrupt a run during startup and verify cleanup"
INTERRUPT_DIR="$WORK_DIR/interrupted"
bench run --suite load_modes --benchmark-config "$FIXTURE" --output-dir "$INTERRUPT_DIR"
INTERRUPT_GATEWAY_PID="$(wait_for_gateway)"
INTERRUPT_WORKER_PID="$(pgrep -P "$INTERRUPT_GATEWAY_PID" | head -n 1 || true)"
kill -INT "$COMMAND_PID"
if wait_command "interrupted benchmark"; then
    echo "interrupted benchmark unexpectedly exited successfully" >&2
    exit 1
fi
python3 - "$INTERRUPT_DIR/results.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["status"] == "interrupted", data
assert data["failure_stage"] in {"server_startup", "readiness"}, data
assert data["error"]["kind"] == "signal", data
PY
test -f "$INTERRUPT_DIR/report.md"
test -f "$INTERRUPT_DIR/logs/gateway.log"
test -f "$INTERRUPT_DIR/logs/worker.log"
assert_reaped "$INTERRUPT_GATEWAY_PID" "interrupted gateway"
assert_reaped "$INTERRUPT_WORKER_PID" "interrupted worker"
assert_no_staged_gateway
echo "interrupt_cleanup_ok=1"

echo "[4/7] Cache states: cold, warm prefix, and cache pressure with runtime deltas"
CACHE_DIR="$WORK_DIR/cache-states"
bench run --suite cache_states --benchmark-config "$FIXTURE" --output-dir "$CACHE_DIR"
wait_command "cache-state benchmark"
python3 - "$CACHE_DIR/results.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["status"] == "succeeded", data
assert [
    (entry["round"], entry["model"], entry["runtime_configuration"])
    for entry in data["applied_order"]
] == [(1, "primary", "block_hash")], data["applied_order"]
assert [warmup["workload"] for warmup in data["warmups"]] == ["cold", "shared", "pressure"], data["warmups"]

trials = {trial["workload_name"]: trial for trial in data["trials"]}
assert {name: trial["request_count"] for name, trial in trials.items()} == {
    "cold": 2,
    "shared": 2,
    "pressure": 4,
}, trials
assert all(request["cached_tokens"] == 0 for request in trials["cold"]["requests"]), trials["cold"]
assert all(
    0 < request["cached_tokens"] < request["prompt_tokens"]
    for request in trials["shared"]["requests"]
), trials["shared"]


def deltas(trial, prefix):
    return [
        value["delta"]
        for name, value in trial["runtime_metrics"].items()
        if name.startswith(prefix)
    ]


assert sum(deltas(trials["shared"], "mlx_prefix_cache_hits_by_backend")) > 0, trials["shared"]
assert sum(deltas(trials["shared"], "mlx_prefix_cache_reused_tokens_by_backend")) > 0, trials["shared"]
assert sum(deltas(trials["pressure"], "mlx_prefix_cache_evictions_by_backend")) > 0, trials["pressure"]
for trial in trials.values():
    reset = trial["cache_preparation"]["reset"]
    assert reset["scheduler_idle"] is True, reset
    assert reset["model_preserved"] is True, reset
    assert reset["graphs_preserved"] is True, reset
    assert deltas(trial, "mlx_requests_total") == [trial["request_count"]], trial
PY
assert_no_staged_gateway
echo "cache_states_ok=1"

echo "[5/7] Benchmark reset: absent without authorization, 409 while busy, preserves the model when idle"
start_manual_gateway unauthorized
python3 - "$PORT" <<'PY'
import sys
import urllib.error
import urllib.request

request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/internal/benchmark/reset",
    data=b"{}",
    method="POST",
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(request, timeout=5)
except urllib.error.HTTPError as error:
    assert error.code == 404, error
else:
    raise AssertionError("regular gateway exposed benchmark reset")
PY
stop_manual_gateway
echo "reset_authorization_ok=1"

start_manual_gateway authorized MLX_AIR_BENCHMARK_ENABLED=1
MODEL="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["versions"]["model"]["name"])' "$CACHE_DIR/results.json")"
python3 - "$PORT" "$MODEL" <<'PY' &
import json
import sys
import urllib.request

payload = json.dumps({
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": "Write a detailed 128-step numbered sequence."}],
    "max_tokens": 128,
    "temperature": 0.0,
    "top_p": 1.0,
}).encode()
request = urllib.request.Request(
    f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions",
    data=payload,
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=300) as response:
    assert response.status == 200
PY
REQUEST_PID=$!
python3 - "$PORT" <<'PY'
import sys
import time
import urllib.error
import urllib.request

base = f"http://127.0.0.1:{sys.argv[1]}"
deadline = time.monotonic() + 30
while True:
    with urllib.request.urlopen(f"{base}/metrics", timeout=2) as response:
        if "mlx_requests_active 1" in response.read().decode():
            break
    if time.monotonic() > deadline:
        raise SystemExit("request did not become active within 30 seconds")
    time.sleep(0.05)

request = urllib.request.Request(
    f"{base}/internal/benchmark/reset",
    data=b"{}",
    method="POST",
    headers={"Content-Type": "application/json"},
)
try:
    urllib.request.urlopen(request, timeout=5)
except urllib.error.HTTPError as error:
    assert error.code == 409, error
else:
    raise AssertionError("active benchmark reset unexpectedly succeeded")
PY
wait "$REQUEST_PID"
unset REQUEST_PID
python3 - "$PORT" "$MODEL" <<'PY'
import json
import sys
import urllib.request

base = f"http://127.0.0.1:{sys.argv[1]}"
with urllib.request.urlopen(f"{base}/live", timeout=5) as response:
    pid_before = json.load(response)["pid"]
with urllib.request.urlopen(f"{base}/ready", timeout=5) as response:
    ready_before = json.load(response)
request = urllib.request.Request(
    f"{base}/internal/benchmark/reset",
    data=b'{"clear_cache":true,"reset_counters":true}',
    method="POST",
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    reset = json.load(response)
assert reset["scheduler_idle"] is True, reset
assert reset["model_preserved"] is True, reset
assert reset["graphs_preserved"] is True, reset
with urllib.request.urlopen(f"{base}/live", timeout=5) as response:
    assert json.load(response)["pid"] == pid_before
with urllib.request.urlopen(f"{base}/ready", timeout=5) as response:
    ready_after = json.load(response)
assert ready_before["model"] == ready_after["model"] == sys.argv[2]
assert ready_after["ready"] is True
with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
    metrics = response.read().decode()
assert "mlx_requests_total 0" in metrics, metrics
PY
stop_manual_gateway
assert_no_staged_gateway
echo "reset_idle_enforcement_ok=1"

echo "[6/7] Diagnostics: representative profile after measurement, then one explicit family"
DIAGNOSTICS_DIR="$WORK_DIR/diagnostics"
bench run --suite diagnostics --benchmark-config "$FIXTURE" --profile representative \
    --output-dir "$DIAGNOSTICS_DIR"
wait_command "representative profile"
RESULT_FILE="$DIAGNOSTICS_DIR/results.json"
python3 - "$RESULT_FILE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["diagnostics"]["status"] == "succeeded", data["diagnostics"]
assert len(data["trials"]) == 2, data["trials"]
processes = data["processes"]
assert [process["purpose"] for process in processes] == [
    "measurement",
    "diagnostic:gateway",
    "diagnostic:decode",
], processes
measured = processes[0]
for diagnostic in processes[1:]:
    assert measured["pid"] != diagnostic["pid"], processes
    assert measured["stopped_monotonic_ns"] <= diagnostic["started_monotonic_ns"], processes
for key in (
    "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE",
    "MLX_RUNTIME_NATIVE_GRAPH_PROFILE",
    "MLX_RUNTIME_NATIVE_METAL_CAPTURE",
):
    assert measured["profiling_environment"][key] == "0", measured
assert processes[1]["profiling_environment"]["MLX_RUNTIME_NATIVE_PIPELINE_PROFILE"] == "1", processes[1]

attempts = data["diagnostics"]["attempts"]
assert [(attempt["family"], attempt["profilers"]) for attempt in attempts] == [
    ("gateway", ["pipeline"]),
    ("decode", ["graph"]),
], attempts
for attempt in attempts:
    assert attempt["status"] == "succeeded", attempt
    for artifact in attempt["artifacts"]:
        assert artifact["status"] == "succeeded", artifact
        assert (result_path.parent / artifact["path"]).is_file(), artifact

report = (result_path.parent / "report.md").read_text()
assert "## Diagnostic artifacts" in report, report
assert "pipeline-report.md" in report, report
PY
echo "representative_profile_ok=1"

bench diagnose --result "$RESULT_FILE" --workload-family decode
wait_command "explicit decode diagnostic"
python3 - "$RESULT_FILE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["diagnostics"]["status"] == "succeeded", data["diagnostics"]
attempts = data["diagnostics"]["attempts"]
assert [attempt["family"] for attempt in attempts] == ["gateway", "decode", "decode"], attempts
decode = attempts[-1]
assert decode["profilers"] == ["graph"], decode
assert decode["status"] == "succeeded", decode
graph = next(artifact for artifact in decode["artifacts"] if artifact["kind"] == "graph-profile")
assert graph["status"] == "succeeded", graph
payload = json.loads((result_path.parent / graph["path"]).read_text())
assert payload["diagnostic_only"] is True, payload
assert payload["excluded_from_measurements"] is True, payload
assert payload["metrics"], payload
assert len(data["trials"]) == 2, data["trials"]
PY
assert_no_staged_gateway
echo "explicit_diagnostic_family_ok=1"

# Calibration runs last because it replaces the staged gateway with a proxy
# that adds a known response delay.
echo "[7/7] Calibration: repeated runs see a controlled 120 ms delay"
mv "$STAGE_DIR/bin/mlx_runtime_gateway" "$STAGE_DIR/bin/mlx_runtime_gateway.real"
cp "$DELAY_PROXY" "$STAGE_DIR/bin/mlx_runtime_gateway"
chmod +x "$STAGE_DIR/bin/mlx_runtime_gateway"
: >"$COMMAND_LOG"
export MLX_AIR_BENCH_DELAY_MS=120
bench calibrate --suite statistics --benchmark-config "$FIXTURE" --repetitions 2
wait_command "calibration"
CALIBRATION_FILE="$(tail -n 1 "$COMMAND_LOG")"
python3 - "$CALIBRATION_FILE" <<'PY'
import json
import pathlib
import sys

result_path = pathlib.Path(sys.argv[1])
data = json.loads(result_path.read_text())
assert data["status"] == "succeeded", data
assert data["command"] == "calibrate", data
assert data["configuration"]["workloads"][0]["trials"] == 2
assert data["repetitions"] == {"requested": 2, "completed": 2}
assert len(data["runs"]) == 2, data["runs"]
assert len(data["host_observations"]) == 4, data["host_observations"]
for observation in data["host_observations"]:
    assert observation["phase"] in {"before", "after"}, observation
    assert {"thermal_state", "power_state", "memory_pressure"} <= observation.keys()

measurements = data["repeated_measurements"]
assert len(measurements) == 1, measurements
measurement = measurements[0]
assert (measurement["metric"], measurement["unit"], measurement["better_direction"]) == (
    "ttft",
    "ms",
    "lower",
), measurement
assert len(measurement["run_values"]) == 2, measurement
assert measurement["max_cv_percent"] == 10.0, measurement

for run in data["runs"]:
    child = json.loads((result_path.parent / run["result_path"]).read_text())
    summary = next(entry for entry in child["analysis"]["metrics"] if entry["primary"])
    assert summary["metric"] == "ttft", summary
    assert summary["trial_count"] == 2, summary
    assert summary["request_count"] == 4, summary
    assert summary["mean"] >= 100.0, summary
    assert summary["request_p95"] >= 100.0, summary
PY
test -s "$(dirname "$CALIBRATION_FILE")/report.md"
assert_no_staged_gateway
echo "calibration_ok=1"
echo "cli_bench_ok=1"
