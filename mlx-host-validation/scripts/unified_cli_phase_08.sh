#!/usr/bin/env bash
# Host-only cache-state, warmup-order, and runtime-delta validation for MLX Air.
# Run on Apple Silicon with Metal, uv, and the fixture model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-unified-cli-phase-08.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
RESULT_DIR="$WORK_DIR/result"
COMMAND_LOG="$WORK_DIR/command.log"
MANUAL_LOG="$WORK_DIR/manual-gateway.log"
FIXTURE="$ROOT/mlx-host-validation/fixtures/unified_cli_phase_08.toml"
UV_CACHE="${MLX_AIR_PHASE8_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    for pid in "${COMMAND_PID:-}" "${GATEWAY_PID:-}" "${REQUEST_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ "$status" -ne 0 ]]; then
        echo "phase-8 command log after validation failure:" >&2
        sed -n '1,300p' "$COMMAND_LOG" >&2
        echo "phase-8 manual gateway log after validation failure:" >&2
        sed -n '1,300p' "$MANUAL_LOG" >&2
        echo "phase-8 benchmark gateway log after validation failure:" >&2
        sed -n '1,300p' "$RESULT_DIR/logs/gateway.log" >&2
        echo "phase-8 benchmark worker log after validation failure:" >&2
        sed -n '1,300p' "$RESULT_DIR/logs/worker.log" >&2
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

wait_bounded() {
    local pid="$1"
    local label="$2"
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

reserve_port() {
    python3 - <<'PY'
import socket
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    print(listener.getsockname()[1])
PY
}

wait_ready() {
    local port="$1"
    python3 - "$port" <<'PY'
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

stop_gateway() {
    if [[ -n "${GATEWAY_PID:-}" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill -TERM "$GATEWAY_PID"
        wait "$GATEWAY_PID"
    fi
    unset GATEWAY_PID
}

echo "[1/6] Stage the benchmark-capable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/6] Run one bounded cold, shared-prefix, and cache-pressure sequence"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench run \
    --suite phase8 \
    --benchmark-config "$FIXTURE" \
    --output-dir "$RESULT_DIR" \
    >"$COMMAND_LOG" 2>&1 &
COMMAND_PID=$!
wait_bounded "$COMMAND_PID" "phase-8 benchmark"
unset COMMAND_PID

echo "[3/6] Validate configured order, non-measured warmup, cache states, and deltas"
python3 - "$RESULT_DIR/results.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert data["status"] == "succeeded", data
assert [(entry["configuration_order"], entry["model"], entry["runtime_configuration"]) for entry in data["applied_order"]] == [
    ("primary", "primary", "serial")
], data["applied_order"]
assert len(data["warmups"]) == 1, data["warmups"]
assert data["warmups"][0]["group"] == "compile", data["warmups"]
assert data["warmups"][0]["measured"] is False, data["warmups"]

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
echo "phase8_result_state_ok=1"

echo "[4/6] Reuse the managed environment and prove reset is absent from a regular process"
PORT="$(reserve_port)"
IPC_PATH="$WORK_DIR/unauthorized.sock"
UNAUTHORIZED_CONFIG="$WORK_DIR/unauthorized.toml"
python3 - "$RESULT_DIR/runtime-00.toml" "$UNAUTHORIZED_CONFIG" "$PORT" "$IPC_PATH" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
text = re.sub(r"(?m)^port = \d+$", f"port = {sys.argv[3]}", text)
text = re.sub(r'(?m)^ipc_path = ".*"$', f'ipc_path = "{sys.argv[4]}"', text)
text = re.sub(r"(?m)^max_completion_tokens = \d+$", "max_completion_tokens = 128", text)
pathlib.Path(sys.argv[2]).write_text(text)
PY
MLX_RUNTIME_CONFIG="$UNAUTHORIZED_CONFIG" \
MLX_RUNTIME_CONTINUOUS_BATCHING=1 \
MLX_RUNTIME_NATIVE_EXECUTION_MODE=serial \
MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=268435456 \
MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES=2 \
    "$STAGE_DIR/bin/mlx_runtime_gateway" >>"$MANUAL_LOG" 2>&1 &
GATEWAY_PID=$!
wait_ready "$PORT"
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
stop_gateway
echo "reset_authorization_ok=1"

echo "[5/6] Prove active-work rejection and idle reset preserve the loaded model and graphs"
PORT="$(reserve_port)"
IPC_PATH="$WORK_DIR/authorized.sock"
AUTHORIZED_CONFIG="$WORK_DIR/authorized.toml"
python3 - "$RESULT_DIR/runtime-00.toml" "$AUTHORIZED_CONFIG" "$PORT" "$IPC_PATH" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text()
text = re.sub(r"(?m)^port = \d+$", f"port = {sys.argv[3]}", text)
text = re.sub(r'(?m)^ipc_path = ".*"$', f'ipc_path = "{sys.argv[4]}"', text)
text = re.sub(r"(?m)^max_completion_tokens = \d+$", "max_completion_tokens = 128", text)
pathlib.Path(sys.argv[2]).write_text(text)
PY
MLX_RUNTIME_CONFIG="$AUTHORIZED_CONFIG" \
MLX_AIR_BENCHMARK_ENABLED=1 \
MLX_RUNTIME_CONTINUOUS_BATCHING=1 \
MLX_RUNTIME_NATIVE_EXECUTION_MODE=serial \
MLX_RUNTIME_NATIVE_PREFIX_CACHE_STRATEGY=block-hash \
MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=268435456 \
MLX_RUNTIME_TEXT_CACHE_MAX_ENTRIES=2 \
    "$STAGE_DIR/bin/mlx_runtime_gateway" >>"$MANUAL_LOG" 2>&1 &
GATEWAY_PID=$!
wait_ready "$PORT"
MODEL="$(python3 - "$RESULT_DIR/results.json" <<'PY'
import json
import pathlib
import sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["versions"]["model"]["name"])
PY
)"
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
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/metrics"
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    with urllib.request.urlopen(url, timeout=2) as response:
        if "mlx_requests_active 1" in response.read().decode():
            raise SystemExit(0)
    time.sleep(0.05)
raise SystemExit("request did not become active within 30 seconds")
PY
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
stop_gateway
echo "idle_enforcement_ok=1"
echo "graph_retention_ok=1"

echo "[6/6] Confirm deterministic artifacts and no staged gateway remains"
test -s "$RESULT_DIR/results.json"
test -s "$RESULT_DIR/logs/gateway.log"
test -f "$RESULT_DIR/logs/worker.log"
if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
    echo "a staged phase-8 gateway remains alive" >&2
    exit 1
fi
echo "runtime_metric_deltas_ok=1"
echo "unified_cli_phase_08_ok=1"
