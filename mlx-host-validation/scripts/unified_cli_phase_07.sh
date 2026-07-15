#!/usr/bin/env bash
# Host-only configuration and workload-engine validation for MLX Air.
# Run on Apple Silicon with Metal, uv, and the fixture model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-unified-cli-phase-07.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
RESULT_DIR="$WORK_DIR/result"
COMMAND_LOG="$WORK_DIR/command.log"
FIXTURE="$ROOT/mlx-host-validation/fixtures/unified_cli_phase_07.toml"
UV_CACHE="${MLX_AIR_PHASE7_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cleanup() {
    local status=$?
    set +e
    for pid in "${COMMAND_PID:-}" "${GATEWAY_PID:-}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
            kill -TERM "$pid" >/dev/null 2>&1 || true
        fi
    done
    if [[ -d "$STAGE_DIR" ]]; then
        while read -r pid; do
            if [[ -n "$pid" ]]; then
                kill -TERM "$pid" >/dev/null 2>&1 || true
            fi
        done < <(pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" || true)
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "phase-7 command log after validation failure:" >&2
        sed -n '1,300p' "$COMMAND_LOG" >&2
        echo "phase-7 gateway log after validation failure:" >&2
        sed -n '1,300p' "$RESULT_DIR/logs/gateway.log" >&2
        echo "phase-7 worker log after validation failure:" >&2
        sed -n '1,300p' "$RESULT_DIR/logs/worker.log" >&2
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

wait_for_gateway() {
    local command_pid="$1"
    local gateway_pid=""
    for _ in $(seq 1 600); do
        if ! kill -0 "$command_pid" >/dev/null 2>&1; then
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

wait_bounded() {
    local pid="$1"
    for _ in $(seq 1 36000); do
        if ! kill -0 "$pid" >/dev/null 2>&1; then
            wait "$pid"
            return $?
        fi
        sleep 0.1
    done
    echo "phase-7 benchmark exceeded the 3600 second bound" >&2
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

echo "[1/4] Stage the benchmark-capable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/4] Run one model start and exactly 11 configured requests"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench run \
    --suite smoke \
    --benchmark-config "$FIXTURE" \
    --output-dir "$RESULT_DIR" \
    >"$COMMAND_LOG" 2>&1 &
COMMAND_PID=$!
GATEWAY_PID="$(wait_for_gateway "$COMMAND_PID")"
WORKER_PID="$(pgrep -P "$GATEWAY_PID" | head -n 1 || true)"
wait_bounded "$COMMAND_PID"
unset COMMAND_PID

echo "[3/4] Verify selection, load semantics, prompts, and token accounting"
python3 - "$RESULT_DIR/results.json" "$FIXTURE" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text())
fixture = str(pathlib.Path(sys.argv[2]).resolve())
assert data["status"] == "succeeded", data
configuration = data["configuration"]
assert configuration["benchmark_config"] == fixture, configuration
assert configuration["suite"] == "smoke", configuration
assert configuration["sampling"]["seed"] == 7007, configuration
assert [workload["name"] for workload in configuration["workloads"]] == [
    "sequential_stream",
    "burst_nonstream",
    "closed_stream",
], configuration

trials = data["trials"]
assert [trial["request_count"] for trial in trials] == [2, 3, 6], trials
assert [trial["load_mode"] for trial in trials] == [
    "sequential",
    "burst",
    "closed-loop",
], trials
assert [trial["streaming"] for trial in trials] == [True, False, True], trials
assert [trial["maximum_observed_in_flight"] for trial in trials] == [1, 3, 2], trials
assert trials[2]["submission_policy"] == "replace-each-completed-request", trials[2]

requests = [request for trial in trials for request in trial["requests"]]
assert len(requests) == 11, requests
assert [request["request_order"] for request in requests] == list(range(11)), requests
assert [request["prompt_index"] for request in trials[0]["requests"]] == [0, 1]
assert [request["prompt_index"] for request in trials[1]["requests"]] == [0, 1, 2]
assert [request["prompt_index"] for request in trials[2]["requests"]] == list(range(6))
for request in requests:
    assert request["status"] == "succeeded", request
    assert request["prompt_tokens"] > 0, request
    assert request["completion_tokens"] > 0, request
    assert request["total_tokens"] == (
        request["prompt_tokens"] + request["completion_tokens"]
    ), request
    assert request["output_sha256"], request
PY
test -s "$RESULT_DIR/report.md"
test -s "$RESULT_DIR/logs/gateway.log"
test -s "$RESULT_DIR/logs/worker.log"
echo "configured_workloads_ok=1"

echo "[4/4] Verify the model process group was reaped"
assert_reaped "$GATEWAY_PID" "phase-7 gateway"
assert_reaped "$WORKER_PID" "phase-7 worker"
if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
    echo "a staged phase-7 gateway remains alive" >&2
    exit 1
fi
echo "exact_request_count_ok=1"
echo "load_semantics_ok=1"
echo "token_accounting_ok=1"
echo "unified_cli_phase_07_ok=1"
