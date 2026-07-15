#!/usr/bin/env bash
# Host-only benchmark execution and process-cleanup validation for MLX Air.
# Run on Apple Silicon with Metal, uv, and the built-in smoke model available.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_ROOT="${TMPDIR:-/tmp}"
WORK_DIR="$(mktemp -d "${TMP_ROOT%/}/mlx-air-unified-cli-phase-06.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
TEST_HOME="$WORK_DIR/home"
SUCCESS_DIR="$WORK_DIR/success"
INTERRUPT_DIR="$WORK_DIR/interrupted"
SUCCESS_LOG="$WORK_DIR/success-command.log"
INTERRUPT_LOG="$WORK_DIR/interrupt-command.log"
UV_CACHE="${MLX_AIR_PHASE6_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"
HOST_HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
FIXTURE="$ROOT/mlx-host-validation/fixtures/unified_cli_phase_06.toml"

cleanup() {
    local status=$?
    set +e
    for pid in "${SUCCESS_PID:-}" "${INTERRUPT_PID:-}" "${SUCCESS_GATEWAY_PID:-}" "${INTERRUPT_GATEWAY_PID:-}"; do
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
        echo "successful-run command log after validation failure:" >&2
        sed -n '1,300p' "$SUCCESS_LOG" >&2
        echo "interrupted-run command log after validation failure:" >&2
        sed -n '1,300p' "$INTERRUPT_LOG" >&2
        echo "successful-run gateway log after validation failure:" >&2
        sed -n '1,300p' "$SUCCESS_DIR/logs/gateway.log" >&2
        echo "successful-run worker log after validation failure:" >&2
        sed -n '1,300p' "$SUCCESS_DIR/logs/worker.log" >&2
        echo "successful-run results after validation failure:" >&2
        sed -n '1,500p' "$SUCCESS_DIR/results.json" >&2
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

assert_reaped() {
    local pid="$1"
    local label="$2"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
        echo "$label remains alive: $pid" >&2
        exit 1
    fi
}

echo "[1/6] Stage the benchmark-capable distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR" "$TEST_HOME"
cd "$OUTSIDE_DIR"

echo "[2/6] Run the bounded self-launched smoke workload"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench run \
    --suite phase6 \
    --benchmark-config "$FIXTURE" \
    --output-dir "$SUCCESS_DIR" \
    >"$SUCCESS_LOG" 2>&1 &
SUCCESS_PID=$!
SUCCESS_GATEWAY_PID="$(wait_for_gateway "$SUCCESS_PID")"
SUCCESS_WORKER_PID="$(pgrep -P "$SUCCESS_GATEWAY_PID" | head -n 1 || true)"
wait_bounded "$SUCCESS_PID" "successful benchmark"
unset SUCCESS_PID

echo "[3/6] Verify successful result data and logs"
python3 - "$SUCCESS_DIR/results.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text())
assert data["status"] == "succeeded", data
assert data["failure_stage"] is None, data
assert not pathlib.Path(data["server"]["runtime_configuration"]["worker"]["ipc_path"]).exists(), data
assert len(data["trials"]) == 2, data
assert [trial["request_count"] for trial in data["trials"]] == [2, 4], data
requests = [request for trial in data["trials"] for request in trial["requests"]]
assert len(requests) == 6, data
assert [request["request_order"] for request in requests] == list(range(6)), data
for request in requests:
    assert request["status"] == "succeeded", request
    assert request["prompt_tokens"] > 0, request
    assert request["completion_tokens"] > 0, request
    assert request["output_sha256"], request
    for field in (
        "submitted_monotonic_ns",
        "first_byte_monotonic_ns",
        "first_token_monotonic_ns",
        "final_token_monotonic_ns",
        "completed_monotonic_ns",
    ):
        assert isinstance(request[field], int), request
PY
test -s "$SUCCESS_DIR/report.md"
test -s "$SUCCESS_DIR/logs/gateway.log"
test -f "$SUCCESS_DIR/logs/worker.log"
assert_reaped "$SUCCESS_GATEWAY_PID" "successful-run gateway"
assert_reaped "$SUCCESS_WORKER_PID" "successful-run worker"
echo "bounded_run_ok=1"

echo "[4/6] Interrupt a second self-launched run during startup"
HOME="$TEST_HOME" HF_HOME="$HOST_HF_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" bench run \
    --suite phase6 \
    --benchmark-config "$FIXTURE" \
    --output-dir "$INTERRUPT_DIR" \
    >"$INTERRUPT_LOG" 2>&1 &
INTERRUPT_PID=$!
INTERRUPT_GATEWAY_PID="$(wait_for_gateway "$INTERRUPT_PID")"
INTERRUPT_WORKER_PID="$(pgrep -P "$INTERRUPT_GATEWAY_PID" | head -n 1 || true)"
kill -INT "$INTERRUPT_PID"
if wait_bounded "$INTERRUPT_PID" "interrupted benchmark"; then
    echo "interrupted benchmark unexpectedly exited successfully" >&2
    exit 1
fi
unset INTERRUPT_PID

echo "[5/6] Verify interrupted result and process cleanup"
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
assert_reaped "$INTERRUPT_GATEWAY_PID" "interrupted-run gateway"
assert_reaped "$INTERRUPT_WORKER_PID" "interrupted-run worker"
echo "interrupt_cleanup_ok=1"

echo "[6/6] Confirm no staged gateway remains"
if pgrep -f "$STAGE_DIR/bin/mlx_runtime_gateway" >/dev/null 2>&1; then
    echo "a staged benchmark gateway remains alive" >&2
    exit 1
fi
echo "all_children_reaped=1"
echo "unified_cli_phase_06_ok=1"
