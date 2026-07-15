#!/usr/bin/env bash
# Host-only launchd lifecycle validation for managed MLX Air instances.
# Run from an interactive macOS user session with Apple Silicon, Metal, uv, and curl.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-unified-cli-phase-04.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
TEST_HOME="$WORK_DIR/home"
START_OUTPUT="$WORK_DIR/start.txt"
PS_OUTPUT="$WORK_DIR/ps.txt"
ATTACH_OUTPUT="$WORK_DIR/attach.txt"
RESPONSE="$WORK_DIR/response.json"
MODEL="${MLX_AIR_PHASE4_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
PORT="${MLX_AIR_PHASE4_PORT:-18004}"
NAME="phase04-$$"
LABEL="com.mlx-air.instance.$NAME"
SERVICE_TARGET="gui/$(id -u)/$LABEL"
INSTANCE_DIR="$TEST_HOME/Library/Application Support/mlx-air/instances/$NAME"
STDOUT_LOG="$TEST_HOME/Library/Logs/mlx-air/$NAME.stdout.log"
STDERR_LOG="$TEST_HOME/Library/Logs/mlx-air/$NAME.stderr.log"
SOCKET="/tmp/mlx-air-$(id -u)/instance-$NAME.sock"
UV_CACHE="${MLX_AIR_PHASE4_UV_CACHE_DIR:-$HOME/Library/Caches/uv}"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${ATTACH_PID:-}" ]] && kill -0 "$ATTACH_PID" >/dev/null 2>&1; then
        kill -INT "$ATTACH_PID" >/dev/null 2>&1
        wait "$ATTACH_PID" >/dev/null 2>&1
    fi
    if [[ -x "$STAGE_DIR/bin/mlx-air" ]] && [[ -d "$INSTANCE_DIR" ]]; then
        HOME="$TEST_HOME" "$STAGE_DIR/bin/mlx-air" stop "$NAME" >/dev/null 2>&1
    fi
    if [[ "$status" -ne 0 ]]; then
        echo "managed stdout log after validation failure:" >&2
        sed -n '1,200p' "$STDOUT_LOG" >&2
        echo "managed stderr log after validation failure:" >&2
        sed -n '1,300p' "$STDERR_LOG" >&2
    fi
    rm -rf "$WORK_DIR"
    exit "$status"
}

trap cleanup EXIT

echo "[1/7] Stage the managed-service distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$TEST_HOME"

echo "[2/7] Start one detached instance and wait for readiness"
HOME="$TEST_HOME" UV_CACHE_DIR="$UV_CACHE" \
    "$STAGE_DIR/bin/mlx-air" serve \
    --model "$MODEL" \
    --port "$PORT" \
    --detach \
    --name "$NAME" | tee "$START_OUTPUT"
grep -F "Started $NAME at http://127.0.0.1:$PORT" "$START_OUTPUT" >/dev/null
test -f "$INSTANCE_DIR/config.toml"
test -f "$INSTANCE_DIR/instance.json"
test -f "$INSTANCE_DIR/launch-agent.plist"
SERVER_PID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "$INSTANCE_DIR/instance.json")"
kill -0 "$SERVER_PID"
echo "detached_start_ok=1"

echo "[3/7] Validate managed-instance status"
HOME="$TEST_HOME" "$STAGE_DIR/bin/mlx-air" ps | tee "$PS_OUTPUT"
grep -F "$NAME" "$PS_OUTPUT" >/dev/null
grep -F "http://127.0.0.1:$PORT" "$PS_OUTPUT" >/dev/null
grep -F "running" "$PS_OUTPUT" >/dev/null
echo "ps_ok=1"

echo "[4/7] Bound attach with SIGINT without stopping the service"
HOME="$TEST_HOME" "$STAGE_DIR/bin/mlx-air" attach "$NAME" >"$ATTACH_OUTPUT" 2>&1 &
ATTACH_PID=$!
sleep 1
kill -INT "$ATTACH_PID"
wait "$ATTACH_PID"
unset ATTACH_PID
test -s "$ATTACH_OUTPUT"
curl --silent --fail --max-time 2 "http://127.0.0.1:$PORT/ready" >/dev/null
kill -0 "$SERVER_PID"
echo "attach_interrupt_ok=1"

echo "[5/7] Serve one request through the detached instance"
curl --silent --show-error --fail --max-time 180 \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: managed\"}],\"max_tokens\":16,\"temperature\":0.0,\"stream\":false}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" >"$RESPONSE"
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); assert data["choices"][0]["message"]["content"]' "$RESPONSE"
echo "managed_request_ok=1"

echo "[6/7] Stop the instance while preserving logs"
HOME="$TEST_HOME" "$STAGE_DIR/bin/mlx-air" stop "$NAME"
test ! -d "$INSTANCE_DIR"
test -f "$STDOUT_LOG"
test -f "$STDERR_LOG"
if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "managed gateway remains alive after stop: $SERVER_PID" >&2
    exit 1
fi
test ! -e "$SOCKET"
if launchctl print "$SERVICE_TARGET" >/dev/null 2>&1; then
    echo "launchd service remains loaded after stop: $SERVICE_TARGET" >&2
    exit 1
fi
echo "managed_stop_ok=1"

echo "[7/7] Confirm active registration is gone"
HOME="$TEST_HOME" "$STAGE_DIR/bin/mlx-air" ps | tee "$PS_OUTPUT"
if grep -F "$NAME" "$PS_OUTPUT" >/dev/null; then
    echo "stopped instance remains in mlx-air ps" >&2
    exit 1
fi
echo "registration_removed=1"
echo "logs_preserved=1"
echo "unified_cli_phase_04_ok=1"
