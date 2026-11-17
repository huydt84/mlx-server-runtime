#!/usr/bin/env bash
# Host-only foreground lifecycle validation for MLX Air.
# Run on Apple Silicon with Metal, uv, curl, and the configured model available.
#
# Usage:
#   bash mlx-host-validation/scripts/cli_serve_foreground.sh [options]
#   bash mlx-host-validation/scripts/cli_serve_foreground.sh --help

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
cli_describe "Validate the mlx-air serve foreground lifecycle: readiness, non-streaming and streaming requests, and SIGTERM shutdown."
cli_option MODEL --model MLX_AIR_SERVE_MODEL string "mlx-community/Qwen2.5-7B-Instruct-4bit" "Model to serve."
cli_option PORT --port MLX_AIR_SERVE_PORT port 18003 "Server port."
cli_option READY_TIMEOUT --ready-timeout MLX_AIR_SERVE_READY_TIMEOUT int 300 \
    "Seconds to wait for the server to report ready."
cli_option REQUEST_TIMEOUT --request-timeout MLX_AIR_SERVE_REQUEST_TIMEOUT int 180 "Seconds allowed per chat request."
cli_option KEEP_WORK_DIR --keep-work-dir MLX_AIR_SERVE_KEEP_WORK_DIR bool 0 \
    "Keep the temporary work directory (staged distribution, logs, captures) after exit."
cli_parse "$@"
cli_require_free_port "$PORT" --port
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/mlx-air-cli-foreground-serve.XXXXXX")"
STAGE_DIR="$WORK_DIR/distribution"
OUTSIDE_DIR="$WORK_DIR/outside-repository"
SERVER_LOG="$WORK_DIR/server.log"
NON_STREAM_RESPONSE="$WORK_DIR/non-stream.json"
STREAM_RESPONSE="$WORK_DIR/stream.txt"
SOCKET_DIR="/tmp/mlx-air-$(id -u)"

cleanup() {
    local status=$?
    set +e
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
        sleep 1
        kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "${WORKER_PID:-}" ]] && kill -0 "$WORKER_PID" >/dev/null 2>&1; then
        kill -KILL "$WORKER_PID" >/dev/null 2>&1 || true
    fi
    if [[ "$status" -ne 0 ]] && [[ -f "$SERVER_LOG" ]]; then
        echo "server log after validation failure:" >&2
        sed -n '1,300p' "$SERVER_LOG" >&2
    fi
    if [[ "$KEEP_WORK_DIR" == "1" ]]; then
        echo "work_dir=$WORK_DIR" >&2
    else
        rm -rf "$WORK_DIR"
    fi
    exit "$status"
}

trap cleanup EXIT

echo "[1/6] Stage the foreground server distribution"
"$ROOT/scripts/stage-mlx-air.sh" --output-dir "$STAGE_DIR"
mkdir -p "$OUTSIDE_DIR"
cd "$OUTSIDE_DIR"

echo "[2/6] Start one real model through mlx-air serve"
"$STAGE_DIR/bin/mlx-air" serve \
    --model "$MODEL" \
    --port "$PORT" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 $((READY_TIMEOUT * 10))); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        echo "server exited before readiness" >&2
        sed -n '1,240p' "$SERVER_LOG" >&2
        exit 1
    fi
    WORKER_PID="$(pgrep -P "$SERVER_PID" | head -n 1 || true)"
    if curl --silent --fail --max-time 1 \
        "http://127.0.0.1:$PORT/ready" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

curl --silent --fail --max-time 2 \
    "http://127.0.0.1:$PORT/ready" >/dev/null
test -n "${WORKER_PID:-}"
echo "server_ready=1"

echo "[3/6] Send one non-streaming request"
curl --silent --show-error --fail --max-time "$REQUEST_TIMEOUT" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: ready\"}],\"max_tokens\":16,\"temperature\":0.0,\"stream\":false}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    >"$NON_STREAM_RESPONSE"
python3 -c 'import json,sys; data=json.load(open(sys.argv[1])); assert data["choices"][0]["message"]["content"]' "$NON_STREAM_RESPONSE"
echo "non_stream_request_ok=1"

echo "[4/6] Send one streaming request"
curl --silent --show-error --fail --max-time "$REQUEST_TIMEOUT" --no-buffer \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: stream\"}],\"max_tokens\":16,\"temperature\":0.0,\"stream\":true}" \
    "http://127.0.0.1:$PORT/v1/chat/completions" \
    >"$STREAM_RESPONSE"
grep -F 'data: [DONE]' "$STREAM_RESPONSE" >/dev/null
echo "stream_request_ok=1"

echo "[5/6] Terminate the foreground server"
kill -TERM "$SERVER_PID"
for _ in $(seq 1 100); do
    if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
if kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    echo "server did not terminate within 10 seconds" >&2
    exit 1
fi
wait "$SERVER_PID"

echo "[6/6] Verify worker and socket cleanup"
if kill -0 "$WORKER_PID" >/dev/null 2>&1; then
    echo "worker remains alive after server termination: $WORKER_PID" >&2
    exit 1
fi
if find "$SOCKET_DIR" -maxdepth 1 -name "foreground-$SERVER_PID-*.sock" -print -quit 2>/dev/null | grep -q .; then
    echo "foreground socket remains after server termination" >&2
    exit 1
fi

echo "worker_reaped=1"
echo "socket_removed=1"
echo "cli_serve_foreground_ok=1"
