#!/usr/bin/env bash
#
# Phase 1.5 host-only MLX validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_1-5.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. The Rust gateway exposes `/live`, `/startup`, `/ready`, `/health`, and `/models*`.
#   3. `/live` reports the process as alive while startup is still in progress.
#   4. A pre-ready inference request is rejected with HTTP 503.
#   5. `/ready` and `/health` flip to healthy after warmup completes.
#   6. `/models`, `/models/{model}/status`, and `/models/{model}/ready` reflect the configured model.
#   7. `POST /v1/chat/completions` succeeds once the model is ready.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_lm_import_ok=1`.
#   - It prints `live_status=live`.
#   - It prints `startup_status=starting` or `startup_status=started`.
#   - It prints `early_inference_rejected=1` when the gateway is probed before readiness.
#   - It prints `ready_status=ready`, `health_response=healthy`, and `assistant_content=` with non-empty text.
#   - It prints model status/ready checks for the configured model.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
MODEL_PATH="mlx-community%2FQwen2.5-7B-Instruct-4bit"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-1-5.log"
LIVE_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-live.json"
STARTUP_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-startup.json"
READY_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-ready.json"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-health.txt"
MODELS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-models.json"
STATUS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-status.json"
READY_MODEL_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-model-ready.json"
EARLY_CHAT_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-early-chat.json"
FINAL_CHAT_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-1-5-chat.json"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

echo "[1/8] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/8] Verify Apple Silicon, mlx, and mlx_lm imports"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

import mlx.core as mx
print("mlx_import_ok=1")

from mlx_lm import load  # noqa: F401
print("mlx_lm_import_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/8] Start gateway"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$LIVE_CAPTURE" "$STARTUP_CAPTURE" "$READY_CAPTURE" "$HEALTH_CAPTURE" "$MODELS_CAPTURE" "$STATUS_CAPTURE" "$READY_MODEL_CAPTURE" "$EARLY_CHAT_CAPTURE" "$FINAL_CHAT_CAPTURE"
cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "[4/8] Wait for /live and /startup"
LIVE_OK=0
for _ in $(seq 1 300); do
    if curl -sS -o "$LIVE_CAPTURE" -w '%{http_code}' http://127.0.0.1:8000/live | grep -qx '200'; then
        uv run python - <<'PY' "$LIVE_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["status"] != "live":
    raise SystemExit(f"unexpected live status: {payload}")
if payload["uptime_seconds"] < 0:
    raise SystemExit("uptime must be non-negative")
print("live_status=live")
PY
        LIVE_OK=1
        break
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

if [[ "$LIVE_OK" != "1" ]]; then
    echo "timed out waiting for /live; inspect $GATEWAY_LOG" >&2
    exit 1
fi

STARTUP_STATUS=""
for _ in $(seq 1 300); do
    HTTP_STATUS=$(curl -sS -o "$STARTUP_CAPTURE" -w '%{http_code}' http://127.0.0.1:8000/startup || true)
    if [[ "$HTTP_STATUS" == "200" ]]; then
        STARTUP_STATUS=$(uv run python - <<'PY' "$STARTUP_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = payload["status"]
print(status)
if status not in {"starting", "started"}:
    raise SystemExit(f"unexpected startup status: {payload}")
PY
)
        echo "startup_status=$STARTUP_STATUS"
        if [[ "$STARTUP_STATUS" == "started" ]]; then
            break
        fi

        EARLY_HTTP_STATUS=$(curl -sS -o "$EARLY_CHAT_CAPTURE" -w '%{http_code}' \
            -X POST http://127.0.0.1:8000/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -d '{
              "model": "'"$MODEL_NAME"'",
              "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
              "max_tokens": 8,
              "temperature": 0.0,
              "top_p": 1.0,
              "stream": false
            }' || true)
        if [[ "$EARLY_HTTP_STATUS" != "503" ]]; then
            echo "expected HTTP 503 for early inference request, got $EARLY_HTTP_STATUS; inspect $EARLY_CHAT_CAPTURE and $GATEWAY_LOG" >&2
            exit 1
        fi

        uv run python - <<'PY' "$EARLY_CHAT_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
error = payload["error"]
if error["code"] not in {"MODEL_NOT_READY", "MODEL_LOAD_FAILED"}:
    raise SystemExit(f"unexpected early inference error: {payload}")
print("early_inference_rejected=1")
PY
        break
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

if [[ -z "$STARTUP_STATUS" ]]; then
    echo "timed out waiting for /startup; inspect $GATEWAY_LOG" >&2
    exit 1
fi

echo "[5/8] Wait for /ready to report ready"
READY_OK=0
for _ in $(seq 1 300); do
    HTTP_STATUS=$(curl -sS -o "$READY_CAPTURE" -w '%{http_code}' http://127.0.0.1:8000/ready || true)
    if [[ "$HTTP_STATUS" == "200" ]]; then
        uv run python - <<'PY' "$READY_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["status"] != "ready":
    raise SystemExit(f"unexpected ready payload: {payload}")
print("ready_status=ready")
PY
        READY_OK=1
        break
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

if [[ "$READY_OK" != "1" ]]; then
    echo "timed out waiting for /ready; inspect $GATEWAY_LOG" >&2
    exit 1
fi

echo "[6/8] Verify /health, /models, and model status routes"
curl -fsS http://127.0.0.1:8000/health >"$HEALTH_CAPTURE"
grep -qx 'healthy' "$HEALTH_CAPTURE"
echo "health_response=healthy"

curl -fsS http://127.0.0.1:8000/models >"$MODELS_CAPTURE"
uv run python - <<'PY' "$MODELS_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
models = payload["models"]
if not models:
    raise SystemExit("expected at least one model")
model = models[0]
if model["name"] != "mlx-community/Qwen2.5-7B-Instruct-4bit":
    raise SystemExit(f"unexpected model list: {payload}")
if model["state"] != "ready":
    raise SystemExit(f"expected ready model summary: {payload}")
print(f"models_status={model['state']}")
PY

curl -fsS "http://127.0.0.1:8000/models/$MODEL_PATH/status" >"$STATUS_CAPTURE"
uv run python - <<'PY' "$STATUS_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["state"] != "ready":
    raise SystemExit(f"unexpected model status: {payload}")
if not payload["warmup_passed"]:
    raise SystemExit(f"expected warmup to pass: {payload}")
print(f"model_status={payload['state']}")
PY

curl -fsS "http://127.0.0.1:8000/models/$MODEL_PATH/ready" >"$READY_MODEL_CAPTURE"
uv run python - <<'PY' "$READY_MODEL_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["ready"] is not True:
    raise SystemExit(f"expected model-ready payload: {payload}")
print("model_ready=1")
PY

echo "[7/8] Run one non-streaming chat completion"
HTTP_STATUS=$(curl -sS -o "$FINAL_CHAT_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
      "max_tokens": 32,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status: $HTTP_STATUS; inspect $FINAL_CHAT_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$FINAL_CHAT_CAPTURE"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("assistant content was empty")
print(f"assistant_content={content}")
PY

echo "[8/8] Done"
echo "gateway_log=$GATEWAY_LOG"
