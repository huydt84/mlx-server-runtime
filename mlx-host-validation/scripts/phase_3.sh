#!/usr/bin/env bash
#
# Phase 3 host-only MLX validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_3.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. The Rust gateway starts successfully with a phase-specific config.
#   3. The `/health` endpoint becomes healthy after worker startup.
#   4. A second request gets `HTTP 429 Too Many Requests` while first request occupies active slot.
#   5. Killing first stream frees slot and next request succeeds.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_lm_import_ok=1`.
#   - It prints `health_response=healthy`.
#   - It prints `queue_overflow_rejected=1`.
#   - It prints `disconnect_cancelled=1`.
#   - It prints `assistant_content=` for first successful request.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-phase-3"
CONFIG_PATH="$TMP_ROOT/runtime.toml"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-3.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-3-health.txt"
STREAM_ONE_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-3-stream-1.txt"
STREAM_TWO_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-3-stream-2.txt"
STREAM_THREE_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-3-stream-3.txt"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

mkdir -p "$TMP_ROOT"
cp "$ROOT/config/runtime.toml" "$CONFIG_PATH"
python3 - <<'PY' "$CONFIG_PATH"
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
text = re.sub(r"max_pending_requests = \d+", "max_pending_requests = 0", text)
text = re.sub(r"max_active_requests = \d+", "max_active_requests = 1", text)
text = re.sub(r"max_prompt_tokens = \d+", "max_prompt_tokens = 32768", text)
text = re.sub(r"max_completion_tokens = \d+", "max_completion_tokens = 4096", text)
text = re.sub(
    r"max_total_tokens_per_request = \d+",
    "max_total_tokens_per_request = 65536",
    text,
)
text = re.sub(r"request_timeout_seconds = \d+", "request_timeout_seconds = 5", text)
path.write_text(text)
PY

echo "[1/5] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/5] Verify Apple Silicon, mlx, and mlx_lm imports"
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

echo "[3/5] Start gateway"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE" "$STREAM_ONE_CAPTURE" "$STREAM_TWO_CAPTURE" "$STREAM_THREE_CAPTURE"
MLX_RUNTIME_CONFIG="$CONFIG_PATH" cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "[4/5] Wait for /health to report healthy"
for _ in $(seq 1 300); do
    if curl -fsS http://127.0.0.1:8000/health >"$HEALTH_CAPTURE"; then
        if grep -qx 'healthy' "$HEALTH_CAPTURE"; then
            echo "health_response=healthy"
            break
        fi
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

grep -qx 'healthy' "$HEALTH_CAPTURE"

echo "[5/5] Provoke queue overflow"
# Start streaming request with a prompt that forces multi-token generation.
curl -sS -N -o "$STREAM_ONE_CAPTURE" \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Count from 1 to 30, listing each number on its own line."}],
      "max_tokens": 128,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": true
    }' &
FIRST_PID=$!

# Give first request time to reach server, acquire permit, and start generating.
sleep 1

# Second request should be rejected — all slots busy.
SECOND_STATUS=$(curl -sS -o "$STREAM_TWO_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Write another short greeting."}],
      "max_tokens": 32,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$SECOND_STATUS" != "429" ]]; then
    echo "expected HTTP 429 for queued request, got $SECOND_STATUS; inspect $STREAM_TWO_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

python3 - <<'PY' "$STREAM_TWO_CAPTURE"
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
if payload["error"]["code"] != "QUEUE_FULL":
    raise SystemExit(f"unexpected queue overflow payload: {payload}")
print("queue_overflow_rejected=1")
PY

# Kill first stream → Rust detects disconnect → cancels worker → releases permit.
kill "$FIRST_PID" >/dev/null 2>&1 || true
wait "$FIRST_PID" >/dev/null 2>&1 || true

# Poll until the slot opens (worker may still be generating).
for _ in $(seq 1 30); do
    THIRD_STATUS=$(curl -sS -o "$STREAM_THREE_CAPTURE" -w '%{http_code}' \
        -X POST http://127.0.0.1:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
          "model": "'"$MODEL_NAME"'",
          "messages": [{"role": "user", "content": "Write a short farewell."}],
          "max_tokens": 32,
          "temperature": 0.0,
          "top_p": 1.0,
          "stream": false
        }')
    if [[ "$THIRD_STATUS" == "200" ]]; then
        break
    fi
    sleep 1
done

if [[ "$THIRD_STATUS" != "200" ]]; then
    echo "expected HTTP 200 after disconnect cancel, got $THIRD_STATUS; inspect $STREAM_THREE_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

python3 - <<'PY' "$STREAM_THREE_CAPTURE"
from pathlib import Path
import json
import sys

payload = json.loads(Path(sys.argv[1]).read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("assistant content after cancel was empty")
print("disconnect_cancelled=1")
PY

uv run python - <<'PY' "$STREAM_ONE_CAPTURE"
from pathlib import Path
import json
import sys

content = []
for line in Path(sys.argv[1]).read_text().splitlines():
    if not line.startswith("data: "):
        continue
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        continue
    event = json.loads(payload)
    content.append(event["choices"][0]["delta"].get("content", ""))

assistant_content = "".join(content).strip()
if not assistant_content:
    raise SystemExit("assistant content was empty")
print(f"assistant_content={assistant_content}")
PY

echo "gateway_log=$GATEWAY_LOG"
