#!/usr/bin/env bash
#
# Phase 2 host-only MLX validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_2.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. The Rust gateway starts successfully with the configured Python worker.
#   3. The `/health` endpoint becomes healthy after worker startup.
#   4. A streamed chat completion returns SSE chunks and a terminal `[DONE]` event.
#   5. The streamed response contains non-empty assistant text.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_lm_import_ok=1`.
#   - It prints `health_response=healthy`.
#   - It prints `stream_chunk_count=` with at least one chunk.
#   - It prints `stream_done=1`.
#   - It prints `assistant_content=` with non-empty streamed text.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-2.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-2-health.txt"
STREAM_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-2-stream.txt"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

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
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE" "$STREAM_CAPTURE"
cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
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

echo "[5/5] Run one streaming chat completion"
HTTP_STATUS=$(curl -sS -N -o "$STREAM_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
      "max_tokens": 32,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": true
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status: $HTTP_STATUS; inspect $STREAM_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$STREAM_CAPTURE"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
chunks = []
content = []
done = 0

for line in lines:
    if not line.startswith("data: "):
        continue
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        done += 1
        continue
    event = json.loads(payload)
    if event["object"] != "chat.completion.chunk":
        raise SystemExit(f"unexpected SSE payload: {event}")
    chunks.append(event)
    delta = event["choices"][0]["delta"]
    content.append(delta.get("content", ""))

if not chunks:
    raise SystemExit("expected at least one streamed chunk")
if done != 1:
    raise SystemExit(f"expected exactly one [DONE] marker, saw {done}")

assistant_content = "".join(content).strip()
if not assistant_content:
    raise SystemExit("assistant content was empty")

print(f"stream_chunk_count={len(chunks)}")
print("stream_done=1")
print(f"assistant_content={assistant_content}")
PY

echo "gateway_log=$GATEWAY_LOG"
