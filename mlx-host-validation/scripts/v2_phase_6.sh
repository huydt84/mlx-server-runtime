#!/usr/bin/env bash
#
# native-v2 Phase 6 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_6.sh
#   MLX_V2_PHASE6_CHECKPOINT=mlx-community/Qwen3-4B-Instruct-2507-4bit \
#     bash mlx-host-validation/scripts/v2_phase_6.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - native-v2 public gateway requests against `mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - set `MLX_V2_PHASE6_CHECKPOINT` to probe the registered Qwen3, Gemma3, or LFM2 family
#   - default v1 gateway non-regression request against same checkpoint
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#   - `cargo` toolchain for `mlx_runtime_gateway`
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_lm_import_ok=1`
#   - `native_non_stream_ok=1`
#   - `native_stream_ok=1`
#   - `native_stop_ok=1`
#   - `native_length_ok=1`
#   - `native_cancel_ok=1`
#   - `native_backend_metrics_ok=1`
#   - `v1_non_regression_ok=1`
#   - `phase_6_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly
#   - missing non-empty output, `[DONE]`, `finish_reason`, cancellation cleanup, or backend labels
#   - missing native cancellation metrics or failed v1 follow-up request

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
CHECKPOINT="${MLX_V2_PHASE6_CHECKPOINT:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
NATIVE_LOG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-native.log"
V1_LOG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-v1.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-health.txt"
NON_STREAM_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-non-stream.json"
STREAM_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-stream.txt"
STOP_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-stop.json"
LENGTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-length.json"
CANCEL_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-cancel.txt"
FOLLOW_UP_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-follow-up.json"
METRICS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-metrics.txt"
V1_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-v1.json"
REQUEST_DIR="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-requests"
NATIVE_CONFIG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-native.toml"
V1_CONFIG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-6-v1.toml"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
mkdir -p "$REQUEST_DIR"

GATEWAY_PID=""
STREAM_PID=""

cleanup() {
    if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" >/dev/null 2>&1; then
        kill "$STREAM_PID" >/dev/null 2>&1 || true
        wait "$STREAM_PID" >/dev/null 2>&1 || true
    fi
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_healthy() {
    local log_path="$1"
    rm -f "$HEALTH_CAPTURE"
    for _ in $(seq 1 300); do
        if [[ -n "$GATEWAY_PID" ]] && ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
            echo "gateway exited unexpectedly; inspect $log_path" >&2
            return 1
        fi
        if curl -fsS http://127.0.0.1:8000/health >"$HEALTH_CAPTURE"; then
            if grep -qx 'healthy' "$HEALTH_CAPTURE"; then
                return 0
            fi
        fi
        sleep 1
    done
    echo "gateway did not become healthy; inspect $log_path" >&2
    return 1
}

start_gateway() {
    local log_path="$1"
    shift
    if lsof -nP -iTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port 8000 is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env "$@" "$GATEWAY_BIN"
    ) >"$log_path" 2>&1 &
    GATEWAY_PID=$!
    wait_healthy "$log_path"
}

stop_gateway() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
    GATEWAY_PID=""
}

write_request() {
    local path="$1"
    local prompt="$2"
    local max_tokens="$3"
    local stream_flag="$4"
    local stop_value="${5:-}"
    uv --directory "$PYTHON_DIR" run python - <<'PY' "$path" "$CHECKPOINT" "$prompt" "$max_tokens" "$stream_flag" "$stop_value"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "model": sys.argv[2],
    "messages": [{"role": "user", "content": sys.argv[3]}],
    "max_tokens": int(sys.argv[4]),
    "temperature": 0.0,
    "top_p": 1.0,
    "stream": sys.argv[5].lower() == "true",
}
if sys.argv[6]:
    payload["stop"] = [sys.argv[6]]
path.write_text(json.dumps(payload))
PY
}

echo "[1/6] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev
cargo build --manifest-path "$ROOT/Cargo.toml" -p mlx_runtime_gateway

echo "[2/6] Verify Apple Silicon, mlx, and mlx_lm imports"
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

echo "[3/6] Start native-mlx gateway and run real public requests"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$ROOT/config/runtime.toml" "$NATIVE_CONFIG" "$V1_CONFIG" "$CHECKPOINT"
from __future__ import annotations

import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
native_target = pathlib.Path(sys.argv[2])
v1_target = pathlib.Path(sys.argv[3])
checkpoint = sys.argv[4]
native_target.write_text(
    source.replace('backend = "v1"', 'backend = "native-mlx"').replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
)
v1_target.write_text(
    source.replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
)
PY

write_request "$REQUEST_DIR/non_stream.json" "Say hello in one short sentence." 16 false
write_request "$REQUEST_DIR/stream.json" "Count from one to six with spaces." 16 true
write_request "$REQUEST_DIR/stop.json" "Reply exactly with HELLO STOP_MARKER NOW." 64 false "STOP_MARKER"
write_request "$REQUEST_DIR/length.json" "Count upward forever using short tokens." 1 false
write_request "$REQUEST_DIR/cancel.json" "Write a long alphabetic sequence with many short chunks." 64 true

# Keep enough paged KV capacity for the larger probe families.  The worker's
# 8 MiB default is sufficient for the tiny default smoke test but can exhaust
# before the cancellation request on a 4B model with a 64-token budget.
start_gateway "$NATIVE_LOG" \
    MLX_RUNTIME_CONFIG="$NATIVE_CONFIG" \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES=$((32 * 1024 * 1024))

NON_STREAM_STATUS=$(curl -sS -o "$NON_STREAM_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/non_stream.json")
if [[ "$NON_STREAM_STATUS" != "200" ]]; then
    echo "unexpected native non-stream HTTP status: $NON_STREAM_STATUS; inspect $NON_STREAM_CAPTURE and $NATIVE_LOG" >&2
    exit 1
fi

STREAM_STATUS=$(curl -sS -N -o "$STREAM_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    --data-binary "@$REQUEST_DIR/stream.json")
if [[ "$STREAM_STATUS" != "200" ]]; then
    echo "unexpected native stream HTTP status: $STREAM_STATUS; inspect $STREAM_CAPTURE and $NATIVE_LOG" >&2
    exit 1
fi

STOP_STATUS=$(curl -sS -o "$STOP_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/stop.json")
if [[ "$STOP_STATUS" != "200" ]]; then
    echo "unexpected native stop HTTP status: $STOP_STATUS; inspect $STOP_CAPTURE and $NATIVE_LOG" >&2
    exit 1
fi

LENGTH_STATUS=$(curl -sS -o "$LENGTH_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/length.json")
if [[ "$LENGTH_STATUS" != "200" ]]; then
    echo "unexpected native length HTTP status: $LENGTH_STATUS; inspect $LENGTH_CAPTURE and $NATIVE_LOG" >&2
    exit 1
fi

uv --directory "$PYTHON_DIR" run python - <<'PY' "$REQUEST_DIR/cancel.json" "$CANCEL_CAPTURE"
from __future__ import annotations

import http.client
import pathlib
import sys

request_path = pathlib.Path(sys.argv[1])
capture_path = pathlib.Path(sys.argv[2])
body = request_path.read_text()

conn = http.client.HTTPConnection("127.0.0.1", 8000, timeout=30)
conn.putrequest("POST", "/v1/chat/completions")
conn.putheader("Content-Type", "application/json")
conn.putheader("Accept", "text/event-stream")
conn.putheader("Content-Length", str(len(body.encode("utf-8"))))
conn.endheaders()
conn.send(body.encode("utf-8"))
response = conn.getresponse()
if response.status != 200:
    raise SystemExit(f"unexpected cancel stream HTTP status: {response.status}")

chunks: list[bytes] = []
while True:
    line = response.fp.readline()
    if not line:
        break
    chunks.append(line)
    if line.startswith(b"data: "):
        break

capture_path.write_bytes(b"".join(chunks))
conn.close()
print("native_disconnect_client_closed=1")
PY

sleep 1

FOLLOW_UP_STATUS=$(curl -sS -o "$FOLLOW_UP_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/non_stream.json")
if [[ "$FOLLOW_UP_STATUS" != "200" ]]; then
    echo "unexpected follow-up HTTP status after cancel: $FOLLOW_UP_STATUS; inspect $FOLLOW_UP_CAPTURE and $NATIVE_LOG" >&2
    exit 1
fi

curl -fsS http://127.0.0.1:8000/metrics >"$METRICS_CAPTURE"

uv --directory "$PYTHON_DIR" run python - <<'PY' \
    "$NON_STREAM_CAPTURE" "$STREAM_CAPTURE" "$STOP_CAPTURE" "$LENGTH_CAPTURE" \
    "$FOLLOW_UP_CAPTURE" "$METRICS_CAPTURE"
from __future__ import annotations

import json
import pathlib
import sys

non_stream = json.loads(pathlib.Path(sys.argv[1]).read_text())
stream_text = pathlib.Path(sys.argv[2]).read_text()
stop_payload = json.loads(pathlib.Path(sys.argv[3]).read_text())
length_payload = json.loads(pathlib.Path(sys.argv[4]).read_text())
follow_up = json.loads(pathlib.Path(sys.argv[5]).read_text())
metrics = pathlib.Path(sys.argv[6]).read_text()

non_stream_content = non_stream["choices"][0]["message"]["content"].strip()
if not non_stream_content:
    raise SystemExit("native non-stream response was empty")
print(f"native_non_stream_content={non_stream_content}")
print("native_non_stream_ok=1")

if "data: [DONE]" not in stream_text:
    raise SystemExit("native stream missing [DONE]")
if stream_text.count("chat.completion.chunk") < 2:
    raise SystemExit("native stream emitted too few chunks")
print(f"native_stream_chunk_count={stream_text.count('chat.completion.chunk')}")
print("native_stream_ok=1")

stop_choice = stop_payload["choices"][0]
stop_content = stop_choice["message"]["content"]
if stop_choice["finish_reason"] != "stop":
    raise SystemExit("native stop request did not finish with stop")
if "STOP_MARKER" in stop_content:
    raise SystemExit("native stop request leaked stop marker into response text")
if not stop_content.strip():
    raise SystemExit("native stop request returned empty content")
print(f"native_stop_content={stop_content.strip()}")
print("native_stop_ok=1")

length_choice = length_payload["choices"][0]
if length_choice["finish_reason"] != "length":
    raise SystemExit("native max-token request did not finish with length")
if length_payload["usage"]["completion_tokens"] != 1:
    raise SystemExit("native max-token request returned unexpected completion_tokens")
print("native_length_ok=1")

follow_up_content = follow_up["choices"][0]["message"]["content"].strip()
if not follow_up_content:
    raise SystemExit("follow-up request after cancellation was empty")
if 'backend="native-mlx"' not in metrics:
    raise SystemExit("metrics missing native-mlx backend label")
if 'mlx_requests_cancelled_total' not in metrics:
    raise SystemExit("metrics missing cancellation counter")
if 'mlx_worker_cancellations_by_backend_total{backend="native-mlx"' not in metrics:
    raise SystemExit("metrics missing native worker cancellation label")
print("native_cancel_ok=1")
print("native_backend_metrics_ok=1")
PY

echo "[4/6] Stop native gateway"
stop_gateway

echo "[5/6] Start default v1 gateway for non-regression"
write_request "$REQUEST_DIR/v1.json" "Say hello in one short sentence." 16 false
start_gateway "$V1_LOG" MLX_RUNTIME_CONFIG="$V1_CONFIG"

V1_STATUS=$(curl -sS -o "$V1_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/v1.json")
if [[ "$V1_STATUS" != "200" ]]; then
    echo "unexpected v1 HTTP status: $V1_STATUS; inspect $V1_CAPTURE and $V1_LOG" >&2
    exit 1
fi

uv --directory "$PYTHON_DIR" run python - <<'PY' "$V1_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("v1 non-regression response was empty")
print(f"v1_non_regression_content={content}")
print("v1_non_regression_ok=1")
PY

echo "[6/6] Phase 6 host validation complete"
echo "native_gateway_log=$NATIVE_LOG"
echo "v1_gateway_log=$V1_LOG"
echo "phase_6_validation_ok=1"
