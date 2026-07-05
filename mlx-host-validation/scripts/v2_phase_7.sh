#!/usr/bin/env bash
#
# native-v2 Phase 7 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_7.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - overlapping native-v2 public gateway streaming requests against `mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - one native-v2 cancelled public stream while other requests continue
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
#   - `native_batch_grow_ok=1` (scheduler batch membership)
#   - `native_batch_shrink_ok=1` (scheduler batch membership)
#   - `native_independent_finish_ok=1`
#   - `native_cancel_isolated_ok=1`
#   - `native_scheduler_metrics_ok=1`
#   - `v1_non_regression_ok=1`
#   - `phase_7_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly
#   - missing overlapping decode batch metrics or queue-count shrink
#   - cancelled request corrupts surviving streams or follow-up metrics remain stuck

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
CHECKPOINT="mlx-community/Qwen2.5-7B-Instruct-4bit"
NATIVE_PORT=18180
V1_PORT=18181
NATIVE_LOG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-native.log"
V1_LOG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-v1.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-health.txt"
V1_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-v1.json"
REQUEST_DIR="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-requests"
NATIVE_CONFIG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-native.toml"
V1_CONFIG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-v1.toml"
OVERLAP_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-overlap.json"
METRICS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-7-metrics.txt"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
mkdir -p "$REQUEST_DIR"

GATEWAY_PID=""

cleanup() {
    if [[ -n "$GATEWAY_PID" ]] && kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

wait_healthy() {
    local log_path="$1"
    local port="$2"
    rm -f "$HEALTH_CAPTURE"
    for _ in $(seq 1 300); do
        if [[ -n "$GATEWAY_PID" ]] && ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
            echo "gateway exited unexpectedly; inspect $log_path" >&2
            return 1
        fi
        if curl -fsS "http://127.0.0.1:${port}/health" >"$HEALTH_CAPTURE"; then
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
    local port="$2"
    shift
    shift
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port $port is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env "$@" "$GATEWAY_BIN"
    ) >"$log_path" 2>&1 &
    GATEWAY_PID=$!
    wait_healthy "$log_path" "$port"
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
    uv --directory "$PYTHON_DIR" run python - <<'PY' "$path" "$CHECKPOINT" "$prompt" "$max_tokens" "$stream_flag"
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

echo "[3/6] Start native-mlx gateway and probe overlapping public requests"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$ROOT/config/runtime.toml" "$NATIVE_CONFIG" "$V1_CONFIG" "$CHECKPOINT" "$NATIVE_PORT" "$V1_PORT"
from __future__ import annotations

import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
native_target = pathlib.Path(sys.argv[2])
v1_target = pathlib.Path(sys.argv[3])
checkpoint = sys.argv[4]
native_port = sys.argv[5]
v1_port = sys.argv[6]
native_target.write_text(
    source.replace('port = 8000', f'port = {native_port}').replace(
        'backend = "v1"', 'backend = "native-mlx"'
    ).replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    ).replace(
        'ipc_path = "/tmp/mlx-runtime.sock"',
        f'ipc_path = "/tmp/mlx-runtime-phase7-native-{native_port}.sock"',
    )
)
v1_target.write_text(
    source.replace('port = 8000', f'port = {v1_port}').replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    ).replace(
        'ipc_path = "/tmp/mlx-runtime.sock"',
        f'ipc_path = "/tmp/mlx-runtime-phase7-v1-{v1_port}.sock"',
    )
)
PY

write_request "$REQUEST_DIR/stream_a.json" "Count upward in many short comma-separated tokens until you reach forty." 64 true
write_request "$REQUEST_DIR/stream_b.json" "List uppercase letters with spaces and keep going until you run out of budget." 64 true
write_request "$REQUEST_DIR/cancel.json" "Write many short numbered tokens with spaces until budget ends." 96 true
write_request "$REQUEST_DIR/v1.json" "Say hello in one short sentence." 16 false

start_gateway "$NATIVE_LOG" "$NATIVE_PORT" MLX_RUNTIME_CONFIG="$NATIVE_CONFIG"

uv --directory "$PYTHON_DIR" run python - <<'PY' "$REQUEST_DIR/stream_a.json" "$REQUEST_DIR/stream_b.json" "$REQUEST_DIR/cancel.json" "$OVERLAP_CAPTURE" "$METRICS_CAPTURE" "$NATIVE_PORT"
from __future__ import annotations

import http.client
import json
import pathlib
import threading
import time
from typing import Any


def load_body(path: str) -> bytes:
    return pathlib.Path(path).read_bytes()


PORT = int(__import__('sys').argv[6])


def post_stream(body: bytes, close_after_first_delta: bool, result: dict[str, Any]) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Accept", "text/event-stream")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    result["status"] = response.status
    lines: list[str] = []
    first_delta_seen = False
    while True:
        raw = response.fp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        lines.append(line)
        if line.startswith("data: ") and line != "data: [DONE]":
            first_delta_seen = True
            if close_after_first_delta:
                result["client_closed"] = True
                result["first_delta_seen"] = True
                conn.close()
                result["lines"] = lines
                return
        if line == "data: [DONE]":
            break
    result["first_delta_seen"] = first_delta_seen
    result["lines"] = lines
    conn.close()


def scrape_metrics() -> dict[str, float]:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    conn.close()
    metrics: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


stream_a = {"name": "stream_a"}
stream_b = {"name": "stream_b"}
cancel = {"name": "cancel"}

thread_a = threading.Thread(target=post_stream, args=(load_body(pathlib.Path(__import__('sys').argv[1]).as_posix()), False, stream_a))
thread_a.start()
deadline = time.time() + 60
while not stream_a.get("first_delta_seen"):
    if time.time() > deadline:
        raise SystemExit("stream_a did not produce first delta")
    time.sleep(0.1)

thread_b = threading.Thread(target=post_stream, args=(load_body(pathlib.Path(__import__('sys').argv[2]).as_posix()), False, stream_b))
thread_b.start()
time.sleep(0.5)
thread_cancel = threading.Thread(target=post_stream, args=(load_body(pathlib.Path(__import__('sys').argv[3]).as_posix()), True, cancel))
thread_cancel.start()

metrics_history: list[dict[str, float]] = []
batch_grow_seen = False
batch_shrink_seen = False
metrics_ready_seen = False
max_running = 0.0
max_decode_batch = 0.0
for _ in range(240):
    try:
        snapshot = scrape_metrics()
    except Exception:
        time.sleep(0.25)
        continue
    metrics_history.append(snapshot)
    running = snapshot.get('mlx_scheduler_requests_by_backend{backend="native-mlx",modality="text",state="running"}', 0.0)
    waiting = snapshot.get('mlx_scheduler_requests_by_backend{backend="native-mlx",modality="text",state="waiting"}', 0.0)
    decode_batch = snapshot.get('mlx_decode_batch_size', 0.0)
    decode_metric_present = any(
        key.startswith('mlx_scheduled_tokens_by_backend{backend="native-mlx"')
        and 'phase="decode"' in key
        for key in snapshot
    )
    max_running = max(max_running, running)
    max_decode_batch = max(max_decode_batch, decode_batch)
    if decode_batch >= 2:
        batch_grow_seen = True
    if batch_grow_seen and decode_batch < max_decode_batch:
        batch_shrink_seen = True
    if decode_metric_present:
        metrics_ready_seen = True
    if not thread_a.is_alive() and not thread_b.is_alive() and not thread_cancel.is_alive():
        break
    time.sleep(0.25)

thread_a.join(timeout=120)
thread_b.join(timeout=120)
thread_cancel.join(timeout=120)

if thread_a.is_alive() or thread_b.is_alive() or thread_cancel.is_alive():
    raise SystemExit("one or more overlap probes did not finish")
if stream_a.get("status") != 200 or stream_b.get("status") != 200 or cancel.get("status") != 200:
    raise SystemExit(f"unexpected stream status values: {stream_a.get('status')}, {stream_b.get('status')}, {cancel.get('status')}")
if not stream_a.get("first_delta_seen") or not stream_b.get("first_delta_seen"):
    raise SystemExit("surviving streams did not produce first delta")
if not cancel.get("client_closed"):
    raise SystemExit("cancel probe did not close client after first delta")

pathlib.Path(__import__('sys').argv[5]).write_text(
    "\n".join(
        [
            f"max_running={max_running}",
            f"batch_grow_seen={int(batch_grow_seen)}",
            f"batch_shrink_seen={int(batch_shrink_seen)}",
            f"metrics_ready_seen={int(metrics_ready_seen)}",
        ]
        + [json.dumps(snapshot, sort_keys=True) for snapshot in metrics_history]
    )
)
pathlib.Path(__import__('sys').argv[4]).write_text(
    json.dumps(
        {
            "stream_a": stream_a,
            "stream_b": stream_b,
            "cancel": cancel,
            "batch_grow_seen": batch_grow_seen,
            "batch_shrink_seen": batch_shrink_seen,
            "metrics_ready_seen": metrics_ready_seen,
            "max_running": max_running,
            "max_decode_batch": max_decode_batch,
        },
        indent=2,
        sort_keys=True,
    )
)

print(f"native_batch_grow_ok={int(batch_grow_seen)}")
print(f"native_batch_shrink_ok={int(batch_shrink_seen)}")
print(f"native_independent_finish_ok={int('data: [DONE]' in stream_a['lines'] and 'data: [DONE]' in stream_b['lines'])}")
print(f"native_cancel_isolated_ok={int(cancel.get('client_closed', False) and 'data: [DONE]' in stream_a['lines'] and 'data: [DONE]' in stream_b['lines'])}")
print(f"native_scheduler_metrics_ok={int(metrics_ready_seen and batch_grow_seen)}")
PY

stop_gateway

echo "[4/6] Verify native overlap outputs and metrics"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$OVERLAP_CAPTURE"
from __future__ import annotations

import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not payload["batch_grow_seen"]:
    raise SystemExit("native batch never grew to overlapping decode membership")
if not payload["batch_shrink_seen"]:
    raise SystemExit("native batch never shrank after overlap")
if not payload["metrics_ready_seen"]:
    raise SystemExit("native scheduler metrics were not observed")
if "data: [DONE]" not in payload["stream_a"]["lines"]:
    raise SystemExit("stream_a missing [DONE]")
if "data: [DONE]" not in payload["stream_b"]["lines"]:
    raise SystemExit("stream_b missing [DONE]")
if not payload["cancel"].get("client_closed"):
    raise SystemExit("cancel probe did not close client")
PY

echo "[5/6] Start default v1 gateway and run non-regression request"
start_gateway "$V1_LOG" "$V1_PORT" MLX_RUNTIME_CONFIG="$V1_CONFIG"
V1_STATUS=$(curl -sS -o "$V1_CAPTURE" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${V1_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/v1.json")
if [[ "$V1_STATUS" != "200" ]]; then
    echo "unexpected v1 HTTP status: $V1_STATUS; inspect $V1_CAPTURE and $V1_LOG" >&2
    exit 1
fi
stop_gateway

echo "[6/6] Validate v1 response"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$V1_CAPTURE"
from __future__ import annotations

import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
choice = payload["choices"][0]
text = choice["message"]["content"].strip()
if not text:
    raise SystemExit("v1 response text was empty")
print("v1_non_regression_ok=1")
print("phase_7_validation_ok=1")
PY
