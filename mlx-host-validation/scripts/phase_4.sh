#!/usr/bin/env bash
#
# Phase 4 host-only telemetry validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_4.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. The Rust gateway starts successfully with telemetry enabled.
#   3. `GET /metrics` exposes Prometheus metrics.
#   4. A request produces structured JSON request logs with request_id, TTFT, latency, and token counts.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_lm_import_ok=1`.
#   - It prints `metrics_endpoint_ok=1`.
#   - It prints `structured_log_ok=1`.
#   - It prints `request_id=...`, `ttft_ms=...`, `latency_ms=...`, and `completion_tokens=...`.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-phase-4"
CONFIG_PATH="$TMP_ROOT/runtime.toml"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-4.log"
REQUEST_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-4-request.json"
METRICS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-4-metrics.txt"

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
text = re.sub(r"request_timeout_seconds = \d+", "request_timeout_seconds = 300", text)
text = re.sub(r"enable_prometheus = .*", "enable_prometheus = true", text)
text = re.sub(r'metrics_path = ".*"', 'metrics_path = "/metrics"', text)
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
rm -f "$GATEWAY_LOG" "$REQUEST_CAPTURE" "$METRICS_CAPTURE"
MLX_RUNTIME_CONFIG="$CONFIG_PATH" cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "[4/5] Wait for /health to report healthy"
for _ in $(seq 1 300); do
    if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
        break
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

curl -fsS http://127.0.0.1:8000/health >/dev/null

echo "[5/5] Run request and inspect telemetry"
curl -sS -o "$REQUEST_CAPTURE" \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Say telemetry ok in one short sentence."}],
      "max_tokens": 16,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }'

python3 - <<'PY' "$REQUEST_CAPTURE" "$GATEWAY_LOG"
from pathlib import Path
import json
import sys

request = json.loads(Path(sys.argv[1]).read_text())
log_lines = Path(sys.argv[2]).read_text().splitlines()

if "choices" not in request:
    raise SystemExit(f"unexpected chat completion payload: {request}")

matches = []
for line in log_lines:
    line = line.strip()
    if not line.startswith("{"):
        continue
    payload = json.loads(line)
    if payload.get("request_id"):
        matches.append(payload)

if not matches:
    raise SystemExit(f"no structured request log found in {sys.argv[2]}")

entry = matches[-1]
required = ["request_id", "ttft_ms", "latency_ms", "completion_tokens"]
for field in required:
    if field not in entry:
        raise SystemExit(f"missing field {field} in request log: {entry}")

print(f"request_id={entry['request_id']}")
print(f"ttft_ms={entry['ttft_ms']}")
print(f"latency_ms={entry['latency_ms']}")
print(f"completion_tokens={entry['completion_tokens']}")
print("structured_log_ok=1")
PY

curl -fsS http://127.0.0.1:8000/metrics >"$METRICS_CAPTURE"

python3 - <<'PY' "$METRICS_CAPTURE"
from pathlib import Path
import sys

metrics = Path(sys.argv[1]).read_text()
required = [
    "mlx_requests_total",
    "mlx_requests_active",
    "mlx_requests_failed_total",
    "mlx_requests_cancelled_total",
    "mlx_queue_depth",
    "mlx_queue_rejected_total",
    "mlx_worker_up",
    "mlx_worker_restarts_total",
    "mlx_ttft_ms_bucket",
    "mlx_request_latency_ms_bucket",
    "mlx_prompt_tokens_total",
    "mlx_completion_tokens_total",
    "mlx_decode_tokens_per_second",
    "mlx_prefill_tokens_per_second",
    "mlx_ipc_messages_sent_total",
    "mlx_ipc_messages_received_total",
    "mlx_ipc_roundtrip_latency_ms",
    "mlx_worker_memory_bytes",
    "mlx_kv_cache_bytes",
]

missing = [name for name in required if name not in metrics]
if missing:
    raise SystemExit(f"missing metrics: {missing}")

print("metrics_endpoint_ok=1")
PY

echo "gateway_log=$GATEWAY_LOG"
