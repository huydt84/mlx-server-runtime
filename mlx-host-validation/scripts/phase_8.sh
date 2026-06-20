#!/usr/bin/env bash
#
# Phase 8 host-only MLX validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_8.sh
#
# What this verifies:
#   1. Python can import `mlx`, `mlx_lm`, and `mlx_vlm` on the host.
#   2. The Rust gateway starts with VLM enabled and serves the configured text model.
#   3. A VLM chat completion with one local image returns generated text through `/v1/chat/completions`.
#   4. A VLM chat completion with one loopback web image returns generated text through `/v1/chat/completions`.
#   5. A text-only chat completion still uses the text path and returns generated text.
#   6. Invalid remote image URLs are rejected before model execution.
#   7. VLM lifecycle and metrics endpoints report warmup-ready state and VLM telemetry after the requests.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1`, `mlx_lm_import_ok=1`, and `mlx_vlm_import_ok=1`.
#   - It prints `health_response=healthy`, `vlm_warmup_ready_ok=1`, and `vlm_ready_response=ready`.
#   - It prints `text_response_non_empty=1`, `vlm_text_stream_ok=1`, `vlm_text_stream_done=1`, `vlm_text_stream_response_non_empty=1`, `vlm_local_image_response_non_empty=1`, `vlm_web_image_response_non_empty=1`, and `vlm_stream_metrics_ok=1`.
#   - It prints `invalid_image_rejected=1`.
#   - It prints `metrics_ok=1` and `vlm_metrics_ok=1`.
#   - If validation fails, the script exits non-zero and points to the captured gateway log path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
VLM_MODEL="mlx-community/Qwen2-VL-2B-Instruct-4bit"
VLM_MODEL_PATH="mlx-community%2FQwen2-VL-2B-Instruct-4bit"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-phase-8"
LOCAL_IMAGE_PATH="$TMP_ROOT/phase_8_local.png"
IMAGE_SERVER_PORT=18080
CONFIG_PATH="$TMP_ROOT/runtime.toml"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-gateway-phase-8.log"
IMAGE_SERVER_LOG="${TMPDIR:-/tmp}/mlx-runtime-phase-8-image-server.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-health.txt"
TEXT_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-text.json"
VLM_STREAM_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-vlm-stream.txt"
VLM_JSON_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-vlm-json.json"
BAD_IMAGE_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-bad-image.json"
VLM_READY_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-vlm-ready.json"
VLM_STATUS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-vlm-status.json"
MODELS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-models.json"
METRICS_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-phase-8-metrics.txt"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
    if [[ -n "${IMAGE_SERVER_PID:-}" ]]; then
        kill "${IMAGE_SERVER_PID}" >/dev/null 2>&1 || true
        wait "${IMAGE_SERVER_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

mkdir -p "$TMP_ROOT"
python3 - <<'PY' "$LOCAL_IMAGE_PATH"
from base64 import b64decode
from pathlib import Path
import sys

png = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAOUlEQVR42mP4z8CAFTE0/MeKSFXPMGrBqAVDwAJqGYRT/agFoxYMAQtGi4pRC0YtGK0PRi0YtQCIAJF3/D1SNXLzAAAAAElFTkSuQmCC"
)
Path(sys.argv[1]).write_bytes(png)
PY
cp "$ROOT/config/runtime.toml" "$CONFIG_PATH"
python3 - <<'PY' "$CONFIG_PATH" "$VLM_MODEL"
from pathlib import Path
import sys

path = Path(sys.argv[1])
vlm_model = sys.argv[2]
text = path.read_text()
text = text.replace(
    '# vlm_model = "mlx-community/Qwen3-VL-2B-Instruct-4bit"',
    f'vlm_model = "{vlm_model}"',
)
text = text.replace('# max_vlm_images = 5', 'max_vlm_images = 5')
path.write_text(text)
PY

echo "[1/7] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/7] Verify Apple Silicon and VLM imports"
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

from mlx_vlm import generate as vlm_generate, load as vlm_load, stream_generate as vlm_stream_generate  # noqa: F401
print("mlx_vlm_import_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/7] Start gateway"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE" "$TEXT_CAPTURE" "$VLM_STREAM_CAPTURE" "$VLM_JSON_CAPTURE" "$BAD_IMAGE_CAPTURE" "$VLM_READY_CAPTURE" "$VLM_STATUS_CAPTURE" "$MODELS_CAPTURE" "$METRICS_CAPTURE"
MLX_RUNTIME_CONFIG="$CONFIG_PATH" cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

echo "[4/7] Start local image server"
python3 -m http.server "$IMAGE_SERVER_PORT" --bind 127.0.0.1 --directory "$TMP_ROOT" >"$IMAGE_SERVER_LOG" 2>&1 &
IMAGE_SERVER_PID=$!

echo "[5/7] Wait for gateway and VLM readiness"
for _ in $(seq 1 300); do
    if curl -fsS -o "$HEALTH_CAPTURE" -w '%{http_code}' http://127.0.0.1:8000/health | grep -qx '200'; then
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

for _ in $(seq 1 300); do
    if curl -fsS "http://127.0.0.1:8000/models/$VLM_MODEL_PATH/status" >"$VLM_STATUS_CAPTURE"; then
        if python3 - <<'PY' "$VLM_STATUS_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["state"] != "ready":
    raise SystemExit(1)
if not payload["warmup_passed"]:
    raise SystemExit(2)
if int(payload["last_warmup_latency_ms"] or 0) <= 0:
    raise SystemExit(3)
PY
        then
            echo "vlm_warmup_ready_ok=1"
            break
        fi
    fi

    if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
        echo "gateway exited unexpectedly; inspect $GATEWAY_LOG" >&2
        exit 1
    fi

    sleep 1
done

curl -fsS "http://127.0.0.1:8000/models/$VLM_MODEL_PATH/ready" >"$VLM_READY_CAPTURE"
python3 - <<'PY' "$VLM_READY_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["ready"] is not True:
    raise SystemExit(f"expected VLM ready payload: {payload}")
print("vlm_ready_response=ready")
PY

echo "[6/7] Run text and VLM completions"
HTTP_STATUS=$(curl -sS -o "$TEXT_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$MODEL_NAME"'",
      "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
      "max_tokens": 4,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status for text request: $HTTP_STATUS; inspect $TEXT_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$TEXT_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("assistant content was empty")
print("text_response_non_empty=1")
PY

HTTP_STATUS=$(curl -sS -N -o "$VLM_STREAM_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    -d '{
      "model": "'"$VLM_MODEL"'",
      "messages": [{"role": "user", "content": "Write one short greeting."}],
      "max_tokens": 4,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": true
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status for VLM stream request: $HTTP_STATUS; inspect $VLM_STREAM_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

python3 - <<'PY' "$VLM_STREAM_CAPTURE"
from pathlib import Path
import json
import sys

path = Path(sys.argv[1])
chunks: list[str] = []
done_seen = False
for line in path.read_text().splitlines():
    if not line.startswith("data: "):
        continue
    payload = line.removeprefix("data: ").strip()
    if payload == "[DONE]":
        done_seen = True
        continue
    event = json.loads(payload)
    chunks.append(event["choices"][0]["delta"].get("content", ""))

assistant_content = "".join(chunks).strip()
if not assistant_content:
    raise SystemExit("assistant content was empty")
if not done_seen:
    raise SystemExit("stream never finished")
    print("vlm_text_stream_ok=1")
    print("vlm_text_stream_done=1")
    print("vlm_text_stream_response_non_empty=1")
PY

HTTP_STATUS=$(curl -sS -o "$VLM_JSON_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$VLM_MODEL"'",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe these images in one short sentence."},
        {"type": "image_url", "image_url": {"url": "'"$LOCAL_IMAGE_PATH"'"}}
      ]}],
      "max_tokens": 4,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status for VLM image request: $HTTP_STATUS; inspect $VLM_JSON_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$VLM_JSON_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("assistant content was empty")
print("vlm_local_image_response_non_empty=1")
PY

HTTP_STATUS=$(curl -sS -o "$VLM_JSON_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$VLM_MODEL"'",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {"url": "http://127.0.0.1:'"$IMAGE_SERVER_PORT"'/phase_8_local.png"}}
      ]}],
      "max_tokens": 4,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status for VLM web-image request: $HTTP_STATUS; inspect $VLM_JSON_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$VLM_JSON_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
content = payload["choices"][0]["message"]["content"].strip()
if not content:
    raise SystemExit("assistant content was empty")
print("vlm_web_image_response_non_empty=1")
PY

curl -fsS http://127.0.0.1:8000/metrics >"$METRICS_CAPTURE"
python3 - <<'PY' "$METRICS_CAPTURE"
import pathlib
import re
import sys

metrics = pathlib.Path(sys.argv[1]).read_text()

def metric_value(name: str) -> int:
    match = re.search(rf"^{re.escape(name)}\s+(\d+)$", metrics, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing metric {name}")
    return int(match.group(1))

def metric_value_float(name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\s+([0-9]+(?:\.[0-9]+)?)$", metrics, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing metric {name}")
    return float(match.group(1))

if metric_value("mlx_vlm_requests_total") != 3:
    raise SystemExit("expected exactly three VLM requests so far")
if metric_value("mlx_vlm_image_count_total") != 2:
    raise SystemExit("expected exactly two VLM images so far")
if metric_value("mlx_vlm_load_errors_total") != 0:
    raise SystemExit("expected zero VLM load errors")
if metric_value("mlx_vlm_image_preprocess_latency_ms") <= 0:
    raise SystemExit("expected positive VLM image preprocessing latency")
if metric_value("mlx_vlm_prompt_template_latency_ms") <= 0:
    raise SystemExit("expected positive VLM prompt/template latency")
if metric_value("mlx_ttft_ms_count") <= 0:
    raise SystemExit("expected positive TTFT histogram count")
if metric_value("mlx_request_latency_ms_count") <= 0:
    raise SystemExit("expected positive request latency histogram count")
if metric_value_float("mlx_decode_tokens_per_second") <= 0:
    raise SystemExit("expected positive decode throughput")
for marker in ("mlx_ttft_ms_bucket", "mlx_request_latency_ms_bucket"):
    if marker not in metrics:
        raise SystemExit(f"missing histogram output for {marker}")
print("vlm_stream_metrics_ok=1")
PY

curl -fsS "http://127.0.0.1:8000/models/$VLM_MODEL_PATH/ready" >"$VLM_READY_CAPTURE"
uv run python - <<'PY' "$VLM_READY_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["ready"] is not True:
    raise SystemExit(f"expected VLM ready payload: {payload}")
print("vlm_ready_response=ready")
PY

echo "[7/7] Run remote failure checks"

HTTP_STATUS=$(curl -sS -o "$BAD_IMAGE_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "'"$VLM_MODEL"'",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe this image in one short sentence."},
        {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}}
      ]}],
      "max_tokens": 32,
      "temperature": 0.0,
      "top_p": 1.0,
      "stream": false
    }')

if [[ "$HTTP_STATUS" != "400" ]]; then
    echo "expected HTTP 400 for invalid image URL, got $HTTP_STATUS; inspect $BAD_IMAGE_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv run python - <<'PY' "$BAD_IMAGE_CAPTURE"
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
if payload["error"]["code"] != "INVALID_IMAGE_URL":
    raise SystemExit(f"unexpected invalid-image payload: {payload}")
print("invalid_image_rejected=1")
PY

echo "Verify metrics"
curl -fsS http://127.0.0.1:8000/models >"$MODELS_CAPTURE"
curl -fsS http://127.0.0.1:8000/metrics >"$METRICS_CAPTURE"
uv run python - <<'PY' "$MODELS_CAPTURE" "$METRICS_CAPTURE"
import json
import pathlib
import re
import sys

models_payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
models = {item["name"]: item for item in models_payload["models"]}
if models.get("mlx-community/Qwen2.5-7B-Instruct-4bit", {}).get("state") != "ready":
    raise SystemExit(f"text model missing or not ready: {models_payload}")
if models.get("mlx-community/Qwen2-VL-2B-Instruct-4bit", {}).get("state") != "ready":
    raise SystemExit(f"VLM model missing or not ready: {models_payload}")

metrics = pathlib.Path(sys.argv[2]).read_text()

def metric_value(name: str) -> int:
    match = re.search(rf"^{re.escape(name)}\s+(\d+)$", metrics, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing metric {name}")
    return int(match.group(1))

def metric_value_float(name: str) -> float:
    match = re.search(rf"^{re.escape(name)}\s+([0-9]+(?:\.[0-9]+)?)$", metrics, re.MULTILINE)
    if not match:
        raise SystemExit(f"missing metric {name}")
    return float(match.group(1))

if metric_value("mlx_vlm_requests_total") != 3:
    raise SystemExit("expected exactly three VLM requests")
if metric_value("mlx_vlm_image_count_total") != 2:
    raise SystemExit("expected exactly two VLM images")
if metric_value("mlx_vlm_load_errors_total") != 0:
    raise SystemExit("expected zero VLM load errors")
if metric_value("mlx_vlm_image_preprocess_latency_ms") <= 0:
    raise SystemExit("expected positive VLM image preprocessing latency")
if metric_value("mlx_vlm_prompt_template_latency_ms") <= 0:
    raise SystemExit("expected positive VLM prompt/template latency")
if metric_value("mlx_ttft_ms_count") <= 0:
    raise SystemExit("expected positive TTFT histogram count")
if metric_value("mlx_request_latency_ms_count") <= 0:
    raise SystemExit("expected positive request latency histogram count")
if metric_value_float("mlx_decode_tokens_per_second") <= 0:
    raise SystemExit("expected positive decode throughput")
for metric_name in (
    "mlx_vlm_image_preprocess_latency_ms",
    "mlx_vlm_prompt_template_latency_ms",
    "mlx_vlm_load_errors_total",
    "mlx_decode_tokens_per_second",
):
    if metric_name not in metrics:
        raise SystemExit(f"missing metric output for {metric_name}")

for marker in ("mlx_ttft_ms_bucket", "mlx_request_latency_ms_bucket"):
    if marker not in metrics:
        raise SystemExit(f"missing histogram output for {marker}")

print("metrics_ok=1")
print("vlm_metrics_ok=1")
PY

cleanup

echo "gateway_log=$GATEWAY_LOG"
