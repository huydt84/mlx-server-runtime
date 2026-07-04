#!/usr/bin/env bash
#
# native-v2 Phase 1 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_1.sh
#
# What this verifies:
#   1. Python environment can import `mlx` and `mlx_lm` on Apple Silicon.
#   2. Explicit `native-mlx` selection reaches native startup boundary and
#      reports structured `supported_class_bug` for supported architecture
#      placeholder startup.
#   3. Explicit `native-mlx` selection rejects unsupported architecture with
#      structured `unsupported_class` startup failure.
#   4. Default v1 backend still reaches healthy gateway startup.
#   5. One real v1 public API request succeeds after shared startup.
#
# Expected verification signal:
#   - Script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_lm_import_ok=1`.
#   - It prints `native_probe_label=` lines for both native startup probes.
#   - It prints `native_probe_category=` and `native_probe_stage=` matching the
#     expected structured startup failure.
#   - It prints `health_response=healthy` for v1 gateway startup.
#   - It prints `assistant_content=` with non-empty v1 output.
#   - If validation fails, script exits non-zero and points to captured logs.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
MODEL_NAME="mlx-community/Qwen2.5-7B-Instruct-4bit"
GATEWAY_LOG="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-1-gateway.log"
HEALTH_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-1-health.txt"
CHAT_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-1-chat.json"
REQUEST_CAPTURE="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-1-request.json"

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

echo "[3/5] Probe native-mlx structured startup failures"
uv run python - <<'PY'
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from mlx_worker.ipc import ModelStatus, WorkerError, decode_bootstrap_message


def run_probe(
    label: str,
    architecture_class: str,
    expected_category: str,
    expected_stage: str,
    expected_code: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"{label}-") as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": [architecture_class]})
        )
        socket_path = root / "worker.sock"
        log_path = root / "worker.log"

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)

        env = os.environ.copy()
        env["MLX_RUNTIME_SOCKET"] = str(socket_path)
        env["MLX_RUNTIME_BACKEND"] = "native-mlx"
        env["MLX_RUNTIME_MODEL"] = str(model_dir)
        env["MLX_RUNTIME_VLM_MODEL"] = ""

        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", "mlx_worker.main"],
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            try:
                conn, _ = listener.accept()
            finally:
                listener.close()

            decoded = []
            with conn:
                buffer = b""
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line:
                            decoded.append(decode_bootstrap_message(line + b"\n"))

            rc = proc.wait(timeout=30)
            if rc != 1:
                raise SystemExit(
                    f"{label}: expected worker exit 1, saw {rc}; inspect {log_path}"
                )

        failed_status = next(
            item
            for item in decoded
            if isinstance(item, ModelStatus) and item.state == "failed"
        )
        if failed_status.last_error is None:
            raise SystemExit(f"{label}: missing last_error; inspect {log_path}")
        terminal_error = decoded[-1]
        if not isinstance(terminal_error, WorkerError) or terminal_error.error is None:
            raise SystemExit(f"{label}: missing structured worker error; inspect {log_path}")

        error = terminal_error.error
        if error.category != expected_category:
            raise SystemExit(
                f"{label}: expected category {expected_category}, saw {error.category}"
            )
        if error.stage != expected_stage:
            raise SystemExit(
                f"{label}: expected stage {expected_stage}, saw {error.stage}"
            )
        if error.code != expected_code:
            raise SystemExit(f"{label}: expected code {expected_code}, saw {error.code}")

        print(f"native_probe_label={label}")
        print(f"native_probe_category={error.category}")
        print(f"native_probe_stage={error.stage}")
        print(f"native_probe_code={error.code}")


run_probe(
    label="supported-class-placeholder",
    architecture_class="Qwen2ForCausalLM",
    expected_category="supported_class_bug",
    expected_stage="native_executor_construction",
    expected_code="NATIVE_EXECUTOR_NOT_IMPLEMENTED",
)
run_probe(
    label="unsupported-class-rejection",
    architecture_class="LlamaForCausalLM",
    expected_category="unsupported_class",
    expected_stage="architecture_detection",
    expected_code="UNSUPPORTED_ARCHITECTURE_CLASS",
)
PY

echo "[4/5] Start gateway with default v1 backend"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE" "$CHAT_CAPTURE" "$REQUEST_CAPTURE"

uv --directory "$PYTHON_DIR" run python - <<'PY' "$REQUEST_CAPTURE" "$MODEL_NAME"
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
model_name = sys.argv[2]
path.write_text(
    json.dumps(
        {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    )
)
PY

cargo run -p mlx_runtime_gateway >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

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

echo "[5/5] Run one non-streaming v1 chat completion"
HTTP_STATUS=$(curl -sS -o "$CHAT_CAPTURE" -w '%{http_code}' \
    -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_CAPTURE")

if [[ "$HTTP_STATUS" != "200" ]]; then
    echo "unexpected HTTP status: $HTTP_STATUS; inspect $CHAT_CAPTURE and $GATEWAY_LOG" >&2
    exit 1
fi

uv --directory "$PYTHON_DIR" run python - <<'PY' "$CHAT_CAPTURE"
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

echo "gateway_log=$GATEWAY_LOG"
