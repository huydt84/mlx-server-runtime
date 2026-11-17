#!/usr/bin/env bash
#
# native-v2 startup-failure host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_startup_failures.sh [options]
#   bash mlx-host-validation/scripts/native_startup_failures.sh --help
#
# What this verifies:
#   1. Python environment can import `mlx` and `mlx_lm` on Apple Silicon.
#   2. Explicit `native-mlx` selection reaches native startup boundary and
#      reports structured `supported_class_bug` for supported architecture
#      placeholder startup.
#   3. Explicit `native-mlx` selection rejects unsupported architecture with
#      structured `unsupported_class` startup failure.
#   4. Native KV memory planning refuses to start with `KV_MEMORY_INSUFFICIENT`
#      when the memory utilization cannot hold the real checkpoint's weights,
#      and when an explicit per-request token limit exceeds the KV pool.
#   5. The v1 backend still reaches healthy gateway startup.
#   6. One real v1 public API request succeeds after shared startup.
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
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
# shellcheck source=lib/gateway.sh
source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
cli_describe "Validate structured native-mlx startup failures, then a healthy v1 gateway and one real request."
cli_option MODEL_NAME --checkpoint MLX_STARTUP_FAILURES_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served by the v1 gateway."
cli_option PORT --port MLX_STARTUP_FAILURES_PORT port 18191 "v1 gateway port."
cli_option HEALTH_TIMEOUT --health-timeout MLX_STARTUP_FAILURES_HEALTH_TIMEOUT int 300 \
    "Seconds to wait for the v1 gateway to report healthy."
cli_option ARTIFACTS_DIR --artifacts-dir MLX_STARTUP_FAILURES_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-startup-failures" "Directory for the gateway log, config, and captures."
cli_option SKIP_BUILD --skip-build MLX_STARTUP_FAILURES_SKIP_BUILD bool 0 \
    "Skip uv sync and cargo build; reuse the existing environment and gateway binary."
cli_parse "$@"
cli_require_free_port "$PORT" --port

GATEWAY_LOG="$ARTIFACTS_DIR/gateway.log"
GATEWAY_CONFIG="$ARTIFACTS_DIR/v1.toml"
HEALTH_CAPTURE="$ARTIFACTS_DIR/health.txt"
CHAT_CAPTURE="$ARTIFACTS_DIR/chat.json"
REQUEST_CAPTURE="$ARTIFACTS_DIR/request.json"
GATEWAY_BIN="$ROOT/target/debug/mlx_runtime_gateway"
mkdir -p "$ARTIFACTS_DIR"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "${GATEWAY_PID}" >/dev/null 2>&1 || true
        wait "${GATEWAY_PID}" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

echo "[1/5] Sync Python dev environment"
cd "$PYTHON_DIR"
if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "build_skipped=1"
else
    uv sync --group dev
    cargo build --manifest-path "$ROOT/Cargo.toml" -p mlx_runtime_gateway
fi

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
MLX_STARTUP_FAILURES_REAL_MODEL="$MODEL_NAME" uv run python - <<'PY'
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
    *,
    model: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix="probe-") as temp_dir:
        root = Path(temp_dir)
        model_dir = root / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"architectures": [architecture_class]})
        )
        model_ref = model or str(model_dir)
        socket_path = root / "worker.sock"
        log_path = root / "worker.log"

        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen(1)

        env = os.environ.copy()
        env["MLX_RUNTIME_SOCKET"] = str(socket_path)
        env["MLX_RUNTIME_BACKEND"] = "native-mlx"
        env["MLX_RUNTIME_MODEL"] = model_ref
        env["MLX_RUNTIME_VLM_MODEL"] = ""
        for name in ("MLX_RUNTIME_KV_CACHE_BYTES", "MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES"):
            env.pop(name, None)
        env.update(extra_env or {})

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

            rc = proc.wait(timeout=300)
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
    label="malformed-supported-class-config",
    architecture_class="Qwen2ForCausalLM",
    expected_category="malformed_checkpoint",
    expected_stage="artifact_validation",
    expected_code="INVALID_NATIVE_CONFIG",
)
run_probe(
    label="unsupported-class-rejection",
    architecture_class="LlamaForCausalLM",
    expected_category="unsupported_class",
    expected_stage="architecture_detection",
    expected_code="UNSUPPORTED_ARCHITECTURE_CLASS",
)
real_model = os.environ["MLX_STARTUP_FAILURES_REAL_MODEL"]
run_probe(
    label="kv-memory-utilization-below-weights",
    architecture_class="unused",
    expected_category="invalid_configuration",
    expected_stage="kv_memory_planning",
    expected_code="KV_MEMORY_INSUFFICIENT",
    model=real_model,
    extra_env={"MLX_RUNTIME_MEMORY_UTILIZATION": "0.01"},
)
run_probe(
    label="kv-memory-explicit-limit-exceeds-pool",
    architecture_class="unused",
    expected_category="invalid_configuration",
    expected_stage="kv_memory_planning",
    expected_code="KV_MEMORY_INSUFFICIENT",
    model=real_model,
    extra_env={
        "MLX_RUNTIME_KV_CACHE_BYTES": str(64 * 1024 * 1024),
        "MLX_RUNTIME_MAX_TOTAL_TOKENS_PER_REQUEST": "65536",
        "MLX_RUNTIME_MAX_TOTAL_TOKENS_EXPLICIT": "1",
    },
)
PY

echo "[4/5] Start gateway with the v1 backend"
cd "$ROOT"
rm -f "$GATEWAY_LOG" "$HEALTH_CAPTURE" "$CHAT_CAPTURE" "$REQUEST_CAPTURE"
if ! grep -qx 'port = 8000' "$ROOT/config/runtime.toml"; then
    echo "config/runtime.toml no longer contains 'port = 8000'" >&2
    exit 1
fi
sed -e "s/^port = 8000$/port = $PORT/" \
    -e 's/^backend = "native-mlx"$/backend = "v1"/' \
    -e "s#^model = \"mlx-community/Qwen2.5-7B-Instruct-4bit\"#model = \"$MODEL_NAME\"#" \
    "$ROOT/config/runtime.toml" >"$GATEWAY_CONFIG"

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

MLX_RUNTIME_CONFIG="$GATEWAY_CONFIG" "$GATEWAY_BIN" >"$GATEWAY_LOG" 2>&1 &
GATEWAY_PID=$!

gateway_wait_ready "$PORT" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$GATEWAY_LOG"
curl -fsS "http://127.0.0.1:$PORT/health" >"$HEALTH_CAPTURE"
grep -qx 'healthy' "$HEALTH_CAPTURE"
echo "health_response=healthy"

echo "[5/5] Run one non-streaming v1 chat completion"
HTTP_STATUS=$(curl -sS -o "$CHAT_CAPTURE" -w '%{http_code}' \
    -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
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
