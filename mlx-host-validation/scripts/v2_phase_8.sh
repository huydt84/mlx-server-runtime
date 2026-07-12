#!/usr/bin/env bash
#
# native-v2 Phase 8 host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/v2_phase_8.sh
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - native-v2 public gateway mixed workload with chunk sizes `32` and `96`
#   - one long prompt plus two active short decode requests against `mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - default v1 public request against same checkpoint
#
# Host requirements:
#   - Apple Silicon (`arm64`)
#   - Metal-capable MLX environment
#   - `uv` environment for `python/`
#   - known-good checkpoint already available to local Hugging Face cache
#   - `cargo` toolchain for `mlx_runtime_gateway`
#   - optional `MLX_PHASE8_TEXT_CACHE_BUDGET_BYTES` override for native KV budget
#
# Expected success signals:
#   - `mlx_import_ok=1`
#   - `mlx_lm_import_ok=1`
#   - `chunk_32_prefill_metric_ok=1`
#   - `chunk_32_decode_metric_ok=1`
#   - `chunk_32_interleave_ok=1`
#   - `chunk_32_outputs_ok=1`
#   - `native_phase8_physical_batch_observed=1`
#   - `native_phase8_single_forward_per_batch_ok=1`
#   - `native_phase8_unequal_lengths_ok=1`
#   - `native_phase8_mixed_single_forward_ok=1`
#   - `native_phase8_mixed_parity_ok=1`
#   - `native_phase8_request_failure_isolation_ok=1`
#   - `chunk_96_prefill_metric_ok=1`
#   - `chunk_96_decode_metric_ok=1`
#   - `chunk_96_interleave_ok=1`
#   - `chunk_96_outputs_ok=1`
#   - `v1_non_regression_ok=1`
#   - `phase_8_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly
#   - long prompt never overlaps active short decode progress
#   - native metrics missing backend-labeled prefill/decode evidence
#   - TTFT/ITL summary missing for either chunk size

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
CHECKPOINT="mlx-community/Qwen2.5-7B-Instruct-4bit"
TEXT_CACHE_BUDGET_BYTES="${MLX_PHASE8_TEXT_CACHE_BUDGET_BYTES:-268435456}"
NATIVE_PORT_BASE=18082
V1_PORT=18083
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-v2-phase-8"
REQUEST_DIR="$TMP_ROOT/requests"
NATIVE_CONFIG="$TMP_ROOT/runtime-native.toml"
V1_CONFIG="$TMP_ROOT/runtime-v1.toml"
HEALTH_CAPTURE="$TMP_ROOT/health.txt"
V1_CAPTURE="$TMP_ROOT/v1.json"
NATIVE_LOG="$TMP_ROOT/native.log"
V1_LOG="$TMP_ROOT/v1.log"
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
    local config_path="$3"
    local chunk_size="$4"
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port $port is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env \
            MLX_RUNTIME_CONFIG="$config_path" \
            MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$chunk_size" \
            MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES" \
            "$GATEWAY_BIN"
    ) >"$log_path" 2>&1 &
    GATEWAY_PID=$!
    wait_healthy "$log_path" "$port"
}

start_v1_gateway() {
    local log_path="$1"
    local port="$2"
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port $port is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env MLX_RUNTIME_CONFIG="$V1_CONFIG" "$GATEWAY_BIN"
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

echo "[1/7] Sync Python dev environment and build gateway"
cd "$PYTHON_DIR"
uv sync --group dev
cd "$ROOT"
cargo build -p mlx_runtime_gateway

echo "[2/7] Verify Apple Silicon, mlx, and mlx_lm imports"
uv --directory "$PYTHON_DIR" run python - <<'PY'
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

echo "[3/7] Build runtime configs and request fixtures"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$ROOT/config/runtime.toml" "$NATIVE_CONFIG" "$V1_CONFIG" "$CHECKPOINT" "$NATIVE_PORT_BASE" "$V1_PORT" "$REQUEST_DIR"
from __future__ import annotations

import json
import pathlib
import sys
import time

from mlx_worker.native_mlx.bootstrap import (
    build_finalized_token_ids,
    resolve_model_path,
)

source = pathlib.Path(sys.argv[1]).read_text()
native_target = pathlib.Path(sys.argv[2])
v1_target = pathlib.Path(sys.argv[3])
checkpoint = sys.argv[4]
native_port = sys.argv[5]
v1_port = sys.argv[6]
request_dir = pathlib.Path(sys.argv[7])
request_dir.mkdir(parents=True, exist_ok=True)

native_target.write_text(
    source.replace('port = 8000', f'port = {native_port}').replace(
        'backend = "v1"', 'backend = "native-mlx"'
    ).replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
)
v1_target.write_text(
    source.replace('port = 8000', f'port = {v1_port}').replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
)

model_path = resolve_model_path(checkpoint)
long_words: list[str] = []
while True:
    start = len(long_words)
    long_words.extend(f"phase8_token_{index:04d}" for index in range(start, start + 128))
    candidate = " ".join(long_words)
    token_count = len(
        build_finalized_token_ids(
            model_path,
            [{"role": "user", "content": candidate}],
        )
    )
    if token_count >= 320:
        long_prompt = candidate
        break
payloads = {
    "short_a.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": "Emit many short uppercase letters separated by spaces until budget ends."}],
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    },
    "short_b.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": "Count upward in many short comma-separated tokens until you run out of budget."}],
        "max_tokens": 48,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    },
    "long.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": long_prompt}],
        "max_tokens": 24,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    },
    "v1.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 16,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    },
}
for name, payload in payloads.items():
    (request_dir / name).write_text(json.dumps(payload))
PY

probe_chunk_size() {
    local chunk_size="$1"
    local port="$2"
    local capture_json="$TMP_ROOT/chunk-${chunk_size}.json"
    local capture_metrics="$TMP_ROOT/chunk-${chunk_size}.metrics.txt"
    local config_path="$TMP_ROOT/runtime-native-${chunk_size}.toml"

    echo "[5/7] Probe native-mlx chunk size ${chunk_size}"
    uv --directory "$PYTHON_DIR" run python - <<'PY' "$NATIVE_CONFIG" "$config_path" "$port" "$chunk_size"
from __future__ import annotations

import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
target = pathlib.Path(sys.argv[2])
target.write_text(
    source.replace('port = 18082', f'port = {sys.argv[3]}').replace(
        'ipc_path = "/tmp/mlx-runtime.sock"',
        f'ipc_path = "/tmp/mlx-runtime-phase8-{sys.argv[4]}.sock"',
    )
)
PY
    start_gateway "$NATIVE_LOG" "$port" "$config_path" "$chunk_size"

    uv --directory "$PYTHON_DIR" run python - <<'PY' "$REQUEST_DIR/short_a.json" "$REQUEST_DIR/short_b.json" "$REQUEST_DIR/long.json" "$capture_json" "$capture_metrics" "$port" "$chunk_size"
from __future__ import annotations

import http.client
import json
import pathlib
import statistics
import sys
import threading
import time
from typing import Any

PORT = int(sys.argv[6])
CHUNK_SIZE = int(sys.argv[7])


def load_body(path: str) -> bytes:
    return pathlib.Path(path).read_bytes()


def post_stream(name: str, body: bytes, result: dict[str, Any]) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=240)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Accept", "text/event-stream")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    result["name"] = name
    result["status"] = response.status
    result["lines"] = []
    result["delta_timestamps"] = []
    result["text_fragments"] = []
    start = time.perf_counter()
    while True:
        raw = response.fp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        result["lines"].append(line)
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = json.loads(line[6:])
            choice = payload.get("choices", [{}])[0]
            delta = choice.get("delta", {}).get("content")
            if delta:
                now = time.perf_counter()
                timestamp = (now - start) * 1000.0
                result["delta_timestamps"].append(timestamp)
                result.setdefault("delta_timestamps_abs", []).append(now)
                result["text_fragments"].append(delta)
                if "first_delta_at_ms" not in result:
                    result["first_delta_at_ms"] = timestamp
                    result["first_delta_at_abs"] = now
        if line == "data: [DONE]":
            break
    conn.close()
    result["done"] = "data: [DONE]" in result["lines"]
    result["text"] = "".join(result["text_fragments"]).strip()
    timestamps = result["delta_timestamps"]
    if len(timestamps) >= 2:
        gaps = [right - left for left, right in zip(timestamps, timestamps[1:])]
        result["mean_itl_ms"] = statistics.fmean(gaps)
    else:
        result["mean_itl_ms"] = None


def scrape_metrics_text() -> str:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    conn.close()
    return body


def parse_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.rsplit(" ", 1)
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics


short_a: dict[str, Any] = {}
short_b: dict[str, Any] = {}
long_result: dict[str, Any] = {}

thread_a = threading.Thread(
    target=post_stream,
    args=("short_a", load_body(sys.argv[1]), short_a),
)
thread_b = threading.Thread(
    target=post_stream,
    args=("short_b", load_body(sys.argv[2]), short_b),
)
thread_a.start()
thread_b.start()

deadline = time.time() + 120
while "first_delta_at_ms" not in short_a or "first_delta_at_ms" not in short_b:
    if time.time() > deadline:
        raise SystemExit("short decode probes did not produce first delta")
    time.sleep(0.05)

long_started_at = time.perf_counter()
thread_long = threading.Thread(
    target=post_stream,
    args=("long", load_body(sys.argv[3]), long_result),
)
thread_long.start()

metric_snapshots: list[dict[str, float]] = []
metric_texts: list[str] = []
for _ in range(320):
    try:
        metric_text = scrape_metrics_text()
    except Exception:
        time.sleep(0.1)
        continue
    metric_texts.append(metric_text)
    metric_snapshots.append(parse_metrics(metric_text))
    if not thread_a.is_alive() and not thread_b.is_alive() and not thread_long.is_alive():
        break
    time.sleep(0.1)

thread_a.join(timeout=240)
thread_b.join(timeout=240)
thread_long.join(timeout=240)

if thread_a.is_alive() or thread_b.is_alive() or thread_long.is_alive():
    raise SystemExit("one or more native-v2 Phase 8 probes did not finish")
if short_a.get("status") != 200 or short_b.get("status") != 200 or long_result.get("status") != 200:
    raise SystemExit(
        f"unexpected stream status values: {short_a.get('status')}, {short_b.get('status')}, {long_result.get('status')}"
    )

prefill_key = 'mlx_scheduled_tokens_by_backend{backend="native-mlx",modality="text",phase="prefill"}'
decode_key = 'mlx_scheduled_tokens_by_backend{backend="native-mlx",modality="text",phase="decode"}'
prefill_metric_seen = any(snapshot.get(prefill_key, 0.0) > 0 for snapshot in metric_snapshots)
decode_metric_seen = any(snapshot.get(decode_key, 0.0) > 0 for snapshot in metric_snapshots)
prefill_chunk_sized_seen = any(snapshot.get(prefill_key, 0.0) >= CHUNK_SIZE for snapshot in metric_snapshots)

long_first_delta_ms = long_result.get("first_delta_at_ms")
long_first_delta_abs = long_result.get("first_delta_at_abs")
if long_first_delta_ms is None:
    raise SystemExit("long prompt never produced first delta")
if long_first_delta_abs is None:
    raise SystemExit("long prompt missing absolute first-delta timestamp")

short_progress_before_long_first = 0
for result in (short_a, short_b):
    short_progress_before_long_first += sum(
        1
        for value in result.get("delta_timestamps_abs", [])
        if value > long_started_at and value < long_first_delta_abs
    )

if not short_a.get("text") or not short_b.get("text") or not long_result.get("text"):
    raise SystemExit("one or more native-v2 responses were empty")

summary = {
    "chunk_size": CHUNK_SIZE,
    "prefill_metric_seen": prefill_metric_seen,
    "decode_metric_seen": decode_metric_seen,
    "prefill_chunk_sized_seen": prefill_chunk_sized_seen,
    "short_progress_before_long_first": short_progress_before_long_first,
    "short_a": short_a,
    "short_b": short_b,
    "long": long_result,
}
pathlib.Path(sys.argv[4]).write_text(json.dumps(summary, indent=2, sort_keys=True))
pathlib.Path(sys.argv[5]).write_text("\n\n".join(metric_texts))

itl_values = [value for value in [short_a.get("mean_itl_ms"), short_b.get("mean_itl_ms")] if value is not None]
if not itl_values:
    raise SystemExit("short decode probes did not produce ITL samples")
mean_itl_ms = statistics.fmean(itl_values)
print(f"chunk_{CHUNK_SIZE}_prefill_metric_ok={int(prefill_metric_seen and prefill_chunk_sized_seen)}")
print(f"chunk_{CHUNK_SIZE}_decode_metric_ok={int(decode_metric_seen)}")
print(f"chunk_{CHUNK_SIZE}_interleave_ok={int(short_progress_before_long_first >= 2 and prefill_metric_seen and decode_metric_seen)}")
print(f"chunk_{CHUNK_SIZE}_outputs_ok={int(bool(short_a.get('text')) and bool(short_b.get('text')) and bool(long_result.get('text')))}")
print(f"chunk_{CHUNK_SIZE}_ttft_ms={long_first_delta_ms:.2f}")
print(f"chunk_{CHUNK_SIZE}_mean_itl_ms={mean_itl_ms:.2f}")
PY

    stop_gateway

    uv --directory "$PYTHON_DIR" run python - <<'PY' "$capture_json" "$chunk_size"
from __future__ import annotations

import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
chunk_size = int(sys.argv[2])
if not payload["prefill_metric_seen"]:
    raise SystemExit(f"missing native prefill metric for chunk size {chunk_size}")
if not payload["decode_metric_seen"]:
    raise SystemExit(f"missing native decode metric for chunk size {chunk_size}")
if not payload["prefill_chunk_sized_seen"]:
    raise SystemExit(f"missing chunk-sized prefill metric for chunk size {chunk_size}")
if payload["short_progress_before_long_first"] < 2:
    raise SystemExit(f"short decode requests did not progress before long TTFT for chunk size {chunk_size}")
for name in ("short_a", "short_b", "long"):
    if payload[name]["done"] is not True:
        raise SystemExit(f"{name} missing [DONE] for chunk size {chunk_size}")
    if not payload[name]["text"]:
        raise SystemExit(f"{name} response was empty for chunk size {chunk_size}")
if payload["long"].get("mean_itl_ms") is None and len(payload["long"]["delta_timestamps"]) < 2:
    print(f"chunk_{chunk_size}_long_itl_samples=1")
PY
}

echo "[4/7] Prove real executor physical batching on unequal lengths"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import sys
import time

import mlx.core as mx

from mlx_worker.native_mlx.bootstrap import (
    build_finalized_token_ids,
    build_native_artifacts,
    resolve_model_path,
)
from mlx_worker.native_mlx.interfaces import (
    ExecutionBatch,
    ExecutionRequest,
    SamplingParams,
)


class RecordingModel:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, inputs, *args, **kwargs):
        self.calls += 1
        self.batch_sizes.append(int(inputs.shape[0]))
        return self.inner(inputs, *args, **kwargs)


def prompt_ids(model_path, prefix: str, min_tokens: int) -> tuple[int, ...]:
    words: list[str] = []
    while True:
        start = len(words)
        words.extend(f"{prefix}_{index:04d}" for index in range(start, start + 64))
        candidate = tuple(
            build_finalized_token_ids(
                model_path,
                [{"role": "user", "content": " ".join(words)}],
            )
        )
        if len(candidate) >= min_tokens:
            return candidate[:min_tokens]


def prefill_request(request_id: str, tokens: tuple[int, ...], handle: str) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        phase="prefill",
        token_ids=tokens,
        positions=tuple(range(len(tokens))),
        cache_handle=handle,
        sampling=SamplingParams(),
    )


def decode_request(request_id: str, token_id: int, position: int, handle: str) -> ExecutionRequest:
    return ExecutionRequest(
        request_id=request_id,
        phase="decode",
        token_ids=(token_id,),
        positions=(position,),
        cache_handle=handle,
        sampling=SamplingParams(),
    )


def result_pair_ok(batched, single, atol: float) -> bool:
    return (
        batched.next_token_id == single.next_token_id
        and batched.cache_length == single.cache_length
    )


checkpoint = sys.argv[1]
model_path = resolve_model_path(checkpoint)
artifacts = build_native_artifacts(checkpoint)
executor = artifacts.executor
cache_coordinator = artifacts.cache_coordinator
prompt_a = prompt_ids(model_path, "phase8a", 3)
prompt_b = prompt_ids(model_path, "phase8b", 1)

handle_a = cache_coordinator.acquire("phase8-a", ()).cache_handle
handle_b = cache_coordinator.acquire("phase8-b", ()).cache_handle
ind_a = cache_coordinator.acquire("phase8-ind-a", ()).cache_handle
ind_b = cache_coordinator.acquire("phase8-ind-b", ()).cache_handle
mixed_decode_handle = cache_coordinator.acquire("phase8-mixed-decode", ()).cache_handle
mixed_prefill_handle = cache_coordinator.acquire("phase8-mixed-prefill", ()).cache_handle
split_decode_handle = cache_coordinator.acquire("phase8-split-decode", ()).cache_handle
split_prefill_handle = cache_coordinator.acquire("phase8-split-prefill", ()).cache_handle
isolation_valid_handle = cache_coordinator.acquire(
    "phase8-isolation-valid", ()
).cache_handle
try:
    recorder = RecordingModel(executor.model)
    executor.model = recorder
    prefill = executor.execute_batch(
        ExecutionBatch(
            requests=(
                prefill_request("phase8-a", prompt_a, handle_a),
                prefill_request("phase8-b", prompt_b, handle_b),
            ),
        )
    )
    prefill_calls = recorder.calls
    prefill_batch_size = max(recorder.batch_sizes)

    independent_prefill_a = executor.execute_batch(
        ExecutionBatch(
            requests=(prefill_request("phase8-ind-a", prompt_a, ind_a),),
        )
    )
    independent_prefill_b = executor.execute_batch(
        ExecutionBatch(
            requests=(prefill_request("phase8-ind-b", prompt_b, ind_b),),
        )
    )

    executor.model = RecordingModel(recorder.inner)
    decode = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "phase8-a",
                    int(prefill.results[0].next_token_id),
                    cache_coordinator.length(handle_a),
                    handle_a,
                ),
                decode_request(
                    "phase8-b",
                    int(prefill.results[1].next_token_id),
                    cache_coordinator.length(handle_b),
                    handle_b,
                ),
            ),
        )
    )
    decode_calls = executor.model.calls
    decode_batch_size = max(executor.model.batch_sizes)

    independent_decode_a = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "phase8-ind-a",
                    int(independent_prefill_a.results[0].next_token_id),
                    cache_coordinator.length(ind_a),
                    ind_a,
                ),
            ),
        )
    )
    independent_decode_b = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "phase8-ind-b",
                    int(independent_prefill_b.results[0].next_token_id),
                    cache_coordinator.length(ind_b),
                    ind_b,
                ),
            ),
        )
    )

    physical_batch_observed = prefill_batch_size == 2 and decode_batch_size == 2
    single_forward_per_batch_ok = prefill_calls == 1 and decode_calls == 1
    unequal_lengths_ok = (
        prefill.results[0].cache_length != prefill.results[1].cache_length
        and decode.results[0].cache_length != decode.results[1].cache_length
    )
    parity_ok = all(
        result_pair_ok(batched, single, atol=0.5)
        for batched, single in zip(
            prefill.results,
            (independent_prefill_a.results[0], independent_prefill_b.results[0]),
            strict=True,
        )
    ) and all(
        result_pair_ok(batched, single, atol=0.5)
        for batched, single in zip(
            decode.results,
            (independent_decode_a.results[0], independent_decode_b.results[0]),
            strict=True,
        )
    )

    base_model = recorder.inner
    mixed_decode_prefill = executor.execute_batch(
        ExecutionBatch(
            requests=(
                prefill_request(
                    "phase8-mixed-decode",
                    prompt_a,
                    mixed_decode_handle,
                ),
            ),
        )
    )
    split_decode_prefill = executor.execute_batch(
        ExecutionBatch(
            requests=(
                prefill_request(
                    "phase8-split-decode",
                    prompt_a,
                    split_decode_handle,
                ),
            ),
        )
    )

    mixed_recorder = RecordingModel(base_model)
    executor.model = mixed_recorder
    mixed_started = time.perf_counter()
    mixed = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "phase8-mixed-decode",
                    int(mixed_decode_prefill.results[0].next_token_id),
                    cache_coordinator.length(mixed_decode_handle),
                    mixed_decode_handle,
                ),
                prefill_request(
                    "phase8-mixed-prefill",
                    prompt_b,
                    mixed_prefill_handle,
                ),
            ),
        )
    )
    mixed_elapsed_ms = max(1, int((time.perf_counter() - mixed_started) * 1000))

    split_recorder = RecordingModel(base_model)
    executor.model = split_recorder
    split_started = time.perf_counter()
    split_decode = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "phase8-split-decode",
                    int(split_decode_prefill.results[0].next_token_id),
                    cache_coordinator.length(split_decode_handle),
                    split_decode_handle,
                ),
            ),
        )
    )
    split_prefill = executor.execute_batch(
        ExecutionBatch(
            requests=(
                prefill_request(
                    "phase8-split-prefill",
                    prompt_b,
                    split_prefill_handle,
                ),
            ),
        )
    )
    split_elapsed_ms = max(1, int((time.perf_counter() - split_started) * 1000))

    mixed_single_forward_ok = (
        mixed.forward_mode.value == "mixed"
        and mixed.model_forward_count == 1
        and mixed.physical_batch_size == 2
        and mixed_recorder.calls == 1
        and mixed_recorder.batch_sizes == [2]
        and [item.phase for item in mixed.results] == ["decode", "prefill"]
        and split_recorder.calls == 2
    )
    mixed_parity_ok = result_pair_ok(
        mixed.results[0],
        split_decode.results[0],
        atol=0.5,
    ) and result_pair_ok(
        mixed.results[1],
        split_prefill.results[0],
        atol=0.5,
    )

    isolation_recorder = RecordingModel(base_model)
    executor.model = isolation_recorder
    isolated = executor.execute_batch(
        ExecutionBatch(
            requests=(
                prefill_request(
                    "phase8-invalid",
                    prompt_b,
                    "missing-phase8-cache",
                ),
                prefill_request(
                    "phase8-isolation-valid",
                    prompt_b,
                    isolation_valid_handle,
                ),
            ),
        )
    )
    invalid_result, valid_result = isolated.results
    request_failure_isolation_ok = (
        invalid_result.error_code == "INVALID_EXECUTION_REQUEST"
        and valid_result.error_code is None
        and valid_result.cache_length == len(prompt_b)
        and cache_coordinator.length(isolation_valid_handle) == len(prompt_b)
        and isolated.physical_batch_size == 1
        and isolated.model_forward_count == 1
        and isolation_recorder.calls == 1
        and isolation_recorder.batch_sizes == [1]
    )

    print(f"native_phase8_physical_batch_observed={int(physical_batch_observed)}")
    print(f"native_phase8_single_forward_per_batch_ok={int(single_forward_per_batch_ok)}")
    print(f"native_phase8_unequal_lengths_ok={int(unequal_lengths_ok and parity_ok)}")
    print(f"native_phase8_mixed_single_forward_ok={int(mixed_single_forward_ok)}")
    print(f"native_phase8_mixed_parity_ok={int(mixed_parity_ok)}")
    print(
        "native_phase8_request_failure_isolation_ok="
        f"{int(request_failure_isolation_ok)}"
    )
    print(f"native_phase8_mixed_step_time_ms={mixed_elapsed_ms}")
    print(f"native_phase8_split_step_time_ms={split_elapsed_ms}")
    if not physical_batch_observed:
        raise SystemExit("phase 8 physical batch size proof failed")
    if not single_forward_per_batch_ok:
        raise SystemExit("phase 8 single-forward proof failed")
    if not unequal_lengths_ok or not parity_ok:
        raise SystemExit("phase 8 unequal-length physical batching proof failed")
    if not mixed_single_forward_ok:
        raise SystemExit("phase 8 mixed step did not use one physical forward")
    if not mixed_parity_ok:
        raise SystemExit("phase 8 mixed and split execution parity failed")
    if not request_failure_isolation_ok:
        raise SystemExit("phase 8 request-local failure isolation failed")
finally:
    cache_coordinator.release(handle_a)
    cache_coordinator.release(handle_b)
    cache_coordinator.release(ind_a)
    cache_coordinator.release(ind_b)
    cache_coordinator.release(mixed_decode_handle)
    cache_coordinator.release(mixed_prefill_handle)
    cache_coordinator.release(split_decode_handle)
    cache_coordinator.release(split_prefill_handle)
    cache_coordinator.release(isolation_valid_handle)
PY

probe_chunk_size 32 18082
probe_chunk_size 96 18084

echo "[6/7] Run default v1 non-regression request"
start_v1_gateway "$V1_LOG" "$V1_PORT"
V1_STATUS=$(curl -sS -o "$V1_CAPTURE" -w '%{http_code}' \
    -X POST "http://127.0.0.1:${V1_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data-binary "@$REQUEST_DIR/v1.json")
if [[ "$V1_STATUS" != "200" ]]; then
    echo "unexpected v1 HTTP status: $V1_STATUS; inspect $V1_CAPTURE and $V1_LOG" >&2
    exit 1
fi
stop_gateway

echo "[7/7] Validate v1 response"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$V1_CAPTURE"
from __future__ import annotations

import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
text = payload["choices"][0]["message"]["content"].strip()
if not text:
    raise SystemExit("v1 response text was empty")
print("v1_non_regression_ok=1")
print("phase_8_validation_ok=1")
PY
