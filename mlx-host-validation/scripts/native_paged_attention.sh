#!/usr/bin/env bash
#
# native-v2 paged-attention host-only validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/native_paged_attention.sh [options]
#   bash mlx-host-validation/scripts/native_paged_attention.sh --help
#
# Known-good checkpoint:
#   - `mlx-community/Qwen2.5-7B-Instruct-4bit`
#
# Probe checkpoints:
#   - public native gateway against `mlx-community/Qwen2.5-7B-Instruct-4bit`
#   - direct native executor with the same checkpoint
#   - dense-reference vs native Metal paged-attention parity probe
#   - paged decode kernel vs a float32 reference for every query/KV dtype
#     pairing (this checkpoint runs float16 queries over a bfloat16 cache)
#   - v1 public gateway request against the same checkpoint
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
#   - `mlx_metal_available=1`
#   - `paged_public_outputs_ok=1`
#   - `paged_public_paged_metrics_ok=1`
#   - `paged_public_decode_ok=1`
#   - `paged_direct_decode_dtype_parity_ok=1`
#   - `paged_direct_model_decode_dtypes=<query>/<kv>` (informational)
#   - `paged_direct_paged_kernel_parity_ok=1`
#   - `paged_direct_page_lifecycle_ok=1`
#   - `paged_direct_capacity_failure_ok=1`
#   - `paged_direct_unsupported_config_ok=1`
#   - `paged_direct_mixed_single_forward_ok=1`
#   - `paged_direct_cancellation_cleanup_ok=1`
#   - `v1_non_regression_ok=1`
#   - `paged_validation_ok=1`
#
# Expected failure signals:
#   - non-zero exit
#   - gateway fails readiness or exits unexpectedly (startup warmup decodes
#     through the checkpoint's own query/KV dtypes, so a kernel that cannot
#     compile for them fails readiness)
#   - a multi-token chat completion returns a non-200 status
#   - public metrics miss `paged-mlx` or `native-metal-paged-sdpa`
#   - direct native executor lacks mixed one-forward evidence
#   - direct parity, page lifecycle, capacity, cancellation, or config probes fail

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
# shellcheck source=lib/cli.sh
source "$ROOT/mlx-host-validation/scripts/lib/cli.sh"
# shellcheck source=lib/gateway.sh
source "$ROOT/mlx-host-validation/scripts/lib/gateway.sh"
cli_describe "Validate native-mlx paged attention: kernel parity, page lifecycle, capacity, cancellation, a public-gateway workload, then a v1 non-regression request."
cli_option CHECKPOINT --checkpoint MLX_PAGED_CHECKPOINT string \
    "mlx-community/Qwen2.5-7B-Instruct-4bit" "Checkpoint served by both gateways and the direct probes."
cli_option NATIVE_PORT --native-port MLX_PAGED_NATIVE_PORT port 18092 "native-mlx gateway port."
cli_option V1_PORT --v1-port MLX_PAGED_V1_PORT port 18093 "v1 gateway port."
cli_option TEXT_CACHE_BUDGET_BYTES --text-cache-budget-bytes MLX_PAGED_TEXT_CACHE_BUDGET_BYTES int \
    268435456 "Native paged KV budget in bytes."
cli_option PREFILL_CHUNK_SIZE --prefill-chunk-size MLX_PAGED_PREFILL_CHUNK_SIZE int 16 \
    "Native prefill chunk size for the public-gateway workload."
cli_option HEALTH_TIMEOUT --health-timeout MLX_PAGED_HEALTH_TIMEOUT int 360 \
    "Seconds to wait for each gateway to report healthy."
cli_option TMP_ROOT --artifacts-dir MLX_PAGED_ARTIFACTS_DIR path \
    "${TMPDIR:-/tmp}/mlx-runtime-v2-paged-attention" "Directory for logs, configs, requests, and captures."
cli_option SKIP_BUILD --skip-build MLX_PAGED_SKIP_BUILD bool 0 \
    "Skip uv sync and cargo build; reuse the existing environment and gateway binary."
cli_parse "$@"
cli_require_free_port "$NATIVE_PORT" --native-port
cli_require_free_port "$V1_PORT" --v1-port

REQUEST_DIR="$TMP_ROOT/requests"
NATIVE_CONFIG="$TMP_ROOT/runtime-native.toml"
V1_CONFIG="$TMP_ROOT/runtime-v1.toml"
NATIVE_LOG="$TMP_ROOT/native.log"
V1_LOG="$TMP_ROOT/v1.log"
PUBLIC_CAPTURE="$TMP_ROOT/public-native.json"
PUBLIC_METRICS="$TMP_ROOT/public-native.metrics.txt"
V1_CAPTURE="$TMP_ROOT/v1.json"
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
    gateway_wait_ready "$port" "$HEALTH_TIMEOUT" "$GATEWAY_PID" "$log_path"
}

start_gateway() {
    local log_path="$1"
    local port="$2"
    local config_path="$3"
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "gateway port $port is already in use" >&2
        return 1
    fi
    rm -f "$log_path"
    (
        cd "$ROOT"
        exec env \
            MLX_RUNTIME_CONFIG="$config_path" \
            MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES" \
            MLX_RUNTIME_TEXT_PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" \
            "$GATEWAY_BIN"
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

echo "[1/7] Sync Python environment and build gateway"
if [[ "$SKIP_BUILD" == "1" ]]; then
    echo "build_skipped=1"
else
    uv --directory "$PYTHON_DIR" sync --group dev
    (cd "$ROOT" && cargo build -p mlx_runtime_gateway)
fi

echo "[2/7] Verify Apple Silicon and MLX Metal"
uv --directory "$PYTHON_DIR" run python - <<'PY'
from __future__ import annotations

import platform

import mlx.core as mx

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")
print("mlx_import_ok=1")
if not mx.metal.is_available():
    raise SystemExit("MLX Metal is not available")
print("mlx_metal_available=1")
PY

echo "[3/7] Build runtime configs and request fixtures"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$ROOT/config/runtime.toml" "$NATIVE_CONFIG" "$V1_CONFIG" "$CHECKPOINT" "$NATIVE_PORT" "$V1_PORT" "$REQUEST_DIR"
from __future__ import annotations

import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1]).read_text()
native_target = pathlib.Path(sys.argv[2])
v1_target = pathlib.Path(sys.argv[3])
checkpoint = sys.argv[4]
native_port = sys.argv[5]
v1_port = sys.argv[6]
request_dir = pathlib.Path(sys.argv[7])
request_dir.mkdir(parents=True, exist_ok=True)

native_target.write_text(
    source.replace('port = 8000', f'port = {native_port}')
    .replace('backend = "v1"', 'backend = "native-mlx"')
    .replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
    .replace(
        'ipc_path = "/tmp/mlx-runtime.sock"',
        'ipc_path = "/tmp/mlx-runtime-paged-native.sock"',
    )
)
v1_target.write_text(
    source.replace('backend = "native-mlx"', 'backend = "v1"').replace('port = 8000', f'port = {v1_port}')
    .replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{checkpoint}"',
    )
    .replace(
        'ipc_path = "/tmp/mlx-runtime.sock"',
        'ipc_path = "/tmp/mlx-runtime-paged-v1.sock"',
    )
)

payloads = {
    "short.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": "Count from one to twenty using short comma separated tokens."}],
        "max_tokens": 2,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    },
    "long.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": " ".join(f"paged_prefill_{index:04d}" for index in range(24))}],
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
    },
    "decode.json": {
        "model": checkpoint,
        "messages": [{"role": "user", "content": "List five colors, one per line."}],
        "max_tokens": 24,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
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

echo "[4/7] Run public native gateway paged-KV workload"
start_gateway "$NATIVE_LOG" "$NATIVE_PORT" "$NATIVE_CONFIG"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$REQUEST_DIR/short.json" "$REQUEST_DIR/long.json" "$PUBLIC_CAPTURE" "$PUBLIC_METRICS" "$NATIVE_PORT" "$REQUEST_DIR/decode.json"
from __future__ import annotations

import http.client
import json
import pathlib
import sys
import threading
import time
from typing import Any

PORT = int(sys.argv[5])


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
    result["text_fragments"] = []
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
                result["text_fragments"].append(delta)
        if line == "data: [DONE]":
            break
    conn.close()
    result["done"] = "data: [DONE]" in result["lines"]
    result["text"] = "".join(result["text_fragments"]).strip()


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


short_result: dict[str, Any] = {}
long_result: dict[str, Any] = {}
short = threading.Thread(target=post_stream, args=("short", load_body(sys.argv[1]), short_result))
long = threading.Thread(target=post_stream, args=("long", load_body(sys.argv[2]), long_result))
short.start()
time.sleep(0.2)
long.start()

metric_texts: list[str] = []
snapshots: list[dict[str, float]] = []
for _ in range(360):
    try:
        text = scrape_metrics_text()
    except Exception:
        time.sleep(0.1)
        continue
    metric_texts.append(text)
    snapshots.append(parse_metrics(text))
    if not short.is_alive() and not long.is_alive():
        break
    time.sleep(0.1)

short.join(timeout=240)
long.join(timeout=240)
if short.is_alive() or long.is_alive():
    raise SystemExit("public native paged-attention probes did not finish")
if short_result.get("status") != 200 or long_result.get("status") != 200:
    raise SystemExit(f"unexpected public statuses: {short_result.get('status')}, {long_result.get('status')}")
if not short_result.get("text") or not long_result.get("text"):
    raise SystemExit("public native output was empty")

decode_conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=240)
decode_conn.request(
    "POST",
    "/v1/chat/completions",
    body=load_body(sys.argv[6]),
    headers={"Content-Type": "application/json"},
)
decode_response = decode_conn.getresponse()
decode_body = decode_response.read().decode("utf-8", errors="replace")
decode_conn.close()
if decode_response.status != 200:
    raise SystemExit(f"multi-token chat returned {decode_response.status}: {decode_body[:500]}")
decode_payload = json.loads(decode_body)
decode_text = decode_payload["choices"][0]["message"]["content"].strip()
decode_tokens = int(decode_payload.get("usage", {}).get("completion_tokens", 0))
if not decode_text or decode_tokens < 2:
    raise SystemExit(f"multi-token chat decoded too little: tokens={decode_tokens} text={decode_text!r}")
print(f"paged_public_decode_completion_tokens={decode_tokens}")
print("paged_public_decode_ok=1")

paged_metric = 'mlx_kv_cache_pages_by_backend{backend="paged-mlx",modality="text",state="used"}'
prefill_metric = 'mlx_scheduled_tokens_by_backend{backend="native-mlx",modality="text",phase="prefill"}'
decode_metric = 'mlx_scheduled_tokens_by_backend{backend="native-mlx",modality="text",phase="decode"}'

paged_metrics_ok = any(snapshot.get(paged_metric, 0.0) > 0 for snapshot in snapshots)
attention_ok = any(
    any(
        key.startswith(
            'mlx_attention_time_by_backend_ms{backend="native-metal-paged-sdpa"'
        )
        for key in snapshot
    )
    for snapshot in snapshots
)
prefill_decode_ok = any(snapshot.get(prefill_metric, 0.0) > 0 for snapshot in snapshots) and any(
    snapshot.get(decode_metric, 0.0) > 0 for snapshot in snapshots
)

summary = {
    "short": short_result,
    "long": long_result,
    "decode": decode_payload,
    "paged_metrics_ok": paged_metrics_ok,
    "attention_ok": attention_ok,
    "prefill_decode_ok": prefill_decode_ok,
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(summary, indent=2, sort_keys=True))
pathlib.Path(sys.argv[4]).write_text("\n\n".join(metric_texts))

print("paged_public_outputs_ok=1")
print(f"paged_public_paged_metrics_ok={int(paged_metrics_ok and attention_ok and prefill_decode_ok)}")
if not paged_metrics_ok:
    raise SystemExit("public metrics did not expose paged-mlx used pages")
if not attention_ok:
    raise SystemExit("public metrics did not expose native-metal-paged-sdpa attention")
if not prefill_decode_ok:
    raise SystemExit("public metrics did not expose prefill and decode phases")
PY
stop_gateway

echo "[5/7] Run direct native paged-KV and attention probes"
uv --directory "$PYTHON_DIR" run python - <<'PY' "$CHECKPOINT"
from __future__ import annotations

import sys
import time

import mlx.core as mx

import mlx_worker.native_mlx.attention as native_attention
from mlx_worker.native_mlx.attention import DenseReferenceAttentionBackend, PagedMetalAttentionBackend
from mlx_worker.native_mlx.bootstrap import build_native_artifacts
from mlx_worker.native_mlx.cache import DenseKVCacheBackend, PagedKVCacheBackend
from mlx_worker.native_mlx.interfaces import ExecutionBatch, ExecutionRequest, ForwardMode, SamplingParams
from mlx_worker.native_mlx.paged_decode import paged_decode_attention


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


class RecordingModel:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, inputs, *args, **kwargs):
        self.calls += 1
        self.batch_sizes.append(int(inputs.shape[0]))
        return self.inner(inputs, *args, **kwargs)


checkpoint = sys.argv[1]

dense_backend = DenseKVCacheBackend(num_layers=1)
dense_cache = dense_backend.get(dense_backend.create("dense"), "dense")
dense_reservation = dense_backend.reserve_batch((dense_cache,), (2,))
paged_backend = PagedKVCacheBackend(
    num_layers=1,
    num_kv_heads=1,
    head_dim=4,
    page_size=8,
    budget_bytes=256,
    dtype=mx.float16,
)
paged_cache = paged_backend.get(paged_backend.create("paged"), "paged")
paged_reservation = paged_backend.reserve_batch((paged_cache,), (2,))
queries = mx.array([[[[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 1, 0], [0, 0, 0, 1]]]], dtype=mx.float16)
keys = mx.array([[[[1, 0, 0, 0], [0, 1, 0, 0]]]], dtype=mx.float16)
values = mx.array([[[[1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=mx.float16)
dense = DenseReferenceAttentionBackend().contexts(dense_reservation, ForwardMode.PREFILL)[0].append_and_attend(
    queries, keys, values, scale=0.5, mask="causal"
)
paged = PagedMetalAttentionBackend().contexts(paged_reservation, ForwardMode.PREFILL)[0].append_and_attend(
    queries, keys, values, scale=0.5, mask="causal"
)
mx.eval(dense, paged)
if not mx.allclose(dense, paged, atol=1e-3, rtol=1e-3).item():
    raise SystemExit("paged Metal attention did not match dense reference")
print("paged_direct_paged_kernel_parity_ok=1")

decode_page_size, decode_heads, decode_kv_heads, decode_head_dim = 16, 28, 4, 128
decode_lengths = [37, 300]
decode_tables = [[5, 2, 9], [int(page) for page in range(40, 59)]]
decode_width = max(len(table) for table in decode_tables)
block_tables = mx.array(
    [table + [0] * (decode_width - len(table)) for table in decode_tables], dtype=mx.int32
)
mx.random.seed(11)
base_keys = mx.random.normal((64, decode_page_size, decode_kv_heads, decode_head_dim))
base_values = mx.random.normal((64, decode_page_size, decode_kv_heads, decode_head_dim))
base_queries = mx.random.normal((len(decode_lengths), decode_heads, 1, decode_head_dim))
for query_dtype, kv_dtype in (
    (mx.float16, mx.float16),
    (mx.bfloat16, mx.bfloat16),
    (mx.float16, mx.bfloat16),
    (mx.bfloat16, mx.float16),
):
    key_pages = base_keys.astype(kv_dtype)
    value_pages = base_values.astype(kv_dtype)
    decode_queries = base_queries.astype(query_dtype)
    output = paged_decode_attention(
        decode_queries,
        key_pages,
        value_pages,
        block_tables,
        mx.array(decode_lengths, dtype=mx.int32),
        max_length=max(decode_lengths),
        scale=decode_head_dim**-0.5,
    )
    if output.dtype != query_dtype:
        raise SystemExit(f"paged decode output dtype {output.dtype} != query dtype {query_dtype}")
    for row, (length, table) in enumerate(zip(decode_lengths, decode_tables, strict=True)):

        def dense(pool, table=table, length=length):
            rows = pool[mx.array(table)].reshape(-1, decode_kv_heads, decode_head_dim)[:length]
            return rows.transpose(1, 0, 2)[None].astype(mx.float32)

        expected = mx.fast.scaled_dot_product_attention(
            decode_queries[row : row + 1].astype(mx.float32),
            dense(key_pages),
            dense(value_pages),
            scale=decode_head_dim**-0.5,
        )
        error = float(mx.max(mx.abs(output[row : row + 1].astype(mx.float32) - expected)).item())
        print(f"paged_direct_decode_max_abs_error[{query_dtype}/{kv_dtype}][{length}]={error:.5f}")
        if error > 2e-2:
            raise SystemExit(f"paged decode {query_dtype}/{kv_dtype} diverged from reference at {length} keys")
print("paged_direct_decode_dtype_parity_ok=1")

lifecycle = PagedKVCacheBackend(
    num_layers=1,
    num_kv_heads=1,
    head_dim=2,
    page_size=8,
    budget_bytes=256,
    dtype=mx.float16,
)
parent_handle = lifecycle.create("parent")
parent = lifecycle.get(parent_handle, "parent")
reservation = lifecycle.reserve_batch((parent,), (3,))
reservation.stage_layer(
    0,
    mx.ones((1, 1, 3, 2), dtype=mx.float16),
    mx.ones((1, 1, 3, 2), dtype=mx.float16),
)
reservation.commit()
child_handle = lifecycle.fork(parent_handle, "child")
child = lifecycle.get(child_handle, "child")
append = lifecycle.reserve_batch((child,), (1,))
append.stage_layer(
    0,
    mx.ones((1, 1, 1, 2), dtype=mx.float16),
    mx.ones((1, 1, 1, 2), dtype=mx.float16),
)
append.commit()
if parent.size() != 3 or child.size() != 4 or parent.block_table == child.block_table:
    raise SystemExit("paged fork/COW lifecycle failed")
lifecycle.release(parent_handle)
lifecycle.release(child_handle)
if lifecycle.metrics()["used_pages"] != 0:
    raise SystemExit("paged page reclamation failed")
print("paged_direct_page_lifecycle_ok=1")

limited = PagedKVCacheBackend(
    num_layers=1,
    num_kv_heads=1,
    head_dim=2,
    page_size=8,
    budget_bytes=64,
    dtype=mx.float16,
)
first = limited.get(limited.create("first"), "first")
second = limited.get(limited.create("second"), "second")
errors = limited.preflight((first, second), (1, 1))
if errors[0] is not None or errors[1] != "native paged KV capacity exhausted before model execution":
    raise SystemExit("paged capacity preflight did not isolate failing request")
if limited.metrics()["used_pages"] != 0:
    raise SystemExit("paged capacity failure mutated cache")
print("paged_direct_capacity_failure_ok=1")

try:
    PagedKVCacheBackend(
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        page_size=7,
        budget_bytes=64,
        dtype=mx.float16,
    )
except ValueError:
    print("paged_direct_unsupported_config_ok=1")
else:
    raise SystemExit("unsupported page size did not fail")

model_decode_dtypes: set[str] = set()
real_paged_decode = native_attention.paged_decode_attention


def recording_paged_decode(queries, key_pages, *args, **kwargs):
    model_decode_dtypes.add(f"{queries.dtype}/{key_pages.dtype}".replace("mlx.core.", ""))
    return real_paged_decode(queries, key_pages, *args, **kwargs)


native_attention.paged_decode_attention = recording_paged_decode
artifacts = build_native_artifacts(checkpoint, cache_budget_bytes=32 * 1024 * 1024, kv_page_size=16)
executor = artifacts.executor
coordinator = artifacts.cache_coordinator
base_model = executor.model
recorder = RecordingModel(base_model)
executor.model = recorder
prompt_a = tuple(range(11, 19))
prompt_b = tuple(range(31, 34))
handle_a = coordinator.acquire("paged-a", prompt_a).cache_handle
handle_b = coordinator.acquire("paged-b", prompt_b).cache_handle
cancel_handle = coordinator.acquire("paged-cancel", (1, 2, 3)).cache_handle
try:
    first_step = executor.execute_batch(
        ExecutionBatch(requests=(prefill_request("paged-a", prompt_a, handle_a),))
    )
    mixed = executor.execute_batch(
        ExecutionBatch(
            requests=(
                decode_request(
                    "paged-a",
                    int(first_step.results[0].next_token_id),
                    coordinator.length(handle_a),
                    handle_a,
                ),
                prefill_request("paged-b", prompt_b, handle_b),
            )
        )
    )
    mixed_ok = (
        mixed.forward_mode.value == "mixed"
        and mixed.model_forward_count == 1
        and mixed.physical_batch_size == 2
        and recorder.calls == 2
        and recorder.batch_sizes[-1] == 2
        and mixed.metrics["cache_backend"] == "paged-mlx"
        and mixed.metrics["attention_backend"] == "native-metal-paged-sdpa"
    )
    if not mixed_ok:
        raise SystemExit("direct mixed native execution did not use one paged forward")
    print("paged_direct_mixed_single_forward_ok=1")

    next_tokens = {result.request_id: int(result.next_token_id) for result in mixed.results}
    decode_step = executor.execute_batch(
        ExecutionBatch(
            requests=tuple(
                decode_request(request_id, next_tokens[request_id], coordinator.length(handle), handle)
                for request_id, handle in (("paged-a", handle_a), ("paged-b", handle_b))
            )
        )
    )
    if decode_step.forward_mode.value != "decode" or len(decode_step.results) != 2:
        raise SystemExit("direct two-row decode step did not run as decode")
    if not model_decode_dtypes:
        raise SystemExit("direct decode step never reached the paged decode kernel")
    print(f"paged_direct_model_decode_dtypes={','.join(sorted(model_decode_dtypes))}")

    executor.execute_batch(
        ExecutionBatch(
            requests=(prefill_request("paged-cancel", (1, 2, 3), cancel_handle),)
        )
    )
    before_cancel_pages = executor.cache_backend.metrics()["used_pages"]
    coordinator.release(cancel_handle)
    after_cancel_pages = executor.cache_backend.metrics()["used_pages"]
    if after_cancel_pages >= before_cancel_pages:
        raise SystemExit("cancellation cleanup did not reclaim pages")
    print("paged_direct_cancellation_cleanup_ok=1")
finally:
    coordinator.release(handle_a)
    coordinator.release(handle_b)
    coordinator.release(cancel_handle)
PY

echo "[6/7] Run v1 non-regression request"
start_gateway "$V1_LOG" "$V1_PORT" "$V1_CONFIG"
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
print("paged_validation_ok=1")
PY
