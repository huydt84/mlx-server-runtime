#!/usr/bin/env bash
#
# Phase 10 host-only validation for text LLM and VLM continuous batching.
# Run on Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_10.sh
#
# Optional overrides:
#   MLX_PHASE10_TEXT_MODEL=<mlx-community/...>
#   MLX_PHASE10_VLM_MODEL=<mlx-community/...>
#   MLX_PHASE10_IMAGE_PATH=/absolute/path/to/local-image.png
#
# What this verifies:
#   1. Python worker imports and gateway start on host with both text and VLM models enabled.
#   2. Serialized baseline and continuous-batch mode both serve real text and VLM `/v1/chat/completions` requests.
#   3. Text LLM continuous mode accepts a new request while another text request is decoding.
#   4. VLM continuous mode accepts a new image-bearing request while another compatible VLM request is decoding.
#   5. VLM text-only and image-bearing requests both route through the configured VLM model.
#   6. Streaming requests return real TTFT and completion usage data for both backend families.
#   7. Shared-prefix and repeated-image workloads leave cache and VLM metrics observable in `/metrics`.
#   8. Mixed text+VLM overlap proves both families make forward progress together.
#   9. Cancellation by client disconnect increments request-cancelled metrics and does not block follow-up work.
#   10. Script records text and VLM p50/p95 TTFT, latency, completion tokens/sec, prompt tokens/sec,
#       decode tokens/sec, cache metrics, VLM request/image counts, image dimensions,
#       APC/vision-feature cache hit rates, and VLM media-preparation timings.
#
# Expected verification signal:
#   - Script exits with status code 0.
#   - It prints `phase_10_validation_ok=1`.
#   - It prints `baseline_ready=1` and `continuous_ready=1`.
#   - It prints text and VLM workload summaries for both modes.
#   - It prints `baseline_metrics_ok=1`, `continuous_metrics_ok=1`, and `vlm_metrics_ok=1`.
#   - It prints `mixed_backend_fairness_ok=1` and `throughput_improved_ok=1`.
#   - It prints `continuous_join_while_decoding_ok=1` and `vlm_continuous_join_while_decoding_ok=1`.
#   - It prints `cache_hits_ok=1`, `vlm_cache_hits_ok=1`, `vlm_local_image_ok=1`, and `cancellation_ok=1`.
#   - If validation fails, script exits non-zero and points to captured gateway log paths.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"
BASE_RUNTIME_CONFIG="$ROOT/config/runtime.toml"
TMP_ROOT="${TMPDIR:-/tmp}/mlx-runtime-phase-10"
PHASE10_RUNTIME_CONFIG="$TMP_ROOT/runtime.phase10.toml"
BASELINE_LOG="$TMP_ROOT/baseline-gateway.log"
CONTINUOUS_LOG="$TMP_ROOT/continuous-gateway.log"
BASELINE_SUMMARY="$TMP_ROOT/baseline-summary.json"
CONTINUOUS_SUMMARY="$TMP_ROOT/continuous-summary.json"
TEXT_CACHE_BUDGET_BYTES=$((64 * 1024 * 1024))
VLM_APC_CACHE_BUDGET_BYTES=$((64 * 1024 * 1024))
VISION_FEATURE_CACHE_BUDGET_BYTES=$((64 * 1024 * 1024))
TEXT_MODEL="${MLX_PHASE10_TEXT_MODEL:-mlx-community/Qwen2.5-7B-Instruct-4bit}"
VLM_MODEL="${MLX_PHASE10_VLM_MODEL:-mlx-community/Qwen2-VL-2B-Instruct-4bit}"
VLM_IMAGE_PATH="${MLX_PHASE10_IMAGE_PATH:-$ROOT/benchmarks/images/fruits.png}"

mkdir -p "$TMP_ROOT"

cleanup() {
    if [[ -n "${GATEWAY_PID:-}" ]]; then
        kill "$GATEWAY_PID" >/dev/null 2>&1 || true
        wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    fi
}

trap cleanup EXIT

if [[ ! -f "$VLM_IMAGE_PATH" ]]; then
    echo "FAIL: local VLM image fixture missing: $VLM_IMAGE_PATH" >&2
    exit 1
fi

python3 - "$BASE_RUNTIME_CONFIG" "$PHASE10_RUNTIME_CONFIG" "$TEXT_MODEL" "$VLM_MODEL" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
dest = Path(sys.argv[2])
text_model = sys.argv[3]
vlm_model = sys.argv[4]

contents = source.read_text(encoding="utf-8")
contents = re.sub(
    r'(?m)^model = ".*"$',
    f'model = "{text_model}"',
    contents,
    count=1,
)
contents = re.sub(
    r'(?m)^# vlm_model = ".*"$',
    f'vlm_model = "{vlm_model}"',
    contents,
    count=1,
)
if 'vlm_model =' not in contents:
    contents = re.sub(
        r'(?m)^(model = ".*")$',
        r'\1\n' + f'vlm_model = "{vlm_model}"',
        contents,
        count=1,
    )
dest.write_text(contents, encoding="utf-8")
PY

echo "[1/6] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

run_mode() {
    local mode_name="$1"
    local continuous_flag="$2"
    local log_path="$3"
    local summary_path="$4"

    echo "[mode:${mode_name}] Start gateway"
    cd "$ROOT"
    rm -f "$log_path" "$summary_path"
    MLX_RUNTIME_CONFIG="$PHASE10_RUNTIME_CONFIG" \
    MLX_RUNTIME_CONTINUOUS_BATCHING="$continuous_flag" \
    MLX_RUNTIME_PROMPT_CONCURRENCY=4 \
    MLX_RUNTIME_DECODE_CONCURRENCY=4 \
    MLX_RUNTIME_PREFILL_CHUNK_SIZE=256 \
    MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES" \
    MLX_RUNTIME_VLM_APC_CACHE_BUDGET_BYTES="$VLM_APC_CACHE_BUDGET_BYTES" \
    MLX_RUNTIME_VISION_FEATURE_CACHE_BUDGET_BYTES="$VISION_FEATURE_CACHE_BUDGET_BYTES" \
    cargo run -p mlx_runtime_gateway >"$log_path" 2>&1 &
    GATEWAY_PID=$!

    echo "[mode:${mode_name}] Wait for readiness"
    if ! MODE="$mode_name" TEXT_MODEL="$TEXT_MODEL" python3 - <<'PY'
import json
import os
import sys
import time
import urllib.request

text_model = os.environ["TEXT_MODEL"]

req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/chat/completions",
    data=json.dumps(
        {
            "model": text_model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    ).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

for _ in range(360):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2):
            pass
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload["choices"][0]["message"]["content"].strip():
            raise SystemExit(0)
    except Exception:
        time.sleep(1)

raise SystemExit(1)
PY
    then
        if ! kill -0 "$GATEWAY_PID" >/dev/null 2>&1; then
            echo "FAIL: gateway exited early; inspect $log_path" >&2
        else
            echo "FAIL: gateway never became ready; inspect $log_path" >&2
        fi
        exit 1
    fi

    MODE="$mode_name" \
    TEXT_MODEL="$TEXT_MODEL" \
    VLM_MODEL="$VLM_MODEL" \
    VLM_IMAGE_PATH="$VLM_IMAGE_PATH" \
    python3 - "$summary_path" <<'PY'
import http.client
import json
import os
import sys
import threading
import time
from pathlib import Path

MODE = os.environ["MODE"]
OUT = Path(sys.argv[1])
HOST = "127.0.0.1"
PORT = 8000
TEXT_MODEL = os.environ["TEXT_MODEL"]
VLM_MODEL = os.environ["VLM_MODEL"]
VLM_IMAGE_PATH = str(Path(os.environ["VLM_IMAGE_PATH"]).resolve())


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def text_payload(prompt, *, stream=True, max_tokens=16):
    return {
        "model": TEXT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def vlm_payload(prompt, *, stream=True, max_tokens=32, with_image=True):
    content = [{"type": "text", "text": prompt}]
    if with_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": VLM_IMAGE_PATH, "detail": "auto"},
            }
        )
    return {
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": content if with_image else prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def request_once(payload, *, first_token_event=None):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    encoded = json.dumps(payload)
    started = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        body = resp.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"unexpected status {resp.status}: {body}")

    first_chunk_at = None
    first_printable_at = None
    usage = None
    text_parts = []
    response_fields = {}
    if payload.get("stream", False):
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            event = json.loads(data.decode("utf-8"))
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
            for key in (
                "prompt_cache_hit",
                "cached_tokens",
                "prompt_cache_bytes",
                "decode_batch_size",
                "prompt_batch_size",
                "configured_prompt_batch_size",
                "configured_decode_batch_size",
                "active_batch_cache_bytes",
                "modality",
                "apc_mode",
                "scheduler_tick_latency_ms",
                "arbitration_delay_ms",
                "worker_cancellation_count",
                "worker_error_count",
                "vision_feature_cache_hit",
                "vision_feature_cache_bytes",
                "vision_feature_cache_entries",
                "vision_feature_cache_evictions",
                "vision_encoder_latency_ms",
                "embedding_latency_ms",
                "prompt_cache_entries",
                "prompt_cache_evictions",
                "peak_memory_bytes",
                "image_width",
                "image_height",
                "backend",
                "scheduler_stage",
                "cancellation_stage",
                "queue_time_ms",
                "prefill_time_ms",
                "ttft_ms",
                "decode_time_ms",
                "completion_time_ms",
                "image_count",
                "image_preprocess_latency_ms",
                "prompt_template_latency_ms",
            ):
                value = event.get(key)
                if value is not None:
                    response_fields[key] = value
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    text_parts.append(delta)
                    if first_printable_at is None and delta.strip():
                        first_printable_at = time.perf_counter()
                    if first_token_event is not None:
                        first_token_event.set()
            else:
                usage = event.get("usage")
    else:
        payload_body = json.loads(resp.read().decode("utf-8"))
        usage = payload_body.get("usage")
        text_parts.append(payload_body["choices"][0]["message"]["content"])
        response_fields.update(
            {
                key: payload_body.get(key)
                for key in (
                    "prompt_cache_hit",
                    "cached_tokens",
                    "prompt_cache_bytes",
                    "decode_batch_size",
                    "prompt_batch_size",
                    "configured_prompt_batch_size",
                    "configured_decode_batch_size",
                    "active_batch_cache_bytes",
                    "modality",
                    "apc_mode",
                    "scheduler_tick_latency_ms",
                    "arbitration_delay_ms",
                    "worker_cancellation_count",
                    "worker_error_count",
                    "vision_feature_cache_hit",
                    "vision_feature_cache_bytes",
                    "vision_feature_cache_entries",
                    "vision_feature_cache_evictions",
                    "vision_encoder_latency_ms",
                    "embedding_latency_ms",
                    "prompt_cache_entries",
                    "prompt_cache_evictions",
                    "peak_memory_bytes",
                    "image_width",
                    "image_height",
                    "backend",
                    "scheduler_stage",
                    "cancellation_stage",
                    "queue_time_ms",
                    "prefill_time_ms",
                    "ttft_ms",
                    "decode_time_ms",
                    "completion_time_ms",
                    "image_count",
                    "image_preprocess_latency_ms",
                    "prompt_template_latency_ms",
                )
            }
        )

    completed_at = time.perf_counter()
    conn.close()
    return {
        "started_at": started,
        "ttft_ms": (
            None
            if first_printable_at is None
            else (first_printable_at - started) * 1000.0
        ),
        "latency_ms": (completed_at - started) * 1000.0,
        "first_chunk_at": first_chunk_at,
        "first_printable_at": first_printable_at,
        "completed_at": completed_at,
        "prompt_tokens": 0 if usage is None else int(usage.get("prompt_tokens", 0)),
        "completion_tokens": 0 if usage is None else int(usage.get("completion_tokens", 0)),
        "text": "".join(text_parts),
        **response_fields,
    }


def concurrent_requests(payloads):
    results = [None] * len(payloads)
    errors = []

    def run(index, payload):
        try:
            results[index] = request_once(payload)
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=run, args=(idx, payload)) for idx, payload in enumerate(payloads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def metrics_snapshot():
    conn = http.client.HTTPConnection(HOST, PORT, timeout=60)
    conn.request("GET", "/metrics")
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"metrics status {resp.status}")
    body = resp.read().decode("utf-8")
    conn.close()
    return body


def metric_value(metrics, name):
    for line in metrics.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return 0.0


def join_while_decoding(first_payload, second_payload):
    first_token = threading.Event()
    first_result = {}
    errors = []

    def run_first():
        try:
            first_result["value"] = request_once(
                first_payload, first_token_event=first_token
            )
        except Exception as exc:
            errors.append(exc)
            first_token.set()

    thread = threading.Thread(target=run_first)
    thread.start()
    if not first_token.wait(timeout=240):
        raise RuntimeError("first request produced no token before join timeout")
    if errors:
        raise errors[0]

    started_while_first_active = thread.is_alive()
    second = request_once(second_payload)
    join_metrics = metrics_snapshot()
    join_decode_batch_size = second.get("decode_batch_size") or metric_value(
        join_metrics, "mlx_decode_batch_size"
    )
    thread.join(timeout=300)
    if thread.is_alive():
        raise RuntimeError("first request did not finish after dynamic join")
    if errors:
        raise errors[0]
    return {
        "first": first_result["value"],
        "second": second,
        "started_while_first_active": started_while_first_active,
        "join_decode_batch_size": join_decode_batch_size,
    }


def cancel_by_disconnect(payload, *, wait_for_first_printable=False):
    conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
    encoded = json.dumps(payload)
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"unexpected cancel status {resp.status}")
    if wait_for_first_printable:
        while True:
            line = resp.readline()
            if not line:
                raise RuntimeError("stream produced no chunk before cancel")
            if line.startswith(b"data: ") and b'"delta"' in line:
                break
    conn.close()


def aggregate_requests(items):
    ttft_values = [item["ttft_ms"] for item in items if item["ttft_ms"] is not None]
    latency_values = [item["latency_ms"] for item in items]
    prompt_tokens = sum(item["prompt_tokens"] for item in items)
    completion_tokens = sum(item["completion_tokens"] for item in items)
    latency_total_ms = sum(latency_values)
    decode_total_ms = sum(
        item["latency_ms"] - (item["ttft_ms"] or 0.0) for item in items
    )
    prompt_cache_hits = sum(1 for item in items if item.get("prompt_cache_hit"))
    prompt_cache_misses = sum(1 for item in items if item.get("prompt_cache_hit") is False)
    vision_cache_hits = sum(1 for item in items if item.get("vision_feature_cache_hit"))
    vision_cache_misses = sum(
        1 for item in items if item.get("vision_feature_cache_hit") is False
    )
    prompt_cache_bytes = max(
        [int(item.get("prompt_cache_bytes") or 0) for item in items] + [0]
    )
    vision_feature_cache_bytes = max(
        [int(item.get("vision_feature_cache_bytes") or 0) for item in items] + [0]
    )
    image_count = sum(int(item.get("image_count") or 0) for item in items)
    image_width = max([int(item.get("image_width") or 0) for item in items] + [0])
    image_height = max([int(item.get("image_height") or 0) for item in items] + [0])
    peak_memory_bytes = max(
        [int(item.get("peak_memory_bytes") or 0) for item in items] + [0]
    )
    worker_cancellations = max(
        [int(item.get("worker_cancellation_count") or 0) for item in items] + [0]
    )
    worker_errors = max([int(item.get("worker_error_count") or 0) for item in items] + [0])
    scheduler_tick_latency_ms = max(
        [int(item.get("scheduler_tick_latency_ms") or 0) for item in items] + [0]
    )
    arbitration_delay_ms = max(
        [int(item.get("arbitration_delay_ms") or 0) for item in items] + [0]
    )
    total_latency_sec = max(sum(latency_values) / 1000.0, 0.001)
    decode_total_sec = max(decode_total_ms / 1000.0, 0.001)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_total_ms": latency_total_ms,
        "decode_total_ms": decode_total_ms,
        "ttft_p50_ms": percentile(ttft_values, 0.5),
        "ttft_p95_ms": percentile(ttft_values, 0.95),
        "latency_p50_ms": percentile(latency_values, 0.5),
        "latency_p95_ms": percentile(latency_values, 0.95),
        "prompt_tokens_per_sec": prompt_tokens / total_latency_sec,
        "completion_tokens_per_sec": completion_tokens / total_latency_sec,
        "decode_tokens_per_sec": completion_tokens / decode_total_sec,
        "prompt_cache_hits": prompt_cache_hits,
        "prompt_cache_misses": prompt_cache_misses,
        "prompt_cache_hit_rate": (
            prompt_cache_hits / max(prompt_cache_hits + prompt_cache_misses, 1)
        ),
        "prompt_cache_bytes": prompt_cache_bytes,
        "vision_cache_hits": vision_cache_hits,
        "vision_cache_misses": vision_cache_misses,
        "vision_cache_hit_rate": (
            vision_cache_hits / max(vision_cache_hits + vision_cache_misses, 1)
        ),
        "vision_feature_cache_bytes": vision_feature_cache_bytes,
        "image_count": image_count,
        "image_width": image_width,
        "image_height": image_height,
        "peak_memory_bytes": peak_memory_bytes,
        "worker_cancellations": worker_cancellations,
        "worker_errors": worker_errors,
        "scheduler_tick_latency_ms": scheduler_tick_latency_ms,
        "arbitration_delay_ms": arbitration_delay_ms,
    }


def run_text_suite():
    started = time.perf_counter()
    decode_only = request_once(text_payload("Say hello in one short sentence.", stream=True))
    short = concurrent_requests(
        [
            text_payload("Short prompt alpha."),
            text_payload("Short prompt beta."),
            text_payload("Short prompt gamma."),
        ]
    )
    long_prompt = " ".join(["Long prompt" for _ in range(128)])
    long = concurrent_requests(
        [
            text_payload(long_prompt),
            text_payload(long_prompt + " extra"),
        ]
    )
    dynamic_join = join_while_decoding(
        text_payload(
            "Count from 1 to 100, writing every number without abbreviating.",
            max_tokens=128,
        ),
        text_payload("Reply with exactly: joined", stream=False, max_tokens=16),
    )
    shared_prefix = concurrent_requests(
        [
            text_payload("Shared prefix: Explain prefix cache in one sentence."),
            text_payload(
                "Shared prefix: Explain prefix cache in one sentence. Use different wording."
            ),
        ]
    )
    cache_hit_seed = request_once(
        text_payload("Deterministic cache hit probe prefix.", stream=False)
    )
    cache_hit_followup = request_once(
        text_payload(
            "Deterministic cache hit probe prefix. Extended follow-up.",
            stream=False,
        )
    )
    cancel_by_disconnect(
        text_payload(
            "Stream this response and then cancel it before the first token.",
            max_tokens=96,
            stream=True,
        )
    )
    cancel_by_disconnect(
        text_payload(
            "Stream this response and then cancel it after the first visible token.",
            max_tokens=128,
            stream=True,
        ),
        wait_for_first_printable=True,
    )
    cancel_followup = request_once(text_payload("Follow-up request after cancellation."))
    cache_pressure = concurrent_requests(
        [
            text_payload("Cache pressure prompt A."),
            text_payload("Cache pressure prompt A."),
            text_payload("Cache pressure prompt A."),
            text_payload("Cache pressure prompt A."),
        ]
    )
    mixed_backend = join_while_decoding(
        text_payload(
            "Count from 1 to 100, writing every number without abbreviating.",
            max_tokens=128,
            stream=True,
        ),
        vlm_payload(
            "Reply with exactly one sentence about the local image.",
            stream=True,
            max_tokens=24,
            with_image=True,
        ),
    )
    all_requests = (
        [decode_only]
        + [cache_hit_seed, cache_hit_followup]
        + short
        + long
        + [dynamic_join["first"], dynamic_join["second"]]
        + shared_prefix
        + [cancel_followup]
        + cache_pressure
    )
    return {
        "suite_wall_clock_ms": (time.perf_counter() - started) * 1000.0,
        "decode_only": decode_only,
        "cache_hit_seed": cache_hit_seed,
        "cache_hit_followup": cache_hit_followup,
        "short_prefill": short,
        "long_prefill": long,
        "dynamic_join": dynamic_join,
        "shared_prefix": shared_prefix,
        "cancel_followup": cancel_followup,
        "cache_pressure": cache_pressure,
        "mixed_backend": mixed_backend,
        **aggregate_requests(all_requests),
    }


def run_vlm_suite():
    started = time.perf_counter()
    decode_only = request_once(
        vlm_payload(
            "Describe this local image in one short sentence.",
            stream=True,
            max_tokens=32,
            with_image=True,
        )
    )
    dynamic_join = join_while_decoding(
        vlm_payload(
            "Describe the local image in exactly 12 short numbered bullet points, each mentioning a visible object or color.",
            max_tokens=96,
            with_image=True,
        ),
        vlm_payload(
            "Reply with exactly one sentence about the same local image.",
            stream=False,
            max_tokens=24,
            with_image=True,
        ),
    )
    mixed_modal = concurrent_requests(
        [
            vlm_payload(
                "Answer this text-only request through the VLM model.",
                with_image=False,
                max_tokens=24,
            ),
            vlm_payload(
                "Describe the local image in one sentence.",
                with_image=True,
                max_tokens=24,
            ),
        ]
    )
    repeated_image_seed = request_once(
        vlm_payload(
            "Describe the local image in one sentence.",
            with_image=True,
            max_tokens=24,
            stream=False,
        )
    )
    repeated_image_followup = request_once(
        vlm_payload(
            "Describe the local image in one sentence.",
            with_image=True,
            max_tokens=24,
            stream=False,
        )
    )
    cancel_by_disconnect(
        vlm_payload(
            "Stream a detailed answer about this image and then cancel before the first token.",
            with_image=True,
            max_tokens=48,
            stream=True,
        )
    )
    cancel_by_disconnect(
        vlm_payload(
            "Stream a detailed answer about this image and then cancel after the first visible token.",
            with_image=True,
            max_tokens=64,
            stream=True,
        ),
        wait_for_first_printable=True,
    )
    shared_apc = concurrent_requests(
        [
            vlm_payload(
                "Describe the same local image in one sentence.",
                with_image=True,
                max_tokens=24,
                stream=False,
            ),
            vlm_payload(
                "Describe the same local image in one sentence.",
                with_image=True,
                max_tokens=24,
                stream=False,
            ),
        ]
    )
    repeat_image_seed = request_once(
        vlm_payload(
            "Describe the same local image in one sentence.",
            with_image=True,
            max_tokens=24,
            stream=False,
        )
    )
    repeat_image_followup = request_once(
        vlm_payload(
            "Describe the same local image in one sentence.",
            with_image=True,
            max_tokens=24,
            stream=False,
        )
    )
    cancel_followup = request_once(
        vlm_payload(
            "Follow-up VLM request after cancellation. Describe the same image briefly.",
            with_image=True,
            max_tokens=24,
        )
    )
    all_requests = (
        [decode_only]
        + [dynamic_join["first"], dynamic_join["second"]]
        + mixed_modal
        + shared_apc
        + [repeat_image_seed, repeat_image_followup]
        + [cancel_followup]
    )
    return {
        "suite_wall_clock_ms": (time.perf_counter() - started) * 1000.0,
        "local_image_path": VLM_IMAGE_PATH,
        "decode_only": decode_only,
        "dynamic_join": dynamic_join,
        "mixed_modal": mixed_modal,
        "shared_apc": shared_apc,
        "repeated_image_seed": repeat_image_seed,
        "repeated_image_followup": repeat_image_followup,
        "cancel_followup": cancel_followup,
        **aggregate_requests(all_requests),
    }


text = run_text_suite()
vlm = run_vlm_suite()
metrics = metrics_snapshot()

summary = {
    "mode": MODE,
    "text": text,
    "vlm": vlm,
    "metrics": {
        "prompt_cache_hits_total": metric_value(metrics, "mlx_prompt_cache_hits_total"),
        "prompt_cache_misses_total": metric_value(metrics, "mlx_prompt_cache_misses_total"),
        "prompt_cache_cached_tokens_total": metric_value(
            metrics, "mlx_prompt_cache_cached_tokens_total"
        ),
        "prompt_cache_bytes": metric_value(metrics, "mlx_prompt_cache_bytes"),
        "active_batch_cache_bytes": metric_value(metrics, "mlx_active_batch_cache_bytes"),
        "worker_memory_bytes": metric_value(metrics, "mlx_worker_memory_bytes"),
        "requests_cancelled_total": metric_value(metrics, "mlx_requests_cancelled_total"),
        "vlm_requests_total": metric_value(metrics, "mlx_vlm_requests_total"),
        "vlm_image_count_total": metric_value(metrics, "mlx_vlm_image_count_total"),
        "vlm_image_preprocess_latency_ms": metric_value(
            metrics, "mlx_vlm_image_preprocess_latency_ms"
        ),
        "vlm_prompt_template_latency_ms": metric_value(
            metrics, "mlx_vlm_prompt_template_latency_ms"
        ),
        "vlm_load_errors_total": metric_value(metrics, "mlx_vlm_load_errors_total"),
    },
}
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

    kill "$GATEWAY_PID" >/dev/null 2>&1 || true
    wait "$GATEWAY_PID" >/dev/null 2>&1 || true
    unset GATEWAY_PID
}

echo "[2/6] Run baseline serialized workload"
run_mode "baseline" 0 "$BASELINE_LOG" "$BASELINE_SUMMARY"
echo "baseline_ready=1"

echo "[3/6] Run continuous-batch workload"
run_mode "continuous" 1 "$CONTINUOUS_LOG" "$CONTINUOUS_SUMMARY"
echo "continuous_ready=1"

echo "[4/6] Compare summaries"
TEXT_CACHE_BUDGET_BYTES="$TEXT_CACHE_BUDGET_BYTES" \
VLM_APC_CACHE_BUDGET_BYTES="$VLM_APC_CACHE_BUDGET_BYTES" \
VISION_FEATURE_CACHE_BUDGET_BYTES="$VISION_FEATURE_CACHE_BUDGET_BYTES" \
python3 - "$ROOT" "$BASELINE_SUMMARY" "$CONTINUOUS_SUMMARY" "$VLM_IMAGE_PATH" <<'PY'
import json
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from pathlib import Path

ROOT = Path(sys.argv[1])
baseline = json.loads(Path(sys.argv[2]).read_text())
continuous = json.loads(Path(sys.argv[3]).read_text())
VLM_IMAGE_PATH = str(Path(sys.argv[4]).resolve())
TEXT_CACHE_BUDGET_BYTES = int(os.environ["TEXT_CACHE_BUDGET_BYTES"])
VLM_APC_CACHE_BUDGET_BYTES = int(os.environ["VLM_APC_CACHE_BUDGET_BYTES"])
VISION_FEATURE_CACHE_BUDGET_BYTES = int(os.environ["VISION_FEATURE_CACHE_BUDGET_BYTES"])

sys.path.insert(0, str(ROOT / "python"))

from mlx_worker.batching import PromptCacheStore
from mlx_worker.ipc import ChatCompletionRequest, ChatMessage, ImageContent, TextContent
from mlx_worker.vlm_engine import MlxVlmEngine, VlmContinuousBatchScheduler


def print_family(mode, label, summary):
    family = summary[label]
    print(f"{mode}_{label}_ttft_p50_ms={family['ttft_p50_ms']:.1f}")
    print(f"{mode}_{label}_ttft_p95_ms={family['ttft_p95_ms']:.1f}")
    print(f"{mode}_{label}_latency_p50_ms={family['latency_p50_ms']:.1f}")
    print(f"{mode}_{label}_latency_p95_ms={family['latency_p95_ms']:.1f}")
    print(
        f"{mode}_{label}_prompt_tokens_per_sec={family['prompt_tokens_per_sec']:.1f}"
    )
    print(
        f"{mode}_{label}_completion_tokens_per_sec={family['completion_tokens_per_sec']:.1f}"
    )
    print(
        f"{mode}_{label}_decode_tokens_per_sec={family['decode_tokens_per_sec']:.1f}"
    )
    print(f"{mode}_{label}_cache_hit_rate={family['prompt_cache_hit_rate']:.3f}")
    print(f"{mode}_{label}_peak_memory_bytes={family['peak_memory_bytes']:.0f}")
    print(f"{mode}_{label}_worker_cancellations={family['worker_cancellations']:.0f}")
    print(f"{mode}_{label}_worker_errors={family['worker_errors']:.0f}")
    print(f"{mode}_{label}_scheduler_tick_latency_ms={family['scheduler_tick_latency_ms']:.0f}")
    print(f"{mode}_{label}_arbitration_delay_ms={family['arbitration_delay_ms']:.0f}")
    if label == "vlm":
        print(f"{mode}_{label}_apc_hit_rate={family['prompt_cache_hit_rate']:.3f}")
        print(f"{mode}_{label}_vision_cache_hit_rate={family['vision_cache_hit_rate']:.3f}")
        print(f"{mode}_{label}_image_count={family['image_count']:.0f}")
        print(f"{mode}_{label}_image_width={family['image_width']:.0f}")
        print(f"{mode}_{label}_image_height={family['image_height']:.0f}")


def throughput(summary, label):
    family = summary[label]
    prompt_tokens = family["prompt_tokens"]
    completion_tokens = family["completion_tokens"]
    latency_sec = max(family["suite_wall_clock_ms"] / 1000.0, 0.001)
    return {
        "prompt_tokens_per_sec": prompt_tokens / latency_sec,
        "completion_tokens_per_sec": completion_tokens / latency_sec,
    }


class FakeTokenizer:
    def encode(self, prompt, add_special_tokens=False):
        return [ord(char) % 97 for char in prompt]

    def decode(self, tokens):
        return "cache-proof"


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        return "VLM processor prompt"


class FakeEmbeddingOutput:
    def __init__(self):
        self.inputs_embeds = [[0.0]]

    def to_dict(self):
        return {}


for mode, summary in (("baseline", baseline), ("continuous", continuous)):
    print_family(mode, "text", summary)
    print_family(mode, "vlm", summary)
    metrics = summary["metrics"]
    print(f"{mode}_prompt_cache_hits_total={metrics['prompt_cache_hits_total']:.0f}")
    print(
        f"{mode}_prompt_cache_cached_tokens_total={metrics['prompt_cache_cached_tokens_total']:.0f}"
    )
    print(f"{mode}_vlm_requests_total={metrics['vlm_requests_total']:.0f}")
    print(f"{mode}_vlm_image_count_total={metrics['vlm_image_count_total']:.0f}")
    print(f"{mode}_vlm_image_width={summary['vlm']['image_width']:.0f}")
    print(f"{mode}_vlm_image_height={summary['vlm']['image_height']:.0f}")
    print(
        f"{mode}_vlm_image_preprocess_latency_ms={metrics['vlm_image_preprocess_latency_ms']:.0f}"
    )
    print(
        f"{mode}_vlm_prompt_template_latency_ms={metrics['vlm_prompt_template_latency_ms']:.0f}"
    )

if not continuous["text"]["cache_hit_followup"].get("prompt_cache_hit"):
    raise SystemExit("continuous cache hit probe did not report a hit")
if continuous["text"]["cache_hit_followup"].get("cached_tokens", 0) <= 0:
    raise SystemExit("continuous cache hit probe did not report cached tokens")
if not continuous["text"]["dynamic_join"]["started_while_first_active"]:
    raise SystemExit("second text request did not start while first request was decoding")
if continuous["text"]["dynamic_join"]["join_decode_batch_size"] < 2:
    raise SystemExit(
        "real mlx_lm.BatchGenerator never reported text decode batch size >= 2 during dynamic join"
    )
if not continuous["text"]["mixed_backend"]["started_while_first_active"]:
    raise SystemExit("mixed text+VLM workload did not overlap decoding")
if not continuous["text"]["mixed_backend"]["second"]["text"].strip():
    raise SystemExit("mixed text+VLM workload returned empty VLM text")
if continuous["text"]["mixed_backend"]["first"]["ttft_ms"] is None:
    raise SystemExit("mixed text+VLM workload never produced first text chunk")
if continuous["text"]["mixed_backend"]["second"]["ttft_ms"] is None:
    raise SystemExit("mixed text+VLM workload never produced first VLM chunk")
if not (
    continuous["text"]["mixed_backend"]["first"]["first_chunk_at"]
    < continuous["text"]["mixed_backend"]["second"]["completed_at"]
    and continuous["text"]["mixed_backend"]["second"]["first_chunk_at"]
    < continuous["text"]["mixed_backend"]["first"]["completed_at"]
):
    raise SystemExit("mixed text+VLM workload did not show concurrent progress")
if not continuous["vlm"]["dynamic_join"]["started_while_first_active"]:
    raise SystemExit("second VLM request did not start while first VLM request was decoding")
if continuous["vlm"]["dynamic_join"]["join_decode_batch_size"] < 2:
    raise SystemExit(
        "real mlx_vlm.generate.BatchGenerator never reported VLM decode batch size >= 2 during dynamic join"
    )
if not continuous["vlm"]["repeated_image_followup"].get("prompt_cache_hit"):
    raise SystemExit("repeated image workload did not report a cache hit")
if continuous["vlm"]["repeated_image_followup"].get("cached_tokens", 0) <= 0:
    raise SystemExit("repeated image workload did not report cached tokens")
if not continuous["vlm"]["repeated_image_followup"].get("vision_feature_cache_hit"):
    raise SystemExit("repeated image workload did not report a vision-feature cache hit")
if continuous["vlm"]["shared_apc"][1].get("prompt_cache_hit") is not True:
    raise SystemExit("shared text+image APC workload did not report a cache hit")
if continuous["vlm"]["shared_apc"][1].get("cached_tokens", 0) <= 0:
    raise SystemExit("shared text+image APC workload did not report cached tokens")
if continuous["metrics"]["requests_cancelled_total"] <= 0:
    raise SystemExit("continuous cancellation metric missing")
if continuous["metrics"]["vlm_requests_total"] <= 0:
    raise SystemExit("continuous VLM request metric missing")
if continuous["metrics"]["vlm_image_count_total"] <= 0:
    raise SystemExit("continuous VLM image-count metric missing")
if continuous["metrics"]["vlm_load_errors_total"] != 0:
    raise SystemExit("continuous VLM load errors were recorded")
if continuous["metrics"]["active_batch_cache_bytes"] <= 0:
    raise SystemExit("combined memory pressure never produced active batch cache usage")
if continuous["text"]["prompt_cache_bytes"] > TEXT_CACHE_BUDGET_BYTES:
    raise SystemExit("text prompt cache exceeded its configured budget")
if continuous["vlm"]["prompt_cache_bytes"] > VLM_APC_CACHE_BUDGET_BYTES:
    raise SystemExit("VLM APC cache exceeded its configured budget")
if continuous["vlm"]["vision_feature_cache_bytes"] > VISION_FEATURE_CACHE_BUDGET_BYTES:
    raise SystemExit("VLM vision-feature cache exceeded its configured budget")
if continuous["text"]["peak_memory_bytes"] <= 0:
    raise SystemExit("text peak memory was never populated")
if continuous["vlm"]["peak_memory_bytes"] <= 0:
    raise SystemExit("VLM peak memory was never populated")
if continuous["vlm"]["image_count"] <= 0:
    raise SystemExit("VLM image count was never populated")
if continuous["vlm"]["image_width"] <= 0 or continuous["vlm"]["image_height"] <= 0:
    raise SystemExit("VLM image dimensions were never populated")
if not continuous["vlm"]["decode_only"]["text"].strip():
    raise SystemExit("continuous VLM decode-only workload returned empty text")
if continuous["text"]["ttft_p50_ms"] is None or baseline["text"]["ttft_p50_ms"] is None:
    raise SystemExit("text TTFT summary missing")
if continuous["vlm"]["ttft_p50_ms"] is None or baseline["vlm"]["ttft_p50_ms"] is None:
    raise SystemExit("VLM TTFT summary missing")
if continuous["text"]["ttft_p95_ms"] is not None and baseline["text"]["ttft_p95_ms"] is not None:
    if continuous["text"]["ttft_p95_ms"] > baseline["text"]["ttft_p95_ms"] * 1.25:
        raise SystemExit("text p95 TTFT regressed beyond the allowed threshold")
if continuous["vlm"]["ttft_p95_ms"] is not None and baseline["vlm"]["ttft_p95_ms"] is not None:
    if continuous["vlm"]["ttft_p95_ms"] > baseline["vlm"]["ttft_p95_ms"] * 1.25:
        raise SystemExit("VLM p95 TTFT regressed beyond the allowed threshold")

baseline_text_throughput = throughput(baseline, "text")
continuous_text_throughput = throughput(continuous, "text")
baseline_vlm_throughput = throughput(baseline, "vlm")
continuous_vlm_throughput = throughput(continuous, "vlm")
if (
    continuous_text_throughput["completion_tokens_per_sec"]
    < baseline_text_throughput["completion_tokens_per_sec"]
):
    raise SystemExit("continuous text throughput did not beat serialized baseline")
if (
    continuous_text_throughput["prompt_tokens_per_sec"]
    < baseline_text_throughput["prompt_tokens_per_sec"]
):
    raise SystemExit("continuous text prompt throughput did not beat serialized baseline")
if (
    continuous_vlm_throughput["completion_tokens_per_sec"]
    < baseline_vlm_throughput["completion_tokens_per_sec"]
):
    raise SystemExit("continuous VLM throughput did not beat serialized baseline")
if (
    continuous_vlm_throughput["prompt_tokens_per_sec"]
    < baseline_vlm_throughput["prompt_tokens_per_sec"]
):
    raise SystemExit("continuous VLM prompt throughput did not beat serialized baseline")

print("baseline_metrics_ok=1")
print("continuous_metrics_ok=1")
print("vlm_metrics_ok=1")
print("mixed_backend_fairness_ok=1")
print("continuous_join_while_decoding_ok=1")
print("vlm_continuous_join_while_decoding_ok=1")
print("cache_hits_ok=1")
print("vlm_cache_hits_ok=1")
print("vlm_local_image_ok=1")
print("cancellation_ok=1")
print("throughput_improved_ok=1")
print("phase_10_validation_ok=1")
PY

echo "baseline_summary=$BASELINE_SUMMARY"
echo "continuous_summary=$CONTINUOUS_SUMMARY"
echo "baseline_log=$BASELINE_LOG"
echo "continuous_log=$CONTINUOUS_LOG"
