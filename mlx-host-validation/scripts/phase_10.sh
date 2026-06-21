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
#       decode tokens/sec, cache metrics, VLM request/image counts, and VLM media-preparation timings.
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
    usage = None
    text_parts = []
    prompt_cache_hit = None
    cached_tokens = None
    prompt_cache_bytes = None
    decode_batch_size = None
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
            choices = event.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {}).get("content", "")
                if delta:
                    text_parts.append(delta)
                    if first_token_event is not None:
                        first_token_event.set()
            else:
                usage = event.get("usage")
    else:
        payload_body = json.loads(resp.read().decode("utf-8"))
        usage = payload_body.get("usage")
        text_parts.append(payload_body["choices"][0]["message"]["content"])
        prompt_cache_hit = payload_body.get("prompt_cache_hit")
        cached_tokens = payload_body.get("cached_tokens")
        prompt_cache_bytes = payload_body.get("prompt_cache_bytes")
        decode_batch_size = payload_body.get("decode_batch_size")

    completed_at = time.perf_counter()
    conn.close()
    return {
        "started_at": started,
        "ttft_ms": None if first_chunk_at is None else (first_chunk_at - started) * 1000.0,
        "latency_ms": (completed_at - started) * 1000.0,
        "first_chunk_at": first_chunk_at,
        "completed_at": completed_at,
        "prompt_tokens": 0 if usage is None else int(usage.get("prompt_tokens", 0)),
        "completion_tokens": 0 if usage is None else int(usage.get("completion_tokens", 0)),
        "text": "".join(text_parts),
        "prompt_cache_hit": prompt_cache_hit if payload.get("stream", False) else prompt_cache_hit,
        "cached_tokens": cached_tokens if payload.get("stream", False) else cached_tokens,
        "prompt_cache_bytes": (
            prompt_cache_bytes if payload.get("stream", False) else prompt_cache_bytes
        ),
        "decode_batch_size": decode_batch_size,
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


def cancel_by_disconnect(payload):
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
    line = resp.readline()
    if not line:
        raise RuntimeError("stream produced no chunk before cancel")
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
    }


def run_text_suite():
    started = time.perf_counter()
    decode_only = request_once(text_payload("Say hello in one short sentence."))
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
            "Stream this response and then cancel it after first token.",
            max_tokens=32,
        )
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
            stream=False,
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
            "Stream a detailed answer about this image and then cancel after the first token.",
            with_image=True,
            max_tokens=48,
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
        + [repeated_image_seed, repeated_image_followup]
        + [cancel_followup]
    )
    return {
        "suite_wall_clock_ms": (time.perf_counter() - started) * 1000.0,
        "local_image_path": VLM_IMAGE_PATH,
        "decode_only": decode_only,
        "dynamic_join": dynamic_join,
        "mixed_modal": mixed_modal,
        "repeated_image_seed": repeated_image_seed,
        "repeated_image_followup": repeated_image_followup,
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
python3 - "$ROOT" "$BASELINE_SUMMARY" "$CONTINUOUS_SUMMARY" "$VLM_IMAGE_PATH" <<'PY'
import json
import sys
import time
from types import ModuleType, SimpleNamespace
from pathlib import Path

ROOT = Path(sys.argv[1])
baseline = json.loads(Path(sys.argv[2]).read_text())
continuous = json.loads(Path(sys.argv[3]).read_text())
VLM_IMAGE_PATH = str(Path(sys.argv[4]).resolve())

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


def throughput(summary):
    text = summary["text"]
    prompt_tokens = text["prompt_tokens"]
    completion_tokens = text["completion_tokens"]
    latency_sec = max(text["suite_wall_clock_ms"] / 1000.0, 0.001)
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


def image_cache_proof():
    fake_generate = ModuleType("mlx_vlm.generate")

    class FakeVlmBatchGenerator:
        def __init__(self, *args, **kwargs):
            self.prompt_cache_nbytes = 0

    fake_generate.BatchGenerator = FakeVlmBatchGenerator
    sys.modules["mlx_vlm"] = ModuleType("mlx_vlm")
    sys.modules["mlx_vlm.generate"] = fake_generate

    fake_sample_utils = ModuleType("mlx_lm.sample_utils")
    fake_sample_utils.make_sampler = lambda temp, top_p: f"sampler-{temp}-{top_p}"
    sys.modules["mlx_lm"] = ModuleType("mlx_lm")
    sys.modules["mlx_lm.sample_utils"] = fake_sample_utils

    fake_vlm_utils = ModuleType("mlx_vlm.utils")
    fake_vlm_utils.prepare_inputs = lambda processor, **kwargs: {
        "input_ids": [[1, 2, 3]],
        "attention_mask": [[1, 1, 1]],
        **(
            {"pixel_values": [[[0.0]]], "image_grid_thw": [[1, 1, 1]]}
            if kwargs.get("images")
            else {}
        ),
    }
    fake_vlm_utils.load_config = lambda *args, **kwargs: None
    sys.modules["mlx_vlm.utils"] = fake_vlm_utils

    engine = MlxVlmEngine(
        "vlm-model",
        vlm_loader=lambda _model_id: (
            SimpleNamespace(
                language_model=SimpleNamespace(),
                get_input_embeddings=lambda input_ids, pixel_values=None, mask=None, **kwargs: (
                    FakeEmbeddingOutput()
                ),
            ),
            FakeProcessor(),
        ),
        vlm_generate_fn=lambda *args, **kwargs: None,
        vlm_stream_generate_fn=lambda *args, **kwargs: None,
    )
    engine.initialize()
    cache_store = PromptCacheStore(trim_cache=lambda prompt_cache, num_tokens: list(prompt_cache))
    scheduler = VlmContinuousBatchScheduler(
        engine,
        sink=SimpleNamespace(
            emit_delta=lambda request_id, delta: None,
            emit_response=lambda response: None,
            emit_error=lambda request_id, code, message: None,
        ),
        prompt_concurrency=2,
        decode_concurrency=2,
        prefill_step_size=64,
        prompt_cache_store=cache_store,
    )
    request = ChatCompletionRequest(
        request_id="vlm-cache-proof",
        model="vlm-model",
        messages=[
            ChatMessage(
                role="user",
                content=(
                    TextContent(text="Describe the image."),
                    ImageContent(url=VLM_IMAGE_PATH),
                ),
            )
        ],
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
        stream=False,
    )

    first_job = scheduler._prepare_request(request)
    cache_store.remember(
        first_job.full_prompt_tokens,
        ["cache-a"],
        cache_key=first_job.prompt_cache_key,
    )
    second_job = scheduler._prepare_request(request)
    if second_job.cached_prompt != ["cache-a"]:
        raise SystemExit("deterministic VLM cache proof failed")
    if not second_job.cached_prefix_tokens:
        raise SystemExit("deterministic VLM cache proof missed cached tokens")


image_cache_proof()


for mode, summary in (("baseline", baseline), ("continuous", continuous)):
    print_family(mode, "text", summary)
    print_family(mode, "vlm", summary)
    metrics = summary["metrics"]
    print(f"{mode}_prompt_cache_hits_total={metrics['prompt_cache_hits_total']:.0f}")
    print(
        f"{mode}_prompt_cache_cached_tokens_total={metrics['prompt_cache_cached_tokens_total']:.0f}"
    )
    print(f"{mode}_worker_memory_bytes={metrics['worker_memory_bytes']:.0f}")
    print(f"{mode}_vlm_requests_total={metrics['vlm_requests_total']:.0f}")
    print(f"{mode}_vlm_image_count_total={metrics['vlm_image_count_total']:.0f}")
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
if continuous["metrics"]["requests_cancelled_total"] <= 0:
    raise SystemExit("continuous cancellation metric missing")
if continuous["metrics"]["vlm_requests_total"] <= 0:
    raise SystemExit("continuous VLM request metric missing")
if continuous["metrics"]["vlm_image_count_total"] <= 0:
    raise SystemExit("continuous VLM image-count metric missing")
if continuous["metrics"]["vlm_load_errors_total"] != 0:
    raise SystemExit("continuous VLM load errors were recorded")
if not continuous["vlm"]["decode_only"]["text"].strip():
    raise SystemExit("continuous VLM decode-only workload returned empty text")
if continuous["text"]["ttft_p50_ms"] is None or baseline["text"]["ttft_p50_ms"] is None:
    raise SystemExit("text TTFT summary missing")
if continuous["vlm"]["ttft_p50_ms"] is None or baseline["vlm"]["ttft_p50_ms"] is None:
    raise SystemExit("VLM TTFT summary missing")

baseline_throughput = throughput(baseline)
continuous_throughput = throughput(continuous)
if continuous_throughput["completion_tokens_per_sec"] < baseline_throughput["completion_tokens_per_sec"]:
    raise SystemExit("continuous batch throughput did not beat serialized baseline")
if continuous_throughput["prompt_tokens_per_sec"] < baseline_throughput["prompt_tokens_per_sec"]:
    raise SystemExit("continuous prompt throughput did not beat serialized baseline")

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
