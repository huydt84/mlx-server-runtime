#!/usr/bin/env bash
#
# Phase 5 host-only batching validation for this repository.
# Run this on an Apple Silicon Mac with Metal available.
#
# Usage:
#   bash mlx-host-validation/scripts/phase_5.sh
#
# What this verifies:
#   1. The Python environment can import both `mlx` and `mlx_lm`.
#   2. `mlx_lm.generate.BatchGenerator` is available on the host.
#   3. The worker engine can batch four concurrent completions into a single backend call.
#   4. Prompt cache reuse keeps the most recent prefix cache available for later requests.
#
# Expected verification signal:
#   - The script exits with status code 0.
#   - It prints `mlx_import_ok=1` and `mlx_batch_generator_ok=1`.
#   - It prints `batched_requests=4`, `batch_backend_calls=1`, and `prompt_cache_reused=1`.
#   - If validation fails, the script exits non-zero and prints the failing assertion.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_DIR="$ROOT/python"

echo "[1/3] Sync Python dev environment"
cd "$PYTHON_DIR"
uv sync --group dev

echo "[2/3] Verify Apple Silicon and mlx-lm batch imports"
uv run python - <<'PY'
import platform

machine = platform.machine()
print(f"machine={machine}")
if machine != "arm64":
    raise SystemExit("expected Apple Silicon arm64 host")

import mlx.core as mx
print("mlx_import_ok=1")

from mlx_lm.generate import BatchGenerator  # noqa: F401
print("mlx_batch_generator_ok=1")

values = (mx.array([1.0, 2.0, 3.0]) * 2).tolist()
print(f"mlx_compute_ok={values}")
PY

echo "[3/3] Run worker batching smoke test"
uv run python - <<'PY'
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import sys

from mlx_worker.batching import BatchBackendContext, MlxBatchCompletionBackend, PromptCacheStore
from mlx_worker.engine import MlxWorkerEngine
from mlx_worker.ipc import ChatCompletionRequest, ChatMessage, ChatCompletionResponse


@dataclass
class FakeTokenizer:
    has_chat_template: bool = False

    def __post_init__(self) -> None:
        self.eos_token_ids = [0]

    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(char) for char in prompt]

    def decode(self, tokens: list[int]) -> str:
        return "".join(f"<{token}>" for token in tokens)


class FakeBatchGenerator:
    instances: list["FakeBatchGenerator"] = []

    def __init__(self, model, stop_tokens=None) -> None:
        self.model = model
        self.stop_tokens = stop_tokens
        self.insert_calls: list[dict[str, object]] = []
        self.closed = False
        self._step = 0
        self._uids: list[int] = []
        FakeBatchGenerator.instances.append(self)

    def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
        self.insert_calls.append(
            {
                "prompts": prompts,
                "max_tokens": max_tokens,
                "caches": caches,
                "samplers": samplers,
            }
        )
        self._uids = [42, 7][: len(prompts)]
        return self._uids

    def next_generated(self):
        if self._step == 0:
            self._step += 1
            return [
                SimpleNamespace(uid=42, token=11, finish_reason=None, prompt_cache=None),
                SimpleNamespace(uid=7, token=21, finish_reason=None, prompt_cache=None),
            ]
        if self._step == 1:
            self._step += 1
            return [
                SimpleNamespace(uid=7, token=22, finish_reason="length", prompt_cache=["cache-b"]),
                SimpleNamespace(uid=42, token=12, finish_reason="stop", prompt_cache=["cache-a"]),
            ]
        return []

    @contextmanager
    def stats(self):
        yield SimpleNamespace(
            prompt_tokens=0,
            prompt_tps=0.0,
            generation_tokens=0,
            generation_tps=0.0,
            peak_memory=0.0,
        )

    def close(self) -> None:
        self.closed = True


fake_generate = ModuleType("mlx_lm.generate")
fake_generate.BatchGenerator = FakeBatchGenerator
sys.modules.setdefault("mlx_lm", ModuleType("mlx_lm"))
sys.modules["mlx_lm.generate"] = fake_generate
fake_models = ModuleType("mlx_lm.models")
fake_cache = ModuleType("mlx_lm.models.cache")
fake_cache.trim_prompt_cache = lambda prompt_cache, num_tokens: None
sys.modules["mlx_lm.models"] = fake_models
sys.modules["mlx_lm.models.cache"] = fake_cache

batch_prompt_cache_store = PromptCacheStore()
batch_prompt_cache_store.remember([1, 2], ["cached-prefix"])
batch_context = BatchBackendContext(
    model_id="test-model",
    model=SimpleNamespace(),
    tokenizer=FakeTokenizer(),
    prompt_cache_store=batch_prompt_cache_store,
    build_prompt_tokens=lambda request: {
        "req-1": [1, 2, 3],
        "req-2": [9, 9],
    }[request.request_id],
    validate_token_limits=lambda request, tokens: None,
    make_sampler=lambda temp, top_p: f"sampler-{temp}-{top_p}",
)
batch_backend = MlxBatchCompletionBackend(batch_context)
batch_requests = [
    ChatCompletionRequest(
        request_id="req-1",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=2,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
    ChatCompletionRequest(
        request_id="req-2",
        model="test-model",
        messages=[ChatMessage(role="user", content="again")],
        max_tokens=2,
        temperature=0.2,
        top_p=0.8,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
]

batch_responses = batch_backend.complete_many(batch_requests)
batch_generator = FakeBatchGenerator.instances[-1]
if batch_generator.insert_calls != [
    {
        "prompts": [[3], [9, 9]],
        "max_tokens": [2, 2],
        "caches": [["cached-prefix"], None],
        "samplers": ["sampler-0.0-1.0", "sampler-0.2-0.8"],
    }
]:
    raise SystemExit(f"unexpected batch insert call: {batch_generator.insert_calls}")
if [response.request_id for response in batch_responses] != ["req-1", "req-2"]:
    raise SystemExit("backend response order changed")
if batch_responses[0].text != "<11>":
    raise SystemExit(f"unexpected first batch response: {batch_responses[0]}")
if batch_responses[1].finish_reason != "length":
    raise SystemExit(f"unexpected second batch response: {batch_responses[1]}")
if not batch_prompt_cache_store.lookup([1, 2, 3, 4]):
    raise SystemExit("prompt cache was not retained")
print("batch_backend_calls=1")
print("prompt_cache_reused=1")


class RecordingBackend:
    def __init__(self, context: BatchBackendContext) -> None:
        self.context = context
        self.calls: list[list[str]] = []

    def complete_many(self, requests):
        self.calls.append([request.request_id for request in requests])
        return [
            ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text=f"text-{request.request_id}",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )
            for request in requests
        ]


engine = MlxWorkerEngine(
    "test-model",
    model_loader=lambda _model_id: (SimpleNamespace(), FakeTokenizer()),
    batch_backend_factory=lambda context: RecordingBackend(context),
)

requests = [
    ChatCompletionRequest(
        request_id="req-1",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
    ChatCompletionRequest(
        request_id="req-2",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello again")],
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
    ChatCompletionRequest(
        request_id="req-3",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello again again")],
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
    ChatCompletionRequest(
        request_id="req-4",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello once more")],
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=32,
        max_completion_tokens=32,
        max_total_tokens_per_request=64,
    ),
]

responses = engine.complete_many(requests)
if [response.request_id for response in responses] != [request.request_id for request in requests]:
    raise SystemExit("batch output order changed")

single = engine.complete_chat(requests[0])
if single.text != "text-req-1":
    raise SystemExit(f"unexpected singleton completion: {single}")

recording_backend = engine._batch_backend
if len(recording_backend.calls) != 2:
    raise SystemExit(f"expected two backend calls, got {recording_backend.calls}")
if recording_backend.calls[0] != ["req-1", "req-2", "req-3", "req-4"]:
    raise SystemExit(f"expected first batch of four requests, got {recording_backend.calls[0]}")
if recording_backend.calls[1] != ["req-1"]:
    raise SystemExit(f"expected singleton follow-up batch, got {recording_backend.calls[1]}")

print("batched_requests=4")
print("engine_batch_wrapper_ok=1")
print("phase_5_batching_ok=1")
PY

echo "batch_validation_ok=1"
