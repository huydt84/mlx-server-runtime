from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys

from mlx_worker.batching import (
    BatchBackendContext,
    MlxBatchCompletionBackend,
    PromptCacheStore,
)
from mlx_worker.engine import MlxWorkerEngine
from mlx_worker.ipc import ChatCompletionRequest, ChatMessage, ChatCompletionResponse


@dataclass
class FakeTokenizer:
    has_chat_template: bool = False
    eos_token_ids: list[int] = field(default_factory=lambda: [0])

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
                SimpleNamespace(
                    uid=42,
                    token=11,
                    finish_reason=None,
                    prompt_cache=None,
                ),
                SimpleNamespace(
                    uid=7,
                    token=21,
                    finish_reason=None,
                    prompt_cache=None,
                ),
            ]
        if self._step == 1:
            self._step += 1
            return [
                SimpleNamespace(
                    uid=7,
                    token=22,
                    finish_reason="length",
                    prompt_cache=["cache-b"],
                ),
                SimpleNamespace(
                    uid=42,
                    token=12,
                    finish_reason="stop",
                    prompt_cache=["cache-a"],
                ),
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


def test_prompt_cache_store_returns_longest_prefix() -> None:
    store = PromptCacheStore()
    store.remember([1, 2], ["cache-a"])
    store.remember([1, 2, 3], ["cache-b"])

    cached = store.lookup([1, 2, 3, 4])

    assert cached is not None
    assert cached.tokens == (1, 2, 3)
    assert cached.prompt_cache == ["cache-b"]


def test_batch_backend_batches_requests_and_reuses_prompt_cache(monkeypatch) -> None:
    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = FakeBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)
    FakeBatchGenerator.instances.clear()

    prompt_cache_store = PromptCacheStore()
    prompt_cache_store.remember([1, 2], ["cached-prefix"])
    context = BatchBackendContext(
        model_id="test-model",
        model=SimpleNamespace(),
        tokenizer=FakeTokenizer(),
        prompt_cache_store=prompt_cache_store,
        build_prompt_tokens=lambda request: {
            "req-1": [1, 2, 3],
            "req-2": [9, 9],
        }[request.request_id],
        validate_token_limits=lambda request, tokens: None,
        make_sampler=lambda temp, top_p: f"sampler-{temp}-{top_p}",
    )

    backend = MlxBatchCompletionBackend(context)
    requests = [
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

    responses = backend.complete_many(requests)
    generator = FakeBatchGenerator.instances[-1]

    assert generator.insert_calls == [
        {
            "prompts": [[3], [9, 9]],
            "max_tokens": [2, 2],
            "caches": [["cached-prefix"], None],
            "samplers": ["sampler-0.0-1.0", "sampler-0.2-0.8"],
        }
    ]
    assert generator.closed is True
    assert [response.request_id for response in responses] == ["req-1", "req-2"]
    assert responses[0].text == "<11>"
    assert responses[0].finish_reason == "stop"
    assert responses[0].completion_tokens == 1
    assert responses[1].text == "<21><22>"
    assert responses[1].finish_reason == "length"
    assert responses[1].completion_tokens == 2
    assert prompt_cache_store.lookup([1, 2, 3, 4]) is not None


def test_engine_complete_chat_uses_batch_backend() -> None:
    fake_backend_calls: list[list[str]] = []

    class FakeBackend:
        def __init__(self, _context) -> None:
            self.context = _context

        def complete_many(self, requests):
            fake_backend_calls.append([request.request_id for request in requests])
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
        batch_backend_factory=lambda context: FakeBackend(context),
    )

    response = engine.complete_chat(
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
        )
    )

    assert fake_backend_calls == [["req-1"]]
    assert response.text == "text-req-1"


def test_batch_backend_raises_on_stalled_generator(monkeypatch) -> None:
    class StalledFake:
        def __init__(self, model, stop_tokens=None) -> None:
            pass

        def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
            return [10 + index for index in range(len(prompts))]

        def next_generated(self):
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
            pass

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = StalledFake
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda _: [1],
            validate_token_limits=lambda r, t: None,
            make_sampler=lambda t, p: f"s-{t}-{p}",
        )
    )
    import pytest

    with pytest.raises(RuntimeError, match="stalled"):
        backend.complete_many(
            [
                ChatCompletionRequest(
                    request_id="req-1",
                    model="test-model",
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_prompt_tokens=32,
                    max_completion_tokens=32,
                    max_total_tokens_per_request=64,
                ),
            ]
        )


def test_batch_backend_recovers_from_transient_empty_polls(monkeypatch) -> None:
    empty_call_count = {"n": 0}

    class TransientEmptyFake:
        def __init__(self, model, stop_tokens=None) -> None:
            pass

        def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
            self._uid_offset = 10
            return [self._uid_offset + index for index in range(len(prompts))]

        def next_generated(self):
            if empty_call_count["n"] < 2:
                empty_call_count["n"] += 1
                return []
            return [
                SimpleNamespace(
                    uid=10, token=42, finish_reason="stop", prompt_cache=None
                )
            ]

        @contextmanager
        def stats(self):
            yield SimpleNamespace(
                prompt_tokens=1,
                prompt_tps=1.0,
                generation_tokens=1,
                generation_tps=1.0,
                peak_memory=0.0,
            )

        def close(self) -> None:
            pass

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = TransientEmptyFake
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda _: [1],
            validate_token_limits=lambda r, t: None,
            make_sampler=lambda t, p: f"s-{t}-{p}",
        )
    )
    results = backend.complete_many(
        [
            ChatCompletionRequest(
                request_id="req-1",
                model="test-model",
                messages=[ChatMessage(role="user", content="hi")],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                max_prompt_tokens=32,
                max_completion_tokens=32,
                max_total_tokens_per_request=64,
            ),
        ]
    )
    assert len(results) == 1
    assert results[0].finish_reason == "stop"
    assert results[0].text == ""


def test_batch_backend_raises_on_uid_count_mismatch(monkeypatch) -> None:
    class WrongUidCountFake:
        def __init__(self, model, stop_tokens=None) -> None:
            pass

        def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
            return [1, 2]

        def next_generated(self):
            return [
                SimpleNamespace(uid=1, token=1, finish_reason="stop", prompt_cache=None)
            ]

        def stats(self):
            yield SimpleNamespace(
                prompt_tokens=0,
                prompt_tps=0.0,
                generation_tokens=0,
                generation_tps=0.0,
                peak_memory=0.0,
            )

        def close(self) -> None:
            pass

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = WrongUidCountFake
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda _: [1],
            validate_token_limits=lambda r, t: None,
            make_sampler=lambda t, p: f"s-{t}-{p}",
        )
    )
    import pytest

    with pytest.raises(RuntimeError, match="returned 2 UIDs for 1 requests"):
        backend.complete_many(
            [
                ChatCompletionRequest(
                    request_id="req-1",
                    model="test-model",
                    messages=[ChatMessage(role="user", content="hi")],
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_prompt_tokens=32,
                    max_completion_tokens=32,
                    max_total_tokens_per_request=64,
                ),
            ]
        )


def test_batch_backend_raises_on_duplicate_uids(monkeypatch) -> None:
    class DuplicateUidFake:
        def __init__(self, model, stop_tokens=None) -> None:
            pass

        def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
            return [42, 42]

        def next_generated(self):
            return [
                SimpleNamespace(
                    uid=42, token=1, finish_reason="stop", prompt_cache=None
                )
            ]

        def stats(self):
            yield SimpleNamespace(
                prompt_tokens=0,
                prompt_tps=0.0,
                generation_tokens=0,
                generation_tps=0.0,
                peak_memory=0.0,
            )

        def close(self) -> None:
            pass

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = DuplicateUidFake
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda _: [1],
            validate_token_limits=lambda r, t: None,
            make_sampler=lambda t, p: f"s-{t}-{p}",
        )
    )
    import pytest

    with pytest.raises(RuntimeError, match="duplicate"):
        backend.complete_many(
            [
                ChatCompletionRequest(
                    request_id="req-1",
                    model="test-model",
                    messages=[ChatMessage(role="user", content="hi")],
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
                    messages=[ChatMessage(role="user", content="ho")],
                    max_tokens=1,
                    temperature=0.0,
                    top_p=1.0,
                    max_prompt_tokens=32,
                    max_completion_tokens=32,
                    max_total_tokens_per_request=64,
                ),
            ]
        )
