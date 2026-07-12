from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
import sys

from mlx_worker.batching import (
    BatchEventSink,
    BatchBackendContext,
    ContinuousBatchScheduler,
    MlxBatchCompletionBackend,
    PromptCacheStore,
    validate_continuous_batching_backend,
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

    def insert(
        self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
    ):
        self.insert_calls.append(
            {
                "prompts": prompts,
                "max_tokens": max_tokens,
                "caches": caches,
                "all_tokens": all_tokens,
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
                    prompt_cache=["cache-b"] * 4,
                    all_tokens=[9, 9, 21, 22],
                ),
                SimpleNamespace(
                    uid=42,
                    token=12,
                    finish_reason="stop",
                    prompt_cache=["cache-a"] * 5,
                    all_tokens=[1, 2, 3, 11, 12],
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


def _install_fake_cache_module(monkeypatch) -> None:
    fake_models = ModuleType("mlx_lm.models")
    fake_cache = ModuleType("mlx_lm.models.cache")
    fake_cache.trim_prompt_cache = lambda prompt_cache, num_tokens: None
    monkeypatch.setitem(sys.modules, "mlx_lm.models", fake_models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", fake_cache)


def test_continuous_batching_contract_can_be_validated_at_startup(
    monkeypatch,
) -> None:
    class CompatibleBatchGenerator:
        def __init__(
            self,
            model,
            *,
            completion_batch_size=None,
            prefill_batch_size=None,
            prefill_step_size=None,
        ) -> None:
            pass

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
            return []

        def next_generated(self):
            return []

        def remove(self, uids) -> None:
            pass

        def close(self) -> None:
            pass

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = CompatibleBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    validate_continuous_batching_backend()


def test_legacy_batch_generator_without_all_tokens_remains_compatible(
    monkeypatch,
) -> None:
    _install_fake_cache_module(monkeypatch)

    class LegacyBatchGenerator(FakeBatchGenerator):
        def insert(self, prompts, *, max_tokens, caches=None, samplers=None):
            self.insert_calls.append(
                {
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "caches": caches,
                    "samplers": samplers,
                }
            )
            self._uids = [42][: len(prompts)]
            return self._uids

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = LegacyBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda _request: [1, 2],
            validate_token_limits=lambda _request, _tokens: None,
            make_sampler=lambda _temperature, _top_p: None,
        )
    )
    response = backend.complete_many(
        [
            ChatCompletionRequest(
                request_id="legacy",
                model="test-model",
                messages=[ChatMessage(role="user", content="hello")],
                max_tokens=2,
                temperature=0.0,
                top_p=1.0,
                max_prompt_tokens=32,
                max_completion_tokens=32,
                max_total_tokens_per_request=64,
            )
        ]
    )[0]

    assert response.text == "<11>"


def test_prompt_cache_store_returns_longest_prefix() -> None:
    store = PromptCacheStore()
    store.remember([1, 2], ["layer-a"])
    store.remember([1, 2, 3], ["layer-b-0", "layer-b-1"])

    cached = store.lookup([1, 2, 3, 4])

    assert cached is not None
    assert cached.tokens == (1, 2, 3)
    assert cached.prompt_cache == ["layer-b-0", "layer-b-1"]
    assert cached.match_kind == "shorter_prefix"


def test_prompt_cache_store_exact_hit_uses_full_prompt() -> None:
    store = PromptCacheStore()
    store.remember([1, 2, 3], ["exact-cache"])

    cached = store.lookup([1, 2, 3])

    assert cached is not None
    assert cached.tokens == (1, 2, 3)
    assert cached.prompt_cache == ["exact-cache"]
    assert cached.match_kind == "exact"


def test_prompt_cache_store_trims_longer_cached_prefix() -> None:
    store = PromptCacheStore(
        trim_cache=lambda cache, count: [value[:-count] for value in cache]
    )
    store.remember([1, 2, 3, 4], ["abcd", "wxyz"])

    cached = store.lookup([1, 2, 9])

    assert cached is not None
    assert cached.tokens == (1, 2)
    assert cached.prompt_cache == ["ab", "wx"]
    assert cached.match_kind == "trimmed_longer_prefix"


def test_prompt_cache_store_uses_cache_namespace() -> None:
    store = PromptCacheStore()
    store.remember([1, 2, 3], ["image-cache"], cache_key=["image-a"])
    store.remember([1, 2, 3], ["text-cache"], cache_key=[])

    image_cached = store.lookup([1, 2, 3, 4], cache_key=["image-a"])
    text_cached = store.lookup([1, 2, 3, 4], cache_key=[])
    miss = store.lookup([1, 2, 3, 4], cache_key=["image-b"])

    assert image_cached is not None
    assert image_cached.prompt_cache == ["image-cache"]
    assert text_cached is not None
    assert text_cached.prompt_cache == ["text-cache"]
    assert miss is None


def test_batch_backend_batches_requests_and_reuses_prompt_cache(monkeypatch) -> None:
    _install_fake_cache_module(monkeypatch)
    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = FakeBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)
    FakeBatchGenerator.instances.clear()

    prompt_cache_store = PromptCacheStore()
    prompt_cache_store.remember([1, 2], ["cached-prefix", "cached-prefix"])
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
            "caches": [["cached-prefix", "cached-prefix"], None],
            "all_tokens": [[1, 2], []],
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
    assert prompt_cache_store.lookup([1, 2, 3, 11, 12, 13]) is not None


def test_batch_backend_exact_prompt_cache_hit_avoids_empty_insert(monkeypatch) -> None:
    _install_fake_cache_module(monkeypatch)
    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = FakeBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)
    FakeBatchGenerator.instances.clear()

    prompt_cache_store = PromptCacheStore()
    prompt_cache_store.remember([1, 2, 3], ["cached-full"])
    backend = MlxBatchCompletionBackend(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=prompt_cache_store,
            build_prompt_tokens=lambda request: [1, 2, 3],
            validate_token_limits=lambda request, tokens: None,
            make_sampler=lambda temp, top_p: None,
        )
    )

    response = backend.complete_many(
        [
            ChatCompletionRequest(
                request_id="req-1",
                model="test-model",
                messages=[ChatMessage(role="user", content="repeat")],
                max_tokens=1,
                temperature=0.0,
                top_p=1.0,
                max_prompt_tokens=32,
                max_completion_tokens=32,
                max_total_tokens_per_request=64,
            )
        ]
    )[0]
    generator = FakeBatchGenerator.instances[-1]

    assert generator.insert_calls[0]["prompts"] == [[1, 2, 3]]
    assert generator.insert_calls[0]["caches"] == [None]
    assert generator.insert_calls[0]["all_tokens"] == [[]]
    assert response.prompt_cache_hit is False


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

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
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

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
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

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
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

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
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


def test_continuous_scheduler_admits_new_request_while_decoding(monkeypatch) -> None:
    _install_fake_cache_module(monkeypatch)
    generators: list[object] = []

    class FakeContinuousBatchGenerator:
        def __init__(
            self,
            model,
            stop_tokens=None,
            completion_batch_size=None,
            prefill_batch_size=None,
            prefill_step_size=None,
        ) -> None:
            self.insert_calls: list[dict[str, object]] = []
            self.config = (
                completion_batch_size,
                prefill_batch_size,
                prefill_step_size,
            )
            self._next_uid = 101
            self._step = 0
            self.removed: list[list[int]] = []
            self.prompt_cache_nbytes = 0
            generators.append(self)

        def insert(
            self, prompts, *, max_tokens, caches=None, all_tokens=None, samplers=None
        ):
            uids = [self._next_uid + index for index in range(len(prompts))]
            self._next_uid += len(prompts)
            self.insert_calls.append(
                {
                    "prompts": prompts,
                    "max_tokens": max_tokens,
                    "caches": caches,
                    "all_tokens": all_tokens,
                    "samplers": samplers,
                    "uids": uids,
                }
            )
            return uids

        def next_generated(self):
            if self._step == 0:
                self._step += 1
                return [
                    SimpleNamespace(
                        uid=101,
                        token=11,
                        finish_reason=None,
                        prompt_cache=None,
                    )
                ]
            if self._step == 1:
                self._step += 1
                return [
                    SimpleNamespace(
                        uid=101,
                        token=12,
                        finish_reason="stop",
                        prompt_cache=["layer-cache-1"],
                        all_tokens=[1, 11, 12],
                    ),
                    SimpleNamespace(
                        uid=102,
                        token=21,
                        finish_reason="stop",
                        prompt_cache=["layer-cache-2"],
                        all_tokens=[2, 21],
                    ),
                ]
            return []

        def close(self) -> None:
            return None

        def remove(self, uids) -> None:
            self.removed.append(uids)

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = FakeContinuousBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    events: list[tuple[str, str, str]] = []

    sink = BatchEventSink(
        emit_delta=lambda request_id, delta: events.append(
            ("delta", request_id, delta)
        ),
        emit_response=lambda response: events.append(
            ("response", response.request_id, response.text)
        ),
        emit_error=lambda request_id, code, message: events.append(
            ("error", request_id, f"{code}:{message}")
        ),
    )

    prompt_cache_store = PromptCacheStore()
    scheduler = ContinuousBatchScheduler(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=prompt_cache_store,
            build_prompt_tokens=lambda request: (
                [1] if request.request_id == "req-1" else [2]
            ),
            validate_token_limits=lambda request, tokens: None,
            make_sampler=lambda temp, top_p: f"sampler-{temp}-{top_p}",
        ),
        sink,
        prompt_concurrency=2,
        decode_concurrency=2,
        prefill_step_size=64,
    )

    scheduler.submit(
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
        stream=True,
    )
    scheduler.tick()
    generator = generators[0]
    assert generator.config == (2, 2, 64)
    assert generator.insert_calls[0]["all_tokens"] == [[]]

    scheduler.submit(
        ChatCompletionRequest(
            request_id="req-2",
            model="test-model",
            messages=[ChatMessage(role="user", content="again")],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        ),
        stream=False,
    )
    scheduler.tick()
    scheduler.tick()

    assert generator.insert_calls[1]["all_tokens"] == [[]]
    assert prompt_cache_store.total_bytes > 0

    scheduler.submit(
        ChatCompletionRequest(
            request_id="req-3",
            model="test-model",
            messages=[ChatMessage(role="user", content="cancel")],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        ),
        stream=True,
    )
    scheduler.tick()
    scheduler.cancel("req-3")

    assert ("delta", "req-1", "<11>") in events
    assert any(event[0] == "response" and event[1] == "req-1" for event in events)
    assert any(event[0] == "response" and event[1] == "req-2" for event in events)
    assert generator.removed == [[103]]


def test_continuous_scheduler_exact_prompt_cache_hit_avoids_empty_insert(
    monkeypatch,
) -> None:
    _install_fake_cache_module(monkeypatch)
    sys.modules["mlx_lm.models.cache"].trim_prompt_cache = (
        lambda prompt_cache, num_tokens: list(prompt_cache)
    )
    generators: list[FakeBatchGenerator] = []

    class RecordingBatchGenerator(FakeBatchGenerator):
        def __init__(
            self,
            model,
            stop_tokens=None,
            completion_batch_size=None,
            prefill_batch_size=None,
            prefill_step_size=None,
        ) -> None:
            super().__init__(model, stop_tokens)
            del completion_batch_size, prefill_batch_size, prefill_step_size
            generators.append(self)

        def remove(self, uids) -> None:
            del uids

    fake_generate = ModuleType("mlx_lm.generate")
    fake_generate.BatchGenerator = RecordingBatchGenerator
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", fake_generate)

    scheduler = ContinuousBatchScheduler(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=FakeTokenizer(),
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda request: [1, 2, 3],
            validate_token_limits=lambda request, tokens: None,
            make_sampler=lambda temp, top_p: None,
        ),
        BatchEventSink(
            emit_delta=lambda request_id, delta: None,
            emit_response=lambda response: None,
            emit_error=lambda request_id, code, message: None,
        ),
        prompt_concurrency=1,
        decode_concurrency=1,
        prefill_step_size=64,
    )

    scheduler.submit(
        ChatCompletionRequest(
            request_id="first",
            model="test-model",
            messages=[ChatMessage(role="user", content="repeat")],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        ),
        stream=False,
    )
    scheduler.tick()
    scheduler.tick()
    scheduler.tick()
    scheduler.submit(
        ChatCompletionRequest(
            request_id="second",
            model="test-model",
            messages=[ChatMessage(role="user", content="repeat")],
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        ),
        stream=False,
    )
    scheduler.tick()

    assert generators[0].insert_calls[0]["prompts"] == [[1, 2, 3]]
    assert generators[0].insert_calls[1]["prompts"] == [[1, 2, 3]]
    assert generators[0].insert_calls[1]["caches"] == [None]
    assert generators[0].insert_calls[1]["all_tokens"] == [[]]
