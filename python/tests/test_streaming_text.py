"""Streamed deltas of characters that are split across tokens."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mlx_worker.batching import (
    BatchBackendContext,
    BatchEventSink,
    ContinuousBatchScheduler,
    PromptCacheStore,
)
from mlx_worker.ipc import ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from mlx_worker.native_mlx.interfaces import SchedulerEvent
from mlx_worker.native_mlx.runtime import NativeRuntime
from mlx_worker.vlm_engine import (
    MlxVlmEngine,
    VlmContinuousBatchScheduler,
    _ScheduledVlmRequest,
)

EOS = 0
REPLACEMENT = "�"


class ByteTokenizer:
    """Decode one byte per token, replacing an incomplete sequence."""

    eos_token_ids = (EOS,)

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        del messages, kwargs
        return [1]

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        del kwargs
        return bytes(tokens).decode("utf-8", errors="replace")


class ByteFallbackTokenizer(ByteTokenizer):
    """Replace every byte of a run that is not valid UTF-8 as a whole."""

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        del kwargs
        try:
            return bytes(tokens).decode("utf-8")
        except UnicodeDecodeError:
            return REPLACEMENT * len(tokens)


class ScriptedGenerator:
    """Yield fixed tokens for one request through both batch generator APIs."""

    def __init__(self, tokens: bytes, finish_reason: str) -> None:
        self._steps = [(token, None) for token in tokens]
        if finish_reason == "stop":
            self._steps.append((EOS, "stop"))
        else:
            self._steps[-1] = (tokens[-1], finish_reason)

    def insert(self, prompts: list[list[int]], **kwargs: Any) -> list[int]:
        del kwargs
        return [7] * len(prompts)

    def next_generated(self) -> list[SimpleNamespace]:
        token, finish_reason = self._steps.pop(0)
        return [
            SimpleNamespace(
                uid=7, token=token, finish_reason=finish_reason, prompt_cache=None
            )
        ]

    def next(self) -> tuple[list[Any], list[SimpleNamespace]]:
        (response,) = self.next_generated()
        if response.finish_reason == "stop":
            response.token = None
        return [], [response]

    def close(self) -> None:
        return None


class StreamRecorder:
    """Collect the deltas and the response of one request."""

    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.response: ChatCompletionResponse | None = None

    def sink(self) -> BatchEventSink:
        return BatchEventSink(
            emit_delta=lambda request_id, delta: self.deltas.append(delta),
            emit_response=self._set_response,
            emit_error=lambda request_id, code, message: pytest.fail(message),
        )

    def _set_response(self, response: ChatCompletionResponse) -> None:
        assert self.response is None
        self.response = response


def _request(max_tokens: int, stop: tuple[str, ...] = ()) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id="request",
        model="test-model",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=64,
        max_completion_tokens=64,
        max_total_tokens_per_request=128,
        stop=stop,
        stream=True,
    )


def _drain(scheduler: Any, recorder: StreamRecorder, steps: int) -> None:
    for _ in range(steps + 3):
        if recorder.response is not None:
            return
        scheduler.tick()
    raise AssertionError("request did not finish")


def _stream_text(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: ByteTokenizer,
    tokens: bytes,
    finish_reason: str,
) -> StreamRecorder:
    recorder = StreamRecorder()
    scheduler = ContinuousBatchScheduler(
        BatchBackendContext(
            model_id="test-model",
            model=SimpleNamespace(),
            tokenizer=tokenizer,
            prompt_cache_store=PromptCacheStore(),
            build_prompt_tokens=lambda request: [1],
            validate_token_limits=lambda request, prompt: None,
            make_sampler=lambda temp, top_p: None,
        ),
        recorder.sink(),
    )
    monkeypatch.setattr(
        scheduler,
        "_make_batch_generator",
        lambda: ScriptedGenerator(tokens, finish_reason),
    )
    scheduler.submit(_request(len(tokens)), stream=True)
    _drain(scheduler, recorder, len(tokens))
    return recorder


def _stream_vlm(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: ByteTokenizer,
    tokens: bytes,
    finish_reason: str,
) -> StreamRecorder:
    recorder = StreamRecorder()
    engine = MlxVlmEngine(
        "test-model",
        vlm_loader=lambda model_id: (SimpleNamespace(), SimpleNamespace()),
        vlm_generate_fn=lambda *args, **kwargs: None,
        vlm_stream_generate_fn=lambda *args, **kwargs: iter(()),
    )
    engine.processor = SimpleNamespace(tokenizer=tokenizer)
    scheduler = VlmContinuousBatchScheduler(engine, recorder.sink())
    monkeypatch.setattr(
        scheduler,
        "_make_batch_generator",
        lambda sampling_key: ScriptedGenerator(tokens, finish_reason),
    )
    monkeypatch.setattr(
        scheduler,
        "_prepare_request",
        lambda request, should_cancel: _ScheduledVlmRequest(
            request=request,
            stream=request.stream,
            full_prompt_tokens=[1],
            prompt_tokens=[1],
            prompt="hello",
            prompt_kwargs={},
            image_paths=(),
            apc_cache_key=(),
            cached_prompt=None,
            apc_extra_hash=0,
            vision_feature_cache_hit=False,
            vision_feature_cache_bytes=None,
            image_width=None,
            image_height=None,
        ),
    )
    scheduler.submit(_request(len(tokens)), stream=True)
    _drain(scheduler, recorder, len(tokens))
    return recorder


class ScriptedScheduler:
    """Feed fixed token events to the native runtime."""

    def __init__(self, tokens: bytes) -> None:
        self._tokens = list(tokens)

    def submit(self, request: Any) -> None:
        del request

    def tick(self) -> tuple[SchedulerEvent, ...]:
        return (
            SchedulerEvent(
                kind="token",
                request_id="request",
                token_id=self._tokens.pop(0),
                cache_length=1,
                phase="decode",
            ),
        )

    def finish(self, request_id: str) -> None:
        del request_id


def _run_native(
    tokenizer: ByteTokenizer,
    tokens: bytes,
    *,
    max_tokens: int,
    stop: tuple[str, ...] = (),
) -> StreamRecorder:
    recorder = StreamRecorder()
    runtime = NativeRuntime(
        ScriptedScheduler(tokens),  # type: ignore[arg-type]
        model_ref="test-model",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(EOS,),
    )
    runtime.submit(_request(max_tokens, stop))
    while recorder.response is None:
        for event in runtime.tick():
            if event.kind == "delta":
                recorder.deltas.append(event.payload.delta)
            else:
                recorder.response = event.payload
    return recorder


def _stream_native(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: ByteTokenizer,
    tokens: bytes,
    finish_reason: str,
) -> StreamRecorder:
    del monkeypatch
    if finish_reason == "stop":
        return _run_native(tokenizer, tokens + bytes([EOS]), max_tokens=len(tokens) + 1)
    return _run_native(tokenizer, tokens, max_tokens=len(tokens))


STREAMS = pytest.mark.parametrize(
    "stream",
    [
        pytest.param(_stream_text, id="text"),
        pytest.param(_stream_vlm, id="vlm"),
        pytest.param(_stream_native, id="native"),
    ],
)
FINISH_REASONS = pytest.mark.parametrize("finish_reason", ["stop", "length"])


@STREAMS
@FINISH_REASONS
@pytest.mark.parametrize(
    ("tokens", "deltas"),
    [
        pytest.param("aéb".encode(), ["a", "é", "b"], id="two-tokens"),
        pytest.param("a😀b".encode(), ["a", "😀", "b"], id="four-tokens"),
        pytest.param("aé".encode(), ["a", "é"], id="ends-complete"),
        pytest.param(b"a\xc3", ["a", REPLACEMENT], id="ends-incomplete"),
        pytest.param(b"\xf0\x9f", [REPLACEMENT], id="only-incomplete"),
        pytest.param(b"a\xc3b", ["a", f"{REPLACEMENT}b"], id="invalid-in-the-middle"),
    ],
)
def test_split_character_is_streamed_once(
    monkeypatch: pytest.MonkeyPatch,
    stream: Any,
    finish_reason: str,
    tokens: bytes,
    deltas: list[str],
) -> None:
    recorder = stream(monkeypatch, ByteTokenizer(), tokens, finish_reason)

    assert recorder.deltas == deltas
    assert recorder.response is not None
    assert recorder.response.text == tokens.decode("utf-8", errors="replace")
    assert recorder.response.finish_reason == finish_reason


@STREAMS
@FINISH_REASONS
@pytest.mark.parametrize(
    ("tokens", "text"),
    [
        pytest.param("😀😀".encode(), "😀😀", id="complete"),
        pytest.param("😀".encode() + b"\xf0", "😀" + REPLACEMENT * 4, id="incomplete"),
    ],
)
def test_emitted_text_is_kept_when_the_tokenizer_rewrites_it(
    monkeypatch: pytest.MonkeyPatch,
    stream: Any,
    finish_reason: str,
    tokens: bytes,
    text: str,
) -> None:
    recorder = stream(monkeypatch, ByteFallbackTokenizer(), tokens, finish_reason)

    assert "".join(recorder.deltas) == text
    assert recorder.response is not None
    assert recorder.response.text == text


def test_native_stop_sequence_drops_held_text() -> None:
    class PairTokenizer(ByteTokenizer):
        def decode(self, tokens: list[int], **kwargs: Any) -> str:
            del kwargs
            return "".join({1: "a", 2: f"END{REPLACEMENT}"}[token] for token in tokens)

    recorder = _run_native(PairTokenizer(), bytes([1, 2]), max_tokens=8, stop=("END",))

    assert "".join(recorder.deltas) == "a"
    assert recorder.response is not None
    assert recorder.response.text == "a"
    assert recorder.response.finish_reason == "stop"
