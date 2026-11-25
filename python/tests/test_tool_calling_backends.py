from __future__ import annotations

import json
import socket
import sys
import threading
from contextlib import nullcontext
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mlx_worker.batching import (
    BatchEventSink,
    ContinuousBatchScheduler,
    MlxBatchCompletionBackend,
)
from mlx_worker.config import WorkerConfig
from mlx_worker.engine import MlxWorkerEngine
from mlx_worker.ipc import (
    ChatCompletionDelta,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    TextContent,
    WorkerCommandError,
    decode_event,
    encode_command,
)
from mlx_worker.native_mlx.interfaces import SchedulerEvent
from mlx_worker.native_mlx.runtime import NativeRuntime
from mlx_worker.native_mlx.worker import run_native_worker
from mlx_worker.tool_calling import ToolsNotSupportedError, resolve_tool_format

EOS = 0
TOOLS = ({"type": "function", "function": {"name": "weather"}},)
CALL = '<tool_call>{"name": "weather", "arguments": {"city": "Huế"}}</tool_call>'


class CharTokenizer:
    """Tokenizer whose token ids are code points and whose prompt is fixed."""

    eos_token_id = EOS
    eos_token_ids = (EOS,)
    has_chat_template = True
    think_start = "<think>"
    think_end = "</think>"

    def __init__(self, prompt: str = "p") -> None:
        self.prompt = prompt
        self.template_calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        self.template_calls.append((messages, kwargs))
        return [ord(char) for char in self.prompt]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(token_id) for token_id in token_ids)

    @property
    def detokenizer(self) -> _Detokenizer:
        return _Detokenizer()


class _Detokenizer:
    def __init__(self) -> None:
        self._last = ""

    def add_token(self, token_id: int) -> None:
        self._last = chr(token_id)

    @property
    def last_segment(self) -> str:
        segment, self._last = self._last, ""
        return segment

    def finalize(self) -> None:
        return None


def _request(**overrides: Any) -> ChatCompletionRequest:
    values: dict[str, Any] = {
        "request_id": "req-1",
        "model": "m",
        "messages": [ChatMessage(role="user", content="weather?")],
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_prompt_tokens": 64,
        "max_completion_tokens": 512,
        "max_total_tokens_per_request": 1024,
        "stream": True,
        "tools": TOOLS,
    }
    values.update(overrides)
    return ChatCompletionRequest(**values)


class Reply:
    """What one request produced, assembled as a streaming client would."""

    def __init__(
        self,
        deltas: list[ChatCompletionDelta],
        response: ChatCompletionResponse,
        stopped: bool,
    ) -> None:
        self.deltas = deltas
        self.response = response
        self.stopped = stopped

    @property
    def streamed_text(self) -> str:
        return "".join(delta.delta for delta in self.deltas)

    @property
    def streamed_reasoning(self) -> str:
        return "".join(delta.reasoning or "" for delta in self.deltas)

    @property
    def streamed_calls(self) -> list[str]:
        return [item.call.id for delta in self.deltas for item in delta.tool_calls]

    @property
    def calls(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (call.name, json.loads(call.arguments)) for call in self.response.tool_calls
        ]


class _ScriptedScheduler:
    """Native scheduler stand-in that emits one token of the text per step."""

    def __init__(self, text: str) -> None:
        self._tokens = [ord(char) for char in text]
        self.finished: list[str] = []
        self.generated = 0
        self._cancelled = False
        self._request_id: str | None = None

    def submit(self, work: Any) -> None:
        self._request_id = work.request_id

    def tick(self) -> tuple[SchedulerEvent, ...]:
        if self._cancelled:
            return (SchedulerEvent(kind="cancelled", request_id=self._request_id),)
        if self.finished or self.generated >= len(self._tokens):
            return ()
        token_id = self._tokens[self.generated]
        self.generated += 1
        return (
            SchedulerEvent(
                kind="token",
                request_id=self._request_id,
                token_id=token_id,
                cache_length=self.generated,
                phase="decode",
            ),
        )

    def finish(self, request_id: str) -> None:
        self.finished.append(request_id)

    def cancel(self, request_id: str) -> bool:
        self._cancelled = request_id == self._request_id
        return self._cancelled

    def idle(self) -> bool:
        return self._request_id is None or bool(self.finished)

    def close(self) -> None:
        return None


def _native_runtime(
    scheduler: _ScriptedScheduler, tokenizer: CharTokenizer, parser: str = "json_tools"
) -> NativeRuntime:
    return NativeRuntime(
        scheduler,  # type: ignore[arg-type]
        model_ref="m",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(EOS,),
        tool_format=resolve_tool_format(tokenizer, parser),
    )


def _run_native(
    text: str, request: ChatCompletionRequest, tokenizer: CharTokenizer
) -> Reply:
    scheduler = _ScriptedScheduler(text + chr(EOS))
    runtime = _native_runtime(scheduler, tokenizer)
    runtime.submit(request)
    deltas: list[ChatCompletionDelta] = []
    responses: list[ChatCompletionResponse] = []
    for _ in range(len(text) + 1):
        for event in runtime.tick():
            if event.kind == "delta":
                deltas.append(event.payload)
            elif event.kind == "response":
                responses.append(event.payload)
        if responses:
            break
    (response,) = responses
    return Reply(deltas, response, stopped=scheduler.generated <= len(text))


class _ScriptedBatchGenerator:
    """mlx-lm batch generator stand-in that emits one token per step."""

    text = ""

    def __init__(
        self,
        model: Any,
        stop_tokens: Any = None,
        completion_batch_size: int = 1,
        prefill_batch_size: int = 1,
        prefill_step_size: int = 1,
    ) -> None:
        del model, stop_tokens
        del completion_batch_size, prefill_batch_size, prefill_step_size
        self.removed: list[int] = []
        self.position = 0
        self.prompt_cache_nbytes = 0
        type(self).last = self

    def insert(
        self,
        prompts: Any,
        *,
        max_tokens: Any,
        caches: Any = None,
        all_tokens: Any = None,
        samplers: Any = None,
    ) -> list[int]:
        del prompts, caches, all_tokens, samplers
        (self._max_tokens,) = max_tokens
        return [7]

    def next_generated(self) -> list[Any]:
        if self.removed or self.position > len(self.text):
            return []
        at_end = self.position == len(self.text)
        token = EOS if at_end else ord(self.text[self.position])
        self.position += 1
        finish_reason = "stop" if at_end else None
        if not at_end and self.position >= self._max_tokens:
            finish_reason = "length"
            self.removed.append(7)
        return [
            SimpleNamespace(
                uid=7, token=token, finish_reason=finish_reason, prompt_cache=None
            )
        ]

    def remove(self, uids: list[int]) -> None:
        self.removed.extend(uids)

    def stats(self) -> Any:
        return nullcontext(SimpleNamespace(peak_memory=0.0))

    def close(self) -> None:
        return None


@pytest.fixture
def batch_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    resolve_tool_format(CharTokenizer(), "json_tools")
    generate = ModuleType("mlx_lm.generate")
    generate.BatchGenerator = _ScriptedBatchGenerator  # type: ignore[attr-defined]
    models = ModuleType("mlx_lm.models")
    cache = ModuleType("mlx_lm.models.cache")
    cache.trim_prompt_cache = lambda prompt_cache, count: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", generate)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache)


def _engine(tokenizer: CharTokenizer, parser: str = "json_tools") -> MlxWorkerEngine:
    return MlxWorkerEngine(
        "m",
        model_loader=lambda model_id: (SimpleNamespace(), tokenizer),
        batch_backend_factory=MlxBatchCompletionBackend,
        tool_call_parser=parser,
    )


def _run_v1_batch(
    text: str, request: ChatCompletionRequest, tokenizer: CharTokenizer
) -> Reply:
    _ScriptedBatchGenerator.text = text
    response = _engine(tokenizer).complete_chat(replace(request, stream=False))
    stopped = _ScriptedBatchGenerator.last.position <= len(text)
    return Reply([], response, stopped)


def _run_v1_stream(
    text: str, request: ChatCompletionRequest, tokenizer: CharTokenizer
) -> Reply:
    generated: list[str] = []

    def stream_generate(*args: Any, max_tokens: int, **kwargs: Any) -> Any:
        del args, kwargs
        for index, char in enumerate(text, start=1):
            generated.append(char)
            ends = index >= max_tokens
            yield SimpleNamespace(
                text=char,
                finish_reason="length" if ends else None,
                prompt_tokens=1,
                generation_tokens=index,
            )
            if ends:
                return
        generated.append("")
        yield SimpleNamespace(
            text="",
            finish_reason="stop",
            prompt_tokens=1,
            generation_tokens=len(text) + 1,
        )

    engine = _engine(tokenizer)
    engine._stream_generate = stream_generate
    deltas: list[ChatCompletionDelta] = []

    def emit_delta(delta: str, *, reasoning: Any = None, tool_calls: Any = ()) -> None:
        deltas.append(
            ChatCompletionDelta(request.request_id, delta, reasoning, tool_calls)
        )

    response = engine.stream_chat(request, emit_delta)
    return Reply(deltas, response, stopped=len(generated) <= len(text))


class _Sink:
    def __init__(self) -> None:
        self.deltas: list[ChatCompletionDelta] = []
        self.responses: list[ChatCompletionResponse] = []
        self.errors: list[tuple[str, str]] = []

    def emit_delta(
        self,
        request_id: str,
        delta: str,
        *,
        reasoning: Any = None,
        tool_calls: Any = (),
    ) -> None:
        self.deltas.append(
            ChatCompletionDelta(request_id, delta, reasoning, tool_calls)
        )

    def event_sink(self) -> BatchEventSink:
        return BatchEventSink(
            emit_delta=self.emit_delta,
            emit_response=self.responses.append,
            emit_error=lambda request_id, code, message: self.errors.append(
                (code, message)
            ),
        )


def _continuous(
    tokenizer: CharTokenizer, parser: str = "json_tools"
) -> tuple[ContinuousBatchScheduler, _Sink]:
    sink = _Sink()
    scheduler = ContinuousBatchScheduler(
        _engine(tokenizer, parser).batch_context(),
        sink.event_sink(),
        prompt_concurrency=1,
        decode_concurrency=1,
        prefill_step_size=64,
    )
    return scheduler, sink


def _run_v1_continuous(
    text: str, request: ChatCompletionRequest, tokenizer: CharTokenizer
) -> Reply:
    _ScriptedBatchGenerator.text = text
    scheduler, sink = _continuous(tokenizer)
    scheduler.submit(request, stream=request.stream)
    for _ in range(len(text) + 2):
        scheduler.tick()
    (response,) = sink.responses
    assert sink.errors == []
    stopped = _ScriptedBatchGenerator.last.position <= len(text)
    return Reply(sink.deltas, response, stopped)


STREAMING_RUNNERS = [_run_native, _run_v1_stream, _run_v1_continuous]
RUNNERS = [*STREAMING_RUNNERS, _run_v1_batch]

pytestmark = pytest.mark.usefixtures("batch_generator")


@pytest.mark.parametrize("run", RUNNERS)
def test_reply_is_split_into_thinking_text_and_calls(run: Any) -> None:
    tokenizer = CharTokenizer(prompt="assistant <think>\n")

    reply = run(f"plan\n</think>\n\nOn it.\n{CALL}", _request(), tokenizer)

    assert reply.response.text == "On it."
    assert reply.response.reasoning == "plan"
    assert reply.calls == [("weather", {"city": "Huế"})]
    assert reply.response.finish_reason == "tool_calls"
    assert reply.response.tool_parser == "json_tools"
    assert reply.response.tool_parse_failures == 0
    assert tokenizer.template_calls[0][1]["tools"] == list(TOOLS)


@pytest.mark.parametrize("run", STREAMING_RUNNERS)
def test_streamed_pieces_add_up_to_the_response(run: Any) -> None:
    tokenizer = CharTokenizer(prompt="assistant <think>\n")

    reply = run(f"plan</think>On it.{CALL}{CALL}", _request(), tokenizer)

    assert reply.streamed_text == reply.response.text == "On it."
    assert reply.streamed_reasoning == reply.response.reasoning == "plan"
    assert reply.streamed_calls == [call.id for call in reply.response.tool_calls]
    assert len(reply.streamed_calls) == 2


@pytest.mark.parametrize("run", RUNNERS)
def test_malformed_call_is_returned_as_text(run: Any) -> None:
    text = "<tool_call>{oops}</tool_call>"

    reply = run(text, _request(), CharTokenizer())

    assert reply.response.text == text
    assert reply.calls == []
    assert reply.response.finish_reason == "stop"
    assert reply.response.tool_parse_failures == 1


@pytest.mark.parametrize("run", RUNNERS)
def test_call_cut_by_the_token_limit_is_returned_as_text(run: Any) -> None:
    cut = len(CALL) - 4

    reply = run(CALL, _request(max_tokens=cut), CharTokenizer())

    assert reply.response.text == CALL[:cut]
    assert reply.calls == []
    assert reply.response.finish_reason == "length"
    assert reply.response.tool_parse_failures == 0


@pytest.mark.parametrize("run", RUNNERS)
def test_call_without_an_end_marker_counts_when_the_model_ended_its_turn(
    run: Any,
) -> None:
    reply = run(CALL.removesuffix("</tool_call>"), _request(), CharTokenizer())

    assert reply.calls == [("weather", {"city": "Huế"})]
    assert reply.response.finish_reason == "tool_calls"


@pytest.mark.parametrize("run", RUNNERS)
def test_generation_stops_after_one_call_without_parallel_calls(run: Any) -> None:
    reply = run(
        f"{CALL} and {CALL}", _request(parallel_tool_calls=False), CharTokenizer()
    )

    assert reply.calls == [("weather", {"city": "Huế"})]
    assert reply.response.text == ""
    assert reply.response.finish_reason == "tool_calls"
    assert reply.response.completion_tokens == len(CALL)
    assert reply.stopped


@pytest.mark.parametrize("run", RUNNERS)
def test_one_call_at_the_token_limit_still_reports_tool_calls(run: Any) -> None:
    request = _request(parallel_tool_calls=False, max_tokens=len(CALL))

    reply = run(f"{CALL}{CALL}", request, CharTokenizer())

    assert len(reply.calls) == 1
    assert reply.response.finish_reason == "tool_calls"


@pytest.mark.parametrize("run", RUNNERS)
def test_requests_without_tools_keep_their_output(run: Any) -> None:
    text = f"<think>x</think> \n{CALL}"

    reply = run(text, _request(tools=()), CharTokenizer())

    assert reply.response.text == text
    assert reply.response.reasoning is None
    assert reply.response.tool_calls == ()
    assert reply.response.tool_parser is None
    assert reply.response.finish_reason == "stop"


@pytest.mark.parametrize("run", RUNNERS)
def test_text_parts_are_joined(run: Any) -> None:
    tokenizer = CharTokenizer()
    parts = (TextContent("a"), TextContent("b"))

    run("x", _request(messages=[ChatMessage("user", parts)], tools=()), tokenizer)

    assert tokenizer.template_calls[0][0] == [{"role": "user", "content": "ab"}]


def test_native_stop_string_inside_a_call_returns_it_as_text() -> None:
    reply = _run_native(CALL, _request(stop=("Huế",)), CharTokenizer())

    assert reply.response.text == CALL.split("Huế")[0]
    assert reply.streamed_text == reply.response.text
    assert reply.calls == []
    assert reply.response.finish_reason == "stop"


def test_native_stop_string_after_a_call_keeps_the_call() -> None:
    reply = _run_native(f"{CALL}END more", _request(stop=("END",)), CharTokenizer())

    assert len(reply.calls) == 1
    assert reply.response.text == ""
    assert reply.response.finish_reason == "tool_calls"


def test_native_cancellation_drops_a_pending_call() -> None:
    scheduler = _ScriptedScheduler(f"Hi {CALL}")
    runtime = _native_runtime(scheduler, CharTokenizer())
    runtime.submit(_request())
    deltas = [
        event.payload
        for _ in range(len("Hi ") + len(CALL) - 1)
        for event in runtime.tick()
        if event.kind == "delta"
    ]

    assert runtime.cancel("req-1")
    events = runtime.tick()

    assert "".join(delta.delta for delta in deltas) == "Hi"
    responses = [event.payload for event in events if event.kind == "response"]
    assert [event for event in events if event.kind == "delta"] == []
    assert [response.finish_reason for response in responses] == ["cancelled"]
    assert responses[0].tool_calls == ()
    assert responses[0].text == "Hi"


def test_v1_stream_cancellation_drops_a_pending_call() -> None:
    engine = _engine(CharTokenizer())
    text = f"Hi {CALL}"
    engine._stream_generate = lambda *args, **kwargs: (
        SimpleNamespace(
            text=char, finish_reason=None, prompt_tokens=1, generation_tokens=1
        )
        for char in text
    )
    seen: list[str] = []
    polls = iter(range(10_000))

    response = engine.stream_chat(
        _request(),
        lambda delta, **kwargs: seen.append(delta),
        lambda: next(polls) >= 2 * (len(text) - 2),
    )

    assert "".join(seen) == "Hi"
    assert response.finish_reason == "cancelled"
    assert response.tool_calls == ()
    assert response.text == "Hi"


def test_v1_continuous_cancellation_drops_a_pending_call() -> None:
    _ScriptedBatchGenerator.text = f"Hi {CALL}"
    scheduler, sink = _continuous(CharTokenizer())
    scheduler.submit(_request(), stream=True)
    for _ in range(len(CALL)):
        scheduler.tick()

    scheduler.cancel("req-1")

    (response,) = sink.responses
    assert response.finish_reason == "cancelled"
    assert response.tool_calls == ()
    assert response.text == "Hi"
    assert all(not delta.tool_calls for delta in sink.deltas)


def test_native_model_without_a_parser_rejects_tools() -> None:
    runtime = _native_runtime(_ScriptedScheduler(""), CharTokenizer(), "none")

    with pytest.raises(ToolsNotSupportedError, match="'m'"):
        runtime.submit(_request())
    runtime.submit(_request(request_id="req-2", tools=()))


def test_v1_model_without_a_parser_rejects_tools() -> None:
    engine = _engine(CharTokenizer(), "none")

    with pytest.raises(ToolsNotSupportedError, match="'m'"):
        engine.complete_chat(_request())
    with pytest.raises(ToolsNotSupportedError, match="'m'"):
        engine.stream_chat(_request(), lambda delta, **kwargs: None)


def test_v1_continuous_reports_tools_not_supported() -> None:
    scheduler, sink = _continuous(CharTokenizer(), "none")

    scheduler.submit(_request(), stream=True)

    assert [code for code, _ in sink.errors] == ["TOOLS_NOT_SUPPORTED"]


def test_v1_continuous_reports_template_errors() -> None:
    class BrokenTokenizer(CharTokenizer):
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
            raise RuntimeError("roles must alternate")

    scheduler, sink = _continuous(BrokenTokenizer())

    scheduler.submit(_request(), stream=True)

    assert [code for code, _ in sink.errors] == ["INVALID_REQUEST"]


def test_v1_continuous_waits_for_complete_characters() -> None:
    class BytesTokenizer(CharTokenizer):
        def decode(
            self, token_ids: list[int], *, skip_special_tokens: bool = False
        ) -> str:
            del skip_special_tokens
            return bytes(token_ids).decode("utf-8", errors="replace")

    encoded = f"Huế {CALL}".encode()
    reply = _run_v1_continuous(
        "".join(chr(byte) for byte in encoded), _request(), BytesTokenizer()
    )

    assert reply.streamed_text == reply.response.text == "Huế"
    assert reply.calls == [("weather", {"city": "Huế"})]


class _WarmRuntime(NativeRuntime):
    def warmup(self) -> None:
        return None


def _native_worker_events(
    reply: str, request: ChatCompletionRequest, parser: str
) -> list[Any]:
    client, gateway = socket.socketpair()
    tokenizer = CharTokenizer()
    runtime = _WarmRuntime(
        _ScriptedScheduler(reply + chr(EOS)),  # type: ignore[arg-type]
        model_ref="m",
        prompt_tokenizer=tokenizer,
        decode_target=tokenizer,
        eos_token_ids=(EOS,),
        tool_format=resolve_tool_format(tokenizer, parser),
    )
    with client, gateway:
        gateway.sendall(encode_command(request))
        worker = threading.Thread(
            target=run_native_worker,
            args=(client, WorkerConfig(socket_path="unused", model="m")),
            kwargs={"native_worker_factory": lambda *args, **kwargs: runtime},
        )
        worker.start()
        events: list[Any] = []
        gateway.settimeout(10)
        with gateway.makefile("rb") as lines:
            for line in lines:
                if not line.startswith(b"{"):
                    continue
                event = decode_event(line)
                events.append(event)
                if isinstance(event, ChatCompletionResponse | WorkerCommandError):
                    break
        gateway.shutdown(socket.SHUT_RDWR)
        worker.join(timeout=10)
        assert not worker.is_alive()
    return events


def test_native_worker_sends_tool_calls_over_ipc() -> None:
    events = _native_worker_events(f"On it. {CALL}", _request(), "json_tools")

    *deltas, response = events
    assert isinstance(response, ChatCompletionResponse)
    assert "".join(delta.delta for delta in deltas) == response.text == "On it."
    streamed = [item.call for delta in deltas for item in delta.tool_calls]
    assert tuple(streamed) == response.tool_calls
    assert response.finish_reason == "tool_calls"
    assert response.tool_parser == "json_tools"


def test_native_worker_reports_tools_not_supported_over_ipc() -> None:
    (error,) = _native_worker_events("", _request(), "none")

    assert isinstance(error, WorkerCommandError)
    assert error.code == "TOOLS_NOT_SUPPORTED"
    assert error.request_id == "req-1"
    assert "'m'" in error.message
