from __future__ import annotations

import io
from typing import Callable
from types import SimpleNamespace

from mlx_worker.ipc import (
    ChatCompletionDelta,
    ChatCompletionResponse,
    ModelError,
    ModelStatus,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_bootstrap_message,
    decode_event,
)


class FakeSocket:
    def __init__(self, reader: io.BytesIO) -> None:
        self.reader = reader
        self.sent: list[bytes] = []
        self.connected_to: str | None = None
        self.shutdown_called = False

    def __enter__(self) -> "FakeSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def connect(self, socket_path: str) -> None:
        self.connected_to = socket_path

    def sendall(self, data: bytes) -> None:
        self.sent.append(data)

    def recv(self, size: int) -> bytes:
        return self.reader.read(size)

    def shutdown(self, how: int) -> None:
        self.shutdown_called = True


def test_main_emits_statuses_before_ready(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(io.BytesIO(b""))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.warmed = False

        def warmup(self):
            self.warmed = True
            return SimpleNamespace()

        def complete_chat(self, request):
            return SimpleNamespace()

        def stream_chat(self, request, emit_delta, should_cancel=None):
            emit_delta("hel")
            emit_delta("lo")
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    monkeypatch.setattr(worker_main, "MlxWorkerEngine", FakeEngine, raising=False)

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 0
    decoded = [decode_bootstrap_message(chunk) for chunk in fake_socket.sent]
    assert isinstance(decoded[0], ModelStatus)
    assert decoded[0].state == "loading_weights"
    assert decoded[1].state == "initializing_runtime"
    assert decoded[2].state == "warming_up"
    assert decoded[3].state == "ready"
    assert isinstance(decoded[4], WorkerReady)
    assert fake_socket.shutdown_called


def test_main_streams_deltas_before_final_response(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(
        io.BytesIO(
            b'{"type":"chat_completion","request":{"request_id":"req-1","model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
        )
    )
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return SimpleNamespace(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

        def stream_chat(self, request, emit_delta, should_cancel=None):
            emit_delta("hel")
            emit_delta("lo")
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 0
    bootstrap = [decode_bootstrap_message(chunk) for chunk in fake_socket.sent[:5]]
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    assert any(isinstance(item, WorkerReady) for item in bootstrap)
    assert any(isinstance(item, ChatCompletionDelta) for item in events)


def test_main_skips_final_event_when_cancelled(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(
        io.BytesIO(
            b'{"type":"chat_completion","request":{"request_id":"req-1","model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
            b'{"type":"cancel_request","request_id":"req-1"}\n'
        )
    )
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )
    monkeypatch.setattr(
        worker_main.select, "select", lambda *args, **kwargs: ([fake_socket], [], [])
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

        def stream_chat(self, request, emit_delta, should_cancel=None):
            emit_delta("hel")
            assert should_cancel is not None
            assert should_cancel()
            return None

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 0
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    assert any(isinstance(item, ChatCompletionDelta) for item in events)
    assert not any(isinstance(item, ChatCompletionResponse) for item in events)


def test_main_detects_buffered_cancel_without_socket_readable(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(
        io.BytesIO(
            b'{"type":"chat_completion","request":{"request_id":"req-1","model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
            b'{"type":"cancel_request","request_id":"req-1"}\n'
        )
    )
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )
    monkeypatch.setattr(
        worker_main.select, "select", lambda *args, **kwargs: ([], [], [])
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

        def stream_chat(self, request, emit_delta, should_cancel=None):
            emit_delta("hel")
            assert should_cancel is not None
            assert should_cancel()
            return None

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 0
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    assert any(isinstance(item, ChatCompletionDelta) for item in events)
    assert not any(isinstance(item, ChatCompletionResponse) for item in events)


def test_main_preserves_unmatched_buffered_command(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(
        io.BytesIO(
            b'{"type":"chat_completion","request":{"request_id":"req-1","model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
            b'{"type":"chat_completion","request":{"request_id":"req-2","model":"test-model","messages":[{"role":"user","content":"again"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
        )
    )
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )
    monkeypatch.setattr(
        worker_main.select, "select", lambda *args, **kwargs: ([fake_socket], [], [])
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.seen_requests: list[str] = []

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            self.seen_requests.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="second",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

        def stream_chat(self, request, emit_delta, should_cancel=None):
            self.seen_requests.append(request.request_id)
            emit_delta("hel")
            assert should_cancel is not None
            assert not should_cancel()
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    engine = FakeEngine("test-model")
    exit_code = worker_main.main(engine_factory=lambda _model: engine)

    assert exit_code == 0
    assert engine.seen_requests == ["req-1", "req-2"]
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    responses = [item for item in events if isinstance(item, ChatCompletionResponse)]
    assert [item.request_id for item in responses] == ["req-1", "req-2"]


def test_main_preserves_stream_when_buffered_frame_is_malformed(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(
        io.BytesIO(
            b'{"type":"chat_completion","request":{"request_id":"req-1","model":"test-model","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0.0,"top_p":1.0,"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
            b'{"type":"chat_completion","request":\n'
        )
    )
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )
    monkeypatch.setattr(
        worker_main.select, "select", lambda *args, **kwargs: ([fake_socket], [], [])
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="second",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

        def stream_chat(self, request, emit_delta, should_cancel=None):
            emit_delta("hel")
            assert should_cancel is not None
            assert not should_cancel()
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="hello",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 0
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    assert isinstance(events[0], ChatCompletionDelta)
    assert isinstance(events[1], ChatCompletionResponse)
    assert isinstance(events[2], WorkerCommandError)
    assert events[2].code == "INVALID_REQUEST"


def test_main_continuous_batching_validates_text_and_vlm_backends(
    monkeypatch,
) -> None:
    from mlx_worker import main as worker_main
    import mlx_worker.vlm_engine as vlm_mod

    validations: list[str] = []

    monkeypatch.setattr(
        worker_main,
        "validate_continuous_batching_backend",
        lambda: validations.append("text"),
    )
    monkeypatch.setattr(
        vlm_mod,
        "validate_vlm_continuous_batching_backend",
        lambda: validations.append("vlm"),
    )

    fake_socket = FakeSocket(io.BytesIO(b""))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
            continuous_batching=True,
        ),
    )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def batch_context(self):
            return SimpleNamespace()

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert validations == ["text", "vlm"]


def test_main_routes_vlm_request_to_vlm_engine(monkeypatch) -> None:
    """Requests with image content are routed to the VLM engine."""
    from mlx_worker import main as worker_main

    vlm_calls: list[str] = []
    vlm_model_ids: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            vlm_model_ids.append(model_id)
            self.model_id = model_id
            self.warmed = False

        def warmup(self):
            self.warmed = True
            return SimpleNamespace()

        def complete_chat(
            self, request, should_cancel: Callable[[], bool] | None = None
        ):
            vlm_calls.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm output",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.warmed = False
            self.seen: list[str] = []

        def warmup(self):
            self.warmed = True
            return SimpleNamespace()

        def complete_chat(self, request):
            self.seen.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text output",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    vlm_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-1","model":"vlm-model"'
        b',"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"img.jpg","detail":"auto"}}]}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(vlm_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert vlm_model_ids == ["vlm-model"]
    assert vlm_calls == ["vlm-1"]


def test_main_routes_text_request_to_text_engine_when_vlm_available(
    monkeypatch,
) -> None:
    """Requests targeting text model route to text engine (model-first)."""
    from mlx_worker import main as worker_main

    text_seen: list[str] = []
    vlm_seen: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            vlm_seen.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            text_seen.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    text_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"text-1","model":"text-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(text_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert text_seen == ["text-1"]
    assert vlm_seen == []


def test_main_routes_text_and_vlm_requests_through_continuous_batch_loop(
    monkeypatch,
) -> None:
    from mlx_worker import main as worker_main
    import mlx_worker.vlm_engine as vlm_mod

    class FakeTextScheduler:
        instances: list["FakeTextScheduler"] = []

        def __init__(self, *args, **kwargs) -> None:
            self.submitted: list[str] = []
            self.pending: list[tuple[object, bool]] = []
            self.closed = False
            FakeTextScheduler.instances.append(self)

        def submit(self, request, stream):
            self.submitted.append(request.request_id)
            self.pending.append((request, stream))
            return True

        def cancel(self, request_id: str) -> bool:
            return False

        def tick(self) -> None:
            if not self.pending:
                return
            request, stream = self.pending.pop(0)
            if stream:
                self_sink.emit_delta(request.request_id, "text-delta")
            self_sink.emit_response(
                ChatCompletionResponse(
                    request_id=request.request_id,
                    model=request.model,
                    text=f"text-{request.request_id}",
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            )

        def idle(self) -> bool:
            return not self.pending

        def close(self) -> None:
            self.closed = True

    class FakeVlmScheduler:
        instances: list["FakeVlmScheduler"] = []

        def __init__(self, engine, sink, **kwargs) -> None:
            self.engine = engine
            self.sink = sink
            self.submitted: list[str] = []
            self.pending: list[tuple[object, bool]] = []
            self.closed = False
            FakeVlmScheduler.instances.append(self)

        def submit(self, request, stream):
            self.submitted.append(request.request_id)
            self.pending.append((request, stream))
            return True

        def cancel(self, request_id: str) -> bool:
            return False

        def tick(self) -> None:
            if not self.pending:
                return
            request, stream = self.pending.pop(0)
            if stream:
                self_sink.emit_delta(request.request_id, "vlm-delta")
            self_sink.emit_response(
                ChatCompletionResponse(
                    request_id=request.request_id,
                    model=request.model,
                    text=f"vlm-{request.request_id}",
                    finish_reason="stop",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            )

        def idle(self) -> bool:
            return not self.pending

        def close(self) -> None:
            self.closed = True

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def batch_context(self):
            return SimpleNamespace()

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self.max_images_per_request = None

        def warmup(self):
            return SimpleNamespace()

    text_request = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"text-1","model":"text-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":true}}\n'
    )
    vlm_request = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-1","model":"vlm-model"'
        b',"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"benchmarks/images/fruits.png","detail":"auto"}}]}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(text_request + vlm_request))
    self_sink = SimpleNamespace()
    sink_events: list[tuple[str, str, str]] = []
    self_sink.emit_delta = lambda request_id, delta: sink_events.append(
        ("delta", request_id, delta)
    )
    self_sink.emit_response = lambda response: sink_events.append(
        ("response", response.request_id, response.text)
    )
    self_sink.emit_error = lambda request_id, code, message: sink_events.append(
        ("error", request_id, f"{code}:{message}")
    )

    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
            continuous_batching=True,
            max_vlm_images=5,
        ),
    )
    monkeypatch.setattr(worker_main, "ContinuousBatchScheduler", FakeTextScheduler)
    monkeypatch.setattr(vlm_mod, "VlmContinuousBatchScheduler", FakeVlmScheduler)
    monkeypatch.setattr(
        worker_main, "validate_continuous_batching_backend", lambda: None
    )
    monkeypatch.setattr(
        vlm_mod, "validate_vlm_continuous_batching_backend", lambda: None
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert FakeTextScheduler.instances[0].submitted == ["text-1"]
    assert FakeVlmScheduler.instances[0].submitted == ["vlm-1"]
    assert any(event[0] == "delta" and event[1] == "text-1" for event in sink_events)
    assert any(event[0] == "response" and event[1] == "vlm-1" for event in sink_events)


def test_main_rejects_image_request_to_text_model(monkeypatch) -> None:
    """Text-only model receives image content request → INVALID_REQUEST error."""
    from mlx_worker import main as worker_main

    text_seen: list[str] = []

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            text_seen.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    image_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"img-req","model":"text-model"'
        b',"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"img.jpg","detail":"auto"}}]}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(image_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model=None,
        ),
    )

    exit_code = worker_main.main(engine_factory=FakeTextEngine)

    assert exit_code == 0
    # Text engine should NOT have been called.
    assert text_seen == []
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    errors = [e for e in events if isinstance(e, WorkerCommandError)]
    assert len(errors) >= 1
    assert errors[0].code == "INVALID_REQUEST"
    assert "does not support image content" in errors[0].message


def test_main_routes_text_only_vlm_request_to_vlm_engine(monkeypatch) -> None:
    """Text-only request targeting VLM model is routed to VLM engine (model-first)."""
    from mlx_worker import main as worker_main

    vlm_calls: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(
            self, request, should_cancel: Callable[[], bool] | None = None
        ):
            vlm_calls.append(request.request_id)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm output",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    # Text-only request targeting VLM model.
    text_only_vlm_request = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-text","model":"vlm-model"'
        b',"messages":[{"role":"user","content":"hello from text"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(text_only_vlm_request))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert vlm_calls == ["vlm-text"]


def test_main_rejects_unknown_model(monkeypatch) -> None:
    """Request for a model that is neither text nor VLM is rejected."""
    from mlx_worker import main as worker_main

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    unknown_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"bad","model":"unknown-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(unknown_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(engine_factory=FakeTextEngine)

    assert exit_code == 0
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    errors = [e for e in events if isinstance(e, WorkerCommandError)]
    assert len(errors) >= 1
    assert errors[0].code == "INVALID_REQUEST"
    assert "not served" in errors[0].message


def test_main_emits_failure_status_when_warmup_fails(monkeypatch) -> None:
    from mlx_worker import main as worker_main

    fake_socket = FakeSocket(io.BytesIO(b""))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(socket_path="/tmp/test.sock", model="test-model"),
    )

    class FakeEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            raise RuntimeError("warmup failed")

    exit_code = worker_main.main(engine_factory=FakeEngine)

    assert exit_code == 1
    decoded = [decode_bootstrap_message(chunk) for chunk in fake_socket.sent]
    assert isinstance(decoded[0], ModelStatus)
    assert decoded[0].state == "loading_weights"
    assert isinstance(decoded[-1], WorkerError)
    assert "warmup failed" in decoded[-1].message
    failed_status = next(
        item
        for item in decoded
        if isinstance(item, ModelStatus) and item.state == "failed"
    )
    assert failed_status.last_error == ModelError(
        code="MODEL_LOAD_FAILED",
        message="warmup failed",
        at=failed_status.last_error.at,
    )


def test_main_does_not_construct_vlm_engine_without_vlm_config(
    monkeypatch,
) -> None:
    """Text-only config with no vlm_model does not construct VLM engine."""
    from mlx_worker import main as worker_main

    vlm_constructed: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            vlm_constructed.append(model_id)

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    text_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"text-1","model":"text-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(text_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model=None,
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert vlm_constructed == [], "VLM engine should not be constructed"


def test_main_vlm_engine_not_warmed_eagerly(monkeypatch) -> None:
    """VLM engine is NOT constructed or warmed during bootstrap.

    VLM should be initialized lazily on first request targeting the VLM
    model.
    """
    from mlx_worker import main as worker_main

    constructed: list[str] = []
    warmup_called: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            constructed.append(model_id)

        def warmup(self):
            warmup_called.append(self.model_id)
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    # Request targets the text model — VLM should stay uninitialized.
    text_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"text-1","model":"text-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(text_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert constructed == [], "VLM engine should not be constructed eagerly"
    assert warmup_called == [], "VLM engine should not be warmed eagerly"


def test_main_default_vlm_engine_lazy_construction(monkeypatch) -> None:
    """When vlm_model is configured without injection, MlxVlmEngine is
    lazily constructed on first VLM request, not during bootstrap."""
    from mlx_worker import main as worker_main

    # Must patch before main() triggers the local import.
    import mlx_worker.vlm_engine as vlm_mod

    constructed_ids: list[str] = []
    constructed_engines: list[object] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            constructed_ids.append(model_id)
            self.model_id = model_id
            self.max_images_per_request = None
            constructed_engines.append(self)

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(
            self, request, should_cancel: Callable[[], bool] | None = None
        ):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm-default",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    monkeypatch.setattr(vlm_mod, "MlxVlmEngine", FakeVlmEngine)

    vlm_request_json = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-1","model":"vlm-model"'
        b',"messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"img.jpg","detail":"auto"}}]}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )

    fake_socket = FakeSocket(io.BytesIO(vlm_request_json))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
            max_vlm_images=7,
        ),
    )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    exit_code = worker_main.main(engine_factory=FakeTextEngine)

    assert exit_code == 0
    # VLM engine was lazily constructed once on first VLM request.
    assert constructed_ids == ["vlm-model"], (
        "VLM engine should be constructed lazily on first VLM request"
    )
    assert constructed_engines[0].max_images_per_request == 7
    # Verify the VLM request actually routed to the VLM engine.
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    responses = [e for e in events if isinstance(e, ChatCompletionResponse)]
    assert len(responses) >= 1
    assert responses[0].text == "vlm-default"


def test_main_vlm_cancel_before_init_skips_initialize(monkeypatch) -> None:
    """Cancel before first VLM request dispatch skips model load entirely.

    When ``should_cancel`` returns True before the engine calls
    ``initialize()``, the blocking ``mlx_vlm.load`` must NOT be
    invoked.  The engine returns a cancelled response immediately.
    """
    from mlx_worker import main as worker_main

    init_called: list[bool] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def complete_chat(
            self, request, should_cancel: Callable[[], bool] | None = None
        ) -> ChatCompletionResponse:
            # Simulate the real engine: check cancel before initialize().
            if should_cancel is not None and should_cancel():
                return ChatCompletionResponse(
                    request_id=request.request_id,
                    model=request.model,
                    text="",
                    finish_reason="cancelled",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            init_called.append(True)
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm output",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    # VLM request followed immediately by cancel for the same request.
    vlm_request = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-1","model":"vlm-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )
    cancel = b'{"type":"cancel_request","request_id":"vlm-1"}\n'

    fake_socket = FakeSocket(io.BytesIO(vlm_request + cancel))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )
    # Force select to return readable so should_cancel picks up the cancel.
    monkeypatch.setattr(
        worker_main.select, "select", lambda *args, **kwargs: ([fake_socket], [], [])
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert init_called == [], "initialize() should NOT be called when cancelled"
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    responses = [e for e in events if isinstance(e, ChatCompletionResponse)]
    assert len(responses) >= 1
    assert responses[0].finish_reason == "cancelled"
    assert responses[0].text == ""


def test_main_vlm_non_stream_cancellation(monkeypatch) -> None:
    """Non-stream VLM request cancelled via cancel_request returns cancelled response."""
    from mlx_worker import main as worker_main

    vlm_seen: list[str] = []

    class FakeVlmEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id
            self._initialized = True  # skip deferred init for this test

        @property
        def is_initialized(self) -> bool:
            return self._initialized

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(
            self, request, should_cancel: Callable[[], bool] | None = None
        ):
            vlm_seen.append(request.request_id)
            # Simulate cancellation check inside the engine.
            if should_cancel is not None and should_cancel():
                return ChatCompletionResponse(
                    request_id=request.request_id,
                    model=request.model,
                    text="",
                    finish_reason="cancelled",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="vlm output",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    class FakeTextEngine:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def warmup(self):
            return SimpleNamespace()

        def complete_chat(self, request):
            return ChatCompletionResponse(
                request_id=request.request_id,
                model=request.model,
                text="text",
                finish_reason="stop",
                prompt_tokens=1,
                completion_tokens=1,
            )

    # VLM non-stream request followed by cancel.
    vlm_request = (
        b'{"type":"chat_completion"'
        b',"request":{"request_id":"vlm-1","model":"vlm-model"'
        b',"messages":[{"role":"user","content":"hello"}]'
        b',"max_tokens":16,"temperature":0.0,"top_p":1.0'
        b',"max_prompt_tokens":32,"max_completion_tokens":32,"max_total_tokens_per_request":64,"stream":false}}\n'
    )
    cancel = b'{"type":"cancel_request","request_id":"vlm-1"}\n'

    fake_socket = FakeSocket(io.BytesIO(vlm_request + cancel))
    monkeypatch.setattr(
        worker_main.socket, "socket", lambda *args, **kwargs: fake_socket
    )
    monkeypatch.setattr(
        worker_main,
        "load_config",
        lambda: SimpleNamespace(
            socket_path="/tmp/test.sock",
            model="text-model",
            vlm_model="vlm-model",
        ),
    )

    exit_code = worker_main.main(
        engine_factory=FakeTextEngine,
        vlm_engine_factory=FakeVlmEngine,
    )

    assert exit_code == 0
    assert vlm_seen == ["vlm-1"]
    events = [decode_event(chunk) for chunk in fake_socket.sent[5:]]
    responses = [e for e in events if isinstance(e, ChatCompletionResponse)]
    assert len(responses) >= 1
    # The VLM engine saw cancel and returned cancelled.
    assert responses[0].finish_reason == "cancelled"
    assert responses[0].text == ""
