from __future__ import annotations

import io
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
