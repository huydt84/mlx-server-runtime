"""Entry point for the Python MLX worker bootstrap."""

from __future__ import annotations

import json
import signal
import socket
import select
import time
from contextlib import suppress
from typing import Callable

from .config import load_config
from .ipc import (
    ChatCompletionResponse,
    ChatCompletionDelta,
    CancelRequest,
    ModelError,
    ModelLoadProgress,
    ModelStatus,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_command,
    encode_bootstrap_message,
    encode_event,
)


def _pop_buffered_line(read_buffer: bytearray) -> bytes | None:
    """Pop one newline-delimited frame from buffer."""

    newline = read_buffer.find(b"\n")
    if newline < 0:
        return None
    line = bytes(read_buffer[: newline + 1])
    del read_buffer[: newline + 1]
    return line


def _read_command_line(
    client: socket.socket,
    read_buffer: bytearray,
    *,
    block: bool,
) -> bytes | None:
    """Read one newline-delimited frame from socket.

    Returns `None` only for non-blocking polls with no full frame available.
    Returns `b""` on EOF.
    """

    line = _pop_buffered_line(read_buffer)
    if line is not None:
        return line

    while True:
        if not block and not select.select([client], [], [], 0)[0]:
            return None

        chunk = client.recv(4096)
        if not chunk:
            return b""

        read_buffer.extend(chunk)
        line = _pop_buffered_line(read_buffer)
        if line is not None:
            return line

        if not block:
            return None


def _is_matching_cancel_request(raw_line: bytes, request_id: str) -> bool:
    """Return true when raw frame is cancel for active request."""

    try:
        payload = json.loads(raw_line)
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("type") == "cancel_request"
        and payload.get("request_id") == request_id
    )


def main(
    engine_factory: Callable[[str], object] | None = None,
) -> int:
    """Run the readiness handshake and Phase 1 worker loop."""

    config = load_config()
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    if engine_factory is None:
        from .engine import MlxWorkerEngine

        engine_factory = MlxWorkerEngine

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(config.socket_path)
        bootstrap_started_at = _now_seconds()
        engine: object | None = None
        try:
            _send_status(
                client,
                _status(
                    model=config.model,
                    state="loading_weights",
                    started_loading_at=bootstrap_started_at,
                    last_transition_at=bootstrap_started_at,
                ),
            )
            engine = engine_factory(config.model)
            runtime_started_at = _now_seconds()
            _send_status(
                client,
                _status(
                    model=config.model,
                    state="initializing_runtime",
                    started_loading_at=bootstrap_started_at,
                    last_transition_at=runtime_started_at,
                ),
            )
            warmup_started = time.perf_counter()
            _send_status(
                client,
                _status(
                    model=config.model,
                    state="warming_up",
                    started_loading_at=bootstrap_started_at,
                    last_transition_at=_now_seconds(),
                ),
            )
            engine.warmup()
            warmup_latency_ms = int((time.perf_counter() - warmup_started) * 1000)
            ready_at = _now_seconds()
            _send_status(
                client,
                _status(
                    model=config.model,
                    state="ready",
                    ready=True,
                    servable=True,
                    started_loading_at=bootstrap_started_at,
                    loaded_at=ready_at,
                    warmup_passed=True,
                    last_warmup_at=ready_at,
                    last_warmup_latency_ms=warmup_latency_ms,
                    last_transition_at=ready_at,
                ),
            )
        except Exception as exc:
            failed_at = _now_seconds()
            _send_status(
                client,
                _status(
                    model=config.model,
                    state="failed",
                    started_loading_at=bootstrap_started_at,
                    last_transition_at=failed_at,
                    last_error=ModelError(
                        code="MODEL_LOAD_FAILED",
                        message=str(exc),
                        at=failed_at,
                    ),
                ),
            )
            client.sendall(encode_bootstrap_message(WorkerError(str(exc))))
            return 1

        client.sendall(encode_bootstrap_message(WorkerReady()))
        read_buffer = bytearray()
        pending_lines: list[bytes] = []

        while not stop:
            if pending_lines:
                raw_line = pending_lines.pop(0)
            else:
                raw_line = _read_command_line(client, read_buffer, block=True)
            if not raw_line:
                break

            try:
                request = decode_command(raw_line)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                event: ChatCompletionResponse | WorkerCommandError | None = (
                    WorkerCommandError(
                        code="INVALID_REQUEST",
                        request_id="unknown",
                        message=str(exc),
                    )
                )
            else:
                if request is None:
                    event = WorkerCommandError(
                        code="INVALID_REQUEST",
                        request_id="unknown",
                        message="unsupported worker command",
                    )
                elif isinstance(request, CancelRequest):
                    continue
                else:
                    try:
                        assert engine is not None
                        if request.stream:
                            cancelled = False

                            def should_cancel() -> bool:
                                nonlocal cancelled
                                if cancelled:
                                    return True
                                pending = _read_command_line(
                                    client,
                                    read_buffer,
                                    block=False,
                                )
                                if pending is None:
                                    return False
                                if not pending:
                                    cancelled = True
                                    return True
                                if _is_matching_cancel_request(
                                    pending,
                                    request.request_id,
                                ):
                                    cancelled = True
                                    return True
                                pending_lines.append(pending)
                                return False

                            def emit_delta(delta: str) -> None:
                                client.sendall(
                                    encode_event(
                                        ChatCompletionDelta(
                                            request_id=request.request_id,
                                            delta=delta,
                                        )
                                    )
                                )

                            event = engine.stream_chat(  # type: ignore[attr-defined]
                                request,
                                emit_delta,
                                should_cancel,
                            )
                        else:
                            event = engine.complete_chat(request)  # type: ignore[attr-defined]
                    except Exception as exc:
                        code = (
                            "INVALID_REQUEST"
                            if isinstance(exc, ValueError)
                            else "WORKER_ERROR"
                        )
                        event = WorkerCommandError(
                            code=code,
                            request_id=request.request_id,
                            message=str(exc),
                        )

            if event is not None:
                client.sendall(encode_event(event))

        with suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)

    return 0


def _status(
    *,
    model: str,
    state: str,
    ready: bool = False,
    servable: bool = False,
    started_loading_at: int | None = None,
    loaded_at: int | None = None,
    warmup_passed: bool = False,
    last_warmup_at: int | None = None,
    last_warmup_latency_ms: int | None = None,
    last_transition_at: int | None = None,
    last_error: ModelError | None = None,
) -> ModelStatus:
    """Build a full model status snapshot."""

    now = _now_seconds()
    return ModelStatus(
        model=model,
        revision=model,
        state=state,  # type: ignore[arg-type]
        ready=ready,
        servable=servable,
        progress=_status_progress(state),
        device=None,
        dtype=None,
        loaded_at=loaded_at,
        started_loading_at=started_loading_at,
        last_transition_at=last_transition_at
        if last_transition_at is not None
        else now,
        last_error=last_error,
        warmup_passed=warmup_passed,
        last_warmup_at=last_warmup_at,
        last_warmup_latency_ms=last_warmup_latency_ms,
    )


def _status_progress(state: str) -> ModelLoadProgress | None:
    if state in {"loading_weights", "initializing_runtime", "warming_up"}:
        return ModelLoadProgress(current_phase=state)
    return None


def _send_status(client: socket.socket, status: ModelStatus) -> None:
    client.sendall(encode_bootstrap_message(status))


def _now_seconds() -> int:
    return int(time.time())


if __name__ == "__main__":
    raise SystemExit(main())
