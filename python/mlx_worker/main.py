"""Entry point for the Python MLX worker bootstrap, including Phase 8 VLM dispatch."""

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
    request_has_images,
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


def _make_should_cancel(
    client: socket.socket,
    read_buffer: bytearray,
    pending_lines: list[bytes],
    request_id: str,
) -> Callable[[], bool]:
    """Build a ``should_cancel`` callback for the active request.

    Returns a closure that the engine should call before expensive
    operations.  Each invocation peeks at the socket without blocking.
    On a matched ``cancel_request``, EOF, or internal error the closure
    returns ``True`` and caches the result so subsequent calls stay
    cancelled.
    """
    cancelled: bool = False

    def should_cancel() -> bool:
        nonlocal cancelled
        if cancelled:
            return True
        pending = _read_command_line(client, read_buffer, block=False)
        if pending is None:
            return False
        if not pending:
            cancelled = True
            return True
        if _is_matching_cancel_request(pending, request_id):
            cancelled = True
            return True
        pending_lines.append(pending)
        return False

    return should_cancel


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
    vlm_engine_factory: Callable[[str], object] | None = None,
) -> int:
    """Run the readiness handshake and Phase 1 worker loop with VLM routing."""

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

    # Default VLM engine factory when env var is set (production path).
    # Engine is NOT constructed eagerly — lazy init on first VLM request.
    if vlm_engine_factory is None:
        vlm_model = getattr(config, "vlm_model", None)
        if vlm_model is not None:
            from .vlm_engine import MlxVlmEngine

            vlm_engine_factory = MlxVlmEngine

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(config.socket_path)
        bootstrap_started_at = _now_seconds()
        engine: object | None = None
        vlm_engine: object | None = None
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
                        # Model-first dispatch: route by model name, not image presence.
                        vlm_model_cfg: str | None = getattr(config, "vlm_model", None)
                        should_cancel = _make_should_cancel(
                            client, read_buffer, pending_lines, request.request_id
                        )
                        if vlm_model_cfg is not None and request.model == vlm_model_cfg:
                            # Lazy initialize VLM engine on first request.
                            if vlm_engine is None:
                                vlm_engine = vlm_engine_factory(config.vlm_model)
                                setattr(
                                    vlm_engine,
                                    "max_images_per_request",
                                    getattr(config, "max_vlm_images", 5),
                                )
                            active_engine: object | None = vlm_engine
                        elif request.model == config.model:
                            active_engine = engine
                        else:
                            raise ValueError(
                                f"model '{request.model}' is not served by this worker "
                                f"(serves text='{config.model}'"
                                + (
                                    f", vlm='{vlm_model_cfg}')"
                                    if vlm_model_cfg
                                    else ")"
                                )
                            )

                        # Text-only engine cannot process image content.
                        if active_engine is engine and request_has_images(request):
                            raise ValueError(
                                f"model '{config.model}' is a text-only model "
                                "and does not support image content"
                            )

                        # VLM engine initialization is deferred to
                        # ``_generate_vlm`` / ``_stream_vlm`` where the
                        # ``should_cancel`` closure is already available.
                        # The engine checks cancellation before the blocking
                        # ``mlx_vlm.load`` call so a cancelled first request
                        # never blocks on cold-start model loading.

                        if request.stream:

                            def emit_delta(delta: str) -> None:
                                client.sendall(
                                    encode_event(
                                        ChatCompletionDelta(
                                            request_id=request.request_id,
                                            delta=delta,
                                        )
                                    )
                                )

                            event = active_engine.stream_chat(  # type: ignore[attr-defined]
                                request,
                                emit_delta,
                                should_cancel,
                            )
                        else:
                            # Non-stream: only VLM engine accepts should_cancel.
                            # Text engine's complete_chat does not accept it.
                            kwargs: dict[str, object] = {}
                            if active_engine is vlm_engine:
                                kwargs["should_cancel"] = should_cancel
                            event = active_engine.complete_chat(  # type: ignore[attr-defined]
                                request, **kwargs
                            )
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
