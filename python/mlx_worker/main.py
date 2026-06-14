"""Entry point for the Python MLX worker bootstrap."""

from __future__ import annotations

import signal
import socket
from contextlib import suppress

from .config import load_config
from .ipc import (
    ChatCompletionResponse,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_command,
    encode_bootstrap_message,
    encode_event,
)


def main() -> int:
    """Run the readiness handshake and Phase 1 worker loop."""

    config = load_config()
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(config.socket_path)
        try:
            from .engine import MlxWorkerEngine

            engine = MlxWorkerEngine(config.model)
        except Exception as exc:
            client.sendall(encode_bootstrap_message(WorkerError(str(exc))))
            return 1

        client.sendall(encode_bootstrap_message(WorkerReady()))
        reader = client.makefile("rb")

        while not stop:
            raw_line = reader.readline()
            if not raw_line:
                break

            request = decode_command(raw_line)
            if request is None:
                event: ChatCompletionResponse | WorkerCommandError = WorkerCommandError(
                    request_id="unknown",
                    message="unsupported worker command",
                )
            else:
                try:
                    event = engine.complete_chat(request)
                except Exception as exc:
                    event = WorkerCommandError(
                        request_id=request.request_id,
                        message=str(exc),
                    )

            client.sendall(encode_event(event))

        with suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
