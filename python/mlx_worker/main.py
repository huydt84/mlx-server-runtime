"""Entry point for the Python MLX worker bootstrap."""

from __future__ import annotations

import signal
import socket
import time
from contextlib import suppress

from .config import load_config
from .ipc import WorkerReady, encode_message


def main() -> int:
    """Run the readiness handshake and then keep the worker alive."""

    config = load_config()
    stop = False

    def handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(config.socket_path)
        client.sendall(encode_message(WorkerReady()))

        while not stop:
            time.sleep(1.0)

        with suppress(OSError):
            client.shutdown(socket.SHUT_RDWR)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

