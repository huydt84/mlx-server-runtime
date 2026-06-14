"""Minimal line-based IPC helpers for the bootstrap handshake."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerReady:
    """Worker ready signal."""


@dataclass(frozen=True)
class WorkerError:
    """Worker startup error."""

    message: str


def encode_message(message: WorkerReady | WorkerError) -> bytes:
    """Encode a worker bootstrap message."""

    if isinstance(message, WorkerReady):
        return b"READY\n"
    sanitized = message.message.replace("\n", " ")
    return f"ERROR\t{sanitized}\n".encode("utf-8")


def decode_message(raw_line: bytes) -> WorkerReady | WorkerError | None:
    """Decode a worker bootstrap message."""

    line = raw_line.decode("utf-8", errors="replace").strip()
    if line == "READY":
        return WorkerReady()

    if line.startswith("ERROR\t"):
        return WorkerError(message=line.split("\t", 1)[1])

    return None
