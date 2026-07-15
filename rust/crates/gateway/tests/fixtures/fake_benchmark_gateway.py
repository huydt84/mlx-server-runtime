#!/usr/bin/env python3
"""Fake gateway used by command-level benchmark lifecycle tests."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


if sys.argv[1:] == ["--version"]:
    print("0.1.0")
    raise SystemExit(0)


def config_values(path: Path) -> dict[str, str]:
    """Read the simple scalar values needed from the generated TOML file."""
    values: dict[str, str] = {}
    section = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
        elif "=" in line:
            key, raw_value = line.split("=", 1)
            values[f"{section}.{key.strip()}"] = raw_value.strip().strip('"')
    return values


configuration = config_values(Path(os.environ["MLX_RUNTIME_CONFIG"]))
port = int(configuration["server.port"])
model = configuration["worker.model"]
started = time.monotonic()
counts = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0}
counts_lock = threading.Lock()
worker = subprocess.Popen(["/bin/sleep", "60"])

pid_file = os.environ.get("FAKE_GATEWAY_PID_FILE")
if pid_file:
    Path(pid_file).write_text(f"{os.getpid()}\n{worker.pid}\n", encoding="utf-8")
worker_log = os.environ.get("MLX_AIR_WORKER_LOG")
if worker_log:
    Path(worker_log).write_text("fake worker started\n", encoding="utf-8")
print(f"fake gateway listening on {port}", file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        if self.path == "/ready":
            delayed = os.environ.get("FAKE_GATEWAY_DELAY_READY") == "1"
            if delayed and time.monotonic() - started < 60:
                self._json(503, {"ready": False, "status": "starting"})
            else:
                self._json(
                    200,
                    {
                        "ready": True,
                        "status": "ready",
                        "model": model,
                        "revision": "fake-revision",
                    },
                )
        elif self.path == "/version":
            self._json(200, {"gateway_version": "0.1.0"})
        elif self.path == "/metrics":
            with counts_lock:
                body = (
                    f"mlx_requests_total {counts['requests']}\n"
                    f"mlx_prompt_tokens_total {counts['prompt_tokens']}\n"
                    f"mlx_completion_tokens_total {counts['completion_tokens']}\n"
                ).encode()
            self._send(200, "text/plain", body)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(content_length))
        if os.environ.get("FAKE_GATEWAY_RUN_MODE") == "request-failure":
            self._json(500, {"error": {"message": "injected request failure"}})
            return
        with counts_lock:
            counts["requests"] += 1
            counts["prompt_tokens"] += 5
            counts["completion_tokens"] += 2
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
        }
        if request.get("stream"):
            events = [
                {"choices": [{"delta": {"content": "fake"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": " output"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": usage},
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            body += "data: [DONE]\n\n"
            self._send(200, "text/event-stream", body.encode())
        else:
            self._json(
                200,
                {
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": "fake output"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": usage,
                },
            )

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _json(self, status: int, value: object) -> None:
        self._send(status, "application/json", json.dumps(value).encode())

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)


def interrupt(_signum: int, _frame: object) -> None:
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, interrupt)
signal.signal(signal.SIGTERM, interrupt)
try:
    server.serve_forever(poll_interval=0.05)
except KeyboardInterrupt:
    pass
finally:
    server.server_close()
    worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()
        worker.wait()
