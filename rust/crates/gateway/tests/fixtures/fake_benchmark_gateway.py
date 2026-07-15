#!/usr/bin/env python3
"""Fake gateway used by command-level benchmark lifecycle tests."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
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
counts = {
    "requests": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "active": 0,
    "prefix_hits": 0,
    "prefix_misses": 0,
    "prefix_reused_tokens": 0,
    "prefix_evictions": 0,
}
primed_prefixes: set[str] = set()
counts_lock = threading.Lock()
wire_requests = 0
dropped_connection = False
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
                    f"mlx_requests_active {counts['active']}\n"
                    f"mlx_prefix_cache_hits_by_backend{{backend=\"fake\",modality=\"text\",strategy=\"fake\"}} {counts['prefix_hits']}\n"
                    f"mlx_prefix_cache_misses_by_backend{{backend=\"fake\",modality=\"text\",strategy=\"fake\"}} {counts['prefix_misses']}\n"
                    f"mlx_prefix_cache_reused_tokens_by_backend{{backend=\"fake\",modality=\"text\",strategy=\"fake\"}} {counts['prefix_reused_tokens']}\n"
                    f"mlx_prefix_cache_evictions_by_backend{{backend=\"fake\",modality=\"text\",strategy=\"fake\"}} {counts['prefix_evictions']}\n"
                ).encode()
            self._send(200, "text/plain", body)
        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        global dropped_connection, wire_requests
        content_length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(content_length))
        if self.path == "/internal/benchmark/reset":
            if os.environ.get("MLX_AIR_BENCHMARK_ENABLED") != "1":
                self._send(404, "text/plain", b"not found")
                return
            with counts_lock:
                if counts["active"]:
                    self._json(
                        409,
                        {"error": {"code": "BENCHMARK_BUSY", "message": "busy"}},
                    )
                    return
                if request.get("clear_cache", True):
                    primed_prefixes.clear()
                if request.get("reset_counters", True):
                    for key in counts:
                        counts[key] = 0
                cache_state = {
                    "prefix_entries": len(primed_prefixes),
                    "prefix_hits": counts["prefix_hits"],
                    "prefix_misses": counts["prefix_misses"],
                    "prefix_evictions": counts["prefix_evictions"],
                }
            self._json(
                200,
                {
                    "request_id": "fake-reset",
                    "scheduler_idle": True,
                    "cache_state": cache_state,
                    "model_preserved": True,
                    "graphs_preserved": True,
                },
            )
            return
        content = str(request.get("messages", [{}])[0].get("content", ""))
        with counts_lock:
            wire_requests += 1
            request_count_file = os.environ.get("FAKE_GATEWAY_REQUEST_COUNT_FILE")
            if request_count_file:
                Path(request_count_file).write_text(
                    f"{wire_requests}\n", encoding="utf-8"
                )
            should_drop = (
                os.environ.get("FAKE_GATEWAY_RUN_MODE") == "drop-first-request"
                and " warmup " not in content
                and not dropped_connection
            )
            if should_drop:
                dropped_connection = True
        if should_drop:
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
            return
        if (
            os.environ.get("FAKE_GATEWAY_RUN_MODE") == "request-failure"
            and " warmup " not in content
        ):
            self._json(500, {"error": {"message": "injected request failure"}})
            return
        shared_prefix = content.partition(". Unique suffix ")[0]
        is_prime = "Benchmark-only priming suffix" in content
        with counts_lock:
            counts["active"] += 1
            counts["requests"] += 1
            counts["prompt_tokens"] += 5
            counts["completion_tokens"] += 2
            cached_tokens = 0
            if is_prime:
                primed_prefixes.add(content.partition(". Benchmark-only")[0])
            elif ". Unique suffix " in content and shared_prefix in primed_prefixes:
                cached_tokens = 3
                counts["prefix_hits"] += 1
                counts["prefix_reused_tokens"] += cached_tokens
            else:
                counts["prefix_misses"] += 1
            if "unique-prefix-" in content:
                counts["prefix_evictions"] += 1
        usage = {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
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
        with counts_lock:
            counts["active"] -= 1

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
