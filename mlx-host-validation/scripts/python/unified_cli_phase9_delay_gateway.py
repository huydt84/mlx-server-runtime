#!/usr/bin/env python3
"""Host-validation proxy that adds a declared gateway response delay."""

from __future__ import annotations

from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


REAL_GATEWAY = Path(__file__).with_name("mlx_runtime_gateway.real")

if sys.argv[1:] == ["--version"]:
    os.execv(REAL_GATEWAY, [str(REAL_GATEWAY), "--version"])


def reserve_port() -> int:
    """Reserve and return one loopback TCP port."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


source_config = Path(os.environ["MLX_RUNTIME_CONFIG"])
source_text = source_config.read_text(encoding="utf-8")
proxy_port = int(re.search(r"(?m)^port = (\d+)$", source_text).group(1))
upstream_port = reserve_port()
socket_directory = Path("/tmp") / f"mlx-air-{os.getuid()}"
socket_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
proxy_socket = socket_directory / f"phase9-proxy-{os.getpid()}.sock"
proxy_config = source_config.with_suffix(".proxy.toml")
proxy_text = re.sub(r"(?m)^port = \d+$", f"port = {upstream_port}", source_text)
proxy_text = re.sub(
    r'(?m)^ipc_path = ".*"$', f'ipc_path = "{proxy_socket}"', proxy_text
)
proxy_config.write_text(proxy_text, encoding="utf-8")
environment = os.environ.copy()
environment["MLX_RUNTIME_CONFIG"] = str(proxy_config)
upstream = subprocess.Popen([str(REAL_GATEWAY)], env=environment)
delay_seconds = int(os.environ.get("MLX_AIR_PHASE9_DELAY_MS", "120")) / 1000


class Handler(BaseHTTPRequestHandler):
    """Forward gateway traffic and delay completion response bodies."""

    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length else None
        connection = HTTPConnection("127.0.0.1", upstream_port, timeout=300)
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in {"connection", "host", "content-length"}
        }
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            self.send_response(response.status)
            for name, value in response.getheaders():
                if name.lower() not in {
                    "connection",
                    "content-length",
                    "transfer-encoding",
                }:
                    self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if self.path == "/v1/chat/completions":
                time.sleep(delay_seconds)
            while chunk := response.read(4096):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (ConnectionError, OSError) as error:
            self.send_error(502, str(error))
        finally:
            self.close_connection = True
            connection.close()

    def log_message(self, _format: str, *_args: object) -> None:
        pass


server = ThreadingHTTPServer(("127.0.0.1", proxy_port), Handler)


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
    upstream.terminate()
    try:
        upstream.wait(timeout=15)
    except subprocess.TimeoutExpired:
        upstream.kill()
        upstream.wait()
    proxy_config.unlink(missing_ok=True)
    proxy_socket.unlink(missing_ok=True)
