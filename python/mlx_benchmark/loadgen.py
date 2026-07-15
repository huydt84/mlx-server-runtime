"""Asynchronous HTTP load generation for declarative benchmark workloads."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import hashlib
import json
import ssl
import time
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

from mlx_benchmark.prompts import Prompt


class LoadGenerationError(RuntimeError):
    """An HTTP protocol or load-generation failure."""


class _HttpConnection:
    def __init__(self, host: str, port: int, use_tls: bool) -> None:
        self.host = host
        self.port = port
        self.use_tls = use_tls
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except (ConnectionError, OSError):
                pass
        self.reader = None
        self.writer = None

    async def post_json(
        self, path: str, payload: dict[str, Any], streaming: bool
    ) -> tuple[dict[str, Any], bool]:
        # A transport failure after write/drain is ambiguous: the server may
        # already have executed the request. Retrying would duplicate measured
        # inference work and corrupt request, cache, and runtime-counter totals.
        return await self._post_json(path, payload, streaming)

    async def _post_json(
        self, path: str, payload: dict[str, Any], streaming: bool
    ) -> tuple[dict[str, Any], bool]:
        if self.reader is None or self.writer is None:
            context = ssl.create_default_context() if self.use_tls else None
            self.reader, self.writer = await asyncio.open_connection(
                self.host,
                self.port,
                ssl=context,
                server_hostname=self.host if self.use_tls else None,
            )
        body = json.dumps(payload, separators=(",", ":")).encode()
        accept = "text/event-stream" if streaming else "application/json"
        request = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            f"Accept: {accept}\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: keep-alive\r\n\r\n"
        ).encode() + body
        self.writer.write(request)
        await self.writer.drain()

        status_line = await self.reader.readline()
        if not status_line:
            raise ConnectionError("server closed the connection before the response")
        first_byte_ns = time.monotonic_ns()
        try:
            version, raw_status, _reason = status_line.decode("latin-1").split(" ", 2)
            status = int(raw_status)
        except (UnicodeDecodeError, ValueError) as error:
            raise LoadGenerationError(
                f"invalid HTTP status line: {status_line!r}"
            ) from error
        headers: dict[str, str] = {}
        while True:
            line = await self.reader.readline()
            if line in {b"\r\n", b"\n"}:
                break
            if not line:
                raise ConnectionError(
                    "server closed the connection in response headers"
                )
            try:
                key, value = line.decode("latin-1").split(":", 1)
            except (UnicodeDecodeError, ValueError) as error:
                raise LoadGenerationError(f"invalid HTTP header: {line!r}") from error
            headers[key.strip().lower()] = value.strip()

        reusable = (
            version == "HTTP/1.1"
            and headers.get("connection", "").lower() != "close"
            and (
                "content-length" in headers
                or "chunked" in headers.get("transfer-encoding", "").lower()
            )
        )
        if streaming:
            response = await _read_stream_response(
                status, headers, self.reader, first_byte_ns
            )
        else:
            response = await _read_json_response(
                status, headers, self.reader, first_byte_ns
            )
        return response, reusable


class _AsyncHttpPool:
    def __init__(self, base_url: str, maximum_connections: int) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise LoadGenerationError(f"invalid benchmark base URL: {base_url}")
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.use_tls = parsed.scheme == "https"
        self.path = f"{parsed.path.rstrip('/')}/v1/chat/completions"
        self._available: list[_HttpConnection] = []
        self._all: list[_HttpConnection] = []
        self._semaphore = asyncio.Semaphore(maximum_connections)
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        await asyncio.gather(*(connection.close() for connection in self._all))

    async def post_json(
        self, payload: dict[str, Any], streaming: bool, timeout_seconds: int
    ) -> dict[str, Any]:
        await self._semaphore.acquire()
        async with self._lock:
            if self._available:
                connection = self._available.pop()
            else:
                connection = _HttpConnection(self.host, self.port, self.use_tls)
                self._all.append(connection)
        reusable = False
        try:
            response, reusable = await asyncio.wait_for(
                connection.post_json(self.path, payload, streaming),
                timeout=timeout_seconds,
            )
            return response
        finally:
            if reusable:
                async with self._lock:
                    self._available.append(connection)
            else:
                await connection.close()
            self._semaphore.release()


async def execute_workloads(
    base_url: str,
    configuration: dict[str, Any],
    prompt_bank: dict[str, list[Prompt]],
    on_trial: Callable[[dict[str, Any]], None] | None = None,
    before_trial: (
        Callable[[dict[str, Any], int, list[Prompt]], Awaitable[dict[str, Any]]] | None
    ) = None,
    after_trial: (
        Callable[[dict[str, Any], dict[str, Any]], Awaitable[None]] | None
    ) = None,
    initial_request_order: int = 0,
) -> list[dict[str, Any]]:
    """Execute every selected workload with one persistent connection pool.

    Args:
        base_url: Ready server base URL.
        configuration: Fully selected benchmark configuration.
        prompt_bank: Generated prompts keyed by group name.
        on_trial: Optional synchronous callback after each completed trial.

    Returns:
        Completed trial records in deterministic workload/trial order.
    """
    maximum_connections = max(
        int(workload["concurrency"]) for workload in configuration["workloads"]
    )
    pool = _AsyncHttpPool(base_url, maximum_connections)
    request_order = initial_request_order
    trials: list[dict[str, Any]] = []
    try:
        for workload in configuration["workloads"]:
            for trial_index in range(int(workload["trials"])):
                context = (
                    await before_trial(
                        workload,
                        trial_index,
                        prompt_bank[workload["prompt_group"]],
                    )
                    if before_trial is not None
                    else {}
                )
                trial, request_order = await _execute_trial(
                    pool,
                    configuration,
                    workload,
                    prompt_bank[workload["prompt_group"]],
                    trial_index,
                    request_order,
                )
                if after_trial is not None:
                    await after_trial(trial, context)
                trials.append(trial)
                if on_trial is not None:
                    on_trial(trial)
    finally:
        await pool.close()
    return trials


async def execute_setup_prompts(
    base_url: str,
    configuration: dict[str, Any],
    prompts: list[Prompt],
    *,
    output_tokens: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    """Execute non-measured warmup or prefix-priming prompts."""

    if not prompts:
        return []
    pool = _AsyncHttpPool(base_url, max(1, concurrency))
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def execute(prompt: Prompt) -> dict[str, Any]:
        async with semaphore:
            model = configuration["models"][0]
            sampling = configuration["sampling"]
            payload = {
                "model": model["checkpoint"],
                "messages": [{"role": "user", "content": prompt.text}],
                "max_tokens": output_tokens,
                "temperature": sampling["temperature"],
                "top_p": sampling["top_p"],
                "stream": False,
            }
            response = await pool.post_json(
                payload,
                False,
                int(sampling["request_timeout_seconds"]),
            )
            return {
                "prompt_name": prompt.name,
                "prompt_sha256": prompt.sha256,
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
                "cached_tokens": response["cached_tokens"],
            }

    try:
        return list(await asyncio.gather(*(execute(prompt) for prompt in prompts)))
    finally:
        await pool.close()


async def _execute_trial(
    pool: _AsyncHttpPool,
    configuration: dict[str, Any],
    workload: dict[str, Any],
    prompts: list[Prompt],
    trial_index: int,
    request_order: int,
) -> tuple[dict[str, Any], int]:
    request_count = int(workload["requests_per_trial"])
    state = _TrialState(workload, trial_index)

    def submit(request_index: int) -> asyncio.Task[dict[str, Any]]:
        nonlocal request_order
        prompt_index = (trial_index * request_count + request_index) % len(prompts)
        prompt = prompts[prompt_index]
        submission_ns, gap_ns, in_flight = state.submitted(request_index)
        task = asyncio.create_task(
            _execute_request(
                pool,
                configuration,
                workload,
                prompt,
                trial_index,
                request_index,
                request_order,
                submission_ns,
                gap_ns,
                in_flight,
                state,
            )
        )
        request_order += 1
        return task

    mode = workload["load_mode"]
    if mode == "sequential":
        requests = [await submit(index) for index in range(request_count)]
        policy = "submit-one-after-previous-completes"
    elif mode == "burst":
        tasks = [submit(index) for index in range(request_count)]
        requests = list(await asyncio.gather(*tasks))
        policy = "submit-declared-count-at-once"
    elif mode == "closed-loop":
        requests = []
        next_index = 0
        active: set[asyncio.Task[dict[str, Any]]] = set()
        while next_index < min(int(workload["concurrency"]), request_count):
            active.add(submit(next_index))
            next_index += 1
        while active:
            completed, active = await asyncio.wait(
                active, return_when=asyncio.FIRST_COMPLETED
            )
            completed_results = [task.result() for task in completed]
            completed_results.sort(key=lambda result: result["completed_monotonic_ns"])
            for result in completed_results:
                requests.append(result)
                if next_index < request_count:
                    active.add(submit(next_index))
                    next_index += 1
        policy = "replace-each-completed-request"
    else:  # Configuration validation makes this unreachable.
        raise AssertionError(f"unsupported load mode: {mode}")

    requests.sort(key=lambda request: request["request_index"])
    window_start = min(request["submitted_monotonic_ns"] for request in requests)
    window_end = max(request["completed_monotonic_ns"] for request in requests)
    elapsed_seconds = max(1, window_end - window_start) / 1_000_000_000
    completion_tokens = sum(request["completion_tokens"] for request in requests)
    return (
        {
            "workload_name": workload["name"],
            "trial_index": trial_index,
            "load_mode": mode,
            "submission_policy": policy,
            "streaming": workload["streaming"],
            "configured_concurrency": workload["concurrency"],
            "request_count": request_count,
            "started_monotonic_ns": window_start,
            "completed_monotonic_ns": window_end,
            "declared_window_ns": window_end - window_start,
            "aggregate_output_tokens_per_second": completion_tokens / elapsed_seconds,
            "maximum_observed_in_flight": state.maximum_in_flight,
            "in_flight_observations": state.observations,
            "submission_gaps_ns": [
                request["submission_gap_ns"] for request in requests
            ],
            "success_count": sum(
                request["status"] == "succeeded" for request in requests
            ),
            "error_count": sum(request["status"] == "failed" for request in requests),
            "requests": requests,
        },
        request_order,
    )


class _TrialState:
    def __init__(self, workload: dict[str, Any], trial_index: int) -> None:
        self.workload_name = str(workload["name"])
        self.trial_index = trial_index
        self.in_flight = 0
        self.maximum_in_flight = 0
        self.last_submission_ns: int | None = None
        self.observations: list[dict[str, Any]] = []

    def submitted(self, request_index: int) -> tuple[int, int | None, int]:
        now = time.monotonic_ns()
        gap = None if self.last_submission_ns is None else now - self.last_submission_ns
        self.last_submission_ns = now
        self.in_flight += 1
        self.maximum_in_flight = max(self.maximum_in_flight, self.in_flight)
        self.observations.append(
            {
                "event": "submitted",
                "request_index": request_index,
                "monotonic_ns": now,
                "in_flight": self.in_flight,
            }
        )
        return now, gap, self.in_flight

    def completed(self, request_index: int, now: int) -> None:
        self.in_flight -= 1
        self.observations.append(
            {
                "event": "completed",
                "request_index": request_index,
                "monotonic_ns": now,
                "in_flight": self.in_flight,
            }
        )


async def _execute_request(
    pool: _AsyncHttpPool,
    configuration: dict[str, Any],
    workload: dict[str, Any],
    prompt: Prompt,
    trial_index: int,
    request_index: int,
    request_order: int,
    submission_ns: int,
    submission_gap_ns: int | None,
    in_flight_at_submission: int,
    state: _TrialState,
) -> dict[str, Any]:
    model = configuration["models"][0]
    sampling = configuration["sampling"]
    record: dict[str, Any] = {
        "workload_name": workload["name"],
        "trial_index": trial_index,
        "request_index": request_index,
        "prompt_group": prompt.group,
        "prompt_name": prompt.name,
        "prompt_index": prompt.index,
        "prompt_target_tokens": prompt.target_tokens,
        "prompt_sha256": prompt.sha256,
        "request_order": request_order,
        "streaming": workload["streaming"],
        "submitted_monotonic_ns": submission_ns,
        "submission_gap_ns": submission_gap_ns,
        "in_flight_at_submission": in_flight_at_submission,
        "first_byte_monotonic_ns": None,
        "first_token_monotonic_ns": None,
        "final_token_monotonic_ns": None,
        "completed_monotonic_ns": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_tokens": 0,
        "output_sha256": None,
        "finish_reason": None,
        "status": "running",
        "error": None,
    }
    payload = {
        "model": model["checkpoint"],
        "messages": [{"role": "user", "content": prompt.text}],
        "max_tokens": workload["output_tokens"],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "stream": workload["streaming"],
    }
    if workload["streaming"]:
        payload["stream_options"] = {"include_usage": True}
    try:
        response = await pool.post_json(
            payload,
            bool(workload["streaming"]),
            int(sampling["request_timeout_seconds"]),
        )
        record.update(response)
        if (
            record["prompt_tokens"] + record["completion_tokens"]
            != record["total_tokens"]
        ):
            raise LoadGenerationError("response usage token total is inconsistent")
        if record["completion_tokens"] <= 0:
            raise LoadGenerationError("response contains no completion tokens")
        record["status"] = "succeeded"
    except Exception as error:
        now = time.monotonic_ns()
        record["first_byte_monotonic_ns"] = record["first_byte_monotonic_ns"] or now
        record["first_token_monotonic_ns"] = record["first_token_monotonic_ns"] or now
        record["final_token_monotonic_ns"] = record["final_token_monotonic_ns"] or now
        record["completed_monotonic_ns"] = now
        record["output_sha256"] = hashlib.sha256(b"").hexdigest()
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
    state.completed(request_index, int(record["completed_monotonic_ns"]))
    return record


async def _read_stream_response(
    status: int,
    headers: dict[str, str],
    reader: asyncio.StreamReader,
    first_byte_ns: int,
) -> dict[str, Any]:
    output: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    first_token_ns: int | None = None
    final_token_ns: int | None = None
    error_body = bytearray()
    buffer = bytearray()
    async for chunk in _body_chunks(headers, reader):
        if status != 200:
            error_body.extend(chunk)
            continue
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw_line, _, remainder = buffer.partition(b"\n")
            buffer = bytearray(remainder)
            line = raw_line.rstrip(b"\r")
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if content:
                    now = time.monotonic_ns()
                    first_token_ns = first_token_ns or now
                    final_token_ns = now
                    output.append(str(content))
                if choices[0].get("finish_reason") is not None:
                    finish_reason = str(choices[0]["finish_reason"])
            if event.get("usage") is not None:
                usage = event["usage"]
    if status != 200:
        raise LoadGenerationError(
            f"HTTP {status}: {error_body.decode(errors='replace')}"
        )
    if first_token_ns is None or final_token_ns is None:
        raise LoadGenerationError("stream completed without a generated token")
    if usage is None:
        raise LoadGenerationError("stream completed without usage")
    completed_ns = time.monotonic_ns()
    return _response_record(
        first_byte_ns,
        first_token_ns,
        final_token_ns,
        completed_ns,
        output,
        finish_reason,
        usage,
    )


async def _read_json_response(
    status: int,
    headers: dict[str, str],
    reader: asyncio.StreamReader,
    first_byte_ns: int,
) -> dict[str, Any]:
    body = b"".join([chunk async for chunk in _body_chunks(headers, reader)])
    if status != 200:
        raise LoadGenerationError(f"HTTP {status}: {body.decode(errors='replace')}")
    payload = json.loads(body)
    choices = payload.get("choices") or []
    if not choices:
        raise LoadGenerationError("non-streaming response contains no choice")
    content = (choices[0].get("message") or {}).get("content")
    if content is None:
        raise LoadGenerationError("non-streaming response contains no content")
    usage = payload.get("usage")
    if usage is None:
        raise LoadGenerationError("non-streaming response contains no usage")
    completed_ns = time.monotonic_ns()
    return _response_record(
        first_byte_ns,
        completed_ns,
        completed_ns,
        completed_ns,
        [str(content)],
        choices[0].get("finish_reason"),
        usage,
    )


def _response_record(
    first_byte_ns: int,
    first_token_ns: int,
    final_token_ns: int,
    completed_ns: int,
    output: list[str],
    finish_reason: Any,
    usage: dict[str, Any],
) -> dict[str, Any]:
    text = "".join(output)
    return {
        "first_byte_monotonic_ns": first_byte_ns,
        "first_token_monotonic_ns": first_token_ns,
        "final_token_monotonic_ns": final_token_ns,
        "completed_monotonic_ns": completed_ns,
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "cached_tokens": int(
            (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        ),
        "output_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "finish_reason": finish_reason,
    }


async def _body_chunks(
    headers: dict[str, str], reader: asyncio.StreamReader
) -> AsyncIterator[bytes]:
    if "chunked" in headers.get("transfer-encoding", "").lower():
        while True:
            size_line = await reader.readline()
            if not size_line:
                raise ConnectionError("server closed a chunked response")
            try:
                size = int(size_line.split(b";", 1)[0].strip(), 16)
            except ValueError as error:
                raise LoadGenerationError(
                    f"invalid HTTP chunk size: {size_line!r}"
                ) from error
            if size == 0:
                while True:
                    trailer = await reader.readline()
                    if trailer in {b"\r\n", b"\n"}:
                        return
                    if not trailer:
                        raise ConnectionError("server closed chunked trailers")
            data = await reader.readexactly(size)
            ending = await reader.readexactly(2)
            if ending != b"\r\n":
                raise LoadGenerationError("HTTP chunk is missing CRLF")
            yield data
    elif "content-length" in headers:
        remaining = int(headers["content-length"])
        while remaining:
            chunk = await reader.readexactly(min(65_536, remaining))
            remaining -= len(chunk)
            yield chunk
    else:
        while True:
            chunk = await reader.read(65_536)
            if not chunk:
                return
            yield chunk
