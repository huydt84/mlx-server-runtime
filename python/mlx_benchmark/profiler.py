"""Capture, view, and compare native MLX pipeline profiles for ``mlx-air profile``.

``run`` launches its own gateway with pipeline profiling enabled, drives a
bounded set of chat requests through the public HTTP API, stops the gateway,
and writes the joined timeline, a stage report, and a self-contained HTML
view. ``view`` rebuilds the HTML for an existing profile and ``compare``
reports per-stage deltas between two profiles. Profiled timings are diagnostic
evidence and never replace ``mlx-air bench run`` results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mlx_benchmark import profile_analysis as analysis
from mlx_benchmark.runner import (
    _interrupts_raise,
    _render_runtime_toml,
    _reserve_loopback_port,
    _RunFailure,
    _RunInterrupted,
    _ServerProcess,
    _wait_for_readiness,
)

_GATEWAY_EXECUTABLE_ENV = "MLX_AIR_GATEWAY_EXECUTABLE"
_INVOCATION_DIRECTORY_ENV = "MLX_AIR_INVOCATION_DIRECTORY"
_PROFILE_FAILURE = 50
_DEFAULT_PROMPT = (
    "Explain in three short paragraphs how a CPU cache hierarchy reduces "
    "memory latency."
)


class _RequestFailure(RuntimeError):
    pass


def _bounded_int(low: int, high: int) -> Any:
    def parse(value: str) -> int:
        parsed = int(value)
        if not low <= parsed <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return parsed

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-air profile",
        description=(
            "Capture and visualize native MLX inference-pipeline profiles. "
            "Profiled timings are diagnostic and are not benchmark results."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    run = actions.add_parser(
        "run",
        help="serve a model with profiling enabled and profile bounded requests",
    )
    run.add_argument("--model", required=True, help="checkpoint or local model path")
    run.add_argument(
        "--prompt", default=_DEFAULT_PROMPT, help="user message sent with each request"
    )
    run.add_argument(
        "--max-tokens",
        type=_bounded_int(1, 8192),
        default=64,
        metavar="N",
        help="completion tokens per request (default: 64)",
    )
    run.add_argument(
        "--requests",
        type=_bounded_int(1, 1000),
        default=1,
        metavar="N",
        help="profiled requests to send (default: 1)",
    )
    run.add_argument(
        "--concurrency",
        type=_bounded_int(1, 64),
        default=1,
        metavar="N",
        help="requests in flight at once (default: 1)",
    )
    run.add_argument(
        "--stream", action="store_true", help="use streaming chat completions"
    )
    run.add_argument(
        "--metal",
        action="store_true",
        help="also capture pipeline.gputrace for the first request "
        "(requires MTL_CAPTURE_ENABLED=1)",
    )
    run.add_argument(
        "--output-dir",
        metavar="PATH",
        help="new or empty artifact directory (default: ./profiles/<run id>)",
    )
    run.add_argument(
        "--readiness-timeout",
        type=_bounded_int(1, 7200),
        default=900,
        metavar="SECONDS",
        help="seconds to wait for the server to become ready (default: 900)",
    )
    run.add_argument(
        "--request-timeout",
        type=_bounded_int(1, 7200),
        default=600,
        metavar="SECONDS",
        help="seconds allowed for each request (default: 600)",
    )
    run.add_argument(
        "--open", action="store_true", help="open profile.html when finished"
    )

    view = actions.add_parser(
        "view", help="build the HTML view for an existing profile"
    )
    view.add_argument("profile", metavar="PATH", help="profile directory or JSONL file")
    view.add_argument(
        "--output", metavar="PATH", help="HTML path (default: <profile>/profile.html)"
    )
    view.add_argument(
        "--include-warmup",
        action="store_true",
        help="include worker warmup requests",
    )
    view.add_argument("--open", action="store_true", help="open the HTML view")

    compare = actions.add_parser(
        "compare", help="compare per-stage time between two profiles"
    )
    compare.add_argument("baseline", metavar="BASELINE", help="baseline profile")
    compare.add_argument("candidate", metavar="CANDIDATE", help="candidate profile")
    compare.add_argument(
        "--output", metavar="PATH", help="also write the comparison as Markdown"
    )
    compare.add_argument(
        "--include-warmup",
        action="store_true",
        help="include worker warmup requests",
    )
    return parser


def _invocation_path(value: str) -> Path:
    invocation = Path(os.environ.get(_INVOCATION_DIRECTORY_ENV, os.getcwd()))
    requested = Path(value).expanduser()
    return requested if requested.is_absolute() else invocation / requested


def _validate_gateway(parser: argparse.ArgumentParser) -> Path:
    value = os.environ.get(_GATEWAY_EXECUTABLE_ENV)
    if value is None:
        parser.error(f"{_GATEWAY_EXECUTABLE_ENV} is not set")
    gateway = Path(value)
    if not gateway.is_absolute():
        parser.error(f"{_GATEWAY_EXECUTABLE_ENV} must be an absolute path")
    if not gateway.is_file():
        parser.error(f"gateway executable does not exist: {gateway}")
    return gateway


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and run one ``mlx-air profile`` action.

    Args:
        argv: Optional leaf command arguments. Process arguments are used when
            omitted.

    Returns:
        The command exit status: 0 on success, 50 on profile failure, or 128
        plus a signal number when interrupted.
    """
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.action == "run":
            if arguments.concurrency > arguments.requests:
                parser.error("--concurrency must not exceed --requests")
            return run_profile(arguments, _validate_gateway(parser))
        if arguments.action == "view":
            return view_profile(arguments)
        return compare_profiles(arguments)
    except analysis.ProfileError as error:
        print(f"error: {error}", file=sys.stderr)
        return _PROFILE_FAILURE


def view_profile(arguments: argparse.Namespace) -> int:
    """Write the HTML view for an existing profile.

    Args:
        arguments: Parsed ``view`` arguments.

    Returns:
        Zero on success.
    """
    source = _invocation_path(arguments.profile)
    events = analysis.select_events(
        analysis.load_events(source), include_warmup=arguments.include_warmup
    )
    if not events:
        raise analysis.ProfileError(f"{source} contains no profiled requests")
    directory = source if source.is_dir() else source.parent
    output = (
        _invocation_path(arguments.output)
        if arguments.output
        else directory / analysis.HTML_FILE
    )
    summary = analysis.summarize(events)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(analysis.render_html(summary, events), encoding="utf-8")
    _print_summary(summary)
    print(output)
    if arguments.open:
        _open(output)
    return 0


def compare_profiles(arguments: argparse.Namespace) -> int:
    """Print a per-stage comparison of two profiles.

    Args:
        arguments: Parsed ``compare`` arguments.

    Returns:
        Zero on success.
    """
    summaries = []
    for value in (arguments.baseline, arguments.candidate):
        path = _invocation_path(value)
        events = analysis.select_events(
            analysis.load_events(path), include_warmup=arguments.include_warmup
        )
        if not events:
            raise analysis.ProfileError(f"{path} contains no profiled requests")
        summaries.append(analysis.summarize(events))
    report = analysis.compare(
        summaries[0],
        summaries[1],
        baseline_label=arguments.baseline,
        candidate_label=arguments.candidate,
    )
    if arguments.output:
        output = _invocation_path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


def run_profile(arguments: argparse.Namespace, gateway: Path) -> int:
    """Serve a model with profiling enabled and profile bounded requests.

    Args:
        arguments: Parsed ``run`` arguments.
        gateway: Absolute path to the version-matched gateway executable.

    Returns:
        Zero when every request succeeded and artifacts were written, 50 on
        failure, or 128 plus a signal number when interrupted.
    """
    if arguments.metal and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        print(
            "error: --metal requires MTL_CAPTURE_ENABLED=1 in the environment",
            file=sys.stderr,
        )
        return _PROFILE_FAILURE
    run_id = f"profile-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    output = _invocation_path(arguments.output_dir or f"profiles/{run_id}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f"error: output directory is not empty: {output}", file=sys.stderr)
        return _PROFILE_FAILURE
    output.mkdir(parents=True, exist_ok=True)
    print(f"Profiling {arguments.model} into {output}", file=sys.stderr)

    records: list[dict[str, Any]] = []
    server: _ServerProcess | None = None
    try:
        with _interrupts_raise():
            base_url, server = _start_profiled_server(
                gateway, output, arguments, run_id
            )
            print("Waiting for the server to become ready", file=sys.stderr)
            _wait_for_readiness(
                base_url,
                server,
                {
                    "sampling": {
                        "readiness_timeout_seconds": arguments.readiness_timeout
                    }
                },
            )
            print(
                f"Sending {arguments.requests} request(s) at concurrency "
                f"{arguments.concurrency}",
                file=sys.stderr,
            )
            records = _drive_requests(base_url, arguments)
    except _RunInterrupted as interrupted:
        print(f"error: {interrupted}", file=sys.stderr)
        return 128 + interrupted.signum
    except _RunFailure as error:
        print(f"error: {error}; see {output / 'gateway.log'}", file=sys.stderr)
        return _PROFILE_FAILURE
    finally:
        if server is not None:
            server.stop()

    try:
        worker_events = analysis.load_events(output)
    except analysis.ProfileError as error:
        print(f"error: worker wrote no profile: {error}", file=sys.stderr)
        return _PROFILE_FAILURE
    events = worker_events + [
        _gateway_event(record, run_id, arguments) for record in records if record["id"]
    ]
    events.sort(key=lambda event: int(event["monotonic_ns"]))
    analysis.write_events(events, output / analysis.EVENTS_FILE)
    analysis.write_trace(events, output / analysis.TRACE_FILE)
    selected = analysis.select_events(events)
    summary, html_path = analysis.write_profile_outputs(selected, output)
    _print_summary(summary)
    failures = [record for record in records if record["error"]]
    for record in failures:
        print(
            f"error: request {record['index']} failed: {record['error']}",
            file=sys.stderr,
        )
    print(html_path)
    if arguments.open:
        _open(html_path)
    return _PROFILE_FAILURE if failures or not selected else 0


def _start_profiled_server(
    gateway: Path, output: Path, arguments: argparse.Namespace, run_id: str
) -> tuple[str, _ServerProcess]:
    port = _reserve_loopback_port()
    socket_directory = Path("/tmp") / f"mlx-air-{os.getuid()}"
    socket_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    suffix = hashlib.sha256(str(output).encode()).hexdigest()[:12]
    ipc_path = socket_directory / f"profile-{os.getpid()}-{suffix}.sock"
    prompt_limit = 32_768
    config_path = output / "runtime.toml"
    config_path.write_text(
        _render_runtime_toml(
            {
                "server": {"host": "127.0.0.1", "port": port},
                "worker": {
                    "python": sys.executable,
                    "module": "mlx_worker.main",
                    "backend": "native-mlx",
                    "model": arguments.model,
                    "ipc_path": str(ipc_path),
                },
                "generation": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": arguments.max_tokens,
                },
                "limits": {
                    "max_pending_requests": max(arguments.requests, 8),
                    "max_active_requests": arguments.concurrency,
                    "max_prompt_tokens": prompt_limit,
                    "max_completion_tokens": arguments.max_tokens,
                    "max_total_tokens_per_request": prompt_limit + arguments.max_tokens,
                    "request_timeout_seconds": arguments.request_timeout,
                },
                "telemetry": {"enable_prometheus": True, "metrics_path": "/metrics"},
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(
        {
            "MLX_RUNTIME_CONFIG": str(config_path),
            "MLX_AIR_WORKER_LOG": str(output / "worker.log"),
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE": "1",
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR": str(output),
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID": run_id,
            "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD": "mlx-air-profile",
            "MLX_RUNTIME_NATIVE_GRAPH_PROFILE": "0",
            "MLX_RUNTIME_NATIVE_METAL_CAPTURE": "1" if arguments.metal else "0",
        }
    )
    log_handle = (output / "gateway.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [str(gateway)],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
    except OSError as error:
        log_handle.close()
        raise _RunFailure(
            "server_startup", f"failed to launch {gateway}: {error}"
        ) from error
    return f"http://127.0.0.1:{port}", _ServerProcess(
        process, log_handle, (ipc_path,), (socket_directory,), {}
    )


def _drive_requests(
    base_url: str, arguments: argparse.Namespace
) -> list[dict[str, Any]]:
    stop = threading.Event()

    def send(index: int) -> dict[str, Any]:
        if stop.is_set():
            return _request_record(index, error="not sent after an interrupt")
        return _send_request(base_url, index, arguments)

    pool = ThreadPoolExecutor(max_workers=arguments.concurrency)
    try:
        futures = [pool.submit(send, index) for index in range(arguments.requests)]
        records = []
        for future in futures:
            record = future.result()
            records.append(record)
            status = "failed" if record["error"] else "ok"
            print(
                f"  request {record['index'] + 1}/{arguments.requests}: {status} "
                f"({record['duration_us'] / 1000:.1f} ms)",
                file=sys.stderr,
            )
        return records
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)


def _request_record(index: int, **values: Any) -> dict[str, Any]:
    record = {
        "index": index,
        "id": None,
        "started_ns": 0,
        "duration_us": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "error": None,
    }
    record.update(values)
    return record


def _send_request(
    base_url: str, index: int, arguments: argparse.Namespace
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": arguments.model,
        "messages": [{"role": "user", "content": arguments.prompt}],
        "max_tokens": arguments.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": arguments.stream,
    }
    if arguments.stream:
        payload["stream_options"] = {"include_usage": True}
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if arguments.stream else "application/json",
        },
    )
    # Same clock as the worker's perf_counter_ns event timestamps.
    started_ns = time.perf_counter_ns()
    record = _request_record(index, started_ns=started_ns)
    try:
        with urlopen(request, timeout=arguments.request_timeout) as response:
            if arguments.stream:
                response_id, usage = _read_stream(response)
            else:
                body = json.loads(response.read())
                response_id, usage = body.get("id"), body.get("usage") or {}
        if not response_id:
            raise _RequestFailure("response has no id")
        record["id"] = str(response_id).removeprefix("chatcmpl-")
        record["prompt_tokens"] = usage.get("prompt_tokens")
        record["completion_tokens"] = usage.get("completion_tokens")
    except HTTPError as error:
        body = error.read().decode(errors="replace")[:500]
        record["error"] = f"HTTP {error.code}: {body}"
    except (URLError, OSError, ValueError, _RequestFailure) as error:
        record["error"] = f"{type(error).__name__}: {error}"
    record["duration_us"] = (time.perf_counter_ns() - started_ns) // 1_000
    return record


def _read_stream(response: Any) -> tuple[str | None, dict[str, Any]]:
    response_id: str | None = None
    usage: dict[str, Any] = {}
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        chunk = json.loads(data)
        response_id = response_id or chunk.get("id")
        if chunk.get("usage"):
            usage = chunk["usage"]
    return response_id, usage


def _gateway_event(
    record: dict[str, Any], run_id: str, arguments: argparse.Namespace
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "request_id": record["id"],
        "backend": "native-mlx",
        "model": arguments.model,
        "workload": "mlx-air-profile",
        "component": "gateway",
        "stage": "http_round_trip",
        "monotonic_ns": record["started_ns"],
        "offset_us": 0,
        "duration_us": record["duration_us"],
        "phase": None,
        "prompt_tokens": record["prompt_tokens"],
        "completion_tokens": record["completion_tokens"],
        "state": "failed" if record["error"] else "completed",
        "error": record["error"],
        "details": {"stream": arguments.stream, "client": "mlx-air profile"},
    }


def _print_summary(summary: dict[str, Any]) -> None:
    totals = summary["totals"]

    def ms(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1f} ms"

    print(
        f"Profiled {totals['request_count']} request(s); "
        f"mean request latency {ms(totals['mean_latency_ms'])}, "
        f"mean TTFT {ms(totals['mean_ttft_ms'])} (lower is better; "
        "diagnostic only)",
        file=sys.stderr,
    )
    if totals["metal_captured_requests"]:
        handling = (
            "excluded from means"
            if totals["metal_capture_excluded"]
            else "included in means: no uncaptured request to measure"
        )
        print(
            f"  {totals['metal_captured_requests']} request(s) ran under Metal "
            f"capture ({handling})",
            file=sys.stderr,
        )
    for row in summary["stages"][:5]:
        print(
            f"  {row['component']}/{row['stage']}: "
            f"{row['mean_ms_per_request']:.2f} ms per request",
            file=sys.stderr,
        )


def _open(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", str(path)], check=False)
    else:
        webbrowser.open(path.as_uri())


if __name__ == "__main__":
    raise SystemExit(main())
