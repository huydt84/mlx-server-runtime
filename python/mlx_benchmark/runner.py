"""Execute the bounded MLX Air benchmark and persist self-contained results."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Any, Iterator
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

_MLX_AIR_VERSION_ENV = "MLX_AIR_VERSION"
_INVOCATION_DIRECTORY_ENV = "MLX_AIR_INVOCATION_DIRECTORY"
_WORKER_LOG_ENV = "MLX_AIR_WORKER_LOG"
_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
_WORKLOAD = "bounded_stream_smoke"
_PROMPTS = (
    "Reply with the single word amber.",
    "Reply with the single word cedar.",
    "Reply with the single word indigo.",
    "Reply with the single word quartz.",
    "Reply with the single word silver.",
    "Reply with the single word violet.",
)
_TRIALS = (
    {"trial_index": 0, "load_mode": "sequential", "request_count": 2},
    {"trial_index": 1, "load_mode": "concurrent", "request_count": 4},
)
_REQUEST_TIMEOUT_SECONDS = 300
_READINESS_TIMEOUT_SECONDS = 1800
_SHUTDOWN_TIMEOUT_SECONDS = 15
_BENCHMARK_EXECUTION_FAILURE = 50
_REQUIRED_COUNTERS = (
    "mlx_requests_total",
    "mlx_prompt_tokens_total",
    "mlx_completion_tokens_total",
)


class _RunFailure(RuntimeError):
    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


class _RunInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


class _ServerProcess:
    def __init__(
        self,
        process: subprocess.Popen[bytes],
        log_handle: Any,
        cleanup_paths: tuple[Path, ...],
    ) -> None:
        self.process = process
        self.log_handle = log_handle
        self.cleanup_paths = cleanup_paths

    def stop(self) -> None:
        """Stop and reap the server process group."""
        try:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
            while (
                _process_group_exists(self.process.pid) and time.monotonic() < deadline
            ):
                self.process.poll()
                time.sleep(0.05)
            if _process_group_exists(self.process.pid):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.process.wait(timeout=5)
        finally:
            self.log_handle.close()
            for path in self.cleanup_paths:
                path.unlink(missing_ok=True)


def run_benchmark(arguments: argparse.Namespace, gateway: Path) -> int:
    """Run the built-in benchmark selected by parsed CLI arguments.

    Args:
        arguments: Validated ``mlx-air bench run`` arguments.
        gateway: Absolute path to the version-matched gateway executable.

    Returns:
        Zero on success, 50 on benchmark failure, or 128 plus the interrupt
        signal number.
    """
    run_directory = _resolve_run_directory(arguments.output_dir)
    _create_artifact_tree(run_directory)
    results = _initial_results(arguments, gateway, run_directory)
    _write_artifacts(run_directory, results)
    server: _ServerProcess | None = None

    try:
        with _interrupts_raise():
            if arguments.server_mode == "self-launched":
                results["failure_stage"] = "server_configuration"
                base_url, server = _start_server(gateway, run_directory, results)
            else:
                base_url = _normalize_base_url(arguments.base_url)

            results["server"]["base_url"] = base_url
            results["failure_stage"] = "readiness"
            readiness = _wait_for_readiness(base_url, server)
            _record_server_identity(results, base_url, readiness, gateway, arguments)
            results["failure_stage"] = "measurement"

            for trial_definition in _TRIALS:
                trial = _execute_trial(base_url, trial_definition)
                results["trials"].append(trial)
                _validate_completed_trial(results, trial, trial_definition)
                _write_artifacts(run_directory, results)

            results["runtime_counters"] = _fetch_runtime_counters(base_url)
            _validate_final_results(results)
            if any(trial["error_count"] for trial in results["trials"]):
                raise _RunFailure("measurement", "one or more requests failed")
            results["status"] = "succeeded"
            results["failure_stage"] = None
            results["completed_at"] = _utc_now()
            _write_artifacts(run_directory, results)
            print(run_directory / "results.json")
            return 0
    except _RunInterrupted as interrupted:
        results["status"] = "interrupted"
        results["error"] = {
            "kind": "signal",
            "message": str(interrupted),
            "signal": interrupted.signum,
        }
        results["completed_at"] = _utc_now()
        _write_artifacts(run_directory, results)
        return 128 + interrupted.signum
    except _RunFailure as error:
        results["status"] = "failed"
        results["failure_stage"] = error.stage
        results["error"] = {"kind": "benchmark", "message": str(error)}
        results["completed_at"] = _utc_now()
        _write_artifacts(run_directory, results)
        print(f"error: {error}", file=sys.stderr)
        return _BENCHMARK_EXECUTION_FAILURE
    except Exception as error:
        results["status"] = "failed"
        results["error"] = {
            "kind": "unexpected",
            "message": f"{type(error).__name__}: {error}",
        }
        results["completed_at"] = _utc_now()
        _write_artifacts(run_directory, results)
        print(f"error: {error}", file=sys.stderr)
        return _BENCHMARK_EXECUTION_FAILURE
    finally:
        if server is not None:
            server.stop()


def _resolve_run_directory(output_dir: str | None) -> Path:
    invocation = Path(os.environ.get(_INVOCATION_DIRECTORY_ENV, os.getcwd()))
    if output_dir is not None:
        requested = Path(output_dir).expanduser()
        return requested if requested.is_absolute() else invocation / requested
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return invocation / "artifacts" / "benchmark" / f"{timestamp}-{os.getpid()}"


def _create_artifact_tree(run_directory: Path) -> None:
    try:
        (run_directory / "logs").mkdir(parents=True, exist_ok=False)
        (run_directory / "logs" / "gateway.log").touch()
        (run_directory / "logs" / "worker.log").touch()
    except OSError as error:
        raise _RunFailure(
            "artifact_setup",
            f"failed to create artifact directory {run_directory}: {error}",
        ) from error


def _initial_results(
    arguments: argparse.Namespace, gateway: Path, run_directory: Path
) -> dict[str, Any]:
    final_configuration = {
        "suite": arguments.suite,
        "focus": arguments.focus,
        "benchmark_config": arguments.benchmark_config,
        "profile": arguments.profile,
        "server_mode": arguments.server_mode,
        "base_url": arguments.base_url,
        "output_directory": str(run_directory),
        "model": _MODEL,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 8,
        "readiness_timeout_seconds": _READINESS_TIMEOUT_SECONDS,
        "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
        "workloads": [
            {
                "name": _WORKLOAD,
                "streaming": True,
                "trials": [dict(trial) for trial in _TRIALS],
            }
        ],
    }
    return {
        "schema_version": 1,
        "run_id": run_directory.name,
        "status": "running",
        "failure_stage": "artifact_setup",
        "error": None,
        "started_at": _utc_now(),
        "completed_at": None,
        "configuration": final_configuration,
        "versions": {
            "mlx_air": os.environ.get(_MLX_AIR_VERSION_ENV, "not_exposed"),
            "gateway": "not_exposed",
            "model": {"name": _MODEL, "revision": "not_exposed"},
            "tokenizer": {"name": _MODEL, "revision": "not_exposed"},
        },
        "host": _host_information(),
        "server": {
            "mode": arguments.server_mode,
            "base_url": arguments.base_url,
            "gateway_executable": str(gateway)
            if arguments.server_mode == "self-launched"
            else None,
            "runtime_configuration": None,
        },
        "runtime_counters": {},
        "trials": [],
        "validation_failures": [],
    }


def _host_information() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor()
        or _command_output(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "macos_version": platform.mac_ver()[0] or "not_macos",
        "power_state": _command_output(["pmset", "-g", "batt"]),
    }


def _command_output(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    output = completed.stdout.strip() or completed.stderr.strip()
    return output if completed.returncode == 0 and output else "unavailable"


def _start_server(
    gateway: Path, run_directory: Path, results: dict[str, Any]
) -> tuple[str, _ServerProcess]:
    port = _reserve_loopback_port()
    socket_directory = Path("/tmp") / f"mlx-air-{os.getuid()}"
    socket_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    socket_suffix = hashlib.sha256(str(run_directory).encode()).hexdigest()[:12]
    ipc_path = socket_directory / f"benchmark-{os.getpid()}-{socket_suffix}.sock"
    runtime_config = {
        "server": {"host": "127.0.0.1", "port": port},
        "worker": {
            "python": sys.executable,
            "module": "mlx_worker.main",
            "backend": "v1",
            "model": _MODEL,
            "ipc_path": str(ipc_path),
        },
        "generation": {"temperature": 0.0, "top_p": 1.0, "max_tokens": 8},
        "limits": {
            "max_pending_requests": 8,
            "max_active_requests": 4,
            "max_prompt_tokens": 1024,
            "max_completion_tokens": 8,
            "max_total_tokens_per_request": 1032,
            "request_timeout_seconds": _REQUEST_TIMEOUT_SECONDS,
        },
        "telemetry": {"enable_prometheus": True, "metrics_path": "/metrics"},
    }
    config_path = run_directory / "runtime.toml"
    config_path.write_text(_render_runtime_toml(runtime_config), encoding="utf-8")
    results["server"]["runtime_configuration"] = runtime_config
    results["configuration"]["base_url"] = f"http://127.0.0.1:{port}"
    results["failure_stage"] = "server_startup"
    _write_artifacts(run_directory, results)

    log_handle = (run_directory / "logs" / "gateway.log").open("ab", buffering=0)
    environment = os.environ.copy()
    environment["MLX_RUNTIME_CONFIG"] = str(config_path)
    environment[_WORKER_LOG_ENV] = str(run_directory / "logs" / "worker.log")
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
    return f"http://127.0.0.1:{port}", _ServerProcess(process, log_handle, (ipc_path,))


def _reserve_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _render_runtime_toml(configuration: dict[str, Any]) -> str:
    server = configuration["server"]
    worker = configuration["worker"]
    generation = configuration["generation"]
    limits = configuration["limits"]
    telemetry = configuration["telemetry"]
    return (
        "[server]\n"
        f"host = {json.dumps(server['host'])}\n"
        f"port = {server['port']}\n\n"
        "[worker]\n"
        f"python = {json.dumps(worker['python'])}\n"
        f"module = {json.dumps(worker['module'])}\n"
        f"backend = {json.dumps(worker['backend'])}\n"
        f"model = {json.dumps(worker['model'])}\n"
        f"ipc_path = {json.dumps(worker['ipc_path'])}\n\n"
        "[generation]\n"
        f"temperature = {generation['temperature']:.1f}\n"
        f"top_p = {generation['top_p']:.1f}\n"
        f"max_tokens = {generation['max_tokens']}\n\n"
        "[limits]\n"
        + "".join(f"{key} = {value}\n" for key, value in limits.items())
        + "\n[telemetry]\n"
        f"enable_prometheus = {str(telemetry['enable_prometheus']).lower()}\n"
        f"metrics_path = {json.dumps(telemetry['metrics_path'])}\n"
    )


def _normalize_base_url(value: str | None) -> str:
    if value is None:
        raise _RunFailure("argument_validation", "external server base URL is missing")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise _RunFailure(
            "argument_validation", f"invalid external server URL: {value}"
        )
    return value.rstrip("/")


def _wait_for_readiness(base_url: str, server: _ServerProcess | None) -> dict[str, Any]:
    deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
    last_error = "readiness endpoint did not respond"
    while time.monotonic() < deadline:
        if server is not None:
            status = server.process.poll()
            if status is not None:
                raise _RunFailure(
                    "readiness", f"gateway exited before readiness with status {status}"
                )
        try:
            status, payload = _get_json(f"{base_url}/ready", timeout=2)
            if status == 200 and payload.get("ready") is True:
                return payload
            last_error = f"readiness returned HTTP {status}: {payload}"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.2)
    raise _RunFailure(
        "readiness",
        f"server did not become ready within {_READINESS_TIMEOUT_SECONDS} seconds: {last_error}",
    )


def _record_server_identity(
    results: dict[str, Any],
    base_url: str,
    readiness: dict[str, Any],
    gateway: Path,
    arguments: argparse.Namespace,
) -> None:
    if arguments.server_mode == "self-launched":
        gateway_version = _command_output([str(gateway), "--version"])
    else:
        gateway_version = _optional_gateway_version(base_url)
    model = str(readiness.get("model") or _MODEL)
    revision = str(readiness.get("revision") or "not_exposed")
    results["versions"]["gateway"] = gateway_version
    results["versions"]["model"] = {"name": model, "revision": revision}
    results["versions"]["tokenizer"] = {"name": model, "revision": revision}
    results["server"]["readiness"] = readiness


def _optional_gateway_version(base_url: str) -> str:
    try:
        status, payload = _get_json(f"{base_url}/version", timeout=2)
    except (OSError, ValueError, json.JSONDecodeError):
        return "not_exposed"
    if status != 200:
        return "not_exposed"
    return str(
        payload.get("gateway_version") or payload.get("version") or "not_exposed"
    )


def _get_json(url: str, timeout: int) -> tuple[int, dict[str, Any]]:
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except Exception as error:
        if hasattr(error, "code") and hasattr(error, "read"):
            body = error.read().decode("utf-8", errors="replace")
            try:
                return int(error.code), json.loads(body)
            except json.JSONDecodeError:
                return int(error.code), {"body": body}
        raise


def _execute_trial(base_url: str, definition: dict[str, Any]) -> dict[str, Any]:
    trial_index = int(definition["trial_index"])
    request_count = int(definition["request_count"])
    started = time.monotonic_ns()
    if definition["load_mode"] == "sequential":
        requests = [
            _execute_request(base_url, trial_index, request_index)
            for request_index in range(request_count)
        ]
    else:
        executor = ThreadPoolExecutor(max_workers=request_count)
        try:
            futures = {
                executor.submit(
                    _execute_request, base_url, trial_index, request_index
                ): request_index
                for request_index in range(request_count)
            }
            requests = [future.result() for future in as_completed(futures)]
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown()
        requests.sort(key=lambda request: request["request_index"])
    completed = time.monotonic_ns()
    return {
        "workload_name": _WORKLOAD,
        "trial_index": trial_index,
        "load_mode": definition["load_mode"],
        "request_count": request_count,
        "started_monotonic_ns": started,
        "completed_monotonic_ns": completed,
        "success_count": sum(request["status"] == "succeeded" for request in requests),
        "error_count": sum(request["status"] == "failed" for request in requests),
        "requests": requests,
    }


def _execute_request(
    base_url: str, trial_index: int, request_index: int
) -> dict[str, Any]:
    global_index = (
        sum(int(trial["request_count"]) for trial in _TRIALS[:trial_index])
        + request_index
    )
    prompt_index = global_index % len(_PROMPTS)
    prompt = _PROMPTS[prompt_index]
    submission = time.monotonic_ns()
    record: dict[str, Any] = {
        "workload_name": _WORKLOAD,
        "trial_index": trial_index,
        "request_index": request_index,
        "prompt_group": "bounded_smoke",
        "prompt_name": f"prompt-{prompt_index}",
        "prompt_index": prompt_index,
        "request_order": global_index,
        "submitted_monotonic_ns": submission,
        "first_byte_monotonic_ns": None,
        "first_token_monotonic_ns": None,
        "final_token_monotonic_ns": None,
        "completed_monotonic_ns": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "output_sha256": None,
        "finish_reason": None,
        "status": "running",
        "error": None,
    }
    try:
        _stream_completion(base_url, prompt, record)
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
    return record


def _stream_completion(base_url: str, prompt: str, record: dict[str, Any]) -> None:
    parsed = urlsplit(base_url)
    connection_type = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    connection = connection_type(
        parsed.hostname, port, timeout=_REQUEST_TIMEOUT_SECONDS
    )
    path_prefix = parsed.path.rstrip("/")
    payload = json.dumps(
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    output_parts: list[str] = []
    try:
        connection.request(
            "POST",
            f"{path_prefix}/v1/chat/completions",
            body=payload,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        response = connection.getresponse()
        record["first_byte_monotonic_ns"] = time.monotonic_ns()
        if response.status != 200:
            detail = response.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    now = time.monotonic_ns()
                    if record["first_token_monotonic_ns"] is None:
                        record["first_token_monotonic_ns"] = now
                    record["final_token_monotonic_ns"] = now
                    output_parts.append(str(content))
                finish_reason = choices[0].get("finish_reason")
                if finish_reason is not None:
                    record["finish_reason"] = finish_reason
            usage = event.get("usage")
            if usage is not None:
                record["prompt_tokens"] = int(usage.get("prompt_tokens", 0))
                record["completion_tokens"] = int(usage.get("completion_tokens", 0))
                record["total_tokens"] = int(usage.get("total_tokens", 0))
        if record["first_token_monotonic_ns"] is None:
            raise RuntimeError("stream completed without a generated token")
        record["completed_monotonic_ns"] = time.monotonic_ns()
        record["output_sha256"] = hashlib.sha256(
            "".join(output_parts).encode("utf-8")
        ).hexdigest()
    finally:
        connection.close()


def _fetch_runtime_counters(base_url: str) -> dict[str, float]:
    request = Request(f"{base_url}/metrics", headers={"Accept": "text/plain"})
    try:
        with urlopen(request, timeout=5) as response:
            text = response.read().decode("utf-8")
    except OSError:
        return {}
    counters: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or " " not in line:
            continue
        name, raw_value = line.rsplit(" ", 1)
        if "{" in name:
            continue
        try:
            counters[name] = float(raw_value)
        except ValueError:
            continue
    return counters


def _validate_completed_trial(
    results: dict[str, Any],
    trial: dict[str, Any],
    definition: dict[str, Any],
) -> None:
    expected_count = int(definition["request_count"])
    if len(trial["requests"]) != expected_count:
        _validation_failure(
            results,
            "request_count",
            f"trial {trial['trial_index']} expected {expected_count} requests, recorded {len(trial['requests'])}",
        )
    expected_indexes = list(range(expected_count))
    indexes = [request["request_index"] for request in trial["requests"]]
    if indexes != expected_indexes:
        _validation_failure(
            results,
            "request_order",
            f"trial {trial['trial_index']} request indexes were {indexes}",
        )
    for request in trial["requests"]:
        _validate_request(results, request)


def _validate_request(results: dict[str, Any], request: dict[str, Any]) -> None:
    timestamps = [
        request["submitted_monotonic_ns"],
        request["first_byte_monotonic_ns"],
        request["first_token_monotonic_ns"],
        request["final_token_monotonic_ns"],
        request["completed_monotonic_ns"],
    ]
    if any(timestamp is None for timestamp in timestamps) or timestamps != sorted(
        timestamps
    ):
        _validation_failure(
            results,
            "request_timestamps",
            f"trial {request['trial_index']} request {request['request_index']} has invalid timestamps",
        )
    token_total = request["prompt_tokens"] + request["completion_tokens"]
    if request["total_tokens"] != token_total:
        _validation_failure(
            results,
            "token_totals",
            f"trial {request['trial_index']} request {request['request_index']} token total mismatch",
        )
    if request["output_sha256"] is None:
        _validation_failure(
            results,
            "output_hash",
            f"trial {request['trial_index']} request {request['request_index']} has no output hash",
        )


def _validate_final_results(results: dict[str, Any]) -> None:
    if results["versions"]["gateway"] == "not_exposed":
        _validation_failure(
            results, "gateway_version", "gateway version was not exposed"
        )
    if results["versions"]["model"]["revision"] == "not_exposed":
        _validation_failure(results, "model_revision", "model revision was not exposed")
    if results["versions"]["tokenizer"]["revision"] == "not_exposed":
        _validation_failure(
            results, "tokenizer_revision", "tokenizer revision was not exposed"
        )
    for counter in _REQUIRED_COUNTERS:
        if counter not in results["runtime_counters"]:
            _validation_failure(
                results,
                "runtime_counter",
                f"required runtime counter {counter} is missing",
            )
    orders = [
        request["request_order"]
        for trial in results["trials"]
        for request in trial["requests"]
    ]
    if orders != list(range(len(orders))):
        _validation_failure(results, "request_order", f"run request order was {orders}")


def _validation_failure(results: dict[str, Any], code: str, message: str) -> None:
    results["validation_failures"].append({"code": code, "message": message})


def _write_artifacts(run_directory: Path, results: dict[str, Any]) -> None:
    _atomic_write_json(run_directory / "results.json", results)
    _atomic_write_text(run_directory / "report.md", _render_report(results))


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# MLX Air Benchmark Report",
        "",
        f"- Run: `{results['run_id']}`",
        f"- Status: `{results['status']}`",
        f"- Suite: `{results['configuration']['suite']}`",
        f"- Server mode: `{results['server']['mode']}`",
        f"- Model: `{results['versions']['model']['name']}`",
        "",
        "| workload | trial | load mode | requested | succeeded | errors |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for trial in results["trials"]:
        lines.append(
            f"| {trial['workload_name']} | {trial['trial_index']} | {trial['load_mode']} | "
            f"{trial['request_count']} | {trial['success_count']} | {trial['error_count']} |"
        )
    if results["validation_failures"]:
        lines.extend(["", "## Validation failures", ""])
        lines.extend(
            f"- `{failure['code']}`: {failure['message']}"
            for failure in results["validation_failures"]
        )
    if results["error"] is not None:
        lines.extend(["", "## Error", "", f"`{results['error']['message']}`"])
    return "\n".join(lines) + "\n"


@contextmanager
def _interrupts_raise() -> Iterator[None]:
    previous: dict[int, Any] = {}

    def raise_interrupt(signum: int, _frame: FrameType | None) -> None:
        raise _RunInterrupted(signum)

    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, raise_interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
