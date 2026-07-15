"""Execute selected MLX Air workloads and persist self-contained results."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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

from mlx_benchmark.configuration import SelectedConfiguration
from mlx_benchmark.loadgen import execute_setup_prompts, execute_workloads
from mlx_benchmark.prompts import (
    generate_prompt_bank,
    shared_prefix_prime_prompts,
)


_MLX_AIR_VERSION_ENV = "MLX_AIR_VERSION"
_INVOCATION_DIRECTORY_ENV = "MLX_AIR_INVOCATION_DIRECTORY"
_WORKER_LOG_ENV = "MLX_AIR_WORKER_LOG"
_SHUTDOWN_TIMEOUT_SECONDS = 15
_BENCHMARK_EXECUTION_FAILURE = 50
_REQUIRED_COUNTERS = (
    "mlx_requests_total",
    "mlx_prompt_tokens_total",
    "mlx_completion_tokens_total",
)
_RUNTIME_METRIC_PREFIXES = (
    "mlx_requests_",
    "mlx_queue_",
    "mlx_prompt_tokens_",
    "mlx_completion_tokens_",
    "mlx_prompt_cache_",
    "mlx_cache_",
    "mlx_prefix_cache_",
    "mlx_kv_cache_",
    "mlx_scheduler_",
    "mlx_scheduled_tokens_",
    "mlx_batch_size_",
    "mlx_executor_",
    "mlx_native_execution_mode",
    "mlx_worker_cancellations_",
    "mlx_worker_errors_",
    "mlx_worker_memory_bytes",
    "mlx_peak_memory_",
    "mlx_model_graph_",
    "mlx_ipc_",
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


def run_benchmark(
    arguments: argparse.Namespace,
    gateway: Path,
    selected: SelectedConfiguration,
) -> int:
    """Run the selected benchmark configuration.

    Args:
        arguments: Validated ``mlx-air bench run`` arguments.
        gateway: Absolute path to the version-matched gateway executable.
        selected: Loaded and fully selected benchmark configuration.

    Returns:
        Zero on success, 50 on benchmark failure, or 128 plus the interrupt
        signal number.
    """
    run_directory = _resolve_run_directory(arguments.output_dir)
    _create_artifact_tree(run_directory)
    results = _initial_results(arguments, gateway, run_directory, selected)
    _write_artifacts(run_directory, results)
    server: _ServerProcess | None = None

    try:
        with _interrupts_raise():
            prompts = generate_prompt_bank(results["configuration"])
            _validate_warmup_material(results["configuration"], prompts)

            def completed_trial(trial: dict[str, Any]) -> None:
                results["trials"].append(trial)
                _validate_completed_trial(results, trial)
                _write_artifacts(run_directory, results)

            for execution in _execution_plan(
                results["configuration"], arguments.server_mode
            ):
                execution_configuration = _configuration_for_execution(
                    results["configuration"], execution
                )
                if not execution_configuration["workloads"]:
                    continue
                if arguments.server_mode == "self-launched":
                    results["failure_stage"] = "server_configuration"
                    base_url, server = _start_server(
                        gateway,
                        run_directory,
                        results,
                        execution_configuration,
                        execution,
                    )
                else:
                    base_url = _normalize_base_url(arguments.base_url)

                results["server"]["base_url"] = base_url
                results["configuration"]["base_url"] = base_url
                results["failure_stage"] = "readiness"
                readiness = _wait_for_readiness(
                    base_url, server, execution_configuration
                )
                _record_server_identity(
                    results, base_url, readiness, gateway, arguments
                )
                results["applied_order"].append(
                    {
                        **execution,
                        "sequence": len(results["applied_order"]),
                        "base_url": base_url,
                    }
                )
                results["failure_stage"] = "measurement"
                asyncio.run(
                    _execute_phase8_state(
                        base_url,
                        execution_configuration,
                        prompts,
                        execution,
                        results,
                        completed_trial,
                    )
                )
                results["runtime_counters"] = _fetch_runtime_counters(base_url)
                if server is not None:
                    server.stop()
                    server = None
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
    arguments: argparse.Namespace,
    gateway: Path,
    run_directory: Path,
    selected: SelectedConfiguration,
) -> dict[str, Any]:
    final_configuration = json.loads(json.dumps(selected.values))
    final_configuration.update(
        {
            "benchmark_config": str(selected.source_path),
            "base_url": arguments.base_url,
            "output_directory": str(run_directory),
        }
    )
    model = final_configuration["models"][0]
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
            "model": {
                "name": model["checkpoint"],
                "revision": model["revision"],
            },
            "tokenizer": {
                "name": model["tokenizer"],
                "revision": model["revision"],
            },
        },
        "host": _host_information(),
        "server": {
            "mode": arguments.server_mode,
            "base_url": arguments.base_url,
            "gateway_executable": str(gateway)
            if arguments.server_mode == "self-launched"
            else None,
            "runtime_configuration": None,
            "selected_model": model["name"],
            "selected_runtime_configuration": final_configuration[
                "runtime_configurations"
            ][0]["name"],
        },
        "runtime_counters": {},
        "applied_order": [],
        "warmups": [],
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


def _execution_plan(
    configuration: dict[str, Any], server_mode: str
) -> list[dict[str, str]]:
    plan = [
        {
            "configuration_order": order["name"],
            "model": order["model"],
            "runtime_configuration": runtime,
        }
        for order in configuration["configuration_orders"]
        for runtime in order["runtime_configurations"]
    ]
    if server_mode == "external":
        return plan[:1]
    return plan


def _configuration_for_execution(
    configuration: dict[str, Any], execution: dict[str, str]
) -> dict[str, Any]:
    selected = json.loads(json.dumps(configuration))
    model_name = execution["model"]
    runtime_name = execution["runtime_configuration"]
    covered_workloads = {
        workload
        for coverage in configuration["coverage"]
        if coverage["model"] == model_name
        for workload in coverage["workloads"]
    }
    selected["models"] = [
        model for model in configuration["models"] if model["name"] == model_name
    ]
    selected["runtime_configurations"] = [
        runtime
        for runtime in configuration["runtime_configurations"]
        if runtime["name"] == runtime_name
    ]
    selected["workloads"] = [
        workload
        for workload in configuration["workloads"]
        if workload["name"] in covered_workloads
        and runtime_name in workload["runtime_configurations"]
    ]
    return selected


def _validate_warmup_material(
    configuration: dict[str, Any], prompts: dict[str, list[Any]]
) -> None:
    measured_groups = {
        workload["prompt_group"] for workload in configuration["workloads"]
    }
    warmup_groups = {
        warmup["prompt_group"] for warmup in configuration["warmup_groups"]
    }
    measured_hashes = {
        prompt.sha256 for group in measured_groups for prompt in prompts[group]
    }
    warmup_hashes = {
        prompt.sha256 for group in warmup_groups for prompt in prompts[group]
    }
    overlap = measured_hashes.intersection(warmup_hashes)
    if overlap:
        raise _RunFailure(
            "configuration_validation",
            "warmup prompts overlap measured prompt material",
        )


async def _execute_phase8_state(
    base_url: str,
    configuration: dict[str, Any],
    prompts: dict[str, list[Any]],
    execution: dict[str, str],
    results: dict[str, Any],
    completed_trial: Any,
) -> None:
    for warmup in configuration["warmup_groups"]:
        group = prompts[warmup["prompt_group"]]
        count = min(int(warmup["concurrency"]), len(group))
        records = await execute_setup_prompts(
            base_url,
            configuration,
            group[:count],
            output_tokens=int(warmup["output_tokens"]),
            concurrency=int(warmup["concurrency"]),
        )
        results["warmups"].append(
            {
                **execution,
                "group": warmup["name"],
                "measured": False,
                "request_count": len(records),
                "requests": records,
            }
        )

    async def before_trial(
        workload: dict[str, Any], trial_index: int, group: list[Any]
    ) -> dict[str, Any]:
        cache_state = configuration["cache_states"][workload["cache_state"]]
        reset = await asyncio.to_thread(
            _reset_benchmark_state,
            base_url,
            True,
            False,
        )
        priming: list[dict[str, Any]] = []
        if cache_state["mode"] == "warm-prefix":
            prime_prompts = shared_prefix_prime_prompts(
                group,
                trial_index=trial_index,
                request_count=int(workload["requests_per_trial"]),
            )
            priming = await execute_setup_prompts(
                base_url,
                configuration,
                prime_prompts,
                output_tokens=1,
                concurrency=min(len(prime_prompts), int(workload["concurrency"])),
            )
        before = await asyncio.to_thread(_fetch_runtime_counters, base_url)
        return {
            "reset": reset,
            "cache_state": cache_state,
            "priming": priming,
            "metrics_before": before,
        }

    async def after_trial(trial: dict[str, Any], context: dict[str, Any]) -> None:
        after = await asyncio.to_thread(_fetch_runtime_counters, base_url)
        trial.update(execution)
        trial["cache_state"] = context["cache_state"]
        trial["cache_preparation"] = {
            "reset": context["reset"],
            "priming": context["priming"],
        }
        trial["runtime_metrics"] = _runtime_metric_delta(
            context["metrics_before"], after
        )

    await execute_workloads(
        base_url,
        configuration,
        prompts,
        on_trial=completed_trial,
        before_trial=before_trial,
        after_trial=after_trial,
        initial_request_order=sum(
            int(trial["request_count"]) for trial in results["trials"]
        ),
    )


def _start_server(
    gateway: Path,
    run_directory: Path,
    results: dict[str, Any],
    configuration: dict[str, Any],
    execution: dict[str, str],
) -> tuple[str, _ServerProcess]:
    model = configuration["models"][0]
    selected_runtime = configuration["runtime_configurations"][0]
    workloads = configuration["workloads"]
    port = _reserve_loopback_port()
    socket_directory = Path("/tmp") / f"mlx-air-{os.getuid()}"
    socket_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    socket_suffix = hashlib.sha256(str(run_directory).encode()).hexdigest()[:12]
    ipc_path = socket_directory / f"benchmark-{os.getpid()}-{socket_suffix}.sock"
    maximum_concurrency = max(int(workload["concurrency"]) for workload in workloads)
    maximum_prompt_tokens = max(
        int(configuration["prompt_bank"][workload["prompt_group"]]["target_tokens"])
        for workload in workloads
    )
    maximum_output_tokens = max(
        int(workload["output_tokens"]) for workload in workloads
    )
    sampling = configuration["sampling"]
    runtime_config = {
        "server": {"host": "127.0.0.1", "port": port},
        "worker": {
            "python": sys.executable,
            "module": "mlx_worker.main",
            "backend": selected_runtime["backend"],
            "model": model["checkpoint"],
            "ipc_path": str(ipc_path),
        },
        "generation": {
            "temperature": sampling["temperature"],
            "top_p": sampling["top_p"],
            "max_tokens": maximum_output_tokens,
        },
        "limits": {
            "max_pending_requests": max(maximum_concurrency * 2, 8),
            "max_active_requests": maximum_concurrency,
            "max_prompt_tokens": min(maximum_prompt_tokens * 2 + 256, 65_536),
            "max_completion_tokens": maximum_output_tokens,
            "max_total_tokens_per_request": min(
                maximum_prompt_tokens * 2 + maximum_output_tokens + 256, 65_536
            ),
            "request_timeout_seconds": sampling["request_timeout_seconds"],
        },
        "telemetry": {"enable_prometheus": True, "metrics_path": "/metrics"},
    }
    sequence = len(results["applied_order"])
    config_path = run_directory / f"runtime-{sequence:02d}.toml"
    config_path.write_text(_render_runtime_toml(runtime_config), encoding="utf-8")
    results["server"]["runtime_configuration"] = runtime_config
    results["configuration"]["base_url"] = f"http://127.0.0.1:{port}"
    results["failure_stage"] = "server_startup"
    _write_artifacts(run_directory, results)

    log_handle = (run_directory / "logs" / "gateway.log").open("ab", buffering=0)
    environment = os.environ.copy()
    environment["MLX_RUNTIME_CONFIG"] = str(config_path)
    environment["MLX_AIR_BENCHMARK_ENABLED"] = "1"
    environment[_WORKER_LOG_ENV] = str(run_directory / "logs" / "worker.log")
    environment.update(selected_runtime["environment"])
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
    results["server"]["selected_model"] = execution["model"]
    results["server"]["selected_runtime_configuration"] = execution[
        "runtime_configuration"
    ]
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
        f"temperature = {float(generation['temperature']):.1f}\n"
        f"top_p = {float(generation['top_p']):.1f}\n"
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


def _wait_for_readiness(
    base_url: str,
    server: _ServerProcess | None,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    timeout_seconds = int(configuration["sampling"]["readiness_timeout_seconds"])
    deadline = time.monotonic() + timeout_seconds
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
        f"server did not become ready within {timeout_seconds} seconds: {last_error}",
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
    configured_model = results["configuration"]["models"][0]
    model = str(readiness.get("model") or configured_model["checkpoint"])
    revision = str(readiness.get("revision") or configured_model["revision"])
    tokenizer_revision = str(readiness.get("tokenizer_revision") or revision)
    results["versions"]["gateway"] = gateway_version
    results["versions"]["model"] = {"name": model, "revision": revision}
    results["versions"]["tokenizer"] = {
        "name": configured_model["tokenizer"],
        "revision": tokenizer_revision,
    }
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


def _reset_benchmark_state(
    base_url: str, clear_cache: bool, reset_counters: bool
) -> dict[str, Any]:
    body = json.dumps(
        {"clear_cache": clear_cache, "reset_counters": reset_counters}
    ).encode()
    request = Request(
        f"{base_url}/internal/benchmark/reset",
        data=body,
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as error:
        if hasattr(error, "code") and hasattr(error, "read"):
            detail = error.read().decode("utf-8", errors="replace")
            raise _RunFailure(
                "cache_reset",
                f"benchmark reset returned HTTP {error.code}: {detail}",
            ) from error
        raise _RunFailure("cache_reset", f"benchmark reset failed: {error}") from error
    if payload.get("scheduler_idle") is not True:
        raise _RunFailure("cache_reset", "benchmark reset did not report idle state")
    if (
        payload.get("model_preserved") is not True
        or payload.get("graphs_preserved") is not True
    ):
        raise _RunFailure(
            "cache_reset", "benchmark reset did not preserve model and graph state"
        )
    return payload


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
        try:
            counters[name] = float(raw_value)
        except ValueError:
            continue
    return counters


def _runtime_metric_delta(
    before: dict[str, float], after: dict[str, float]
) -> dict[str, dict[str, float]]:
    names = sorted(
        name
        for name in set(before).union(after)
        if name.startswith(_RUNTIME_METRIC_PREFIXES)
    )
    return {
        name: {
            "before": before.get(name, 0.0),
            "after": after.get(name, 0.0),
            "delta": after.get(name, 0.0) - before.get(name, 0.0),
        }
        for name in names
    }


def _validate_completed_trial(results: dict[str, Any], trial: dict[str, Any]) -> None:
    workload = next(
        workload
        for workload in results["configuration"]["workloads"]
        if workload["name"] == trial["workload_name"]
    )
    expected_count = int(workload["requests_per_trial"])
    if len(trial["requests"]) != expected_count:
        _validation_failure(
            results,
            "request_count",
            f"{trial['workload_name']} trial {trial['trial_index']} expected "
            f"{expected_count} requests, recorded {len(trial['requests'])}",
        )
    expected_indexes = list(range(expected_count))
    indexes = [request["request_index"] for request in trial["requests"]]
    if indexes != expected_indexes:
        _validation_failure(
            results,
            "request_order",
            f"{trial['workload_name']} trial {trial['trial_index']} request indexes were {indexes}",
        )
    mode = workload["load_mode"]
    expected_maximum = 1 if mode == "sequential" else int(workload["concurrency"])
    if trial["maximum_observed_in_flight"] != expected_maximum:
        _validation_failure(
            results,
            "load_concurrency",
            f"{trial['workload_name']} trial {trial['trial_index']} expected maximum "
            f"in-flight {expected_maximum}, observed {trial['maximum_observed_in_flight']}",
        )
    if trial["streaming"] != workload["streaming"]:
        _validation_failure(
            results,
            "streaming_mode",
            f"{trial['workload_name']} recorded the wrong streaming mode",
        )
    for request in trial["requests"]:
        _validate_request(results, request)
    cache_mode = trial["cache_state"]["mode"]
    cached_tokens = [int(request["cached_tokens"]) for request in trial["requests"]]
    if cache_mode == "cold" and any(cached_tokens):
        _validation_failure(
            results,
            "cold_cache_hit",
            f"{trial['workload_name']} cold trial recorded cached tokens {cached_tokens}",
        )
    if cache_mode == "warm-prefix" and any(
        cached <= 0 or cached >= int(request["prompt_tokens"])
        for cached, request in zip(cached_tokens, trial["requests"], strict=True)
    ):
        _validation_failure(
            results,
            "shared_prefix_state",
            f"{trial['workload_name']} did not record partial prefix hits for every request",
        )
    if not trial.get("runtime_metrics"):
        _validation_failure(
            results,
            "runtime_metric_delta",
            f"{trial['workload_name']} trial {trial['trial_index']} has no runtime metric delta",
        )


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
            f"{request['workload_name']} trial {request['trial_index']} request "
            f"{request['request_index']} has invalid timestamps",
        )
    token_total = request["prompt_tokens"] + request["completion_tokens"]
    if request["total_tokens"] != token_total:
        _validation_failure(
            results,
            "token_totals",
            f"{request['workload_name']} trial {request['trial_index']} request "
            f"{request['request_index']} token total mismatch",
        )
    if request["output_sha256"] is None:
        _validation_failure(
            results,
            "output_hash",
            f"{request['workload_name']} trial {request['trial_index']} request "
            f"{request['request_index']} has no output hash",
        )


def _validate_final_results(results: dict[str, Any]) -> None:
    if results["versions"]["gateway"] == "not_exposed":
        _validation_failure(
            results, "gateway_version", "gateway version was not exposed"
        )
    if results["versions"]["model"]["revision"] in {"not_exposed", "resolve-at-run"}:
        _validation_failure(results, "model_revision", "model revision was not exposed")
    if results["versions"]["tokenizer"]["revision"] in {
        "not_exposed",
        "resolve-at-run",
    }:
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
    expected_order = [
        (
            entry["configuration_order"],
            entry["model"],
            entry["runtime_configuration"],
        )
        for entry in _execution_plan(
            results["configuration"], results["server"]["mode"]
        )
    ]
    applied_order = [
        (
            entry["configuration_order"],
            entry["model"],
            entry["runtime_configuration"],
        )
        for entry in results["applied_order"]
    ]
    if applied_order != expected_order:
        _validation_failure(
            results,
            "configuration_order",
            f"expected applied order {expected_order}, recorded {applied_order}",
        )


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
        f"- Focus: `{results['configuration']['focus']}`",
        f"- Server mode: `{results['server']['mode']}`",
        f"- Model: `{results['versions']['model']['name']}`",
        "",
        "| workload | trial | load mode | streaming | concurrency | requested | succeeded | errors |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for trial in results["trials"]:
        lines.append(
            f"| {trial['workload_name']} | {trial['trial_index']} | {trial['load_mode']} | "
            f"{trial['streaming']} | {trial['configured_concurrency']} | "
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
