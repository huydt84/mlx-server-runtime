from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import pytest

from mlx_benchmark.__main__ import main as benchmark_main


FIXTURES = Path(__file__).parent / "fixtures" / "mlx_benchmark"
PYTHON_ROOT = Path(__file__).parents[1]


@dataclass(frozen=True)
class _BenchmarkEnvironment:
    gateway: Path
    config: Path
    invocation: Path


@pytest.fixture
def benchmark_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> _BenchmarkEnvironment:
    gateway = tmp_path / "mlx_runtime_gateway"
    shutil.copyfile(FIXTURES / "fake_gateway.py", gateway)
    gateway.chmod(0o755)
    config = FIXTURES / "phase7_benchmark.toml"
    monkeypatch.setenv("MLX_AIR_GATEWAY_EXECUTABLE", str(gateway))
    monkeypatch.setenv("MLX_AIR_DEFAULT_BENCHMARK_CONFIG", str(config))
    monkeypatch.setenv("MLX_AIR_INVOCATION_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("MLX_AIR_VERSION", "0.1.0")
    monkeypatch.delenv("FAKE_GATEWAY_DELAY_READY", raising=False)
    monkeypatch.delenv("FAKE_GATEWAY_PID_FILE", raising=False)
    monkeypatch.delenv("FAKE_GATEWAY_RUN_MODE", raising=False)
    return _BenchmarkEnvironment(gateway, config, tmp_path)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["run", "--suite", "smoke", "--server-mode", "external"],
            "--base-url is required with --server-mode external",
        ),
        (
            [
                "run",
                "--suite",
                "smoke",
                "--base-url",
                "http://127.0.0.1:9000",
            ],
            "--base-url is not valid with --server-mode self-launched",
        ),
    ],
)
def test_run_rejects_inconsistent_server_arguments(
    benchmark_environment: _BenchmarkEnvironment,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
    message: str,
) -> None:
    del benchmark_environment

    with pytest.raises(SystemExit) as error:
        benchmark_main(arguments)

    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("original", "replacement", "field"),
    [
        (
            'cache_state = "cold"',
            'cache_state = "missing"',
            "workloads.sequential_stream.cache_state",
        ),
        (
            'metric_unit = "ms"',
            'metric_unit = "furlongs"',
            "workloads.sequential_stream.metric_unit",
        ),
        (
            'metric_direction = "lower"',
            'metric_direction = "sideways"',
            "workloads.sequential_stream.metric_direction",
        ),
        (
            "requests_per_trial = 2",
            "requests_per_trial = 0",
            "workloads.sequential_stream.requests_per_trial",
        ),
        (
            'load_mode = "sequential"',
            'load_mode = "ramp"',
            "workloads.sequential_stream.load_mode",
        ),
        (
            "[warmup_groups.short]\n"
            'prompt_group = "warmup_short"\n'
            "concurrency = 1\n"
            "output_tokens = 4",
            '[warmup_groups.short]\nprompt_group = "warmup_short"\nconcurrency = 1',
            "warmup_groups.short.output_tokens",
        ),
    ],
)
def test_invalid_configuration_reports_exact_field_before_server_startup(
    benchmark_environment: _BenchmarkEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    original: str,
    replacement: str,
    field: str,
) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        benchmark_environment.config.read_text(encoding="utf-8").replace(
            original, replacement, 1
        ),
        encoding="utf-8",
    )
    pid_file = tmp_path / "gateway-pids"
    monkeypatch.setenv("FAKE_GATEWAY_PID_FILE", str(pid_file))

    status = benchmark_main(
        ["run", "--suite", "smoke", "--benchmark-config", str(invalid)]
    )

    assert status == 50
    assert field in capsys.readouterr().err
    assert not pid_file.exists()


def test_self_launched_run_writes_exact_trials_and_reaps_process_group(
    benchmark_environment: _BenchmarkEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "successful-artifacts"
    pid_file = tmp_path / "successful-pids"
    monkeypatch.setenv("FAKE_GATEWAY_PID_FILE", str(pid_file))

    status = benchmark_main(
        ["run", "--suite", "smoke", "--output-dir", str(output_directory)]
    )

    assert status == 0
    results = _read_results(output_directory)
    assert results["status"] == "succeeded"
    assert results["configuration"]["suite"] == "smoke"
    assert results["configuration"]["benchmark_config"] == str(
        benchmark_environment.config
    )
    assert results["configuration"]["sampling"]["seed"] == 7
    assert results["configuration"]["workloads"][2]["load_mode"] == "closed-loop"
    assert [trial["request_count"] for trial in results["trials"]] == [2, 3, 6]
    assert [trial["maximum_observed_in_flight"] for trial in results["trials"]] == [
        1,
        3,
        2,
    ]
    assert [trial["submission_policy"] for trial in results["trials"]] == [
        "submit-one-after-previous-completes",
        "submit-declared-count-at-once",
        "replace-each-completed-request",
    ]
    requests = _all_requests(results)
    assert len(requests) == 11
    assert all(
        request["first_byte_monotonic_ns"] is not None
        and request["first_token_monotonic_ns"] is not None
        and request["final_token_monotonic_ns"] is not None
        and request["completed_monotonic_ns"] is not None
        and request["prompt_tokens"] == 5
        and request["completion_tokens"] == 2
        and request["total_tokens"] == 7
        and request["output_sha256"] is not None
        for request in requests
    )
    assert all(
        request["streaming"] is True for request in results["trials"][0]["requests"]
    )
    assert all(
        request["streaming"] is False for request in results["trials"][1]["requests"]
    )
    assert (output_directory / "report.md").is_file()
    assert (output_directory / "logs" / "gateway.log").is_file()
    assert (output_directory / "logs" / "worker.log").is_file()
    _assert_processes_reaped(pid_file)


def test_phase8_run_applies_order_cache_state_and_trial_metric_deltas(
    benchmark_environment: _BenchmarkEnvironment, tmp_path: Path
) -> None:
    output_directory = tmp_path / "phase8-artifacts"
    config = FIXTURES / "phase8_benchmark.toml"

    status = benchmark_main(
        [
            "run",
            "--suite",
            "phase8",
            "--benchmark-config",
            str(config),
            "--output-dir",
            str(output_directory),
        ]
    )

    assert status == 0
    results = _read_results(output_directory)
    assert [entry["runtime_configuration"] for entry in results["applied_order"]] == [
        "serial",
        "overlap",
    ]
    assert len(results["warmups"]) == 2
    assert all(warmup["measured"] is False for warmup in results["warmups"])
    assert len(results["trials"]) == 6
    for trial in results["trials"]:
        assert trial["configuration_order"] == "round"
        assert "mlx_requests_total" in trial["runtime_metrics"]
        if trial["workload_name"] == "cold":
            assert all(request["cached_tokens"] == 0 for request in trial["requests"])
        elif trial["workload_name"] == "shared":
            assert all(request["cached_tokens"] == 3 for request in trial["requests"])
            assert _metric_delta(
                trial, "mlx_prefix_cache_hits_by_backend"
            ) == pytest.approx(2.0)
        elif trial["workload_name"] == "pressure":
            assert _metric_delta(
                trial, "mlx_prefix_cache_evictions_by_backend"
            ) == pytest.approx(2.0)


def test_repeated_runs_preserve_workload_prompt_trial_and_request_order(
    benchmark_environment: _BenchmarkEnvironment, tmp_path: Path
) -> None:
    directories = [tmp_path / "first", tmp_path / "second"]
    for directory in directories:
        assert (
            benchmark_main(["run", "--suite", "smoke", "--output-dir", str(directory)])
            == 0
        )

    assert _request_identity(_read_results(directories[0])) == _request_identity(
        _read_results(directories[1])
    )


def test_request_failure_preserves_measurements_and_reaps_process_group(
    benchmark_environment: _BenchmarkEnvironment,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_directory = tmp_path / "failure-artifacts"
    pid_file = tmp_path / "failure-pids"
    monkeypatch.setenv("FAKE_GATEWAY_PID_FILE", str(pid_file))
    monkeypatch.setenv("FAKE_GATEWAY_RUN_MODE", "request-failure")

    status = benchmark_main(
        ["run", "--suite", "smoke", "--output-dir", str(output_directory)]
    )

    assert status == 50
    results = _read_results(output_directory)
    assert results["status"] == "failed"
    assert results["failure_stage"] == "measurement"
    requests = _all_requests(results)
    assert len(requests) == 11
    assert all(request["status"] == "failed" for request in requests)
    _assert_processes_reaped(pid_file)


def test_sigint_preserves_failure_stage_and_reaps_process_group(
    benchmark_environment: _BenchmarkEnvironment, tmp_path: Path
) -> None:
    output_directory = tmp_path / "interrupt-artifacts"
    pid_file = tmp_path / "interrupt-pids"
    environment = os.environ.copy()
    environment["FAKE_GATEWAY_PID_FILE"] = str(pid_file)
    environment["FAKE_GATEWAY_DELAY_READY"] = "1"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlx_benchmark",
            "run",
            "--suite",
            "smoke",
            "--output-dir",
            str(output_directory),
        ],
        cwd=PYTHON_ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_file(pid_file)
        process.send_signal(signal.SIGINT)
        status = process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    assert status == 130
    results = _read_results(output_directory)
    assert results["status"] == "interrupted"
    assert results["failure_stage"] == "readiness"
    assert (output_directory / "logs" / "gateway.log").is_file()
    assert (output_directory / "logs" / "worker.log").is_file()
    _assert_processes_reaped(pid_file)


def test_external_mode_records_identity_without_stopping_server(
    benchmark_environment: _BenchmarkEnvironment, tmp_path: Path
) -> None:
    output_directory = tmp_path / "external-artifacts"
    port = _reserve_port()
    runtime_config = tmp_path / "external.toml"
    runtime_config.write_text(
        f"[server]\nport = {port}\n"
        "[worker]\n"
        'model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"\n',
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["MLX_RUNTIME_CONFIG"] = str(runtime_config)
    environment["MLX_AIR_BENCHMARK_ENABLED"] = "1"
    server = subprocess.Popen(
        [str(benchmark_environment.gateway)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_port(port)
        status = benchmark_main(
            [
                "run",
                "--suite",
                "smoke",
                "--server-mode",
                "external",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--output-dir",
                str(output_directory),
            ]
        )

        assert status == 0
        assert server.poll() is None
        results = _read_results(output_directory)
        assert results["server"]["mode"] == "external"
        assert results["versions"]["gateway"] == "0.1.0"
    finally:
        if server.poll() is None:
            server.terminate()
        server.wait(timeout=5)


def _read_results(directory: Path) -> dict[str, object]:
    return json.loads((directory / "results.json").read_text(encoding="utf-8"))


def _all_requests(results: dict[str, object]) -> list[dict[str, object]]:
    return [request for trial in results["trials"] for request in trial["requests"]]


def _request_identity(results: dict[str, object]) -> list[tuple[object, ...]]:
    return [
        (
            request["workload_name"],
            request["trial_index"],
            request["request_index"],
            request["prompt_index"],
            request["prompt_target_tokens"],
            request["prompt_sha256"],
        )
        for request in _all_requests(results)
    ]


def _metric_delta(trial: dict[str, object], prefix: str) -> float:
    metric = next(
        value
        for name, value in trial["runtime_metrics"].items()
        if name.startswith(prefix)
    )
    return float(metric["delta"])


def _assert_processes_reaped(pid_file: Path) -> None:
    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").splitlines()]
    assert len(pids) == 2
    for pid in pids:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _wait_for_file(path: Path) -> None:
    deadline = time.monotonic() + 5
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.02)


def _reserve_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.02)
    raise AssertionError(f"timed out waiting for port {port}")
