from __future__ import annotations

import json

import pytest

from mlx_worker.native_mlx import pipeline_profile
from mlx_worker.native_mlx.pipeline_profile import (
    PipelineEvent,
    PipelineProfiler,
    render_pipeline_report,
    validate_pipeline_event,
    write_pipeline_artifacts,
)


def test_pipeline_profiler_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", raising=False)

    assert PipelineProfiler.from_environment("model") is None


def test_pipeline_profiler_requires_output_directory(monkeypatch) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "1")
    monkeypatch.delenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR", raising=False)

    with pytest.raises(ValueError, match="PIPELINE_PROFILE_DIR"):
        PipelineProfiler.from_environment("model")


def test_pipeline_profiler_rejects_metal_capture_without_startup_flag(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "1")
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR", str(tmp_path))
    monkeypatch.setenv("MLX_RUNTIME_NATIVE_METAL_CAPTURE", "1")
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)

    with pytest.raises(ValueError, match="MTL_CAPTURE_ENABLED=1"):
        PipelineProfiler.from_environment("model")


def test_pipeline_profiler_writes_correlated_artifacts(tmp_path) -> None:
    profiler = PipelineProfiler(tmp_path, model="model", run_id="run-1")
    profiler.record("req-1", "gateway", "http", duration_us=10)
    profiler.record("req-1", "runtime", "terminal", duration_us=20, state="cancelled")

    jsonl, trace, report = profiler.write_artifacts()

    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert {row["run_id"] for row in rows} == {"run-1"}
    assert {row["request_id"] for row in rows} == {"req-1"}
    assert len(json.loads(trace.read_text())["traceEvents"]) == 2
    assert "not benchmark rows" in report.read_text()
    assert "inference-graph tracing" in report.read_text()


def test_pipeline_event_schema_rejects_empty_correlation_id() -> None:
    event = PipelineEvent(
        schema_version=1,
        run_id="run",
        request_id="",
        backend="native-mlx",
        model="model",
        workload="workload",
        component="runtime",
        stage="terminal",
        monotonic_ns=1,
        offset_us=0,
        duration_us=0,
    )

    with pytest.raises(ValueError, match="request_id"):
        validate_pipeline_event(event)


def test_report_orders_dominant_stage_first() -> None:
    base = {
        "schema_version": 1,
        "run_id": "run",
        "request_id": "request",
        "backend": "native-mlx",
        "model": "model",
        "workload": "workload",
        "monotonic_ns": 1,
        "offset_us": 0,
    }
    report = render_pipeline_report(
        (
            PipelineEvent(**base, component="runtime", stage="small", duration_us=1),
            PipelineEvent(**base, component="model", stage="dominant", duration_us=10),
        )
    )

    assert report.index("dominant") < report.index("small")


def _record_events(profiler: PipelineProfiler, count: int, start: int = 0) -> None:
    for index in range(start, start + count):
        profiler.record(
            f"req-{index % 3}",
            "mlx" if index % 2 else "runtime",
            "async_dispatch" if index % 2 else "terminal",
            duration_us=index,
            details={"step": index, "nested": {"tokens": [index, index + 1]}},
        )


def test_incremental_artifacts_match_full_rewrite(tmp_path) -> None:
    profiler = PipelineProfiler(tmp_path / "incremental", model="model", run_id="run")
    _record_events(profiler, 3)
    profiler.write_artifacts()
    _record_events(profiler, 1, start=3)
    profiler.write_artifacts()
    _record_events(profiler, 5, start=4)
    paths = profiler.write_artifacts()

    expected = write_pipeline_artifacts(profiler.events(), tmp_path / "full")

    for actual_path, expected_path in zip(paths, expected, strict=True):
        assert actual_path.read_bytes() == expected_path.read_bytes()
    assert len(json.loads(paths[1].read_text())["traceEvents"]) == 9


@pytest.mark.parametrize("stored", [10, 1_000])
def test_flush_cost_does_not_depend_on_stored_events(
    monkeypatch, tmp_path, stored
) -> None:
    profiler = PipelineProfiler(tmp_path, model="model", run_id="run")
    _record_events(profiler, stored)
    jsonl, trace, _ = profiler.write_artifacts()
    calls = {"jsonl": 0, "trace": 0, "full_rewrite": 0}

    def counting(name, function):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        pipeline_profile,
        "_jsonl_line",
        counting("jsonl", pipeline_profile._jsonl_line),
    )
    monkeypatch.setattr(
        pipeline_profile,
        "_trace_entry",
        counting("trace", pipeline_profile._trace_entry),
    )
    monkeypatch.setattr(
        pipeline_profile,
        "_write_event_artifacts",
        counting("full_rewrite", pipeline_profile._write_event_artifacts),
    )
    jsonl_size = jsonl.stat().st_size

    for flush in range(1, 4):
        _record_events(profiler, 1, start=stored + flush)
        profiler.write_artifacts()
        assert calls == {"jsonl": flush, "trace": flush, "full_rewrite": 0}

    profiler.write_artifacts()

    assert calls == {"jsonl": 3, "trace": 3, "full_rewrite": 0}
    new_lines = jsonl.read_bytes()[jsonl_size:].decode().splitlines()
    assert [json.loads(line)["duration_us"] for line in new_lines] == [
        stored + 1,
        stored + 2,
        stored + 3,
    ]
    assert len(json.loads(trace.read_text())["traceEvents"]) == stored + 3


def test_incremental_flush_keeps_externally_joined_events(tmp_path) -> None:
    profiler = PipelineProfiler(tmp_path, model="model", run_id="run")
    _record_events(profiler, 2)
    profiler.write_artifacts()
    gateway = PipelineEvent(
        schema_version=1,
        run_id="run",
        request_id="req-0",
        backend="native-mlx",
        model="model",
        workload="workload",
        component="gateway",
        stage="http_round_trip",
        monotonic_ns=1,
        offset_us=0,
        duration_us=5,
    )
    write_pipeline_artifacts((*profiler.events(), gateway), tmp_path)

    _record_events(profiler, 1, start=2)
    _, trace, _ = profiler.write_artifacts()

    names = [event["cat"] for event in json.loads(trace.read_text())["traceEvents"]]
    assert names == ["runtime", "mlx", "gateway", "runtime"]


def test_incremental_flush_rewrites_unrecognized_trace(tmp_path) -> None:
    profiler = PipelineProfiler(tmp_path, model="model", run_id="run")
    _record_events(profiler, 2)
    profiler.write_artifacts()
    write_pipeline_artifacts((), tmp_path)

    _record_events(profiler, 1, start=2)
    paths = profiler.write_artifacts()

    expected = write_pipeline_artifacts(profiler.events(), tmp_path / "full")
    for actual_path, expected_path in zip(paths, expected, strict=True):
        assert actual_path.read_bytes() == expected_path.read_bytes()
