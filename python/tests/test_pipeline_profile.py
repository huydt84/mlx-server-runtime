from __future__ import annotations

import json

import pytest

from mlx_worker.native_mlx.pipeline_profile import (
    PipelineEvent,
    PipelineProfiler,
    render_pipeline_report,
    validate_pipeline_event,
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
    assert "semantic inference-graph tracing" in report.read_text()


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
    base = dict(
        schema_version=1,
        run_id="run",
        request_id="request",
        backend="native-mlx",
        model="model",
        workload="workload",
        monotonic_ns=1,
        offset_us=0,
    )
    report = render_pipeline_report(
        (
            PipelineEvent(**base, component="runtime", stage="small", duration_us=1),
            PipelineEvent(**base, component="model", stage="dominant", duration_us=10),
        )
    )

    assert report.index("dominant") < report.index("small")
