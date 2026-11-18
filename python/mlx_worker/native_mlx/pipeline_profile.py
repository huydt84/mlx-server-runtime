"""Low-overhead, request-correlated native MLX pipeline profiling."""

from __future__ import annotations

import json
import os
import textwrap
import threading
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PipelineEvent:
    """One stage event in a correlated inference-pipeline profile."""

    schema_version: int
    run_id: str
    request_id: str
    backend: str
    model: str
    workload: str
    component: str
    stage: str
    monotonic_ns: int
    offset_us: int
    duration_us: int
    phase: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    state: str | None = None
    error: str | None = None
    details: dict[str, Any] | None = None


class PipelineProfiler:
    """Append bounded stage events and render durable diagnostic artifacts.

    Artifacts are written incrementally: each call to `write_artifacts` appends
    only the events recorded since the previous call and re-renders the report
    from running stage totals, so the cost of a flush depends on the number of
    new events, not on how many events the run has already stored.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        model: str,
        workload: str = "public-gateway",
        run_id: str | None = None,
        max_events: int = 50_000,
    ) -> None:
        self.output_dir = output_dir
        self.model = model
        self.workload = workload
        self.run_id = run_id or uuid.uuid4().hex
        self.max_events = max_events
        self._origin_ns = time.perf_counter_ns()
        self._events: list[PipelineEvent] = []
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._written = 0
        self._artifacts_started = False
        self._stage_totals: dict[tuple[str, str], int] = {}
        self._metal_capture_requested = os.environ.get(
            "MLX_RUNTIME_NATIVE_METAL_CAPTURE", "0"
        ).lower() in {"1", "true", "yes", "on"}
        self._metal_capture_active = False
        self._metal_capture_finished = False

    @classmethod
    def from_environment(cls, model: str) -> PipelineProfiler | None:
        """Construct the profiler only when explicitly enabled."""

        if os.environ.get("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE", "0").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return None
        output = os.environ.get("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR", "").strip()
        if not output:
            raise ValueError(
                "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR is required when pipeline profiling is enabled"
            )
        if (
            os.environ.get("MLX_RUNTIME_NATIVE_METAL_CAPTURE", "0").lower()
            in {
                "1",
                "true",
                "yes",
                "on",
            }
            and os.environ.get("MTL_CAPTURE_ENABLED") != "1"
        ):
            raise ValueError(
                "Metal capture requires MTL_CAPTURE_ENABLED=1 before process startup"
            )
        return cls(
            Path(output),
            model=model,
            workload=os.environ.get(
                "MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD", "public-gateway"
            ),
            run_id=os.environ.get("MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID"),
        )

    def record(
        self,
        request_id: str,
        component: str,
        stage: str,
        *,
        started_ns: int | None = None,
        duration_us: int = 0,
        phase: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        state: str | None = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record one event; stop at the configured bound instead of growing forever."""

        now_ns = time.perf_counter_ns()
        start_ns = started_ns if started_ns is not None else now_ns
        event = PipelineEvent(
            schema_version=1,
            run_id=self.run_id,
            request_id=request_id,
            backend="native-mlx",
            model=self.model,
            workload=self.workload,
            component=component,
            stage=stage,
            monotonic_ns=start_ns,
            offset_us=max(0, (start_ns - self._origin_ns) // 1_000),
            duration_us=max(0, duration_us),
            phase=phase,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            state=state,
            error=error,
            details=details,
        )
        validate_pipeline_event(event)
        with self._lock:
            if len(self._events) < self.max_events:
                self._events.append(event)

    def events(self) -> tuple[PipelineEvent, ...]:
        """Return an immutable snapshot of collected events."""

        with self._lock:
            return tuple(self._events)

    def begin_metal_capture(self, request_id: str) -> None:
        """Start one bounded MLX Metal capture for the first public request."""

        if (
            not self._metal_capture_requested
            or self._metal_capture_active
            or self._metal_capture_finished
            or request_id == "warmup"
        ):
            return
        import mlx.core as mx

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "pipeline.gputrace"
        mx.metal.start_capture(str(path))
        self._metal_capture_active = True
        self.record(
            request_id,
            "metal",
            "capture_start",
            details={"gputrace_path": str(path), "benchmark_latency": False},
        )

    def end_metal_capture(self, request_id: str) -> None:
        """Stop the active bounded capture and record its artifact path."""

        if not self._metal_capture_active:
            return
        import mlx.core as mx

        mx.metal.stop_capture()
        self._metal_capture_active = False
        self._metal_capture_finished = True
        self.record(
            request_id,
            "metal",
            "capture_stop",
            details={
                "gputrace_path": str(self.output_dir / "pipeline.gputrace"),
                "benchmark_latency": False,
            },
        )

    def write_artifacts(self) -> tuple[Path, Path, Path]:
        """Persist events recorded since the last call to the three artifacts.

        The first call writes all three files from scratch. Later calls append
        new events to the JSONL file and the Chrome Trace event array, then
        rewrite the bounded markdown report from running stage totals. When
        nothing was recorded since the last call, no file is touched.

        Returns:
            Paths of the JSONL events, Chrome Trace JSON, and markdown report.
        """

        paths = _artifact_paths(self.output_dir)
        with self._write_lock:
            with self._lock:
                pending = self._events[self._written :]
            if self._artifacts_started and not pending:
                return paths
            for event in pending:
                key = (event.component, event.stage)
                self._stage_totals[key] = (
                    self._stage_totals.get(key, 0) + event.duration_us
                )
            written = self._written + len(pending)
            if not self._artifacts_started or not _append_artifacts(pending, paths):
                with self._lock:
                    rows = tuple(self._events[:written])
                _write_event_artifacts(rows, paths)
            paths[2].write_text(
                _render_report(self._stage_totals, written), encoding="utf-8"
            )
            self._written = written
            self._artifacts_started = True
        return paths


def validate_pipeline_event(event: PipelineEvent) -> None:
    """Validate fields required for joining events across serving layers."""

    for name in (
        "run_id",
        "request_id",
        "backend",
        "model",
        "workload",
        "component",
        "stage",
    ):
        if not getattr(event, name):
            raise ValueError(f"pipeline event {name} must not be empty")
    if event.offset_us < 0 or event.duration_us < 0:
        raise ValueError("pipeline event timestamps and durations must be non-negative")


def render_pipeline_report(events: Iterable[PipelineEvent]) -> str:
    """Render a stage summary while keeping diagnostic and benchmark claims separate."""

    rows = tuple(events)
    totals: dict[tuple[str, str], int] = {}
    for event in rows:
        key = (event.component, event.stage)
        totals[key] = totals.get(key, 0) + event.duration_us
    return _render_report(totals, len(rows))


def _render_report(totals: dict[tuple[str, str], int], event_count: int) -> str:
    lines = [
        "# Native MLX Inference Pipeline Profile",
        "",
        "> Diagnostic timing only. These profiled measurements are not benchmark rows.",
        "",
        "This report is whole-pipeline timing. Semantic inference-graph tracing and Metal paged-attention capture remain separate, narrower diagnostics.",
        "",
        "## Artifact classes",
        "",
        "- Low-overhead pipeline timing: this report and `pipeline-events.jsonl`.",
        "- Joined CPU timeline: `pipeline-trace.json` (Chrome Trace/Perfetto compatible).",
        "- Heavy Metal frame capture: optional `.gputrace`; captured wall-clock is not benchmark latency.",
        "- Whole-process CPU/GPU evidence: optional Instruments or `xctrace` Metal System Trace.",
        "- Fair benchmark measurements: run separately with profiling disabled.",
        "",
        "## Stage totals",
        "",
        "| Component | Stage | Total ms |",
        "| --- | --- | ---: |",
    ]
    lines.extend(
        f"| {component} | {stage} | {duration / 1000:.3f} |"
        for (component, stage), duration in sorted(
            totals.items(), key=lambda item: item[1], reverse=True
        )
    )
    lines.extend(["", f"Events: {event_count}", ""])
    return "\n".join(lines)


def write_pipeline_artifacts(
    events: Iterable[PipelineEvent], output_dir: Path
) -> tuple[Path, Path, Path]:
    """Write a supplied joined event stream, including gateway-side events."""

    rows = tuple(events)
    paths = _artifact_paths(output_dir)
    _write_event_artifacts(rows, paths)
    paths[2].write_text(render_pipeline_report(rows), encoding="utf-8")
    return paths


# `json.dumps({"traceEvents": [...]}, indent=2) + "\n"` split into a fixed head,
# one block per event, and a fixed tail, so new events can be spliced in before
# the tail without re-serializing the events already on disk.
_TRACE_HEAD = '{\n  "traceEvents": ['
_TRACE_TAIL = "\n  ]\n}\n"
_EMPTY_TRACE = '{\n  "traceEvents": []\n}\n'


def _artifact_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "pipeline-events.jsonl",
        output_dir / "pipeline-trace.json",
        output_dir / "pipeline-report.md",
    )


def _write_event_artifacts(
    rows: tuple[PipelineEvent, ...], paths: tuple[Path, Path, Path]
) -> None:
    jsonl, trace, _ = paths
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text("".join(_jsonl_line(event) for event in rows), encoding="utf-8")
    entries = [_trace_entry(event) for event in rows]
    trace.write_text(
        _TRACE_HEAD + ",".join(f"\n{entry}" for entry in entries) + _TRACE_TAIL
        if entries
        else _EMPTY_TRACE,
        encoding="utf-8",
    )


def _append_artifacts(
    pending: list[PipelineEvent], paths: tuple[Path, Path, Path]
) -> bool:
    """Append events to existing artifacts; return False when a rewrite is needed."""

    jsonl, trace, _ = paths
    tail = _TRACE_TAIL.encode()
    try:
        with trace.open("r+b") as handle:
            size = handle.seek(0, os.SEEK_END)
            if size < len(tail):
                return False
            handle.seek(size - len(tail))
            if handle.read(len(tail)) != tail:
                return False
            handle.seek(size - len(tail))
            handle.write(
                (
                    "".join(f",\n{_trace_entry(event)}" for event in pending)
                    + _TRACE_TAIL
                ).encode()
            )
        with jsonl.open("a", encoding="utf-8") as handle:
            handle.write("".join(_jsonl_line(event) for event in pending))
    except FileNotFoundError:
        return False
    return True


def _jsonl_line(event: PipelineEvent) -> str:
    return json.dumps(asdict(event), sort_keys=True) + "\n"


def _trace_entry(event: PipelineEvent) -> str:
    return textwrap.indent(json.dumps(_trace_event(event), indent=2), "    ")


def _trace_event(event: PipelineEvent) -> dict[str, Any]:
    return {
        "name": event.stage,
        "cat": event.component,
        "ph": "X",
        "ts": event.monotonic_ns // 1_000,
        "dur": event.duration_us,
        "pid": event.run_id,
        "tid": event.request_id,
        "args": {
            "backend": event.backend,
            "model": event.model,
            "phase": event.phase,
            "state": event.state,
            **(event.details or {}),
        },
    }


__all__ = [
    "PipelineEvent",
    "PipelineProfiler",
    "render_pipeline_report",
    "validate_pipeline_event",
    "write_pipeline_artifacts",
]
