"""Tests for ``mlx-air profile`` capture, analysis, and visualization."""

from __future__ import annotations

import json
import os
import stat
import sys
import textwrap
from pathlib import Path

import pytest

from mlx_benchmark import profile_analysis as analysis
from mlx_benchmark import profiler


def _event(
    request_id: str,
    component: str,
    stage: str,
    start_ns: int,
    duration_us: int,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-1",
        "request_id": request_id,
        "backend": "native-mlx",
        "model": "test-model",
        "workload": "test",
        "component": component,
        "stage": stage,
        "monotonic_ns": start_ns,
        "offset_us": 0,
        "duration_us": duration_us,
        **extra,
    }


def _write_profile(directory: Path, events: list[dict[str, object]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    analysis.write_events(events, directory / analysis.EVENTS_FILE)
    return directory


def _request_events(request_id: str, origin: int, forward_us: int) -> list[dict]:
    return [
        _event(request_id, "scheduler", "select_work", origin, 100, phase="prefill"),
        _event(request_id, "model", "forward_dispatch", origin + 200_000, forward_us),
        _event(request_id, "mlx", "synchronize_eval", origin + 400_000, 2_000),
        _event(
            request_id,
            "runtime",
            "terminal",
            origin,
            10_000,
            state="completed",
            prompt_tokens=12,
            completion_tokens=4,
            details={"ttft_ms": 3, "scheduler_queue_wait_ms": 1, "decode_time_ms": 6},
        ),
    ]


def test_summary_keeps_request_envelopes_out_of_stage_totals() -> None:
    events = _request_events("a", 1_000_000, 4_000) + _request_events(
        "b", 2_000_000, 6_000
    )

    summary = analysis.summarize(events)

    assert summary["totals"]["request_count"] == 2
    assert summary["totals"]["mean_latency_ms"] == 10.0
    assert summary["totals"]["mean_ttft_ms"] == 3.0
    assert summary["totals"]["completion_tokens"] == 8
    assert summary["totals"]["failed_requests"] == 0
    stages = {(row["component"], row["stage"]): row for row in summary["stages"]}
    assert ("runtime", "terminal") not in stages
    assert stages[("model", "forward_dispatch")]["total_ms"] == 10.0
    assert stages[("model", "forward_dispatch")]["mean_ms_per_request"] == 5.0
    assert summary["stages"][0]["stage"] == "forward_dispatch"


def test_warmup_requests_are_excluded_unless_requested() -> None:
    events = _request_events("warmup-0", 0, 1_000) + _request_events("a", 0, 1_000)

    assert {e["request_id"] for e in analysis.select_events(events)} == {"a"}
    assert len(analysis.select_events(events, include_warmup=True)) == len(events)


def test_load_events_reports_the_malformed_line(tmp_path: Path) -> None:
    path = tmp_path / analysis.EVENTS_FILE
    path.write_text(json.dumps(_event("a", "mlx", "x", 1, 1)) + "\n{broken\n")

    with pytest.raises(analysis.ProfileError, match=r":2: invalid JSON"):
        analysis.load_events(tmp_path)


def test_load_events_rejects_events_without_required_fields(tmp_path: Path) -> None:
    event = _event("a", "mlx", "x", 1, 1)
    del event["stage"]
    (tmp_path / analysis.EVENTS_FILE).write_text(json.dumps(event) + "\n")

    with pytest.raises(analysis.ProfileError, match="missing stage"):
        analysis.load_events(tmp_path)


def test_compare_names_baseline_candidate_and_direction() -> None:
    baseline = analysis.summarize(_request_events("a", 0, 8_000))
    candidate = analysis.summarize(_request_events("b", 0, 4_000))

    report = analysis.compare(
        baseline, candidate, baseline_label="before", candidate_label="after"
    )

    assert "- Baseline: `before`" in report
    assert "- Candidate: `after`" in report
    assert "lower is better" in report
    assert (
        "| model | forward_dispatch | 8.000 | 4.000 | -4.000 | -50.0% | better (lower) |"
        in report
    )
    assert (
        "| mlx | synchronize_eval | 2.000 | 2.000 | +0.000 | +0.0% | unchanged |"
        in report
    )


def test_compare_reports_stages_present_in_only_one_profile() -> None:
    baseline = analysis.summarize(_request_events("a", 0, 1_000))
    candidate = analysis.summarize(
        _request_events("b", 0, 1_000) + [_event("b", "cache", "commit", 0, 3_000)]
    )

    report = analysis.compare(
        baseline, candidate, baseline_label="a", candidate_label="b"
    )

    assert "| cache | commit | n/a | 3.000 | n/a | n/a | only in candidate |" in report


def test_html_is_self_contained_and_escapes_embedded_data() -> None:
    events = _request_events("</script><b>x", 0, 1_000)

    page = analysis.render_html(analysis.summarize(events), events)

    assert page.count("</script>") == 1
    assert "http://" not in page.split("<script>")[1]
    assert "<script src" not in page
    assert "Diagnostic timing only" in page


def test_timeline_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analysis, "_TIMELINE_REQUEST_LIMIT", 2)
    events = [
        e for index in range(4) for e in _request_events(f"r{index}", index, 1_000)
    ]

    timeline = analysis.timeline_spans(events)

    assert timeline["requests"] == ["r0", "r1"]
    assert timeline["truncated_requests"] == 2
    assert timeline["components"][:3] == ["mlx", "model", "scheduler"]


def test_view_writes_html_for_an_existing_profile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _write_profile(tmp_path / "p", _request_events("a", 0, 1_000))

    assert profiler.main(["view", str(directory)]) == 0

    html_path = directory / analysis.HTML_FILE
    assert capsys.readouterr().out.strip() == str(html_path)
    assert "forward_dispatch" in html_path.read_text()


def test_view_fails_for_a_profile_with_only_warmup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = _write_profile(tmp_path / "p", _request_events("warmup-0", 0, 1))

    assert profiler.main(["view", str(directory)]) == 50
    assert "no profiled requests" in capsys.readouterr().err


def test_compare_command_writes_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = _write_profile(tmp_path / "base", _request_events("a", 0, 2_000))
    cand = _write_profile(tmp_path / "cand", _request_events("b", 0, 1_000))
    output = tmp_path / "compare.md"

    status = profiler.main(["compare", str(base), str(cand), "--output", str(output)])

    assert status == 0
    assert output.read_text() == capsys.readouterr().out
    assert "better (lower)" in output.read_text()


def test_run_rejects_concurrency_above_request_count() -> None:
    with pytest.raises(SystemExit) as error:
        profiler.main(["run", "--model", "m", "--requests", "1", "--concurrency", "2"])

    assert error.value.code == 2


def test_run_requires_metal_capture_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gateway = tmp_path / "gateway"
    gateway.write_text("")
    monkeypatch.setenv("MLX_AIR_GATEWAY_EXECUTABLE", str(gateway))
    monkeypatch.delenv("MTL_CAPTURE_ENABLED", raising=False)

    assert profiler.main(["run", "--model", "m", "--metal"]) == 50
    assert "MTL_CAPTURE_ENABLED=1" in capsys.readouterr().err


def test_run_refuses_a_non_empty_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gateway = tmp_path / "gateway"
    gateway.write_text("")
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_text("x")
    monkeypatch.setenv("MLX_AIR_GATEWAY_EXECUTABLE", str(gateway))

    status = profiler.main(["run", "--model", "m", "--output-dir", str(output)])

    assert status == 50
    assert "not empty" in capsys.readouterr().err


_FAKE_GATEWAY = textwrap.dedent(
    """
    import json, os, re, time, tomllib
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    config = tomllib.loads(open(os.environ["MLX_RUNTIME_CONFIG"]).read())
    out = os.environ["MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR"]
    run_id = os.environ["MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID"]
    counter = [0]

    def record(request_id, component, stage, start, duration_us, **extra):
        event = {"schema_version": 1, "run_id": run_id, "request_id": request_id,
                 "backend": "native-mlx", "model": "m", "workload": "w",
                 "component": component, "stage": stage, "monotonic_ns": start,
                 "offset_us": 0, "duration_us": duration_us, **extra}
        with open(os.path.join(out, "pipeline-events.jsonl"), "a") as handle:
            handle.write(json.dumps(event) + "\\n")

    record("warmup-0", "model", "forward_dispatch", time.perf_counter_ns(), 50)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, payload, content_type="application/json"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"ready": True} if self.path == "/ready" else {"status": "ready"})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            counter[0] += 1
            request_id = f"req{counter[0]}"
            start = time.perf_counter_ns()
            record(request_id, "model", "forward_dispatch", start, 1500, phase="decode")
            record(request_id, "runtime", "terminal", start, 3000, state="completed",
                   prompt_tokens=5, completion_tokens=request["max_tokens"],
                   details={"ttft_ms": 1})
            usage = {"prompt_tokens": 5, "completion_tokens": request["max_tokens"]}
            if request["stream"]:
                chunks = [{"id": "chatcmpl-" + request_id, "choices": []},
                          {"id": "chatcmpl-" + request_id, "choices": [], "usage": usage}]
                body = "".join(f"data: {json.dumps(c)}\\n\\n" for c in chunks)
                self.reply((body + "data: [DONE]\\n\\n").encode(), "text/event-stream")
            else:
                self.reply({"id": "chatcmpl-" + request_id, "usage": usage})

    ThreadingHTTPServer(("127.0.0.1", config["server"]["port"]), Handler).serve_forever()
    """
)


@pytest.mark.parametrize("stream", [False, True])
def test_run_profiles_requests_and_joins_gateway_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stream: bool,
) -> None:
    gateway = tmp_path / "gateway"
    gateway.write_text(f"#!{sys.executable}\n{_FAKE_GATEWAY}")
    gateway.chmod(gateway.stat().st_mode | stat.S_IXUSR)
    output = tmp_path / "profile"
    monkeypatch.setenv("MLX_AIR_GATEWAY_EXECUTABLE", str(gateway))
    arguments = ["run", "--model", "m", "--requests", "3", "--concurrency", "2"]
    arguments += ["--max-tokens", "4", "--output-dir", str(output)]

    status = profiler.main(arguments + (["--stream"] if stream else []))

    captured = capsys.readouterr()
    assert status == 0, captured.err
    assert captured.out.strip() == str(output / analysis.HTML_FILE)
    events = analysis.load_events(output)
    gateway_events = [e for e in events if e["component"] == "gateway"]
    assert sorted(e["request_id"] for e in gateway_events) == ["req1", "req2", "req3"]
    assert [e["monotonic_ns"] for e in events] == sorted(
        e["monotonic_ns"] for e in events
    )
    summary = json.loads((output / analysis.SUMMARY_FILE).read_text())
    assert summary["totals"]["request_count"] == 3
    assert summary["totals"]["mean_http_round_trip_ms"] is not None
    trace = json.loads((output / analysis.TRACE_FILE).read_text())
    assert len(trace["traceEvents"]) == len(events)
    for name in ("profile.html", "pipeline-report.md", "runtime.toml", "gateway.log"):
        assert (output / name).is_file(), name
    assert 'backend = "native-mlx"' in (output / "runtime.toml").read_text()
    assert "max_active_requests = 2" in (output / "runtime.toml").read_text()


def test_run_reports_a_gateway_that_exits_before_readiness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gateway = tmp_path / "gateway"
    gateway.write_text(f"#!{sys.executable}\nraise SystemExit(3)\n")
    gateway.chmod(gateway.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("MLX_AIR_GATEWAY_EXECUTABLE", str(gateway))

    status = profiler.main(
        ["run", "--model", "m", "--output-dir", str(tmp_path / "out")]
    )

    assert status == 50
    assert "exited before readiness" in capsys.readouterr().err
    assert os.path.exists(tmp_path / "out" / "gateway.log")


def _captured(request_id: str, origin: int, forward_us: int) -> list[dict]:
    return [
        _event(request_id, "metal", "capture_start", origin, 0),
        *_request_events(request_id, origin, forward_us),
        _event(request_id, "metal", "capture_stop", origin + 500_000, 0),
    ]


def test_metal_captured_requests_are_excluded_from_means() -> None:
    events = _captured("a", 0, 900_000) + _request_events("b", 1_000_000, 2_000)

    summary = analysis.summarize(events)

    totals = summary["totals"]
    assert totals["request_count"] == 2
    assert totals["measured_request_count"] == 1
    assert totals["metal_captured_requests"] == 1
    assert totals["metal_capture_excluded"] is True
    stages = {(row["component"], row["stage"]): row for row in summary["stages"]}
    assert stages[("model", "forward_dispatch")]["mean_ms_per_request"] == 2.0
    assert ("metal", "capture_start") not in stages
    assert [row["metal_capture"] for row in summary["requests"]] == [True, False]
    assert "excluded from the means" in analysis.render_markdown(summary)


def test_metal_capture_is_kept_when_every_request_was_captured() -> None:
    summary = analysis.summarize(_captured("a", 0, 4_000))

    assert summary["totals"]["metal_capture_excluded"] is False
    assert summary["totals"]["mean_latency_ms"] == 10.0
    assert "include capture overhead" in analysis.render_markdown(summary)
