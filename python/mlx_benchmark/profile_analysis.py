"""Summarize, compare, and visualize native MLX pipeline profiles.

The functions here read the ``pipeline-events.jsonl`` stream written by the
native worker profiler (plus gateway-side events joined by ``mlx-air profile
run``) as plain JSON objects, so they do not import MLX or the worker runtime.
Profiled timings are diagnostic evidence only and never benchmark results.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

EVENTS_FILE = "pipeline-events.jsonl"
TRACE_FILE = "pipeline-trace.json"
REPORT_FILE = "pipeline-report.md"
HTML_FILE = "profile.html"
SUMMARY_FILE = "profile-summary.json"

# Spans that cover a whole request rather than one pipeline stage. They are
# reported as request latency and kept out of the stage breakdown so they do
# not dwarf the stages they contain.
ENVELOPE_STAGES = frozenset({("runtime", "terminal"), ("gateway", "http_round_trip")})

# Fixed component order for the timeline palette. Color follows the component,
# never its rank, so the same component has the same color in every profile.
COMPONENT_ORDER = (
    "mlx",
    "model",
    "scheduler",
    "cache",
    "executor",
    "sampling",
    "streaming",
    "runtime",
)

_REQUIRED_FIELDS = ("request_id", "component", "stage", "monotonic_ns", "duration_us")
_TIMELINE_REQUEST_LIMIT = 64
_TIMELINE_SPAN_LIMIT = 40_000
_UNCHANGED_FRACTION = 0.005


class ProfileError(ValueError):
    """Raised when a profile directory or event stream cannot be used."""


def events_path(path: Path) -> Path:
    """Resolve a profile directory or event file to its JSONL event stream.

    Args:
        path: A profile directory or a ``pipeline-events.jsonl`` file.

    Returns:
        The path of the JSONL event stream.

    Raises:
        ProfileError: The path does not contain a pipeline event stream.
    """
    candidate = path / EVENTS_FILE if path.is_dir() else path
    if not candidate.is_file():
        raise ProfileError(f"no pipeline event stream at {candidate}")
    return candidate


def load_events(path: Path) -> list[dict[str, Any]]:
    """Load pipeline events from a profile directory or JSONL file.

    Args:
        path: A profile directory or a ``pipeline-events.jsonl`` file.

    Returns:
        Events in file order.

    Raises:
        ProfileError: The stream is missing, malformed, or lacks a required field.
    """
    source = events_path(path)
    events: list[dict[str, Any]] = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProfileError(f"{source}:{number}: invalid JSON: {error}") from error
        if not isinstance(event, dict):
            raise ProfileError(f"{source}:{number}: event is not a JSON object")
        missing = [name for name in _REQUIRED_FIELDS if event.get(name) in (None, "")]
        if missing:
            raise ProfileError(
                f"{source}:{number}: event is missing {', '.join(missing)}"
            )
        events.append(event)
    return events


def is_warmup(request_id: str) -> bool:
    """Return whether a request ID belongs to worker startup warmup.

    Args:
        request_id: The event request ID.

    Returns:
        True for warmup requests issued before the server reports ready.
    """
    return request_id.startswith("warmup")


def select_events(
    events: Iterable[dict[str, Any]], *, include_warmup: bool = False
) -> list[dict[str, Any]]:
    """Drop warmup events unless they are explicitly requested.

    Args:
        events: Loaded pipeline events.
        include_warmup: Keep warmup requests in the selection.

    Returns:
        The selected events.
    """
    return [
        event
        for event in events
        if include_warmup or not is_warmup(str(event["request_id"]))
    ]


def summarize(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate events into per-request latency and per-stage totals.

    Stage totals add each event's duration. Batched stages are recorded once
    per request in the batch, so a stage shared by a batch of N requests is
    counted N times; per-request means are therefore comparable between
    profiles captured with the same concurrency.

    A request bracketed by a Metal frame capture runs many times slower than
    normal. When other requests exist, captured requests are kept in the
    request list but excluded from the means and stage totals.

    Args:
        events: Selected pipeline events.

    Returns:
        A JSON-serializable summary with ``totals``, ``requests``, and ``stages``.
    """
    captured = {
        str(event["request_id"])
        for event in events
        if (event["component"], event["stage"]) == ("metal", "capture_start")
    }
    all_ids = {str(event["request_id"]) for event in events}
    excluded = captured if all_ids - captured else set()
    requests: dict[str, dict[str, Any]] = {}
    stages: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        request_id = str(event["request_id"])
        component = str(event["component"])
        stage = str(event["stage"])
        start_ns = int(event["monotonic_ns"])
        duration_us = int(event["duration_us"])
        request = requests.setdefault(
            request_id,
            {
                "request_id": request_id,
                "start_ns": start_ns,
                "end_ns": start_ns,
                "state": None,
                "error": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "latency_ms": None,
                "http_round_trip_ms": None,
                "ttft_ms": None,
                "queue_ms": None,
                "prefill_ms": None,
                "decode_ms": None,
                "metal_capture": request_id in captured,
                "event_count": 0,
            },
        )
        request["start_ns"] = min(request["start_ns"], start_ns)
        request["end_ns"] = max(request["end_ns"], start_ns + duration_us * 1_000)
        request["event_count"] += 1
        if (component, stage) == ("runtime", "terminal"):
            details = event.get("details") or {}
            request["state"] = event.get("state")
            request["error"] = event.get("error")
            request["prompt_tokens"] = event.get("prompt_tokens")
            request["completion_tokens"] = event.get("completion_tokens")
            request["latency_ms"] = duration_us / 1_000 if duration_us else None
            request["ttft_ms"] = details.get("ttft_ms")
            request["queue_ms"] = details.get("scheduler_queue_wait_ms")
            request["prefill_ms"] = details.get("prefill_time_ms")
            request["decode_ms"] = details.get("decode_time_ms")
        elif (component, stage) == ("gateway", "http_round_trip"):
            request["http_round_trip_ms"] = duration_us / 1_000
        if (component, stage) in ENVELOPE_STAGES or request_id in excluded:
            continue
        row = stages.setdefault(
            (component, stage),
            {
                "component": component,
                "stage": stage,
                "total_ms": 0.0,
                "count": 0,
                "requests": set(),
            },
        )
        row["total_ms"] += duration_us / 1_000
        row["count"] += 1
        row["requests"].add(request_id)

    request_rows = sorted(requests.values(), key=lambda item: item["start_ns"])
    request_count = len(request_rows)
    measured = [row for row in request_rows if row["request_id"] not in excluded]
    measured_count = len(measured)
    stage_rows = []
    for row in sorted(stages.values(), key=lambda item: item["total_ms"], reverse=True):
        stage_rows.append(
            {
                "component": row["component"],
                "stage": row["stage"],
                "total_ms": round(row["total_ms"], 3),
                "count": row["count"],
                "request_count": len(row["requests"]),
                "mean_ms_per_request": round(row["total_ms"] / measured_count, 3)
                if measured_count
                else 0.0,
            }
        )
    origin = min((row["start_ns"] for row in request_rows), default=0)
    end = max((row["end_ns"] for row in request_rows), default=0)
    first = events[0] if events else {}
    return {
        "diagnostic_only": True,
        "totals": {
            "request_count": request_count,
            "measured_request_count": measured_count,
            "metal_captured_requests": len(captured),
            "metal_capture_excluded": bool(excluded),
            "event_count": len(events),
            "wall_ms": round((end - origin) / 1_000_000, 3),
            "completion_tokens": sum(
                int(row["completion_tokens"] or 0) for row in request_rows
            ),
            "failed_requests": sum(
                row["state"] not in (None, "completed", "stop", "length")
                for row in request_rows
            ),
            "mean_latency_ms": _mean(row["latency_ms"] for row in measured),
            "mean_http_round_trip_ms": _mean(
                row["http_round_trip_ms"] for row in measured
            ),
            "mean_ttft_ms": _mean(row["ttft_ms"] for row in measured),
            "run_id": first.get("run_id"),
            "model": first.get("model"),
            "workload": first.get("workload"),
        },
        "requests": [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"start_ns", "end_ns"}
                },
                "offset_ms": round((row["start_ns"] - origin) / 1_000_000, 3),
                "span_ms": round((row["end_ns"] - row["start_ns"]) / 1_000_000, 3),
            }
            for row in request_rows
        ],
        "stages": stage_rows,
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return round(sum(present) / len(present), 3) if present else None


def write_trace(events: Sequence[dict[str, Any]], path: Path) -> None:
    """Write a Chrome Trace/Perfetto timeline for joined events.

    Args:
        events: Pipeline events, including gateway-side events.
        path: Destination ``pipeline-trace.json`` path.
    """
    trace = [
        {
            "name": event["stage"],
            "cat": event["component"],
            "ph": "X",
            "ts": int(event["monotonic_ns"]) // 1_000,
            "dur": int(event["duration_us"]),
            "pid": event.get("run_id") or "profile",
            "tid": event["request_id"],
            "args": {
                "backend": event.get("backend"),
                "model": event.get("model"),
                "phase": event.get("phase"),
                "state": event.get("state"),
                **(event.get("details") or {}),
            },
        }
        for event in events
    ]
    path.write_text(json.dumps({"traceEvents": trace}, indent=2) + "\n", "utf-8")


def write_events(events: Sequence[dict[str, Any]], path: Path) -> None:
    """Write events as sorted-key JSON lines.

    Args:
        events: Pipeline events.
        path: Destination ``pipeline-events.jsonl`` path.
    """
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events), "utf-8"
    )


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a stage report for a profile summary.

    Args:
        summary: Output of :func:`summarize`.

    Returns:
        Markdown text.
    """
    totals = summary["totals"]
    lines = [
        "# Native MLX Inference Pipeline Profile",
        "",
        "> Diagnostic timing only. These profiled measurements are not benchmark rows.",
        "",
        f"- Model: `{totals['model']}`",
        f"- Requests: {totals['request_count']} ({totals['failed_requests']} failed)",
        f"- Mean request latency: {_ms(totals['mean_latency_ms'])} (lower is better)",
        f"- Mean TTFT: {_ms(totals['mean_ttft_ms'])} (lower is better)",
        f"- Events: {totals['event_count']}",
        "",
        "Batched stages are recorded once per request in the batch.",
        "",
        *_capture_note(totals),
        "## Stage totals",
        "",
        "| Component | Stage | Total ms | Mean ms / request | Events |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {row['component']} | {row['stage']} | {row['total_ms']:.3f} | "
        f"{row['mean_ms_per_request']:.3f} | {row['count']} |"
        for row in summary["stages"]
    )
    lines.append("")
    return "\n".join(lines)


def _capture_note(totals: dict[str, Any]) -> list[str]:
    captured = totals.get("metal_captured_requests") or 0
    if not captured:
        return []
    if totals.get("metal_capture_excluded"):
        return [
            (
                f"{captured} request(s) ran under Metal frame capture and are excluded "
                "from the means and stage totals."
            ),
            "",
        ]
    return [
        (
            "Every request ran under Metal frame capture; the means and stage totals "
            "include capture overhead."
        ),
        "",
    ]


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f} ms"


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_label: str,
    candidate_label: str,
) -> str:
    """Render a per-stage comparison between two profile summaries.

    Every delta is candidate minus baseline in milliseconds per request. Lower
    time is better, so a negative delta is reported as better.

    Args:
        baseline: Summary of the baseline profile.
        candidate: Summary of the candidate profile.
        baseline_label: Name shown for the baseline.
        candidate_label: Name shown for the candidate.

    Returns:
        Markdown text.
    """
    lines = [
        "# Pipeline Profile Comparison",
        "",
        "> Diagnostic timing only. Profiled time is not benchmark latency; confirm",
        "> any conclusion with `mlx-air bench run` results.",
        "",
        (
            f"- Baseline: `{baseline_label}` "
            f"({baseline['totals']['request_count']} requests)"
        ),
        (
            f"- Candidate: `{candidate_label}` "
            f"({candidate['totals']['request_count']} requests)"
        ),
        "- Metric: mean milliseconds per request; lower is better.",
        "- Delta: candidate minus baseline; percentages are relative to the baseline.",
        "",
        *[
            f"- {name}: {line}"
            for name, summary in (("Baseline", baseline), ("Candidate", candidate))
            for line in _capture_note(summary["totals"])[:1]
        ],
        "",
        "## Request latency",
        "",
        "| Metric | Baseline ms | Candidate ms | Delta ms | Delta % | Verdict |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for label, key in (
        ("request latency (worker)", "mean_latency_ms"),
        ("HTTP round trip", "mean_http_round_trip_ms"),
        ("TTFT", "mean_ttft_ms"),
    ):
        lines.append(
            _comparison_row(label, baseline["totals"][key], candidate["totals"][key])
        )
    lines.extend(
        [
            "",
            "## Stages",
            "",
            "| Component | Stage | Baseline ms | Candidate ms | Delta ms | Delta % | Verdict |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    base_stages = {(row["component"], row["stage"]): row for row in baseline["stages"]}
    cand_stages = {(row["component"], row["stage"]): row for row in candidate["stages"]}
    keys = sorted(
        set(base_stages) | set(cand_stages),
        key=lambda key: max(
            (base_stages.get(key) or {}).get("mean_ms_per_request", 0.0),
            (cand_stages.get(key) or {}).get("mean_ms_per_request", 0.0),
        ),
        reverse=True,
    )
    for component, stage in keys:
        base = base_stages.get((component, stage))
        cand = cand_stages.get((component, stage))
        lines.append(
            f"| {component} "
            + _comparison_row(
                stage,
                base["mean_ms_per_request"] if base else None,
                cand["mean_ms_per_request"] if cand else None,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _comparison_row(label: str, baseline: float | None, candidate: float | None) -> str:
    if baseline is None or candidate is None:
        verdict = "only in candidate" if baseline is None else "only in baseline"
        if baseline is None and candidate is None:
            verdict = "not recorded"
        return (
            f"| {label} | {_cell(baseline)} | {_cell(candidate)} | n/a | n/a | "
            f"{verdict} |"
        )
    delta = candidate - baseline
    percent = f"{delta / baseline * 100:+.1f}%" if baseline else "n/a"
    if abs(delta) <= abs(baseline) * _UNCHANGED_FRACTION:
        verdict = "unchanged"
    elif delta < 0:
        verdict = "better (lower)"
    else:
        verdict = "worse (higher)"
    return (
        f"| {label} | {baseline:.3f} | {candidate:.3f} | {delta:+.3f} | {percent} | "
        f"{verdict} |"
    )


def _cell(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def timeline_spans(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Prepare bounded timeline rows for the HTML view.

    Args:
        events: Selected pipeline events.

    Returns:
        Request IDs, components, and spans as compact arrays with
        microsecond offsets from the earliest event, plus truncation flags.
    """
    starts: dict[str, int] = {}
    for event in events:
        request_id = str(event["request_id"])
        start = int(event["monotonic_ns"])
        starts[request_id] = min(starts.get(request_id, start), start)
    ordered = sorted(starts, key=starts.__getitem__)
    shown = ordered[:_TIMELINE_REQUEST_LIMIT]
    index = {request_id: position for position, request_id in enumerate(shown)}
    origin = min(starts.values(), default=0)
    components: list[str] = []
    spans: list[list[Any]] = []
    for event in sorted(events, key=lambda item: int(item["monotonic_ns"])):
        request_id = str(event["request_id"])
        if request_id not in index:
            continue
        if len(spans) >= _TIMELINE_SPAN_LIMIT:
            break
        component = str(event["component"])
        if component not in components:
            components.append(component)
        spans.append(
            [
                index[request_id],
                component,
                str(event["stage"]),
                (int(event["monotonic_ns"]) - origin) // 1_000,
                int(event["duration_us"]),
                event.get("phase"),
                event.get("state"),
            ]
        )
    ordered_components = [name for name in COMPONENT_ORDER if name in components]
    ordered_components += sorted(set(components) - set(ordered_components))
    return {
        "requests": shown,
        "components": ordered_components,
        "spans": spans,
        "truncated_requests": len(ordered) - len(shown),
        "truncated_spans": len(spans) >= _TIMELINE_SPAN_LIMIT,
    }


def render_html(summary: dict[str, Any], events: Sequence[dict[str, Any]]) -> str:
    """Render a self-contained interactive HTML view of one profile.

    The page embeds its data and script and loads nothing from the network.

    Args:
        summary: Output of :func:`summarize` for ``events``.
        events: Selected pipeline events.

    Returns:
        HTML text.
    """
    payload = {
        "summary": summary,
        "timeline": timeline_spans(events),
        "component_order": list(COMPONENT_ORDER),
    }
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(f"Pipeline profile · {summary['totals'].get('model') or ''}")
    return _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data)


def write_profile_outputs(
    events: Sequence[dict[str, Any]], directory: Path
) -> tuple[dict[str, Any], Path]:
    """Write the report, summary, and HTML view for selected events.

    Args:
        events: Selected pipeline events.
        directory: Directory that receives the outputs.

    Returns:
        The summary and the HTML path.
    """
    summary = summarize(events)
    (directory / REPORT_FILE).write_text(render_markdown(summary), "utf-8")
    (directory / SUMMARY_FILE).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", "utf-8"
    )
    html_path = directory / HTML_FILE
    html_path.write_text(render_html(summary, events), "utf-8")
    return summary, html_path


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10); --bar: #2a78d6; --other: #898781;
  --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100;
  --c5: #e87ba4; --c6: #008300; --c7: #6250d6; --c8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --bar: #3987e5; --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
    --c5: #d55181; --c6: #0ca30c; --c7: #9085e9; --c8: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --bar: #3987e5; --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500;
  --c5: #d55181; --c6: #0ca30c; --c7: #9085e9; --c8: #e66767;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 4px; }
p.note, .meta { color: var(--ink-2); margin: 0 0 12px; overflow-wrap: anywhere; }
section { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px; margin-top: 16px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 12px; margin-top: 16px; }
.tile { background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 12px 16px; }
.tile .label { color: var(--ink-2); font-size: 12px; }
.tile .value { font-size: 24px; font-variant-numeric: tabular-nums; }
.tile .hint { color: var(--muted); font-size: 12px; }
.banner { border-left: 3px solid var(--axis); padding: 8px 12px; color: var(--ink-2);
  background: var(--surface); margin-top: 12px; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--grid);
  white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; }
td.num, th.num { text-align: right; }
.bars { display: grid; grid-template-columns: minmax(120px, 220px) 1fr 90px;
  gap: 4px 12px; align-items: center; }
.bars .name { color: var(--ink-2); overflow: hidden; text-overflow: ellipsis;
  white-space: nowrap; }
.bars .track { height: 14px; }
.bars .fill { height: 100%; background: var(--bar); border-radius: 0 4px 4px 0;
  min-width: 2px; }
.bars .val { text-align: right; font-variant-numeric: tabular-nums; }
.bars .row:hover .fill { filter: brightness(1.1); }
.bars .row { display: contents; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 8px 0;
  color: var(--ink-2); font-size: 12px; }
.legend span::before { content: ""; display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; vertical-align: -1px;
  background: var(--swatch); }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
  color: var(--ink-2); font-size: 12px; }
button { font: inherit; color: var(--ink); background: var(--surface);
  border: 1px solid var(--border); border-radius: 6px; padding: 4px 10px;
  cursor: pointer; }
#timeline-wrap { position: relative; margin-top: 8px; }
canvas { display: block; width: 100%; cursor: crosshair; }
#tip { position: fixed; pointer-events: none; display: none; z-index: 10;
  background: var(--surface); color: var(--ink); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.12); max-width: 320px; }
#tip b { display: block; margin-bottom: 2px; }
#tip .k { color: var(--ink-2); }
code { font-size: 12px; }
</style>
</head>
<body>
<main>
  <h1>Pipeline profile</h1>
  <div class="meta" id="meta"></div>
  <div class="banner">Diagnostic timing only: profiling adds overhead, so these
    numbers are not benchmark latency. Use <code>mlx-air bench run</code> for
    performance conclusions.</div>
  <div class="banner" id="capture-note" hidden></div>
  <div class="tiles" id="tiles"></div>

  <section>
    <h2>Where time goes</h2>
    <p class="note">Mean milliseconds per request by pipeline stage, largest first.
      Lower is better. A stage that runs once per batch is counted once for
      each request in that batch.</p>
    <div class="bars" id="bars"></div>
  </section>

  <section>
    <h2>Timeline</h2>
    <p class="note" id="timeline-note">One block per request, one row per
      component. Scroll to zoom, drag to pan, hover for details.</p>
    <div class="controls">
      <button id="reset" type="button">Reset zoom</button>
      <span id="range"></span>
    </div>
    <div class="legend" id="legend"></div>
    <div id="timeline-wrap"><canvas id="timeline"></canvas></div>
  </section>

  <section>
    <h2>Requests</h2>
    <div class="scroll"><table id="requests"></table></div>
  </section>

  <section>
    <h2>Stage table</h2>
    <div class="scroll"><table id="stages"></table></div>
    <p class="note" style="margin-top:12px">For a deeper look, open
      <code>pipeline-trace.json</code> from this folder in
      <a href="https://ui.perfetto.dev" target="_blank" rel="noopener">ui.perfetto.dev</a>.</p>
  </section>
</main>
<div id="tip"></div>
<script>
const DATA = __DATA__;
const S = DATA.summary, T = DATA.timeline, totals = S.totals;
const fmt = (v, d = 1) => v == null ? "n/a" : Number(v).toLocaleString(undefined,
  {minimumFractionDigits: d, maximumFractionDigits: d});
const esc = s => String(s ?? "").replace(/[&<>"]/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

document.getElementById("meta").textContent = [
  totals.model && `Model ${totals.model}`, totals.run_id && `Run ${totals.run_id}`,
  `${totals.event_count} events`, `${fmt(totals.wall_ms)} ms observed`
].filter(Boolean).join(" · ");

if (totals.metal_captured_requests) {
  const note = document.getElementById("capture-note");
  note.hidden = false;
  note.textContent = totals.metal_capture_excluded
    ? `${totals.metal_captured_requests} request(s) ran under Metal frame capture and are excluded from the means and stage totals; they still appear in the timeline and request table.`
    : "Every request ran under Metal frame capture, so the means and stage totals include capture overhead.";
}
const tiles = [
  ["Requests", totals.request_count, totals.failed_requests
    ? `${totals.failed_requests} failed` : "all completed", 0],
  ["Mean request latency", totals.mean_latency_ms, "ms in worker · lower is better", 1],
  ["Mean HTTP round trip", totals.mean_http_round_trip_ms, "ms at client · lower is better", 1],
  ["Mean TTFT", totals.mean_ttft_ms, "ms · lower is better", 1],
  ["Completion tokens", totals.completion_tokens, "total generated", 0],
];
document.getElementById("tiles").innerHTML = tiles.map(([l, v, h, d]) =>
  `<div class="tile"><div class="label">${esc(l)}</div>` +
  `<div class="value">${fmt(v, d)}</div><div class="hint">${esc(h)}</div></div>`).join("");

const tip = document.getElementById("tip");
function showTip(e, htmlText) {
  tip.innerHTML = htmlText; tip.style.display = "block";
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + w > innerWidth - 8) x = e.clientX - w - 14;
  if (y + h > innerHeight - 8) y = e.clientY - h - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.display = "none"; };

const stages = S.stages.slice(0, 16);
const maxStage = Math.max(1e-9, ...stages.map(r => r.mean_ms_per_request));
const bars = document.getElementById("bars");
bars.innerHTML = stages.map((r, i) =>
  `<div class="row" data-i="${i}"><div class="name">${esc(r.component)} / ${esc(r.stage)}</div>` +
  `<div class="track"><div class="fill" style="width:${(r.mean_ms_per_request / maxStage * 100).toFixed(2)}%"></div></div>` +
  `<div class="val">${fmt(r.mean_ms_per_request, 2)} ms</div></div>`).join("") ||
  `<p class="note">No stage events were recorded.</p>`;
bars.querySelectorAll(".row").forEach(row => {
  const r = stages[+row.dataset.i];
  const text = `<b>${esc(r.component)} / ${esc(r.stage)}</b>` +
    `<span class="k">Mean per request</span> ${fmt(r.mean_ms_per_request, 3)} ms<br>` +
    `<span class="k">Total</span> ${fmt(r.total_ms, 3)} ms<br>` +
    `<span class="k">Events</span> ${r.count} across ${r.request_count} requests`;
  row.querySelectorAll("div").forEach(cell => {
    cell.addEventListener("mousemove", e => showTip(e, text));
    cell.addEventListener("mouseleave", hideTip);
  });
});

function table(id, columns, rows) {
  document.getElementById(id).innerHTML =
    "<thead><tr>" + columns.map(c => `<th class="${c[2] ? "num" : ""}">${esc(c[0])}</th>`).join("") +
    "</tr></thead><tbody>" + rows.map(r => "<tr>" + columns.map(c =>
      `<td class="${c[2] ? "num" : ""}">${c[2] ? fmt(r[c[1]], c[3] ?? 1) : esc(r[c[1]])}</td>`
    ).join("") + "</tr>").join("") + "</tbody>";
}
table("requests", [["Request", "request_id"], ["State", "state"],
  ["Start ms", "offset_ms", true], ["Latency ms", "latency_ms", true],
  ["HTTP ms", "http_round_trip_ms", true], ["TTFT ms", "ttft_ms", true],
  ["Queue ms", "queue_ms", true], ["Prefill ms", "prefill_ms", true],
  ["Decode ms", "decode_ms", true], ["Prompt tok", "prompt_tokens", true, 0],
  ["Output tok", "completion_tokens", true, 0], ["Metal capture", "capture"]],
  S.requests.map(r => ({...r, capture: r.metal_capture ? "yes" : ""})));
table("stages", [["Component", "component"], ["Stage", "stage"],
  ["Mean ms / request", "mean_ms_per_request", true, 3], ["Total ms", "total_ms", true, 3],
  ["Events", "count", true, 0], ["Requests", "request_count", true, 0]], S.stages);

// Timeline: one block per request, one row per component inside it.
const canvas = document.getElementById("timeline");
const ctx = canvas.getContext("2d");
const comps = T.components;
const slot = name => {
  const i = DATA.component_order.indexOf(name);
  return i >= 0 ? `--c${i + 1}` : "--other";
};
const unslotted = comps.filter(c => slot(c) === "--other");
document.getElementById("legend").innerHTML = comps.filter(c => slot(c) !== "--other")
  .map(c => `<span style="--swatch:var(${slot(c)})">${esc(c)}</span>`).join("") +
  (unslotted.length ? `<span style="--swatch:var(--other)">other: ${esc(unslotted.join(", "))}</span>` : "");
if (T.truncated_requests || T.truncated_spans) {
  document.getElementById("timeline-note").textContent +=
    ` Showing the first ${T.requests.length} requests` +
    (T.truncated_spans ? " and a bounded number of spans." : ".");
}
const ROW = 12, GAP = 2, HEADER = 16, BLOCK_GAP = 10, LEFT = 150, TOP = 22;
const rowsPerBlock = Math.max(1, comps.length);
const blockHeight = HEADER + rowsPerBlock * (ROW + GAP) + BLOCK_GAP;
let endUs = 1;
for (const s of T.spans) endUs = Math.max(endUs, s[3] + Math.max(s[4], 1));
let view = [0, endUs];
let layout = [];

function draw() {
  const width = canvas.parentElement.clientWidth;
  const height = TOP + Math.max(1, T.requests.length) * blockHeight;
  const dpr = devicePixelRatio || 1;
  canvas.width = width * dpr; canvas.height = height * dpr;
  canvas.style.height = height + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const plotW = Math.max(10, width - LEFT - 8);
  const x = us => LEFT + (us - view[0]) / (view[1] - view[0]) * plotW;
  ctx.font = "11px system-ui, -apple-system, sans-serif";
  ctx.textBaseline = "middle";
  // Grid and axis ticks.
  const span = view[1] - view[0];
  const step = Math.pow(10, Math.floor(Math.log10(span / 6)));
  const tick = [1, 2, 5, 10].map(m => m * step).find(t => span / t <= 8) || step * 10;
  ctx.strokeStyle = css("--grid"); ctx.fillStyle = css("--muted"); ctx.lineWidth = 1;
  for (let t = Math.ceil(view[0] / tick) * tick; t <= view[1]; t += tick) {
    const px = Math.round(x(t)) + 0.5;
    ctx.beginPath(); ctx.moveTo(px, TOP - 4); ctx.lineTo(px, height); ctx.stroke();
    ctx.fillText(`${fmt(t / 1000, tick < 1000 ? 2 : 0)} ms`, px + 3, 9);
  }
  // Row labels.
  T.requests.forEach((id, r) => {
    const top = TOP + r * blockHeight;
    ctx.fillStyle = css("--ink");
    ctx.fillText(`Request ${id}`, 4, top + HEADER / 2 - 1);
    ctx.fillStyle = css("--muted");
    comps.forEach((c, k) => {
      ctx.fillText(c, 14, top + HEADER + k * (ROW + GAP) + ROW / 2);
    });
  });
  ctx.save(); ctx.beginPath(); ctx.rect(LEFT, 0, plotW, height); ctx.clip();
  layout = [];
  for (const s of T.spans) {
    const [r, comp, , start, dur] = s;
    const x0 = x(start), x1 = x(start + dur);
    if (x1 < LEFT || x0 > LEFT + plotW) continue;
    const y = TOP + r * blockHeight + HEADER + comps.indexOf(comp) * (ROW + GAP);
    const w = Math.max(1.5, x1 - x0);
    ctx.fillStyle = css(slot(comp));
    ctx.fillRect(x0, y, w, ROW);
    layout.push([x0, y, w, s]);
  }
  ctx.restore();
  document.getElementById("range").textContent =
    `${fmt(view[0] / 1000, 2)} – ${fmt(view[1] / 1000, 2)} ms`;
}

function hit(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left, my = e.clientY - rect.top;
  let best = null;
  for (const item of layout) {
    const [x0, y, w] = item;
    if (my >= y - 1 && my <= y + ROW + 1 && mx >= x0 - 3 && mx <= x0 + w + 3) {
      if (!best || w < best[2]) best = item;
    }
  }
  return best;
}
let drag = null;
canvas.addEventListener("mousemove", e => {
  if (drag) {
    const plotW = canvas.clientWidth - LEFT - 8;
    const shift = (drag.x - e.clientX) / plotW * (drag.view[1] - drag.view[0]);
    view = [drag.view[0] + shift, drag.view[1] + shift];
    draw(); hideTip(); return;
  }
  const item = hit(e);
  if (!item) { hideTip(); return; }
  const [r, comp, stage, start, dur, phase, state] = item[3];
  showTip(e, `<b>${esc(comp)} / ${esc(stage)}</b>` +
    `<span class="k">Request</span> ${esc(T.requests[r])}<br>` +
    `<span class="k">Duration</span> ${fmt(dur / 1000, 3)} ms<br>` +
    `<span class="k">Start</span> ${fmt(start / 1000, 3)} ms` +
    (phase ? `<br><span class="k">Phase</span> ${esc(phase)}` : "") +
    (state ? `<br><span class="k">State</span> ${esc(state)}` : ""));
});
canvas.addEventListener("mouseleave", hideTip);
canvas.addEventListener("mousedown", e => { drag = {x: e.clientX, view: view.slice()}; });
addEventListener("mouseup", () => { drag = null; });
canvas.addEventListener("wheel", e => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const plotW = rect.width - LEFT - 8;
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left - LEFT) / plotW));
  const center = view[0] + frac * (view[1] - view[0]);
  const scale = Math.exp(e.deltaY * 0.0015);
  const width = Math.min(endUs * 1.05, Math.max(5, (view[1] - view[0]) * scale));
  view = [center - frac * width, center + (1 - frac) * width];
  draw();
}, {passive: false});
document.getElementById("reset").addEventListener("click", () => { view = [0, endUs]; draw(); });
addEventListener("resize", draw);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
draw();
</script>
</body>
</html>
"""
