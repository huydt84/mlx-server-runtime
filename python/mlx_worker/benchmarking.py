"""Benchmark reporting helpers for MLX runtime comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark measurement for a backend."""

    backend: str
    ttft_ms: float
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkRun:
    """Inputs used to assemble a benchmark report."""

    model: str
    prompt: str
    max_tokens: int
    results: tuple[BenchmarkResult, ...]
    generated_at: str


def now_utc_iso() -> str:
    """Return the current UTC timestamp in RFC 3339-ish form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_results(run: BenchmarkRun) -> str:
    """Render a markdown benchmark report."""

    return "\n".join(_render_run(run, include_heading=True))


def summarize_report(runs: Sequence[BenchmarkRun]) -> str:
    """Render a markdown report for multiple benchmark runs."""

    if not runs:
        raise ValueError("benchmark report requires at least one run")

    lines = ["# Phase 6 Benchmark Report", ""]
    for index, run in enumerate(runs):
        if index:
            lines.append("")
        lines.extend([f"## Model: {run.model}", ""])
        lines.extend(_render_run(run, include_heading=False))
    return "\n".join(lines)


def write_report_suite(path: Path, runs: Sequence[BenchmarkRun]) -> None:
    """Write a combined markdown benchmark report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summarize_report(runs), encoding="utf-8")


def _render_run(run: BenchmarkRun, *, include_heading: bool) -> list[str]:
    if not run.results:
        raise ValueError("benchmark run requires at least one result")

    raw = next(
        (result for result in run.results if result.backend == "raw mlx-lm"), None
    )
    rows = []
    for result in run.results:
        rows.append(
            {
                "backend": result.backend,
                "ttft_ms": _format_number(result.ttft_ms),
                "latency_ms": _format_number(result.latency_ms),
                "prompt_tokens": str(result.prompt_tokens),
                "completion_tokens": str(result.completion_tokens),
                "ttft_overhead_ms": _format_overhead(
                    result.ttft_ms, raw.ttft_ms if raw else None
                ),
                "latency_overhead_ms": _format_overhead(
                    result.latency_ms, raw.latency_ms if raw else None
                ),
                "notes": "; ".join(result.notes) if result.notes else "-",
            }
        )

    lines = [
        "# Phase 6 Benchmark Report",
        "",
        f"- generated_at: {run.generated_at}",
        f"- model: {run.model}",
        f"- max_tokens: {run.max_tokens}",
        f"- prompt: {run.prompt}",
        "",
        "## Results",
        "",
        "| backend | ttft_ms | latency_ms | prompt_tokens | completion_tokens | ttft_overhead_ms | latency_overhead_ms | notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {ttft_ms} | {latency_ms} | {prompt_tokens} | {completion_tokens} | {ttft_overhead_ms} | {latency_overhead_ms} | {notes} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Overhead Summary",
            "",
            _overhead_summary(raw, run.results),
            "",
            "## Observability / Control",
            "",
            "- raw mlx-lm: direct execution path with no HTTP serving surface.",
            "- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.",
            "- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.",
            "",
        ]
    )

    if not include_heading:
        return lines[2:]
    return lines


def write_report(path: Path, run: BenchmarkRun) -> None:
    """Write a markdown benchmark report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summarize_results(run), encoding="utf-8")


def _overhead_summary(
    raw: BenchmarkResult | None, results: Sequence[BenchmarkResult]
) -> str:
    if raw is None:
        return "Raw mlx-lm baseline was not recorded."

    slower = [
        result
        for result in results
        if result.backend != raw.backend and result.latency_ms > raw.latency_ms
    ]
    if not slower:
        return "No backend exceeded the raw mlx-lm baseline in measured latency."

    worst = max(slower, key=lambda result: result.latency_ms - raw.latency_ms)
    return (
        f"{worst.backend} was {worst.latency_ms - raw.latency_ms:.1f} ms slower than raw mlx-lm "
        f"on total latency and {worst.ttft_ms - raw.ttft_ms:.1f} ms slower on TTFT."
    )


def _format_number(value: float) -> str:
    return f"{value:.1f}"


def _format_overhead(value: float, baseline: float | None) -> str:
    if baseline is None:
        return "-"
    delta = value - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}"
