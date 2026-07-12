"""Benchmark reporting helpers for MLX runtime comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


P95_MIN_SAMPLES = 20
P99_MIN_SAMPLES = 100
MEAN_MEDIAN_WARN_FRACTION = 0.25
COMPLETION_TOKENS_WARN_FRACTION = 0.05


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregated benchmark measurements for one backend."""

    backend: str
    samples: int
    errors: int
    error_rate: float
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    ttft_p99_ms: float | None
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    prompt_tokens_mean: float | None
    completion_tokens_mean: float | None
    completion_tokens_p50: float | None
    total_tokens_mean: float | None
    decode_time_mean_ms: float | None
    latency_per_completion_token_ms: float | None
    decode_time_per_completion_token_ms: float | None
    latency_p50_per_completion_token_ms: float | None
    decode_tokens_per_second_mean: float | None
    decode_tokens_per_second_p50: float | None
    end_to_end_tokens_per_second_mean: float | None
    end_to_end_tokens_per_second_p50: float | None
    # VLM-specific fields — None for text-only benchmark runs.
    image_preprocess_latency_ms_mean: float | None = None
    """Mean wall-clock time spent on image content extraction (ms)."""
    image_count_mean: float | None = None
    """Mean number of images per request across successful samples."""
    vlm_load_time_ms: float | None = None
    """VLM model load time (ms) captured on warmup or first request."""
    image_load_ms_mean: float | None = None
    image_load_ms_p50: float | None = None
    image_load_ms_p95: float | None = None
    image_decode_ms_mean: float | None = None
    image_decode_ms_p50: float | None = None
    image_decode_ms_p95: float | None = None
    image_preprocess_latency_ms_p50: float | None = None
    image_preprocess_latency_ms_p95: float | None = None

    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def attempts(self) -> int:
        """Return total measured attempts including errors."""

        return self.samples + self.errors

    @property
    def ttft_ms(self) -> float | None:
        """Compatibility alias for the old ambiguous field name."""

        return self.ttft_mean_ms

    @property
    def latency_ms(self) -> float | None:
        """Compatibility alias for the old ambiguous field name."""

        return self.latency_mean_ms

    @property
    def prompt_tokens(self) -> float | None:
        """Compatibility alias for the old ambiguous field name."""

        return self.prompt_tokens_mean

    @property
    def completion_tokens(self) -> float | None:
        """Compatibility alias for the old ambiguous field name."""

        return self.completion_tokens_mean

    @property
    def prompt_tokens_per_request_mean(self) -> float | None:
        """Clearer label for markdown/report output."""

        return self.prompt_tokens_mean

    @property
    def completion_tokens_per_request_mean(self) -> float | None:
        """Clearer label for markdown/report output."""

        return self.completion_tokens_mean

    @property
    def total_tokens_per_request_mean(self) -> float | None:
        """Clearer label for markdown/report output."""

        return self.total_tokens_mean


@dataclass(frozen=True)
class BenchmarkRun:
    """Inputs used to assemble a benchmark report."""

    model: str
    prompt: str
    max_tokens: int
    results: tuple[BenchmarkResult, ...]
    generated_at: str
    benchmark_mode: str = "smoke"
    warmup_runs_per_fixture: int = 0
    measured_runs_per_fixture: int = 1
    scenario: str = "baseline"
    metadata: dict | None = None


@dataclass(frozen=True)
class VlmFixtureReportRow:
    backend: str
    fixture_name: str
    fixture_category: str
    image_count: int
    total_image_pixels: int | None
    total_megapixels: float | None
    widths_summary: str
    heights_summary: str
    formats_summary: str
    total_file_size_bytes: int | None
    prompt_preview: str | None
    prompt_text_source: str | None
    prompt_tokens_mean: float | None
    completion_tokens_mean: float | None
    total_tokens_mean: float | None
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    image_load_ms_mean: float | None
    image_decode_ms_mean: float | None
    image_preprocess_ms_mean: float | None
    image_preprocess_ms_p50: float | None
    image_preprocess_ms_p95: float | None
    decode_tps_mean: float | None
    e2e_tps_mean: float | None
    latency_per_completion_token_ms: float | None
    latency_per_image_ms: float | None
    latency_per_megapixel_ms: float | None
    samples: int
    errors: int
    error_rate: float
    max_tokens: int
    temperature: float
    top_p: float
    stop_reason_summary: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmStreamingBackendResult:
    backend: str
    mode: str
    streaming_available: bool
    samples: int
    errors: int
    error_rate: float
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    ttft_p95_ms: float | None
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    completion_tokens_mean: float | None
    prompt_tokens_mean: float | None
    sse_chunk_count_mean: float | None
    raw_chunk_count_mean: float | None
    sse_chunk_interval_mean_ms: float | None
    sse_chunk_interval_p50_ms: float | None
    sse_chunk_interval_p95_ms: float | None
    chunk_interval_mean_ms: float | None
    chunk_interval_p50_ms: float | None
    chunk_interval_p95_ms: float | None
    stream_completed_normally: bool | None
    finish_reason_distribution: str | None = None
    parse_errors: int = 0
    notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmCancellationResult:
    backend: str
    cancellation_attempts: int
    server_observed_cancellations: int | None
    cancellation_errors: int
    follow_up_success_count: int
    follow_up_error_count: int
    follow_up_latency_mean_ms: float | None
    worker_health_after_cancel: str | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmConcurrencyResult:
    backend: str
    concurrency: int
    completed_requests: int
    errors: int
    error_rate: float
    wall_clock_duration_ms: float | None
    requests_per_second: float | None
    completion_tokens_per_second: float | None
    ttft_mean_ms: float | None
    ttft_p50_ms: float | None
    latency_mean_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    queue_wait_mean_ms: float | None
    max_queue_depth: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VlmComparisonRow:
    scenario: str
    backend_a: str
    backend_b: str
    fairness_level: str
    comparison_kind: str
    same_model: bool
    same_scenario: bool
    same_benchmark_mode: bool
    same_fixture_set: bool
    same_max_tokens: bool
    same_temperature: bool
    same_top_p: bool
    same_streaming_mode: bool
    same_backend_category: bool
    prompt_tokens_mean_delta_pct: float | None
    completion_tokens_mean_delta_pct: float | None
    token_equivalent: bool | None
    error_rates_comparable: bool
    sufficient_samples: bool
    headline_eligible: bool
    reasons_not_headline_eligible: tuple[str, ...]
    latency_mean_delta_ms: float | None = None
    latency_p50_delta_ms: float | None = None
    latency_p95_delta_ms: float | None = None
    ttft_mean_delta_ms: float | None = None
    decode_tps_delta_pct: float | None = None
    e2e_tps_delta_pct: float | None = None


@dataclass(frozen=True)
class VlmScenarioRun:
    scenario: str
    benchmark_mode: str
    started_at: str
    ended_at: str
    fixture_count: int
    fixture_names: tuple[str, ...]
    warmup_runs_per_fixture: int
    measured_runs_per_fixture: int
    expected_measured_samples_per_backend: int | None
    backend_order: tuple[str, ...]
    order_rounds: int
    order_randomized: bool
    backend_order_seed: int | None
    aggregated_across_order_rounds: bool
    order_round_details: tuple[dict[str, object], ...]
    results: tuple[BenchmarkResult, ...] = ()
    fixture_rows: tuple[VlmFixtureReportRow, ...] = ()
    streaming_rows: tuple[VlmStreamingBackendResult, ...] = ()
    cancellation_rows: tuple[VlmCancellationResult, ...] = ()
    concurrency_rows: tuple[VlmConcurrencyResult, ...] = ()
    comparison_rows: tuple[VlmComparisonRow, ...] = ()
    fairness_warnings: tuple[str, ...] = ()
    interpretation: tuple[str, ...] = ()


def now_utc_iso() -> str:
    """Return the current UTC timestamp in RFC 3339-ish form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def mean(values: Sequence[float]) -> float | None:
    """Return the arithmetic mean or None for an empty sequence."""

    if not values:
        return None
    return sum(values) / len(values)


def percentile(
    values: Sequence[float], p: float, *, min_samples: int = 1
) -> float | None:
    """Return a linear-interpolated percentile or None when unavailable."""

    if len(values) < min_samples:
        return None
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def calculate_decode_tokens_per_second(
    completion_tokens: float, decode_time_ms: float
) -> float | None:
    """Return decode throughput or None when decode time is not positive."""

    if decode_time_ms <= 0:
        return None
    return completion_tokens / (decode_time_ms / 1000.0)


def calculate_end_to_end_tokens_per_second(
    completion_tokens: float, latency_ms: float
) -> float | None:
    """Return end-to-end throughput or None when latency is not positive."""

    if latency_ms <= 0:
        return None
    return completion_tokens / (latency_ms / 1000.0)


def calculate_per_token_latency_ms(
    latency_ms: float | None, completion_tokens: float | None
) -> float | None:
    """Return normalized latency in milliseconds per completion token."""

    if latency_ms is None or completion_tokens is None or completion_tokens <= 0:
        return None
    return latency_ms / completion_tokens


def calculate_overhead(value: float | None, baseline: float | None) -> float | None:
    """Return the absolute delta from baseline."""

    if value is None or baseline is None:
        return None
    return value - baseline


def calculate_overhead_percent(
    value: float | None, baseline: float | None
) -> float | None:
    """Return the percentage delta from baseline."""

    if value is None or baseline is None or baseline == 0:
        return None
    return ((value - baseline) / baseline) * 100.0


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


def write_report(path: Path, run: BenchmarkRun) -> None:
    """Write a markdown benchmark report to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summarize_results(run), encoding="utf-8")


def _render_run(run: BenchmarkRun, *, include_heading: bool) -> list[str]:
    if not run.results:
        raise ValueError("benchmark run requires at least one result")

    raw = next(
        (result for result in run.results if result.backend == "raw mlx-lm"), None
    )
    suite_warnings = _collect_suite_warnings(raw, run.results)

    lines = [
        "# Phase 6 Benchmark Report",
        "",
        "## Benchmark Configuration",
        "",
        f"- generated_at: {run.generated_at}",
        f"- model: {run.model}",
        f"- max_tokens: {run.max_tokens}",
        f"- prompt_suite: {run.prompt}",
        "",
        "## Metric Definitions",
        "",
        "- `samples`: successful measured requests included in aggregate statistics.",
        "- `errors`: measured requests that failed and were excluded from latency and token aggregates.",
        "- `error_rate = errors / (samples + errors)`.",
        "- `ttft_*`: time from request start until the first generated token arrives.",
        "- `latency_*`: end-to-end request time from request start until the final token or final response.",
        "- `decode_time_mean_ms = latency_mean_ms - ttft_mean_ms`.",
        "- `decode_tokens_per_second = completion_tokens / (decode_time_ms / 1000)` when decode time is positive.",
        "- `end_to_end_tokens_per_second = completion_tokens / (latency_ms / 1000)` when latency is positive.",
        "- `latency_per_completion_token_ms = latency_mean_ms / completion_tokens_per_request_mean`.",
        "- `decode_time_per_completion_token_ms = decode_time_mean_ms / completion_tokens_per_request_mean`.",
        "- `*_p95` is reported only when at least 20 successful samples exist; `*_p99` requires at least 100 successful samples.",
        "",
        "## Raw Per-Backend Metrics",
        "",
        "| backend | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_per_request_mean | completion_tokens_per_request_mean | total_tokens_per_request_mean | decode_time_mean_ms | latency_per_completion_token_ms | decode_time_per_completion_token_ms | latency_p50_per_completion_token_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in run.results:
        lines.append(
            "| {backend} | {samples} | {errors} | {error_rate} | {ttft_mean_ms} | {ttft_p50_ms} | {ttft_p95_ms} | {ttft_p99_ms} | {latency_mean_ms} | {latency_p50_ms} | {latency_p95_ms} | {latency_p99_ms} | {prompt_tokens_mean} | {completion_tokens_mean} | {total_tokens_mean} | {decode_time_mean_ms} | {latency_per_completion_token_ms} | {decode_time_per_completion_token_ms} | {latency_p50_per_completion_token_ms} |".format(
                backend=result.backend,
                samples=result.samples,
                errors=result.errors,
                error_rate=_format_percent(result.error_rate),
                ttft_mean_ms=_format_number(result.ttft_mean_ms),
                ttft_p50_ms=_format_number(result.ttft_p50_ms),
                ttft_p95_ms=_format_number(result.ttft_p95_ms),
                ttft_p99_ms=_format_number(result.ttft_p99_ms),
                latency_mean_ms=_format_number(result.latency_mean_ms),
                latency_p50_ms=_format_number(result.latency_p50_ms),
                latency_p95_ms=_format_number(result.latency_p95_ms),
                latency_p99_ms=_format_number(result.latency_p99_ms),
                prompt_tokens_mean=_format_number(
                    result.prompt_tokens_per_request_mean
                ),
                completion_tokens_mean=_format_number(
                    result.completion_tokens_per_request_mean
                ),
                total_tokens_mean=_format_number(result.total_tokens_per_request_mean),
                decode_time_mean_ms=_format_number(result.decode_time_mean_ms),
                latency_per_completion_token_ms=_format_number(
                    result.latency_per_completion_token_ms
                ),
                decode_time_per_completion_token_ms=_format_number(
                    result.decode_time_per_completion_token_ms
                ),
                latency_p50_per_completion_token_ms=_format_number(
                    result.latency_p50_per_completion_token_ms
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Throughput Metrics",
            "",
            "| backend | decode_tokens_per_second_mean | decode_tokens_per_second_p50 | end_to_end_tokens_per_second_mean | end_to_end_tokens_per_second_p50 |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in run.results:
        lines.append(
            "| {backend} | {decode_mean} | {decode_p50} | {e2e_mean} | {e2e_p50} |".format(
                backend=result.backend,
                decode_mean=_format_number(result.decode_tokens_per_second_mean),
                decode_p50=_format_number(result.decode_tokens_per_second_p50),
                e2e_mean=_format_number(result.end_to_end_tokens_per_second_mean),
                e2e_p50=_format_number(result.end_to_end_tokens_per_second_p50),
            )
        )

    lines.extend(
        [
            "",
            "## Overhead Vs Raw MLX-LM",
            "",
            "| backend | ttft_mean_overhead_ms | latency_mean_overhead_ms | ttft_p50_overhead_ms | latency_p50_overhead_ms | ttft_mean_overhead_percent | latency_mean_overhead_percent | decode_tps_delta_percent | e2e_tps_delta_percent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in run.results:
        lines.append(
            "| {backend} | {ttft_mean_delta} | {latency_mean_delta} | {ttft_p50_delta} | {latency_p50_delta} | {ttft_mean_delta_pct} | {latency_mean_delta_pct} | {decode_tps_delta_pct} | {e2e_tps_delta_pct} |".format(
                backend=result.backend,
                ttft_mean_delta=_format_signed_number(
                    calculate_overhead(
                        result.ttft_mean_ms, raw.ttft_mean_ms if raw else None
                    )
                ),
                latency_mean_delta=_format_signed_number(
                    calculate_overhead(
                        result.latency_mean_ms, raw.latency_mean_ms if raw else None
                    )
                ),
                ttft_p50_delta=_format_signed_number(
                    calculate_overhead(
                        result.ttft_p50_ms, raw.ttft_p50_ms if raw else None
                    )
                ),
                latency_p50_delta=_format_signed_number(
                    calculate_overhead(
                        result.latency_p50_ms, raw.latency_p50_ms if raw else None
                    )
                ),
                ttft_mean_delta_pct=_format_signed_percent(
                    calculate_overhead_percent(
                        result.ttft_mean_ms, raw.ttft_mean_ms if raw else None
                    )
                ),
                latency_mean_delta_pct=_format_signed_percent(
                    calculate_overhead_percent(
                        result.latency_mean_ms, raw.latency_mean_ms if raw else None
                    )
                ),
                decode_tps_delta_pct=_format_signed_percent(
                    calculate_overhead_percent(
                        result.decode_tokens_per_second_mean,
                        raw.decode_tokens_per_second_mean if raw else None,
                    )
                ),
                e2e_tps_delta_pct=_format_signed_percent(
                    calculate_overhead_percent(
                        result.end_to_end_tokens_per_second_mean,
                        raw.end_to_end_tokens_per_second_mean if raw else None,
                    )
                ),
            )
        )

    lines.extend(
        [
            "",
            "## Notes / Warnings",
            "",
        ]
    )
    entries = _collect_note_lines(run.results, suite_warnings)
    if entries:
        lines.extend(entries)
    else:
        lines.append("- no warnings")

    lines.extend(
        [
            "",
            "## Observability / Control",
            "",
            "- raw mlx-lm: direct execution path with no HTTP serving surface.",
            "- mlx_lm.server: HTTP serving, but no Rust control plane, queue admission, or gateway metrics in this repository's runtime model.",
            "- this project: Rust HTTP/SSE control plane, `/metrics`, request logs, queueing, cancellation, and worker supervision.",
            "",
            "## Overhead Summary",
            "",
            _overhead_summary(raw, run.results),
            "",
        ]
    )

    if not include_heading:
        return lines[2:]
    return lines


def _collect_note_lines(
    results: Sequence[BenchmarkResult], suite_warnings: Sequence[str]
) -> list[str]:
    lines: list[str] = []
    for warning in suite_warnings:
        lines.append(f"- suite warning: {warning}")
    for result in results:
        for warning in result.warnings:
            lines.append(f"- {result.backend}: {warning}")
        for note in result.notes:
            lines.append(f"- {result.backend}: {note}")
    return lines


def _collect_suite_warnings(
    raw: BenchmarkResult | None, results: Sequence[BenchmarkResult]
) -> tuple[str, ...]:
    warnings: list[str] = []
    if raw and raw.completion_tokens_per_request_mean is not None:
        for result in results:
            if (
                result.backend == raw.backend
                or result.completion_tokens_per_request_mean is None
            ):
                continue
            delta = abs(
                result.completion_tokens_per_request_mean
                - raw.completion_tokens_per_request_mean
            )
            ratio = (
                delta / raw.completion_tokens_per_request_mean
                if raw.completion_tokens_per_request_mean > 0
                else 0.0
            )
            if ratio > COMPLETION_TOKENS_WARN_FRACTION:
                warnings.append(
                    f"{result.backend} completion_tokens_per_request_mean differs from raw mlx-lm by {ratio * 100.0:.1f}%; prefer normalized per-token metrics over raw latency comparisons"
                )
    return tuple(warnings)


def _overhead_summary(
    raw: BenchmarkResult | None, results: Sequence[BenchmarkResult]
) -> str:
    if raw is None:
        return "Raw mlx-lm baseline was not recorded."
    slower = [
        result
        for result in results
        if result.backend != raw.backend
        and result.latency_mean_ms is not None
        and raw.latency_mean_ms is not None
        and result.latency_mean_ms > raw.latency_mean_ms
    ]
    if not slower:
        return "No backend exceeded the raw mlx-lm baseline in measured mean latency."
    worst = max(
        slower,
        key=lambda result: (
            (result.latency_mean_ms or 0.0) - (raw.latency_mean_ms or 0.0)
        ),
    )
    latency_delta = (
        calculate_overhead(worst.latency_mean_ms, raw.latency_mean_ms) or 0.0
    )
    ttft_delta = calculate_overhead(worst.ttft_mean_ms, raw.ttft_mean_ms) or 0.0
    return (
        f"{worst.backend} was {latency_delta:.1f} ms slower than raw mlx-lm "
        f"on mean latency and {ttft_delta:.1f} ms slower on mean TTFT."
    )


def _format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def _format_signed_number(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}"


def _format_overhead(value: float | None, baseline: float | None) -> str:
    """Compatibility helper for rendering absolute deltas."""

    return _format_signed_number(calculate_overhead(value, baseline))


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def _format_signed_percent(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def _format_na_percent(value: float | None) -> str:
    if value is None:
        return "not_available"
    return f"{value:.1f}%"


# ---------------------------------------------------------------------------
# VLM report helpers — separate from Phase 6 text benchmark report.
# ---------------------------------------------------------------------------


def write_vlm_report(path: Path, runs: Sequence[BenchmarkRun]) -> None:
    """Write a Phase 9 VLM benchmark report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_vlm_report(runs), encoding="utf-8")


def _render_vlm_report(runs: Sequence[BenchmarkRun]) -> str:
    """Render markdown VLM benchmark report with scenario isolation."""
    if not runs:
        raise ValueError("VLM benchmark report requires at least one run")

    lines = [
        "# Phase 9 — VLM Benchmark Report",
        "",
        "## Benchmark Configuration",
        "",
        "This report separates baseline, streaming, cancellation, and concurrency scenarios.",
        "Do not infer one single winner from `--scenario all`; each scenario isolates a different behavior.",
        "",
        "Backend roles:",
        "",
        "- **raw mlx-vlm**: direct Python reference. Useful context, not strict HTTP overhead baseline unless model-input/token equivalence is proven.",
        "- **mlx_vlm.server**: HTTP server reference for fair OpenAI-style HTTP comparison.",
        "- **this project**: Rust control plane with queueing, backpressure, cancellation, telemetry, and worker supervision.",
        "",
        "## Headline Fairness Rule",
        "",
        "A comparison may be used as headline speed claim only if all conditions are true:",
        "",
        "1. same model",
        "2. same scenario",
        "3. same benchmark mode",
        "4. same fixture set",
        "5. same max_tokens",
        "6. same temperature",
        "7. same top_p",
        "8. same streaming mode",
        "9. same backend category, or token-equivalent cross-category",
        "10. prompt_tokens_mean delta <= 5%",
        "11. completion_tokens_mean delta <= 5%",
        "12. error rates are comparable",
        "13. comparison is not contaminated by mixed stress scenario execution",
        "14. measured sample count is sufficient for reported statistic",
        "",
        "Fairness caveat: Image sizes, prompt templates, and output lengths can differ across fixtures; do not compare raw latency across different image sizes or token workloads.",
        "",
    ]

    for run_index, run in enumerate(runs):
        if run_index:
            lines.append("")
        lines.append(f"## Model: {run.model}")
        lines.append("")
        lines.append(f"- generated_at: {run.generated_at}")
        lines.append(f"- max_tokens: {run.max_tokens}")
        lines.append(f"- prompt_suite: {run.prompt}")
        lines.append("")
        for scenario_run in _scenario_runs_for_report(run):
            lines.extend(_render_vlm_scenario_section(run, scenario_run))
            lines.append("")

    lines.extend(
        [
            "## Final Interpretation",
            "",
            "Proven:",
            "- this project can serve VLM requests through Rust control plane.",
            "- this project can be compared fairly to `mlx_vlm.server` when HTTP token counts match.",
            "- stable mode can produce 45 measured samples per backend with default 9-fixture suite.",
            "- p95 is available at suite level when sample count is sufficient.",
            "- p99 is unavailable unless sample count reaches configured threshold.",
            "",
            "Directionally suggested:",
            "- Rust control-plane overhead is likely below local benchmark noise when this project matches `mlx_vlm.server`.",
            "- model/image work dominates VLM latency.",
            "- raw direct-call reference gives useful context but not strict HTTP overhead.",
            "",
            "Not proven:",
            "- strict raw-vs-HTTP overhead unless model-input/token parity is proven.",
            "- production concurrency.",
            "- stable p99 with only 45 samples.",
            "- cross-scenario winner when running `--scenario all`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _scenario_runs_for_report(run: BenchmarkRun) -> tuple[VlmScenarioRun, ...]:
    metadata = run.metadata or {}
    scenario_runs = tuple(metadata.get("scenario_runs", ()))
    if scenario_runs:
        return scenario_runs
    return (
        VlmScenarioRun(
            scenario=run.scenario,
            benchmark_mode=run.benchmark_mode,
            started_at=run.generated_at,
            ended_at=run.generated_at,
            fixture_count=0,
            fixture_names=(),
            warmup_runs_per_fixture=run.warmup_runs_per_fixture,
            measured_runs_per_fixture=run.measured_runs_per_fixture,
            expected_measured_samples_per_backend=None,
            backend_order=tuple(result.backend for result in run.results),
            order_rounds=1,
            order_randomized=False,
            backend_order_seed=None,
            aggregated_across_order_rounds=False,
            order_round_details=(),
            results=run.results,
            fixture_rows=tuple(metadata.get("fixture_rows", ())),
            streaming_rows=tuple(metadata.get("streaming_rows", ())),
            cancellation_rows=tuple(metadata.get("cancellation_rows", ())),
            concurrency_rows=tuple(metadata.get("concurrency_rows", ())),
            fairness_warnings=tuple(metadata.get("fairness_warnings", ())),
            interpretation=(),
        ),
    )


def _render_vlm_scenario_section(
    run: BenchmarkRun, scenario_run: VlmScenarioRun
) -> list[str]:
    lines = [
        f"### Scenario: {scenario_run.scenario}",
        "",
        f"- benchmark_mode: {scenario_run.benchmark_mode}",
        f"- started_at: {scenario_run.started_at}",
        f"- ended_at: {scenario_run.ended_at}",
        f"- fixture_count: {scenario_run.fixture_count}",
        f"- warmup_runs_per_fixture: {scenario_run.warmup_runs_per_fixture}",
        f"- measured_runs_per_fixture: {scenario_run.measured_runs_per_fixture}",
        f"- expected_measured_samples_per_backend: {_format_na_int(scenario_run.expected_measured_samples_per_backend)}",
        f"- backend_order: {', '.join(scenario_run.backend_order) if scenario_run.backend_order else 'not_available'}",
        f"- order_rounds: {scenario_run.order_rounds}",
        f"- backend_order_seed: {_format_na_int(scenario_run.backend_order_seed)}",
        f"- order_randomized: {'yes' if scenario_run.order_randomized else 'no'}",
        f"- aggregated_across_order_rounds: {'yes' if scenario_run.aggregated_across_order_rounds else 'no'}",
        "",
    ]
    if scenario_run.order_round_details:
        lines.extend(
            [
                "#### Backend Order Rounds",
                "",
                "| round | backend_order |",
                "| ---: | --- |",
            ]
        )
        for detail in scenario_run.order_round_details:
            lines.append(
                f"| {detail.get('order_round_index', 'not_available')} | {', '.join(detail.get('backend_order', ())) or 'not_available'} |"
            )

    if scenario_run.results:
        lines.extend(
            [
                "",
                "#### Raw Per-Backend Metrics",
                "",
                "| backend | vlm_load_time_ms | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | ttft_p99_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | latency_p99_ms | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | latency_per_completion_token_ms | decode_time_per_completion_token_ms | image_load_ms_mean | image_decode_ms_mean | image_preprocess_ms_mean | decode_tps_mean | e2e_tps_mean | notes | warnings |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for result in scenario_run.results:
            lines.append(
                "| {backend} | {load_time} | {samples} | {errors} | {error_rate} | {ttft_mean} | {ttft_p50} | {ttft_p95} | {ttft_p99} | {latency_mean} | {latency_p50} | {latency_p95} | {latency_p99} | {prompt_mean} | {completion_mean} | {total_mean} | {latency_per_token} | {decode_per_token} | {image_load} | {image_decode} | {image_preprocess} | {decode_tps} | {e2e_tps} | {notes} | {warnings} |".format(
                    backend=result.backend,
                    load_time=_format_na_number(result.vlm_load_time_ms),
                    samples=result.samples,
                    errors=result.errors,
                    error_rate=_format_percent(result.error_rate),
                    ttft_mean=_format_na_number(result.ttft_mean_ms),
                    ttft_p50=_format_na_number(result.ttft_p50_ms),
                    ttft_p95=_format_na_number(result.ttft_p95_ms),
                    ttft_p99=_format_na_number(result.ttft_p99_ms),
                    latency_mean=_format_na_number(result.latency_mean_ms),
                    latency_p50=_format_na_number(result.latency_p50_ms),
                    latency_p95=_format_na_number(result.latency_p95_ms),
                    latency_p99=_format_na_number(result.latency_p99_ms),
                    prompt_mean=_format_na_number(result.prompt_tokens_mean),
                    completion_mean=_format_na_number(result.completion_tokens_mean),
                    total_mean=_format_na_number(result.total_tokens_mean),
                    latency_per_token=_format_na_number(
                        result.latency_per_completion_token_ms
                    ),
                    decode_per_token=_format_na_number(
                        result.decode_time_per_completion_token_ms
                    ),
                    image_load=_format_na_number(result.image_load_ms_mean),
                    image_decode=_format_na_number(result.image_decode_ms_mean),
                    image_preprocess=_format_na_number(
                        result.image_preprocess_latency_ms_mean
                    ),
                    decode_tps=_format_na_number(result.decode_tokens_per_second_mean),
                    e2e_tps=_format_na_number(result.end_to_end_tokens_per_second_mean),
                    notes=_join_or_default(result.notes),
                    warnings=_join_or_default(result.warnings),
                )
            )

    if scenario_run.comparison_rows:
        lines.extend(
            [
                "",
                "#### Headline Eligibility",
                "",
                "| backend_pair | fairness_level | comparison_kind | prompt_tokens_mean_delta_pct | completion_tokens_mean_delta_pct | token_equivalent | headline_eligible | reasons_not_headline_eligible |",
                "| --- | --- | --- | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in scenario_run.comparison_rows:
            lines.append(
                "| {pair} | {fairness_level} | {kind} | {prompt_delta} | {completion_delta} | {token_equivalent} | {headline} | {reasons} |".format(
                    pair=f"{row.backend_a} vs {row.backend_b}",
                    fairness_level=row.fairness_level,
                    kind=row.comparison_kind,
                    prompt_delta=_format_na_percent(row.prompt_tokens_mean_delta_pct),
                    completion_delta=_format_na_percent(
                        row.completion_tokens_mean_delta_pct
                    ),
                    token_equivalent=_format_bool(row.token_equivalent),
                    headline=_format_bool(row.headline_eligible),
                    reasons=_join_or_default(row.reasons_not_headline_eligible),
                )
            )

    if scenario_run.scenario == "baseline":
        lines.extend(_render_baseline_scenario_detail(scenario_run))
    elif scenario_run.scenario == "streaming":
        lines.extend(_render_streaming_scenario_detail(scenario_run))
    elif scenario_run.scenario == "cancellation":
        lines.extend(_render_cancellation_scenario_detail(scenario_run))
    elif scenario_run.scenario == "concurrency":
        lines.extend(_render_concurrency_scenario_detail(scenario_run))

    lines.extend(["", "#### Interpretation", ""])
    if scenario_run.fairness_warnings:
        for warning in scenario_run.fairness_warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- no fairness warnings")
    for item in scenario_run.interpretation:
        lines.append(f"- {item}")
    return lines


def _render_baseline_scenario_detail(scenario_run: VlmScenarioRun) -> list[str]:
    lines: list[str] = []
    http_rows = [
        row
        for row in scenario_run.comparison_rows
        if row.comparison_kind == "http_fair_comparison"
    ]
    direct_rows = [
        row
        for row in scenario_run.comparison_rows
        if row.comparison_kind == "direct_call_reference"
    ]
    if http_rows:
        lines.extend(
            [
                "",
                "#### HTTP Backend Fair Comparison",
                "",
                "| backend_pair | latency_mean_delta_ms | latency_p50_delta_ms | latency_p95_delta_ms | ttft_mean_delta_ms | decode_tps_delta_pct | e2e_tps_delta_pct | token_equivalent | headline_eligible | reasons_not_headline_eligible |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in http_rows:
            lines.append(
                "| {pair} | {latency_mean} | {latency_p50} | {latency_p95} | {ttft_mean} | {decode_tps} | {e2e_tps} | {token_equivalent} | {headline} | {reasons} |".format(
                    pair=f"{row.backend_a} vs {row.backend_b}",
                    latency_mean=_format_signed_number(row.latency_mean_delta_ms),
                    latency_p50=_format_signed_number(row.latency_p50_delta_ms),
                    latency_p95=_format_signed_number(row.latency_p95_delta_ms),
                    ttft_mean=_format_signed_number(row.ttft_mean_delta_ms),
                    decode_tps=_format_signed_percent(row.decode_tps_delta_pct),
                    e2e_tps=_format_signed_percent(row.e2e_tps_delta_pct),
                    token_equivalent=_format_bool(row.token_equivalent),
                    headline=_format_bool(row.headline_eligible),
                    reasons=_join_or_default(row.reasons_not_headline_eligible),
                )
            )
    if direct_rows:
        lines.extend(
            [
                "",
                "#### Direct-Call Reference vs HTTP Backends",
                "",
                "Raw mlx-vlm is a direct Python reference. It is not a strict overhead baseline unless model-input/token equivalence is proven. The primary fair serving comparison is mlx_vlm.server vs this project.",
                "",
                "| backend_pair | latency_mean_delta_ms | latency_p50_delta_ms | latency_p95_delta_ms | decode_tps_delta_pct | e2e_tps_delta_pct | prompt_tokens_mean_delta_pct | completion_tokens_mean_delta_pct | token_equivalent | headline_eligible | reasons_not_headline_eligible |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
            ]
        )
        for row in direct_rows:
            lines.append(
                "| {pair} | {latency_mean} | {latency_p50} | {latency_p95} | {decode_tps} | {e2e_tps} | {prompt_delta} | {completion_delta} | {token_equivalent} | {headline} | {reasons} |".format(
                    pair=f"{row.backend_a} vs {row.backend_b}",
                    latency_mean=_format_signed_number(row.latency_mean_delta_ms),
                    latency_p50=_format_signed_number(row.latency_p50_delta_ms),
                    latency_p95=_format_signed_number(row.latency_p95_delta_ms),
                    decode_tps=_format_signed_percent(row.decode_tps_delta_pct),
                    e2e_tps=_format_signed_percent(row.e2e_tps_delta_pct),
                    prompt_delta=_format_na_percent(row.prompt_tokens_mean_delta_pct),
                    completion_delta=_format_na_percent(
                        row.completion_tokens_mean_delta_pct
                    ),
                    token_equivalent=_format_bool(row.token_equivalent),
                    headline=_format_bool(row.headline_eligible),
                    reasons=_join_or_default(row.reasons_not_headline_eligible),
                )
            )
    if scenario_run.fixture_rows:
        lines.extend(
            [
                "",
                "#### Per-Fixture Breakdown",
                "",
                "| backend | fixture | category | image_count | total_image_pixels | total_megapixels | widths | heights | prompt_tokens_mean | completion_tokens_mean | total_tokens_mean | ttft_mean_ms | ttft_p50_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | image_preprocess_ms_mean | decode_tps_mean | e2e_tps_mean | latency_per_completion_token_ms | latency_per_image_ms | latency_per_megapixel_ms | errors | error_rate | warnings |",
                "| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in scenario_run.fixture_rows:
            lines.append(
                "| {backend} | {fixture} | {category} | {image_count} | {pixels} | {megapixels} | {widths} | {heights} | {prompt_tokens} | {completion_tokens} | {total_tokens} | {ttft_mean} | {ttft_p50} | {latency_mean} | {latency_p50} | {latency_p95} | {image_preprocess} | {decode_tps} | {e2e_tps} | {latency_per_token} | {latency_per_image} | {latency_per_megapixel} | {errors} | {error_rate} | {warnings} |".format(
                    backend=row.backend,
                    fixture=row.fixture_name,
                    category=row.fixture_category,
                    image_count=row.image_count,
                    pixels=_format_na_int(row.total_image_pixels),
                    megapixels=_format_na_number(row.total_megapixels),
                    widths=row.widths_summary,
                    heights=row.heights_summary,
                    prompt_tokens=_format_na_number(row.prompt_tokens_mean),
                    completion_tokens=_format_na_number(row.completion_tokens_mean),
                    total_tokens=_format_na_number(row.total_tokens_mean),
                    ttft_mean=_format_na_number(row.ttft_mean_ms),
                    ttft_p50=_format_na_number(row.ttft_p50_ms),
                    latency_mean=_format_na_number(row.latency_mean_ms),
                    latency_p50=_format_na_number(row.latency_p50_ms),
                    latency_p95=_format_na_number(row.latency_p95_ms),
                    image_preprocess=_format_na_number(row.image_preprocess_ms_mean),
                    decode_tps=_format_na_number(row.decode_tps_mean),
                    e2e_tps=_format_na_number(row.e2e_tps_mean),
                    latency_per_token=_format_na_number(
                        row.latency_per_completion_token_ms
                    ),
                    latency_per_image=_format_na_number(row.latency_per_image_ms),
                    latency_per_megapixel=_format_na_number(
                        row.latency_per_megapixel_ms
                    ),
                    errors=row.errors,
                    error_rate=_format_percent(row.error_rate),
                    warnings=_join_or_default(row.warnings),
                )
            )
    return lines


def _render_streaming_scenario_detail(scenario_run: VlmScenarioRun) -> list[str]:
    lines: list[str] = []
    if scenario_run.streaming_rows:
        lines.extend(
            [
                "",
                "#### Streaming Metrics",
                "",
                "| backend | streaming_available | samples | errors | error_rate | ttft_mean_ms | ttft_p50_ms | ttft_p95_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | prompt_tokens_mean | completion_tokens_mean | sse_chunk_count_mean | raw_chunk_count_mean | chunk_interval_mean_ms | chunk_interval_p50_ms | chunk_interval_p95_ms | stream_completed_normally | parse_errors | finish_reason_distribution | notes | warnings |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
            ]
        )
        for row in scenario_run.streaming_rows:
            lines.append(
                "| {backend} | {available} | {samples} | {errors} | {error_rate} | {ttft_mean} | {ttft_p50} | {ttft_p95} | {latency_mean} | {latency_p50} | {latency_p95} | {prompt_tokens} | {completion_tokens} | {sse_chunk_count} | {raw_chunk_count} | {chunk_interval_mean} | {chunk_interval_p50} | {chunk_interval_p95} | {completed} | {parse_errors} | {finish_reason} | {notes} | {warnings} |".format(
                    backend=row.backend,
                    available=_format_bool(row.streaming_available),
                    samples=row.samples,
                    errors=row.errors,
                    error_rate=_format_percent(row.error_rate),
                    ttft_mean=_format_na_number(row.ttft_mean_ms),
                    ttft_p50=_format_na_number(row.ttft_p50_ms),
                    ttft_p95=_format_na_number(row.ttft_p95_ms),
                    latency_mean=_format_na_number(row.latency_mean_ms),
                    latency_p50=_format_na_number(row.latency_p50_ms),
                    latency_p95=_format_na_number(row.latency_p95_ms),
                    prompt_tokens=_format_na_number(row.prompt_tokens_mean),
                    completion_tokens=_format_na_number(row.completion_tokens_mean),
                    sse_chunk_count=_format_na_number(row.sse_chunk_count_mean),
                    raw_chunk_count=_format_na_number(row.raw_chunk_count_mean),
                    chunk_interval_mean=_format_na_number(row.chunk_interval_mean_ms),
                    chunk_interval_p50=_format_na_number(row.chunk_interval_p50_ms),
                    chunk_interval_p95=_format_na_number(row.chunk_interval_p95_ms),
                    completed=_format_bool(row.stream_completed_normally),
                    parse_errors=row.parse_errors,
                    finish_reason=row.finish_reason_distribution or "not_available",
                    notes=_join_or_default(row.notes),
                    warnings=_join_or_default(row.warnings),
                )
            )
    return lines


def _render_cancellation_scenario_detail(scenario_run: VlmScenarioRun) -> list[str]:
    lines: list[str] = []
    if scenario_run.cancellation_rows:
        lines.extend(
            [
                "",
                "#### Cancellation Results",
                "",
                "| backend | cancellation_attempts | server_observed_cancellations | cancellation_errors | follow_up_success_count | follow_up_error_count | follow_up_latency_mean_ms | worker_health_after_cancel | warnings |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for row in scenario_run.cancellation_rows:
            lines.append(
                "| {backend} | {attempts} | {observed} | {cancel_errors} | {follow_success} | {follow_error} | {follow_latency} | {worker_health} | {warnings} |".format(
                    backend=row.backend,
                    attempts=row.cancellation_attempts,
                    observed=_format_na_int(row.server_observed_cancellations),
                    cancel_errors=row.cancellation_errors,
                    follow_success=row.follow_up_success_count,
                    follow_error=row.follow_up_error_count,
                    follow_latency=_format_na_number(row.follow_up_latency_mean_ms),
                    worker_health=row.worker_health_after_cancel or "not_available",
                    warnings=_join_or_default(row.warnings),
                )
            )
    return lines


def _render_concurrency_scenario_detail(scenario_run: VlmScenarioRun) -> list[str]:
    lines: list[str] = []
    if scenario_run.concurrency_rows:
        lines.extend(
            [
                "",
                "#### Concurrency Metrics",
                "",
                "Concurrency warning: local MacBook behavior only, not production capacity.",
                "",
                "| backend | concurrency | completed_requests | errors | error_rate | wall_clock_duration_ms | requests_per_second | completion_tokens_per_second | ttft_mean_ms | ttft_p50_ms | latency_mean_ms | latency_p50_ms | latency_p95_ms | queue_wait_mean_ms | max_queue_depth | warnings |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for row in scenario_run.concurrency_rows:
            lines.append(
                "| {backend} | {concurrency} | {completed} | {errors} | {error_rate} | {wall_clock} | {rps} | {tokens_ps} | {ttft_mean} | {ttft_p50} | {latency_mean} | {latency_p50} | {latency_p95} | {queue_wait} | {max_queue_depth} | {warnings} |".format(
                    backend=row.backend,
                    concurrency=row.concurrency,
                    completed=row.completed_requests,
                    errors=row.errors,
                    error_rate=_format_percent(row.error_rate),
                    wall_clock=_format_na_number(row.wall_clock_duration_ms),
                    rps=_format_na_number(row.requests_per_second),
                    tokens_ps=_format_na_number(row.completion_tokens_per_second),
                    ttft_mean=_format_na_number(row.ttft_mean_ms),
                    ttft_p50=_format_na_number(row.ttft_p50_ms),
                    latency_mean=_format_na_number(row.latency_mean_ms),
                    latency_p50=_format_na_number(row.latency_p50_ms),
                    latency_p95=_format_na_number(row.latency_p95_ms),
                    queue_wait=_format_na_number(row.queue_wait_mean_ms),
                    max_queue_depth=_format_na_int(row.max_queue_depth),
                    warnings=_join_or_default(row.warnings),
                )
            )
    return lines


def _format_na_number(value: float | None) -> str:
    if value is None:
        return "not_available"
    return f"{value:.1f}"


def _format_na_int(value: int | None) -> str:
    if value is None:
        return "not_available"
    return str(value)


def _format_bool(value: bool | None) -> str:
    if value is None:
        return "not_available"
    return "yes" if value else "no"


def _join_or_default(values: Sequence[str], default: str = "-") -> str:
    if not values:
        return default
    return "; ".join(values).replace("|", "\\|")
