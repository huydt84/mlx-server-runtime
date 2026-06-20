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


# ---------------------------------------------------------------------------
# VLM report helpers — separate from Phase 6 text benchmark report.
# ---------------------------------------------------------------------------


def write_vlm_report(path: Path, runs: Sequence[BenchmarkRun]) -> None:
    """Write a Phase 9 VLM benchmark report to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_vlm_report(runs), encoding="utf-8")


def _render_vlm_report(runs: Sequence[BenchmarkRun]) -> str:
    """Render a markdown VLM benchmark report with fairness notes."""
    if not runs:
        raise ValueError("VLM benchmark report requires at least one run")

    lines = [
        "# Phase 9 — VLM Benchmark Report",
        "",
        "## Fairness Notes",
        "",
        "This benchmark compares VLM inference throughput across backends. "
        "Image sizes, prompt templates, and output lengths may differ "
        "materially across fixtures — do not compare raw latency numbers "
        "between different fixture categories without normalising.",
        "",
        "Backend differences:",
        "",
        "- **raw mlx-vlm**: direct Python call; no HTTP, no queue, no IPC.",
        "- **mlx_vlm.server**: HTTP server with its own queuing.",
        "- **this project**: Rust control plane with queue, backpressure, "
        "cancellation, telemetry, and worker supervision.",
        "",
        "All backends use `temperature=0.0` and `top_p=1.0`; `max_tokens` "
        "comes from the benchmark command line.",
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

        # Per-backend metrics table.
        lines.append("### Raw Per-Backend Metrics")
        lines.append("")
        lines.append(
            "| backend | vlm_load_time_ms | samples | errors | error_rate "
            "| ttft_mean_ms | ttft_p50_ms "
            "| latency_mean_ms | latency_p50_ms "
            "| latency_per_completion_token_ms "
            "| prompt_tokens_mean | completion_tokens_mean "
            "| image_preprocess_ms_mean "
            "| decode_tps_mean | e2e_tps_mean |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: "
            "| ---: | ---: "
            "| ---: | ---: "
            "| ---: "
            "| ---: | ---: "
            "| ---: "
            "| ---: | ---: |"
        )
        for result in run.results:
            lines.append(
                "| {backend} | {load_time} | {samples} | {errors} | {error_rate} "
                "| {ttft_mean} | {ttft_p50} "
                "| {latency_mean} | {latency_p50} "
                "| {latency_per_token} "
                "| {prompt_mean} | {completion_mean} "
                "| {img_preproc} "
                "| {decode_tps} | {e2e_tps} |".format(
                    backend=result.backend,
                    load_time=_format_number(result.vlm_load_time_ms),
                    samples=result.samples,
                    errors=result.errors,
                    error_rate=_format_percent(result.error_rate),
                    ttft_mean=_format_number(result.ttft_mean_ms),
                    ttft_p50=_format_number(result.ttft_p50_ms),
                    latency_mean=_format_number(result.latency_mean_ms),
                    latency_p50=_format_number(result.latency_p50_ms),
                    latency_per_token=_format_number(
                        result.latency_per_completion_token_ms
                    ),
                    prompt_mean=_format_number(result.prompt_tokens_mean),
                    completion_mean=_format_number(result.completion_tokens_mean),
                    img_preproc=_format_number(result.image_preprocess_latency_ms_mean),
                    decode_tps=_format_number(result.decode_tokens_per_second_mean),
                    e2e_tps=_format_number(result.end_to_end_tokens_per_second_mean),
                )
            )

        # Overhead vs raw mlx-vlm.
        raw = next((r for r in run.results if r.backend == "raw mlx-vlm"), None)
        lines.append("")
        lines.append("### Overhead Vs Raw MLX-VLM")
        lines.append("")
        lines.append(
            "| backend "
            "| ttft_mean_delta_ms | latency_mean_delta_ms "
            "| decode_tps_delta_pct | e2e_tps_delta_pct |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for result in run.results:
            lines.append(
                "| {backend} "
                "| {ttft_delta} | {latency_delta} "
                "| {decode_tps_pct} | {e2e_tps_pct} |".format(
                    backend=result.backend,
                    ttft_delta=_format_signed_number(
                        calculate_overhead(
                            result.ttft_mean_ms, raw.ttft_mean_ms if raw else None
                        )
                    ),
                    latency_delta=_format_signed_number(
                        calculate_overhead(
                            result.latency_mean_ms,
                            raw.latency_mean_ms if raw else None,
                        )
                    ),
                    decode_tps_pct=_format_signed_percent(
                        calculate_overhead_percent(
                            result.decode_tokens_per_second_mean,
                            raw.decode_tokens_per_second_mean if raw else None,
                        )
                    ),
                    e2e_tps_pct=_format_signed_percent(
                        calculate_overhead_percent(
                            result.end_to_end_tokens_per_second_mean,
                            raw.end_to_end_tokens_per_second_mean if raw else None,
                        )
                    ),
                )
            )

        # Notes / warnings.
        lines.append("")
        lines.append("### Notes / Warnings")
        lines.append("")
        all_warnings: list[str] = []
        for r in run.results:
            all_warnings.extend(r.warnings)
        if all_warnings:
            for w in dict.fromkeys(all_warnings):
                lines.append(f"- {w}")
        else:
            lines.append("- no warnings")

    return "\n".join(lines) + "\n"
