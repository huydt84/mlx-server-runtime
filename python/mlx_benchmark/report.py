"""Render benchmark and calibration results as Markdown reports."""

from __future__ import annotations

from typing import Any


def render_report(results: dict[str, Any]) -> str:
    """Render the report appropriate for the recorded benchmark command.

    Args:
        results: A benchmark ``run`` or ``calibrate`` result.

    Returns:
        A complete Markdown document ending in a newline.
    """
    if results.get("command") == "calibrate":
        return _render_calibration_report(results)
    return _render_run_report(results)


def _render_run_report(results: dict[str, Any]) -> str:
    lines = [
        "# MLX Air Benchmark Report",
        "",
        f"- Run: `{results['run_id']}`",
        f"- Status: `{results['status']}`",
        f"- Suite: `{results['configuration']['suite']}`",
        f"- Focus: `{results['configuration']['focus']}`",
        f"- Server mode: `{results['server']['mode']}`",
        f"- Model: `{results['versions']['model']['name']}`",
    ]
    analysis = results.get("analysis")
    if analysis:
        lines.extend(
            [
                "",
                "## Primary metrics",
                "",
                "Absolute values are reported first. Lower latency, TTFT, and TPOT are better; higher throughput is better. Bootstrap intervals resample completed trials, never requests from within a concurrent trial.",
                "",
                "| model | runtime | workload | metric | direction | trials | mean | median | standard deviation | CV | bootstrap 95% interval |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for summary in analysis["primary_metrics"]:
            interval = summary["bootstrap_95_interval"]
            lines.append(
                f"| {summary['model']} | {summary['runtime_configuration']} | {summary['workload']} | "
                f"{summary['metric']} ({summary['unit']}) | {summary['better_direction']} is better | "
                f"{summary['independent_trial_count']} | {_number(summary['mean'])} | "
                f"{_number(summary['median'])} | {_number(summary['standard_deviation'])} | "
                f"{_percent(summary['coefficient_of_variation_percent'])} | "
                f"[{_number(interval['lower'])}, {_number(interval['upper'])}] {summary['unit']} |"
            )
        if analysis["secondary_metrics"]:
            lines.extend(
                [
                    "",
                    "## Configured secondary metrics",
                    "",
                    "| model | runtime | workload | metric | direction | trials | mean | median | standard deviation | CV |",
                    "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for summary in analysis["secondary_metrics"]:
                lines.append(
                    f"| {summary['model']} | {summary['runtime_configuration']} | {summary['workload']} | "
                    f"{summary['metric']} ({summary['unit']}) | {summary['better_direction']} is better | "
                    f"{summary['independent_trial_count']} | {_number(summary['mean'])} | "
                    f"{_number(summary['median'])} | {_number(summary['standard_deviation'])} | "
                    f"{_percent(summary['coefficient_of_variation_percent'])} |"
                )
        for tail in analysis["tails"]:
            lines.extend(_render_tail(tail))

    lines.extend(
        [
            "",
            "## Trial execution",
            "",
            "| workload | trial | load mode | streaming | concurrency | requested | succeeded | errors |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial in results["trials"]:
        lines.append(
            f"| {trial['workload_name']} | {trial['trial_index']} | {trial['load_mode']} | "
            f"{trial['streaming']} | {trial['configured_concurrency']} | "
            f"{trial['request_count']} | {trial['success_count']} | {trial['error_count']} |"
        )
    _append_failures_and_error(lines, results)
    return "\n".join(lines) + "\n"


def _render_tail(tail: dict[str, Any]) -> list[str]:
    ttft = tail["request_ttft_p95"]
    latency = tail["request_latency_p95"]
    wall = tail["trial_wall_time_p95"]
    return [
        "",
        f"## Tail distribution: {tail['model']} / {tail['runtime_configuration']} / {tail['workload']}",
        "",
        f"- Request TTFT p95: {_number(ttft['value'])} ms across {ttft['sample_count']} successful requests from {tail['independent_trial_count']} completed trials. Requests within a trial are correlated and are not bootstrap observations.",
        f"- Request latency p95: {_number(latency['value'])} ms across {latency['sample_count']} successful requests from {tail['independent_trial_count']} completed trials.",
        f"- Trial wall-time p95: {_number(wall['value'])} ms across {wall['sample_count']} independent completed trials.",
        f"- Maximum TTFT per trial: {_render_maxima(tail['maximum_ttft_per_trial'])}.",
        f"- Maximum latency per trial: {_render_maxima(tail['maximum_latency_per_trial'])}.",
    ]


def _render_calibration_report(results: dict[str, Any]) -> str:
    repetitions = results["repetitions"]
    lines = [
        "# MLX Air Benchmark Calibration Report",
        "",
        f"- Calibration: `{results['run_id']}`",
        f"- Status: `{results['status']}`",
        f"- Suite: `{results['configuration']['suite']}`",
        f"- Focus: `{results['configuration']['focus']}`",
        f"- Repetitions: {repetitions['completed']} completed of {repetitions['requested']} requested",
        "",
        "## Run-to-run variation",
        "",
        "Each repetition executes the same selected benchmark configuration. Values below are per-run primary-metric means; bootstrap intervals resample complete runs.",
        "",
        "| model | runtime | workload | metric | direction | runs | mean | standard deviation | CV | absolute 95% interval | range | stable |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for summary in results.get("repeated_measurements", []):
        interval = summary["bootstrap_95_interval"]
        value_range = summary["run_to_run_range"]
        lines.append(
            f"| {summary['model']} | {summary['runtime_configuration']} | {summary['workload']} | "
            f"{summary['metric']} ({summary['unit']}) | {summary['better_direction']} is better | "
            f"{summary['completed_repetition_count']} | {_number(summary['mean'])} | "
            f"{_number(summary['standard_deviation'])} | {_percent(summary['coefficient_of_variation_percent'])} | "
            f"[{_number(interval['lower'])}, {_number(interval['upper'])}] {summary['unit']} | "
            f"{_number(value_range['minimum'])}–{_number(value_range['maximum'])} {summary['unit']} | "
            f"{'no' if summary['unstable'] else 'yes'} |"
        )
    unstable = results.get("unstable_workloads", [])
    lines.extend(["", "## Stability", ""])
    if unstable:
        lines.append(
            "The following workload measurements exceeded the configured coefficient-of-variation threshold; supplied trial counts and configuration were retained:"
        )
        lines.extend(
            f"- `{item['model']} / {item['runtime_configuration']} / {item['workload']}`: "
            f"{_percent(item['coefficient_of_variation_percent'])} CV (limit {_number(item['maximum_coefficient_of_variation_percent'])}%)."
            for item in unstable
        )
    else:
        lines.append(
            "No completed workload exceeded the configured variation threshold."
        )
    lines.extend(
        [
            "",
            "## Host observations",
            "",
            "Thermal, power, and memory-pressure observations are recorded before and after each repetition in `results.json`.",
        ]
    )
    _append_failures_and_error(lines, results)
    return "\n".join(lines) + "\n"


def _append_failures_and_error(lines: list[str], results: dict[str, Any]) -> None:
    if results.get("validation_failures"):
        lines.extend(["", "## Validation failures", ""])
        lines.extend(
            f"- `{failure['code']}`: {failure['message']}"
            for failure in results["validation_failures"]
        )
    if results.get("error") is not None:
        lines.extend(["", "## Error", "", f"`{results['error']['message']}`"])


def _render_maxima(values: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{value['configuration_order']} trial {value['trial_index']}: {_number(value['value'])} ms "
        f"({value['request_sample_count']} requests)"
        for value in values
    )


def _number(value: float) -> str:
    return f"{value:.3f}"


def _percent(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.2f}%"
