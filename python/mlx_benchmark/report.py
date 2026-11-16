"""Render benchmark and calibration results as Markdown reports."""

from __future__ import annotations

from typing import Any


def render_report(results: dict[str, Any]) -> str:
    """Render the Markdown report for a ``run`` or ``calibrate`` result."""
    if results.get("command") == "calibrate":
        lines = _calibration_lines(results)
    else:
        lines = _run_lines(results)
    _append_failures_and_error(lines, results)
    return "\n".join(lines) + "\n"


def _run_lines(results: dict[str, Any]) -> list[str]:
    configuration = results["configuration"]
    lines = [
        "# MLX Air Benchmark Report",
        "",
        f"- Run: `{results['run_id']}`",
        f"- Status: `{results['status']}`",
        f"- Suite: `{configuration['suite']}`",
        f"- Focus: `{configuration['focus']}`",
        f"- Server mode: `{results['server']['mode']}`",
    ]
    analysis = results.get("analysis")
    if analysis:
        lines.extend(
            [
                "",
                "## Results",
                "",
                "Lower ms is better; higher tokens/s is better. Statistics are over "
                "trial means; p95 pools individual requests. **Bold** rows are each "
                "workload's primary metric.",
                "",
                "| model | runtime | workload | metric | trials | mean | median | min–max | CV | request p95 |",
                "| --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
            ]
        )
        for summary in analysis["metrics"]:
            metric = f"{summary['metric']} ({summary['unit']})"
            if summary["primary"]:
                metric = f"**{metric}**"
            p95 = summary["request_p95"]
            lines.append(
                f"| {summary['model']} | {summary['runtime_configuration']} | "
                f"{summary['workload']} | {metric} | {summary['trial_count']} | "
                f"{_number(summary['mean'])} | {_number(summary['median'])} | "
                f"{_number(summary['minimum'])}–{_number(summary['maximum'])} | "
                f"{_percent(summary['coefficient_of_variation_percent'])} | "
                f"{'—' if p95 is None else _number(p95)} |"
            )

    lines.extend(
        [
            "",
            "## Trials",
            "",
            "| round | model | runtime | workload | trial | load mode | concurrency | succeeded | errors |",
            "| ---: | --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for trial in results["trials"]:
        lines.append(
            f"| {trial.get('round', '')} | {trial.get('model', '')} | "
            f"{trial.get('runtime_configuration', '')} | {trial['workload_name']} | "
            f"{trial['trial_index']} | {trial['load_mode']} | "
            f"{trial['configured_concurrency']} | {trial['success_count']} | "
            f"{trial['error_count']} |"
        )
    _append_diagnostics(lines, results)
    return lines


def _calibration_lines(results: dict[str, Any]) -> list[str]:
    configuration = results["configuration"]
    repetitions = results["repetitions"]
    lines = [
        "# MLX Air Benchmark Calibration Report",
        "",
        f"- Calibration: `{results['run_id']}`",
        f"- Status: `{results['status']}`",
        f"- Suite: `{configuration['suite']}`",
        f"- Focus: `{configuration['focus']}`",
        f"- Repetitions: {repetitions['completed']} of {repetitions['requested']}",
        "",
        "## Run-to-run variation",
        "",
        "Each value is one repetition's mean for the workload's primary metric. "
        "Host thermal, power, and memory-pressure observations are in "
        "`results.json`.",
        "",
        "| model | runtime | workload | metric | runs | mean | min–max | CV | stable |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for summary in results.get("repeated_measurements", []):
        lines.append(
            f"| {summary['model']} | {summary['runtime_configuration']} | "
            f"{summary['workload']} | {summary['metric']} ({summary['unit']}) | "
            f"{len(summary['run_values'])} | {_number(summary['mean'])} | "
            f"{_number(summary['minimum'])}–{_number(summary['maximum'])} | "
            f"{_percent(summary['coefficient_of_variation_percent'])} | "
            f"{'no' if summary['unstable'] else 'yes'} |"
        )
    return lines


def _append_failures_and_error(lines: list[str], results: dict[str, Any]) -> None:
    if results.get("validation_failures"):
        lines.extend(["", "## Validation failures", ""])
        lines.extend(
            f"- `{failure['code']}`: {failure['message']}"
            for failure in results["validation_failures"]
        )
    if results.get("error") is not None:
        lines.extend(["", "## Error", "", f"`{results['error']['message']}`"])


def _append_diagnostics(lines: list[str], results: dict[str, Any]) -> None:
    diagnostics = results.get("diagnostics")
    if not diagnostics or not diagnostics.get("attempts"):
        return
    lines.extend(
        [
            "",
            "## Diagnostic artifacts",
            "",
            f"Diagnostic status: `{diagnostics['status']}`. Diagnostics ran in "
            "separate processes after measurement; the rows above are unaffected.",
            "",
        ]
    )
    for attempt in diagnostics["attempts"]:
        target = " / ".join(
            str(attempt[key])
            for key in ("family", "model", "runtime_configuration", "workload")
            if attempt.get(key) is not None
        )
        lines.append(f"- `{target}`: `{attempt['status']}`")
        for artifact in attempt.get("artifacts", []):
            if artifact["status"] == "succeeded":
                lines.append(f"  - [{artifact['kind']}]({artifact['path']})")
            else:
                lines.append(f"  - `{artifact['kind']}`: failed ({artifact['error']})")
        if attempt.get("error"):
            lines.append(f"  - Error: `{attempt['error']}`")


def _number(value: float) -> str:
    return f"{value:.3f}"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}%"
