"""Summarize benchmark trials, treating each completed trial as one observation."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics as stdlib_statistics
from typing import Any


# Metric name -> (unit, better direction).
METRICS = {
    "ttft": ("ms", "lower"),
    "tpot": ("ms", "lower"),
    "end_to_end_latency": ("ms", "lower"),
    "aggregate_output_tokens_per_second": ("tokens/s", "higher"),
}
STREAMING_ONLY_METRICS = frozenset({"ttft", "tpot"})
_TRIAL_METRICS = frozenset({"aggregate_output_tokens_per_second"})


def workload_metrics(workload: dict[str, Any]) -> list[str]:
    """Return the metrics measurable for a workload, primary metric first."""
    primary = str(workload["primary_metric"])
    available = [
        metric
        for metric in METRICS
        if workload["streaming"] or metric not in STREAMING_ONLY_METRICS
    ]
    return [primary] + [metric for metric in available if metric != primary]


def analyze_run(results: dict[str, Any]) -> dict[str, Any]:
    """Summarize every measurable metric per model, runtime, and workload.

    Trial-level means are the independent observations. Requests inside one
    trial share load and are only pooled for the request-level p95.
    """
    workloads = {
        workload["name"]: workload for workload in results["configuration"]["workloads"]
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in results.get("trials", []):
        key = (trial["model"], trial["runtime_configuration"], trial["workload_name"])
        grouped[key].append(trial)

    summaries = []
    for (model, runtime, workload_name), trials in sorted(grouped.items()):
        workload = workloads[workload_name]
        for metric in workload_metrics(workload):
            summaries.append(
                {
                    "model": model,
                    "runtime_configuration": runtime,
                    "workload": workload_name,
                    "metric": metric,
                    "primary": metric == workload["primary_metric"],
                    "unit": METRICS[metric][0],
                    "better_direction": METRICS[metric][1],
                    "trial_count": len(trials),
                    "request_count": sum(
                        int(trial["success_count"]) for trial in trials
                    ),
                    "trial_values": [_trial_value(trial, metric) for trial in trials],
                    "request_p95": _request_p95(trials, metric),
                }
            )
            summaries[-1].update(summarize(summaries[-1]["trial_values"]))
    return {"metrics": summaries}


def analyze_calibration(
    run_results: list[dict[str, Any]], max_cv_percent: float
) -> list[dict[str, Any]]:
    """Summarize run-to-run variation of each primary metric's mean."""
    repeated: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    units: dict[tuple[str, str, str, str], tuple[str, str]] = {}
    for result in run_results:
        analysis = result.get("analysis") or analyze_run(result)
        for summary in analysis["metrics"]:
            if not summary["primary"]:
                continue
            key = (
                summary["model"],
                summary["runtime_configuration"],
                summary["workload"],
                summary["metric"],
            )
            repeated[key].append(float(summary["mean"]))
            units[key] = (summary["unit"], summary["better_direction"])

    summaries = []
    for key in sorted(repeated):
        model, runtime, workload, metric = key
        values = repeated[key]
        summary = summarize(values)
        cv = summary["coefficient_of_variation_percent"]
        summaries.append(
            {
                "model": model,
                "runtime_configuration": runtime,
                "workload": workload,
                "metric": metric,
                "unit": units[key][0],
                "better_direction": units[key][1],
                "run_values": values,
                **summary,
                "max_cv_percent": max_cv_percent,
                "unstable": cv is None or cv > max_cv_percent,
            }
        )
    return summaries


def summarize(values: list[float]) -> dict[str, Any]:
    """Return mean, median, spread, and coefficient of variation."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = stdlib_statistics.fmean(values)
    deviation = stdlib_statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "median": stdlib_statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "standard_deviation": deviation,
        "coefficient_of_variation_percent": (
            deviation / abs(mean) * 100.0 if mean != 0.0 else None
        ),
    }


def _trial_value(trial: dict[str, Any], metric: str) -> float:
    if metric in _TRIAL_METRICS:
        return float(trial[metric])
    values = _request_values([trial], metric)
    if not values:
        raise ValueError(
            f"{trial['workload_name']} trial {trial['trial_index']} has no {metric} samples"
        )
    return stdlib_statistics.fmean(values)


def _request_p95(trials: list[dict[str, Any]], metric: str) -> float | None:
    if metric in _TRIAL_METRICS:
        return None
    return _percentile(_request_values(trials, metric), 95.0)


def _request_values(trials: list[dict[str, Any]], metric: str) -> list[float]:
    values = []
    for trial in trials:
        for request in trial["requests"]:
            if request["status"] != "succeeded":
                continue
            value = request_metric(request, metric)
            if value is not None:
                values.append(value)
    return values


def request_metric(request: dict[str, Any], metric: str) -> float | None:
    """Compute one request-level latency metric in milliseconds."""
    submitted = int(request["submitted_monotonic_ns"])
    if metric == "ttft":
        return (int(request["first_token_monotonic_ns"]) - submitted) / 1_000_000
    if metric == "end_to_end_latency":
        return (int(request["completed_monotonic_ns"]) - submitted) / 1_000_000
    if metric == "tpot":
        completion_tokens = int(request["completion_tokens"])
        if completion_tokens <= 1:
            return None
        decode_ns = int(request["final_token_monotonic_ns"]) - int(
            request["first_token_monotonic_ns"]
        )
        return decode_ns / (completion_tokens - 1) / 1_000_000
    raise ValueError(f"metric {metric!r} is not a request metric")


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower, upper = math.floor(rank), math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)
