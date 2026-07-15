"""Compute trial-level benchmark statistics and configured tail distributions."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import random
import statistics as stdlib_statistics
from typing import Any, Iterable


_BOOTSTRAP_RESAMPLES = 10_000
_METRIC_DEFINITIONS = {
    "ttft": {"unit": "ms", "better_direction": "lower"},
    "end_to_end_latency": {"unit": "ms", "better_direction": "lower"},
    "tpot": {"unit": "ms", "better_direction": "lower"},
    "aggregate_output_tokens_per_second": {
        "unit": "tokens/s",
        "better_direction": "higher",
    },
}


def analyze_run(results: dict[str, Any]) -> dict[str, Any]:
    """Analyze completed trials using trials as independent observations.

    Args:
        results: One benchmark run result containing selected configuration and
            completed trials.

    Returns:
        Primary, secondary, and configured tail summaries suitable for JSON
        serialization.
    """
    configuration = results["configuration"]
    workloads = {workload["name"]: workload for workload in configuration["workloads"]}
    grouped_trials = _group_trials(results.get("trials", []))
    seed = int(configuration["sampling"]["seed"])
    primary_metrics: list[dict[str, Any]] = []
    secondary_metrics: list[dict[str, Any]] = []

    for key in sorted(grouped_trials):
        model, runtime_configuration, workload_name = key
        workload = workloads[workload_name]
        trials = grouped_trials[key]
        primary_metrics.append(
            _metric_summary(
                model,
                runtime_configuration,
                workload,
                str(workload["primary_metric"]),
                trials,
                seed,
                include_interval=True,
            )
        )
        for metric in workload.get("secondary_metrics", []):
            secondary_metrics.append(
                _metric_summary(
                    model,
                    runtime_configuration,
                    workload,
                    str(metric),
                    trials,
                    seed,
                    include_interval=False,
                )
            )

    tail_workloads = set(configuration["tail_selection"]["workloads"])
    tails = [
        _tail_summary(key, trials)
        for key, trials in sorted(grouped_trials.items())
        if key[2] in tail_workloads
    ]
    return {
        "independent_observation": "one completed trial",
        "request_samples_are_independent": False,
        "primary_metrics": primary_metrics,
        "secondary_metrics": secondary_metrics,
        "tails": tails,
    }


def analyze_calibration(
    run_results: list[dict[str, Any]], maximum_cv_percent: float
) -> list[dict[str, Any]]:
    """Summarize run-to-run variation for repeated unchanged runs.

    Args:
        run_results: Successful benchmark run results in repetition order.
        maximum_cv_percent: Configured instability threshold.

    Returns:
        One run-level variation summary per model, runtime configuration, and
        workload primary metric.
    """
    repeated: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    definitions: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for result in run_results:
        analysis = result.get("analysis") or analyze_run(result)
        for summary in analysis["primary_metrics"]:
            key = (
                summary["model"],
                summary["runtime_configuration"],
                summary["workload"],
                summary["metric"],
            )
            repeated[key].append(float(summary["mean"]))
            definitions[key] = summary

    seed = int(run_results[0]["configuration"]["sampling"]["seed"])
    summaries: list[dict[str, Any]] = []
    for key in sorted(repeated):
        model, runtime_configuration, workload, metric = key
        values = repeated[key]
        absolute = _absolute_summary(values, seed, "calibration:" + "\x00".join(key))
        coefficient = absolute["coefficient_of_variation_percent"]
        definition = definitions[key]
        summaries.append(
            {
                "model": model,
                "runtime_configuration": runtime_configuration,
                "workload": workload,
                "metric": metric,
                "unit": definition["unit"],
                "better_direction": definition["better_direction"],
                "completed_repetition_count": len(values),
                "repetition_values": values,
                **absolute,
                "run_to_run_range": {
                    "minimum": min(values),
                    "maximum": max(values),
                    "span": max(values) - min(values),
                    "unit": definition["unit"],
                },
                "maximum_coefficient_of_variation_percent": maximum_cv_percent,
                "unstable": coefficient is None or coefficient > maximum_cv_percent,
            }
        )
    return summaries


def metric_definition(metric: str) -> dict[str, str]:
    """Return the standard unit and direction for a supported metric."""
    try:
        return dict(_METRIC_DEFINITIONS[metric])
    except KeyError as error:
        raise ValueError(f"unsupported benchmark metric {metric!r}") from error


def supported_metrics() -> frozenset[str]:
    """Return metric names implemented by the statistics layer."""
    return frozenset(_METRIC_DEFINITIONS)


def _group_trials(
    trials: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        grouped[
            (
                str(trial["model"]),
                str(trial["runtime_configuration"]),
                str(trial["workload_name"]),
            )
        ].append(trial)
    return dict(grouped)


def _metric_summary(
    model: str,
    runtime_configuration: str,
    workload: dict[str, Any],
    metric: str,
    trials: list[dict[str, Any]],
    seed: int,
    *,
    include_interval: bool,
) -> dict[str, Any]:
    definition = metric_definition(metric)
    if metric == workload["primary_metric"]:
        unit = str(workload["metric_unit"])
        direction = str(workload["metric_direction"])
    else:
        unit = definition["unit"]
        direction = definition["better_direction"]
    trial_samples = [
        {
            "configuration_order": trial["configuration_order"],
            "trial_index": trial["trial_index"],
            "request_count": sum(
                request["status"] == "succeeded" for request in trial["requests"]
            ),
            "value": _trial_metric(trial, metric),
        }
        for trial in trials
    ]
    values = [float(sample["value"]) for sample in trial_samples]
    if not values:
        raise ValueError(
            f"{model}/{runtime_configuration}/{workload['name']} has no trial samples"
        )
    summary = {
        "model": model,
        "runtime_configuration": runtime_configuration,
        "workload": workload["name"],
        "metric": metric,
        "unit": unit,
        "better_direction": direction,
        "independent_trial_count": len(values),
        "request_count": sum(sample["request_count"] for sample in trial_samples),
        "trial_samples": trial_samples,
        **_absolute_summary(
            values,
            seed,
            f"{model}\x00{runtime_configuration}\x00{workload['name']}\x00{metric}",
            include_interval=include_interval,
        ),
    }
    return summary


def _absolute_summary(
    values: list[float],
    seed: int,
    label: str,
    *,
    include_interval: bool = True,
) -> dict[str, Any]:
    mean = stdlib_statistics.fmean(values)
    deviation = stdlib_statistics.stdev(values) if len(values) > 1 else 0.0
    summary: dict[str, Any] = {
        "mean": mean,
        "median": stdlib_statistics.median(values),
        "standard_deviation": deviation,
        "coefficient_of_variation_percent": (
            deviation / abs(mean) * 100.0 if mean != 0.0 else None
        ),
    }
    if include_interval:
        lower, upper = _bootstrap_mean_interval(values, seed, label)
        summary["bootstrap_95_interval"] = {
            "lower": lower,
            "upper": upper,
            "confidence": 0.95,
            "statistic": "mean",
            "resampling_unit": "run" if label.startswith("calibration:") else "trial",
            "resamples": _BOOTSTRAP_RESAMPLES,
        }
    return summary


def _bootstrap_mean_interval(
    values: list[float], seed: int, label: str
) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    generator = random.Random(seed ^ int.from_bytes(digest[:8], "big"))
    count = len(values)
    means = sorted(
        stdlib_statistics.fmean(generator.choices(values, k=count))
        for _ in range(_BOOTSTRAP_RESAMPLES)
    )
    return _percentile(means, 2.5), _percentile(means, 97.5)


def _trial_metric(trial: dict[str, Any], metric: str) -> float:
    if metric == "aggregate_output_tokens_per_second":
        return float(trial[metric])
    request_values = [
        value
        for request in trial["requests"]
        if request["status"] == "succeeded"
        and (value := _request_metric(request, metric)) is not None
    ]
    if not request_values:
        raise ValueError(
            f"{trial['workload_name']} trial {trial['trial_index']} has no {metric} samples"
        )
    return stdlib_statistics.fmean(request_values)


def _request_metric(request: dict[str, Any], metric: str) -> float | None:
    if metric == "ttft":
        return (
            int(request["first_token_monotonic_ns"])
            - int(request["submitted_monotonic_ns"])
        ) / 1_000_000
    if metric == "end_to_end_latency":
        return (
            int(request["completed_monotonic_ns"])
            - int(request["submitted_monotonic_ns"])
        ) / 1_000_000
    if metric == "tpot":
        completion_tokens = int(request["completion_tokens"])
        if completion_tokens <= 1:
            return None
        return (
            (
                int(request["final_token_monotonic_ns"])
                - int(request["first_token_monotonic_ns"])
            )
            / (completion_tokens - 1)
            / 1_000_000
        )
    raise ValueError(f"metric {metric!r} is not a request metric")


def _tail_summary(
    key: tuple[str, str, str], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    model, runtime_configuration, workload = key
    requests = [
        request
        for trial in trials
        for request in trial["requests"]
        if request["status"] == "succeeded"
    ]
    ttft_values = [float(_request_metric(request, "ttft")) for request in requests]
    latency_values = [
        float(_request_metric(request, "end_to_end_latency")) for request in requests
    ]
    wall_times = [float(trial["declared_window_ns"]) / 1_000_000 for trial in trials]
    return {
        "model": model,
        "runtime_configuration": runtime_configuration,
        "workload": workload,
        "request_sample_count": len(requests),
        "independent_trial_count": len(trials),
        "request_ttft_p95": _percentile_record(
            ttft_values, 95.0, "successful requests", independent=False
        ),
        "request_latency_p95": _percentile_record(
            latency_values, 95.0, "successful requests", independent=False
        ),
        "maximum_ttft_per_trial": _trial_maxima(trials, "ttft"),
        "maximum_latency_per_trial": _trial_maxima(trials, "end_to_end_latency"),
        "trial_wall_time_p95": _percentile_record(
            wall_times, 95.0, "completed trials", independent=True
        ),
    }


def _trial_maxima(trials: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    maxima = []
    for trial in trials:
        values = [
            float(value)
            for request in trial["requests"]
            if request["status"] == "succeeded"
            and (value := _request_metric(request, metric)) is not None
        ]
        maxima.append(
            {
                "configuration_order": trial["configuration_order"],
                "trial_index": trial["trial_index"],
                "value": max(values),
                "unit": "ms",
                "request_sample_count": len(values),
            }
        )
    return maxima


def _percentile_record(
    values: list[float], percentile: float, population: str, *, independent: bool
) -> dict[str, Any]:
    return {
        "value": _percentile(values, percentile),
        "unit": "ms",
        "percentile": percentile,
        "population": population,
        "sample_count": len(values),
        "independent_samples": independent,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile without samples")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
