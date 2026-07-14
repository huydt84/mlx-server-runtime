from __future__ import annotations

import hashlib

import pytest

from benchmarks.v2_benchmark import (
    DEFAULT_CONFIGURATIONS,
    DEFAULT_MODELS,
    RequestSample,
    Scenario,
    _describe_change,
    _relative_mean_confidence_interval,
    _replace_config_value,
    _split_samples,
    _workload_fingerprint,
    compare_results,
    render_comparison,
    summarize_scenario,
)


def test_default_matrix_covers_four_models_and_serial_overlap() -> None:
    assert len(DEFAULT_MODELS) == 4
    assert DEFAULT_CONFIGURATIONS == ("serial-radix", "overlap-radix")


def test_split_samples_preserves_total() -> None:
    assert _split_samples(20, 3) == [7, 7, 6]
    assert sum(_split_samples(5, 2)) == 5


def test_replace_config_value_requires_and_replaces_key() -> None:
    assert _replace_config_value('backend = "v1"\n', "backend", "native-mlx") == (
        'backend = "native-mlx"\n'
    )
    with pytest.raises(ValueError, match="no model key"):
        _replace_config_value('backend = "v1"\n', "model", "missing")


def test_describe_change_is_explicit_for_reverse_and_forward_metrics() -> None:
    assert _describe_change(100.0, 90.0, "lower") == (
        "100.000 -> 90.000; 10.000 lower (10.00% better)"
    )
    assert _describe_change(100.0, 90.0, "higher") == (
        "100.000 -> 90.000; 10.000 lower (10.00% worse)"
    )


def test_confidence_interval_detects_clear_latency_regression() -> None:
    low, high = _relative_mean_confidence_interval([100.0] * 20, [110.0] * 20)
    assert low == pytest.approx(0.1)
    assert high == pytest.approx(0.1)


def test_summarize_scenario_keeps_raw_samples_and_directions() -> None:
    scenario = Scenario("interactive", "test", (), 1)
    samples = [
        RequestSample(
            scenario="interactive",
            request="one",
            round=1,
            status=200,
            ttft_ms=10.0,
            latency_ms=20.0,
            prompt_tokens=8,
            completion_tokens=4,
            decode_tokens_per_second=300.0,
            end_to_end_tokens_per_second=200.0,
            text_sha256=hashlib.sha256(b"ok").hexdigest(),
            request_id="req-1",
            started_monotonic_ns=1,
            finished_monotonic_ns=2,
        )
    ]
    result = summarize_scenario(scenario, samples, [25.0], {})
    assert result["ttft_mean_ms"] == 10.0
    assert result["latency_mean_ms"] == 20.0
    assert result["completion_tokens_per_second"] == 160.0
    assert result["completion_tokens_per_second_samples"] == [160.0]
    assert result["requests_per_second_samples"] == [40.0]
    assert result["ttft_p95_ms"] is None
    assert len(result["raw_samples"]) == 1


def test_workload_fingerprint_ignores_label_and_profile() -> None:
    manifest = {
        "label": "before",
        "profile": "all",
        "metal_capture": True,
        "models": ["model"],
        "scenarios": [{"name": "short"}],
    }
    changed = {**manifest, "label": "after", "profile": "none", "metal_capture": False}
    assert _workload_fingerprint(manifest) == _workload_fingerprint(changed)


def test_compare_results_reports_lower_latency_as_better() -> None:
    baseline = _synthetic_result(100.0, "before")
    candidate = _synthetic_result(90.0, "after")
    comparison = compare_results(baseline, candidate, max_regression_ratio=0.02)
    latency = next(
        row for row in comparison["rows"] if row["metric"] == "latency_mean_ms"
    )
    assert comparison["passed"] is True
    assert latency["verdict"] == "better"
    assert "10.000 lower" in latency["explicit_change"]
    assert {
        "completion_tokens_per_second",
        "requests_per_second",
    }.issubset({row["metric"] for row in comparison["rows"]})
    assert "lower is better" in render_comparison(comparison)


def test_compare_results_rejects_incomparable_manifest() -> None:
    baseline = _synthetic_result(100.0, "before")
    candidate = _synthetic_result(90.0, "after")
    candidate["manifest"]["models"] = ["different"]
    with pytest.raises(ValueError, match="not comparable"):
        compare_results(baseline, candidate, max_regression_ratio=0.02)


def _synthetic_result(latency: float, label: str) -> dict:
    manifest = {
        "label": label,
        "profile": "all",
        "metal_capture": False,
        "models": ["model"],
        "configurations": ["overlap-radix"],
        "warmups_per_scenario_per_round": 1,
        "target_samples_per_scenario": 20,
        "order_rounds": 2,
        "order_seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "text_cache_budget_bytes": 1,
        "scenarios": [{"name": "short"}],
    }
    manifest["workload_fingerprint"] = _workload_fingerprint(manifest)
    digest = hashlib.sha256(b"same").hexdigest()
    samples = [
        {
            "status": 200,
            "request": "short",
            "prompt_tokens": 8,
            "completion_tokens": 4,
            "text_sha256": digest,
            "latency_ms": latency,
            "ttft_ms": latency / 2,
            "decode_tokens_per_second": 100.0,
            "end_to_end_tokens_per_second": 50.0,
        }
        for _ in range(20)
    ]
    return {
        "manifest": manifest,
        "source": {"git_commit": label},
        "runs": [
            {
                "model": "model",
                "configuration": "overlap-radix",
                "scenarios": [
                    {
                        "scenario": "short",
                        "latency_mean_ms": latency,
                        "ttft_mean_ms": latency / 2,
                        "decode_tokens_per_second_mean": 100.0,
                        "end_to_end_tokens_per_second_mean": 50.0,
                        "completion_tokens_per_second": 80.0,
                        "requests_per_second": 20.0,
                        "completion_tokens_per_second_samples": [80.0] * 20,
                        "requests_per_second_samples": [20.0] * 20,
                        "raw_samples": samples,
                    }
                ],
            }
        ],
    }
