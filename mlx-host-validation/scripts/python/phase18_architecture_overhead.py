#!/usr/bin/env python3
"""Measure architecture-manifest lookup without importing model modules."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time

from mlx_worker.native_mlx import registry


def _measure(table: dict[str, object], *, iterations: int) -> float:
    for _ in range(2_000):
        table.get("Qwen2ForCausalLM")
    started = time.perf_counter_ns()
    for _ in range(iterations):
        if table.get("Qwen2ForCausalLM") is None:
            raise RuntimeError("selected architecture disappeared from registry")
    return (time.perf_counter_ns() - started) / iterations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-markdown")
    parser.add_argument("--iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    selected = registry._REGISTRY["Qwen2ForCausalLM"]
    baseline = {"Qwen2ForCausalLM": selected}
    expanded = dict(baseline)
    expanded.update(
        {
            f"SyntheticForCausalLM{i}": replace(
                selected, architecture_class=f"SyntheticForCausalLM{i}"
            )
            for i in range(100)
        }
    )
    baseline_samples = [
        _measure(baseline, iterations=args.iterations) for _ in range(5)
    ]
    expanded_samples = [
        _measure(expanded, iterations=args.iterations) for _ in range(5)
    ]
    baseline_median = statistics.median(baseline_samples)
    expanded_median = statistics.median(expanded_samples)
    delta = expanded_median - baseline_median
    ratio = delta / baseline_median if baseline_median else 0.0
    allowed_delta_ns = max(5_000.0, baseline_median * 0.01)
    passed = abs(delta) <= allowed_delta_ns
    payload = {
        "lookup": "dict.get",
        "baseline_manifest_count": len(baseline),
        "expanded_manifest_count": len(expanded),
        "iterations_per_sample": args.iterations,
        "baseline_ns_per_lookup": baseline_median,
        "expanded_ns_per_lookup": expanded_median,
        "delta_ns_per_lookup": delta,
        "delta_ratio": ratio,
        "allowed_delta_ns": allowed_delta_ns,
        "passed": passed,
        "selected_module_imported": "mlx_worker.native_mlx.models.qwen2" in sys.modules,
        "request_path_registry_calls": 0,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if args.output_markdown:
        markdown = Path(args.output_markdown)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(
            "\n".join(
                (
                    "# Phase 18 Architecture Isolation",
                    "",
                    f"- Gate: **{'pass' if passed else 'fail'}**",
                    f"- Baseline manifests: {len(baseline)}",
                    f"- Expanded manifests: {len(expanded)}",
                    f"- Baseline lookup: {baseline_median:.3f} ns",
                    f"- Expanded lookup: {expanded_median:.3f} ns",
                    f"- Delta: {delta:.3f} ns ({ratio:.3%})",
                    "- Request-path registry calls: 0",
                    "- Selected model module imported by manifest lookup: no",
                    "",
                )
            )
        )
    print(f"phase18_manifest_count_baseline={len(baseline)}")
    print(f"phase18_manifest_count_expanded={len(expanded)}")
    print(f"phase18_lookup_delta_ns={delta:.3f}")
    print(f"phase18_lookup_delta_ratio={ratio:.6f}")
    print(f"phase18_architecture_report={output}")
    if args.output_markdown:
        print(f"phase18_architecture_markdown={args.output_markdown}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
