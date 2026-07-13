#!/usr/bin/env python3
"""Run and compare serial versus overlap public-gateway workloads."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import http.client
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any
import xml.etree.ElementTree as ET
from importlib.metadata import version


@dataclass(frozen=True)
class RequestSample:
    """One public request observation."""

    workload: str
    status: int
    ttft_ms: float
    latency_ms: float
    text: str
    prompt_tokens: int
    completion_tokens: int


def main() -> None:
    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    run = subcommands.add_parser("run")
    run.add_argument("--port", type=int, required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--mode", choices=("serial", "overlap"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--warmups", type=int, default=3)
    run.add_argument("--samples", type=int, default=5)
    run.add_argument("--max-tokens", type=int, default=8)

    metal_workload = subcommands.add_parser("metal-workload")
    metal_workload.add_argument("--port", type=int, required=True)
    metal_workload.add_argument("--model", required=True)
    metal_workload.add_argument("--mode", choices=("serial", "overlap"), required=True)
    metal_workload.add_argument("--output", type=Path, required=True)
    metal_workload.add_argument("--requests", type=int, default=4)
    metal_workload.add_argument("--max-tokens", type=int, default=128)

    compare = subcommands.add_parser("compare")
    compare.add_argument("--serial", type=Path, required=True)
    compare.add_argument("--overlap", type=Path, required=True)
    compare.add_argument("--output-json", type=Path, required=True)
    compare.add_argument("--output-markdown", type=Path, required=True)
    compare.add_argument("--max-regression", type=float, default=0.02)

    merge = subcommands.add_parser("merge")
    merge.add_argument("--mode", choices=("serial", "overlap"), required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("inputs", type=Path, nargs="+")

    timeline = subcommands.add_parser("timeline")
    timeline.add_argument("--events", type=Path, required=True)
    timeline.add_argument("--output", type=Path, required=True)

    metal = subcommands.add_parser("metal")
    metal.add_argument("--xml", type=Path, required=True)
    metal.add_argument("--output", type=Path, required=True)
    metal.add_argument("--process", default="python")

    metal_compare = subcommands.add_parser("metal-compare")
    metal_compare.add_argument("--serial", type=Path, required=True)
    metal_compare.add_argument("--overlap", type=Path, required=True)
    metal_compare.add_argument("--output", type=Path, required=True)
    metal_compare.add_argument("--max-regression", type=float, default=0.02)
    metal_compare.add_argument("--absolute-noise-us", type=float, default=50.0)

    args = parser.parse_args()
    if args.command == "run":
        run_mode(args)
    elif args.command == "metal-workload":
        run_metal_workload(args)
    elif args.command == "merge":
        merge_runs(args)
    elif args.command == "compare":
        compare_modes(args)
    elif args.command == "timeline":
        analyze_timeline(args)
    elif args.command == "metal":
        analyze_metal_trace(args)
    else:
        compare_metal_traces(args)


def run_mode(args: argparse.Namespace) -> None:
    if args.warmups < 0 or args.samples <= 0 or args.max_tokens <= 0:
        raise SystemExit("warmups must be non-negative; samples/max-tokens positive")
    base = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Write the numbers one through four, separated by spaces.",
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    for _ in range(args.warmups):
        sample = _post(args.port, {**base, "stream": False}, "warmup")
        if sample.status != 200:
            raise SystemExit(f"{args.mode} warmup returned HTTP {sample.status}")

    samples: list[RequestSample] = []
    started = time.perf_counter()
    for _ in range(args.samples):
        samples.append(_post(args.port, {**base, "stream": False}, "nonstream_single"))
        samples.append(
            _post(
                args.port,
                {
                    **base,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                },
                "stream_single",
            )
        )
    single_wall_ms = (time.perf_counter() - started) * 1000

    concurrent_wall_ms: list[float] = []
    for _ in range(args.samples):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = tuple(
                pool.map(
                    lambda _: _post(
                        args.port,
                        {
                            **base,
                            "stream": True,
                            "stream_options": {"include_usage": True},
                        },
                        "stream_concurrency4",
                    ),
                    range(4),
                )
            )
        concurrent_wall_ms.append((time.perf_counter() - started) * 1000)
        samples.extend(concurrent)

    bad = [sample for sample in samples if sample.status != 200]
    if bad:
        raise SystemExit(f"{args.mode} benchmark observed non-200 responses")
    metrics = _metrics(args.port)
    expected_mode_metric = (
        'mlx_native_execution_mode{backend="native-mlx",modality="text",'
        f'mode="{args.mode}"}}'
    )
    if metrics.get(expected_mode_metric) != 1.0:
        observed = sorted(
            key for key in metrics if key.startswith("mlx_native_execution_mode")
        )
        raise SystemExit(
            f"gateway did not report requested execution mode {args.mode}; "
            f"observed={observed}"
        )
    payload = {
        "backend": "native-mlx",
        "mode": args.mode,
        "model": args.model,
        "warmups": args.warmups,
        "samples_per_workload": args.samples,
        "max_tokens": args.max_tokens,
        "environment": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "mlx": version("mlx"),
        },
        "single_wall_ms": single_wall_ms,
        "concurrent_wall_ms": concurrent_wall_ms,
        "metrics": metrics,
        "samples": [asdict(sample) for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_mode={args.mode}")
    print(f"phase17_mode_samples={len(samples)}")
    print(f"phase17_mode_output={args.output}")


def run_metal_workload(args: argparse.Namespace) -> None:
    """Run one sustained concurrent decode window for Metal comparison."""

    if args.requests <= 0 or args.max_tokens <= 0:
        raise SystemExit("requests and max-tokens must be positive")
    warmup = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with the number one."}],
        "max_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    if _post(args.port, warmup, "metal_warmup").status != 200:
        raise SystemExit(f"{args.mode} Metal warmup failed")
    body = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Count upward using short comma-separated numbers and continue "
                    "until the response token budget is exhausted."
                ),
            }
        ],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
    }
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.requests) as pool:
        samples = list(
            pool.map(
                lambda _: _post(args.port, body, "metal_concurrency"),
                range(args.requests),
            )
        )
    if any(sample.status != 200 for sample in samples):
        raise SystemExit(f"{args.mode} sustained Metal workload failed")
    payload = {
        "mode": args.mode,
        "model": args.model,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "elapsed_s": time.perf_counter() - started,
        "samples": [asdict(sample) for sample in samples],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_metal_workload_mode={args.mode}")
    print(f"phase17_metal_workload_requests={args.requests}")
    print(f"phase17_metal_workload_max_tokens={args.max_tokens}")
    print(f"phase17_metal_workload_output={args.output}")


def compare_modes(args: argparse.Namespace) -> None:
    serial = json.loads(args.serial.read_text())
    overlap = json.loads(args.overlap.read_text())
    if serial["model"] != overlap["model"]:
        raise SystemExit("serial and overlap checkpoints differ")
    serial_samples = serial["samples"]
    overlap_samples = overlap["samples"]
    if len(serial_samples) != len(overlap_samples):
        raise SystemExit("serial and overlap sample counts differ")
    for index, (left, right) in enumerate(
        zip(serial_samples, overlap_samples, strict=True)
    ):
        comparable = (
            left["workload"],
            left["text"],
            left["prompt_tokens"],
            left["completion_tokens"],
        )
        candidate = (
            right["workload"],
            right["text"],
            right["prompt_tokens"],
            right["completion_tokens"],
        )
        if comparable != candidate:
            raise SystemExit(f"serial/overlap parity mismatch at sample {index}")

    rows = []
    regressions: list[str] = []
    for workload in ("nonstream_single", "stream_single", "stream_concurrency4"):
        serial_rows = [row for row in serial_samples if row["workload"] == workload]
        overlap_rows = [row for row in overlap_samples if row["workload"] == workload]
        serial_latency = [float(row["latency_ms"]) for row in serial_rows]
        overlap_latency = [float(row["latency_ms"]) for row in overlap_rows]
        serial_mean = statistics.mean(serial_latency)
        overlap_mean = statistics.mean(overlap_latency)
        regression = (overlap_mean - serial_mean) / serial_mean
        latency_ci_low, latency_ci_high = _relative_mean_confidence_interval(
            serial_latency, overlap_latency
        )
        row = {
            "workload": workload,
            "samples": len(serial_rows),
            "serial_latency_mean_ms": serial_mean,
            "overlap_latency_mean_ms": overlap_mean,
            "latency_change_ratio": regression,
            "latency_change_ci95_low": latency_ci_low,
            "latency_change_ci95_high": latency_ci_high,
            "serial_latency_p95_ms": _percentile(serial_latency, 0.95),
            "overlap_latency_p95_ms": _percentile(overlap_latency, 0.95),
        }
        if workload.startswith("stream"):
            serial_ttft = [
                float(item["ttft_ms"])
                for item in serial_rows
                if not math.isnan(float(item["ttft_ms"]))
            ]
            overlap_ttft = [
                float(item["ttft_ms"])
                for item in overlap_rows
                if not math.isnan(float(item["ttft_ms"]))
            ]
            row["serial_ttft_mean_ms"] = statistics.mean(serial_ttft)
            row["overlap_ttft_mean_ms"] = statistics.mean(overlap_ttft)
            row["ttft_change_ratio"] = (
                row["overlap_ttft_mean_ms"] - row["serial_ttft_mean_ms"]
            ) / row["serial_ttft_mean_ms"]
            ttft_ci_low, ttft_ci_high = _relative_mean_confidence_interval(
                serial_ttft, overlap_ttft
            )
            row["ttft_change_ci95_low"] = ttft_ci_low
            row["ttft_change_ci95_high"] = ttft_ci_high
            if ttft_ci_low > args.max_regression:
                regressions.append(
                    f"{workload} TTFT regression lower confidence bound is "
                    f"{ttft_ci_low * 100:.2f}%"
                )
        rows.append(row)
        if latency_ci_low > args.max_regression:
            regressions.append(
                f"{workload} latency regression lower confidence bound is "
                f"{latency_ci_low * 100:.2f}%"
            )

    result = {
        "model": serial["model"],
        "max_regression": args.max_regression,
        "parity": True,
        "passed": not regressions,
        "regressions": regressions,
        "workloads": rows,
        "serial_environment": serial.get("environment", {}),
        "overlap_environment": overlap.get("environment", {}),
        "serial_worker_memory_bytes": serial["metrics"].get("mlx_worker_memory_bytes"),
        "overlap_worker_memory_bytes": overlap["metrics"].get(
            "mlx_worker_memory_bytes"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n")
    args.output_markdown.write_text(_render_report(result))
    print(f"phase17_parity={int(result['parity'])}")
    print(f"phase17_performance_passed={int(result['passed'])}")
    print(f"phase17_comparison={args.output_json}")
    if regressions:
        raise SystemExit("; ".join(regressions))


def merge_runs(args: argparse.Namespace) -> None:
    """Merge equal-mode ABBA rounds while preserving raw observations."""

    runs = [json.loads(path.read_text()) for path in args.inputs]
    if any(run["mode"] != args.mode for run in runs):
        raise SystemExit("merge input execution mode mismatch")
    if len({run["model"] for run in runs}) != 1:
        raise SystemExit("merge input checkpoint mismatch")
    payload = dict(runs[0])
    payload["rounds"] = len(runs)
    payload["warmups"] = sum(int(run["warmups"]) for run in runs)
    payload["samples_per_workload"] = sum(
        int(run["samples_per_workload"]) for run in runs
    )
    payload["single_wall_ms"] = sum(float(run["single_wall_ms"]) for run in runs)
    payload["concurrent_wall_ms"] = [
        value for run in runs for value in run["concurrent_wall_ms"]
    ]
    payload["samples"] = [value for run in runs for value in run["samples"]]
    payload["metrics"] = runs[-1]["metrics"]
    payload["raw_rounds"] = [str(path) for path in args.inputs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_merged_mode={args.mode}")
    print(f"phase17_merged_rounds={len(runs)}")
    print(f"phase17_merged_samples={len(payload['samples'])}")


def analyze_timeline(args: argparse.Namespace) -> None:
    events = [
        json.loads(line)
        for line in args.events.read_text().splitlines()
        if line.strip()
    ]
    dispatches = sorted(
        [event for event in events if event.get("stage") == "async_dispatch"],
        key=lambda event: int(event["monotonic_ns"]),
    )
    synchronizations = sorted(
        [event for event in events if event.get("stage") == "synchronize_eval"],
        key=lambda event: int(event["monotonic_ns"]),
    )
    detokenizations = [
        event for event in events if event.get("stage") == "detokenization"
    ]
    windows = []
    for dispatch in dispatches:
        dispatch_ns = int(dispatch["monotonic_ns"])
        synchronization = next(
            (
                event
                for event in synchronizations
                if int(event["monotonic_ns"]) >= dispatch_ns
            ),
            None,
        )
        if synchronization is None:
            continue
        sync_end_ns = (
            int(synchronization["monotonic_ns"])
            + int(synchronization.get("duration_us", 0)) * 1_000
        )
        cpu_events = [
            event
            for event in detokenizations
            if dispatch_ns <= int(event["monotonic_ns"]) <= sync_end_ns
        ]
        if cpu_events:
            windows.append(
                {
                    "request_id": dispatch.get("request_id"),
                    "dispatch_ns": dispatch_ns,
                    "synchronize_end_ns": sync_end_ns,
                    "cpu_stages": [event["stage"] for event in cpu_events],
                }
            )
    result = {
        "async_dispatch_events": len(dispatches),
        "synchronize_events": len(synchronizations),
        "detokenization_events": len(detokenizations),
        "cpu_overlap_windows": windows,
        "cpu_work_while_mlx_outstanding": bool(windows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"phase17_cpu_overlap_windows={len(windows)}")
    print(f"phase17_timeline_analysis={args.output}")
    if not windows:
        raise SystemExit("no CPU work observed while an MLX step was outstanding")


def _post(port: int, payload: dict[str, Any], workload: str) -> RequestSample:
    encoded = json.dumps(payload).encode()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    started = time.perf_counter()
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream" if payload["stream"] else "application/json",
        },
    )
    response = connection.getresponse()
    ttft_ms = math.nan
    text = ""
    prompt_tokens = 0
    completion_tokens = 0
    if payload["stream"]:
        while raw := response.fp.readline():
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            choices = event.get("choices") or []
            if choices:
                delta = choices[0].get("delta", {}).get("content") or ""
                if delta and math.isnan(ttft_ms):
                    ttft_ms = (time.perf_counter() - started) * 1000
                text += delta
            usage = event.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", prompt_tokens))
            completion_tokens = int(usage.get("completion_tokens", completion_tokens))
    else:
        body = json.loads(response.read())
        if response.status == 200:
            text = str(body["choices"][0]["message"]["content"])
            usage = body.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
    latency_ms = (time.perf_counter() - started) * 1000
    connection.close()
    return RequestSample(
        workload=workload,
        status=response.status,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def _metrics(port: int) -> dict[str, float]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    connection.request("GET", "/metrics")
    response = connection.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    connection.close()
    if response.status != 200:
        raise SystemExit(f"metrics returned HTTP {response.status}")
    wanted = (
        "mlx_scheduler_tick_latency_by_backend_ms",
        "mlx_executor_stage_latency_by_backend_ms",
        "mlx_native_execution_mode",
        "mlx_worker_memory_bytes",
    )
    return {
        line.rpartition(" ")[0]: float(line.rpartition(" ")[2])
        for line in body.splitlines()
        if line.startswith(wanted)
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]


def _relative_mean_confidence_interval(
    baseline: list[float], candidate: list[float]
) -> tuple[float, float]:
    """Return a conservative independent 95% interval for relative mean change."""

    baseline_mean = statistics.mean(baseline)
    candidate_mean = statistics.mean(candidate)
    relative_mean = (candidate_mean - baseline_mean) / baseline_mean
    if len(baseline) < 2 or len(candidate) < 2:
        return relative_mean, relative_mean
    standard_error = math.sqrt(
        statistics.variance(baseline) / len(baseline)
        + statistics.variance(candidate) / len(candidate)
    )
    critical = _student_t_critical_95(min(len(baseline), len(candidate)) - 1)
    margin = critical * standard_error / baseline_mean
    return relative_mean - margin, relative_mean + margin


def _student_t_critical_95(degrees_of_freedom: int) -> float:
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    for upper_bound, critical in table.items():
        if degrees_of_freedom <= upper_bound:
            return critical
    return 1.960


def analyze_metal_trace(args: argparse.Namespace) -> None:
    """Require real GPU intervals for the MLX worker process."""

    root = ET.parse(args.xml).getroot()
    process_names = {
        element.attrib["id"]: element.attrib.get("fmt", "")
        for element in root.iter("process")
        if "id" in element.attrib
    }
    process_match = args.process.casefold()
    intervals: list[tuple[int, int]] = []
    for row in root.iter("row"):
        process = row.find("process")
        if process is None:
            continue
        process_name = process.attrib.get("fmt", "")
        if not process_name and "ref" in process.attrib:
            process_name = process_names.get(process.attrib["ref"], "")
        if process_match not in process_name.casefold():
            continue
        depth = row.find("metal-nesting-level")
        if depth is not None and int(depth.text or "0") != 0:
            continue
        started = row.find("start-time")
        duration = row.find("duration")
        if started is None or duration is None:
            continue
        start_ns = int(started.text or "0")
        duration_ns = int(duration.text or "0")
        if duration_ns > 0:
            intervals.append((start_ns, start_ns + duration_ns))
    if not intervals:
        raise SystemExit(
            f"Metal System Trace has no GPU intervals for process match {args.process!r}"
        )

    merged: list[list[int]] = []
    for start_ns, end_ns in sorted(intervals):
        if merged and start_ns <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_ns)
        else:
            merged.append([start_ns, end_ns])
    gaps_us = [
        (right[0] - left[1]) / 1_000
        for left, right in zip(merged, merged[1:], strict=False)
        if right[0] > left[1]
    ]
    payload = {
        "source": str(args.xml),
        "process_match": args.process,
        "raw_gpu_intervals": len(intervals),
        "merged_gpu_intervals": len(merged),
        "gpu_active_ms": sum(end - start for start, end in merged) / 1_000_000,
        "gpu_span_ms": (merged[-1][1] - merged[0][0]) / 1_000_000,
        "idle_gap_count": len(gaps_us),
        "idle_gap_p50_us": _percentile(gaps_us, 0.50) if gaps_us else 0.0,
        "idle_gap_p90_us": _percentile(gaps_us, 0.90) if gaps_us else 0.0,
        "idle_gap_p95_us": _percentile(gaps_us, 0.95) if gaps_us else 0.0,
        "idle_gap_p99_us": _percentile(gaps_us, 0.99) if gaps_us else 0.0,
        "idle_gap_max_us": max(gaps_us, default=0.0),
        "idle_gap_largest_us": sorted(gaps_us, reverse=True)[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_metal_gpu_intervals={len(intervals)}")
    print(f"phase17_metal_gpu_active_ms={payload['gpu_active_ms']:.3f}")
    print(f"phase17_metal_idle_gap_p95_us={payload['idle_gap_p95_us']:.3f}")
    print(f"phase17_metal_analysis={args.output}")


def compare_metal_traces(args: argparse.Namespace) -> None:
    """Compare serial and overlap GPU interval gaps with bounded noise."""

    if args.max_regression < 0 or args.absolute_noise_us < 0:
        raise SystemExit("Metal comparison tolerances must be non-negative")
    serial = json.loads(args.serial.read_text())
    overlap = json.loads(args.overlap.read_text())
    serial_gap = float(serial["idle_gap_p95_us"])
    overlap_gap = float(overlap["idle_gap_p95_us"])
    allowed_delta = max(args.absolute_noise_us, serial_gap * args.max_regression)
    delta = overlap_gap - serial_gap
    payload = {
        "serial_analysis": str(args.serial),
        "overlap_analysis": str(args.overlap),
        "serial_idle_gap_p95_us": serial_gap,
        "overlap_idle_gap_p95_us": overlap_gap,
        "idle_gap_delta_us": delta,
        "allowed_delta_us": allowed_delta,
        "max_regression": args.max_regression,
        "absolute_noise_us": args.absolute_noise_us,
        "serial_gpu_intervals": int(serial["raw_gpu_intervals"]),
        "overlap_gpu_intervals": int(overlap["raw_gpu_intervals"]),
        "passed": delta <= allowed_delta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"phase17_metal_serial_idle_gap_p95_us={serial_gap:.3f}")
    print(f"phase17_metal_overlap_idle_gap_p95_us={overlap_gap:.3f}")
    print(f"phase17_metal_idle_gap_delta_us={delta:.3f}")
    print(f"phase17_metal_comparison_ok={int(payload['passed'])}")
    print(f"phase17_metal_comparison={args.output}")
    if not payload["passed"]:
        raise SystemExit(
            "overlap Metal p95 idle-gap regression exceeded the bounded noise gate"
        )


def _render_report(result: dict[str, Any]) -> str:
    lines = [
        "# Phase 17 Serial vs Overlap",
        "",
        f"- Model: `{result['model']}`",
        f"- Output parity: **{'pass' if result['parity'] else 'fail'}**",
        f"- Performance gate: **{'pass' if result['passed'] else 'fail'}**",
        "",
        "| Workload | Samples | Serial mean ms | Overlap mean ms | Change (95% CI) | Serial p95 ms | Overlap p95 ms | TTFT change (95% CI) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["workloads"]:
        lines.append(
            "| {workload} | {samples} | {serial_latency_mean_ms:.2f} | "
            "{overlap_latency_mean_ms:.2f} | {latency_change_ratio:.2%} "
            "({latency_change_ci95_low:.2%}, {latency_change_ci95_high:.2%}) | "
            "{serial_latency_p95_ms:.2f} | {overlap_latency_p95_ms:.2f} | "
            "{ttft_change} |".format(
                **row,
                ttft_change=(
                    f"{row['ttft_change_ratio']:.2%} "
                    f"({row['ttft_change_ci95_low']:.2%}, "
                    f"{row['ttft_change_ci95_high']:.2%})"
                    if "ttft_change_ratio" in row
                    else "-"
                ),
            )
        )
    if result["regressions"]:
        lines.extend(["", "## Regressions", ""])
        lines.extend(f"- {item}" for item in result["regressions"])
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
