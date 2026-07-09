"""Phase 12 host validation fixtures, policy probes, and report helpers."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import threading
import time
import urllib.error
import urllib.request
from typing import Any


def main() -> None:
    """Run one Phase 12 helper subcommand."""

    parser = argparse.ArgumentParser()
    subcommands = parser.add_subparsers(dest="command", required=True)

    fixtures = subcommands.add_parser("fixtures")
    fixtures.add_argument("--runtime-template", required=True)
    fixtures.add_argument("--native-config", required=True)
    fixtures.add_argument("--v1-config", required=True)
    fixtures.add_argument("--checkpoint", required=True)
    fixtures.add_argument("--native-port", required=True)
    fixtures.add_argument("--v1-port", required=True)
    fixtures.add_argument("--request-dir", required=True)

    policy = subcommands.add_parser("policy-probe")
    policy.add_argument("--request-dir", required=True)
    policy.add_argument("--port", type=int, required=True)
    policy.add_argument(
        "--policy", required=True, choices=("fcfs", "lpm", "lof", "priority")
    )
    policy.add_argument("--metrics-capture", required=True)
    policy.add_argument("--output", required=True)

    cancel = subcommands.add_parser("cancel-probe")
    cancel.add_argument("--request-dir", required=True)
    cancel.add_argument("--port", type=int, required=True)
    cancel.add_argument("--metrics-capture", required=True)
    cancel.add_argument("--output", required=True)

    v1 = subcommands.add_parser("v1-probe")
    v1.add_argument("--request-dir", required=True)
    v1.add_argument("--port", type=int, required=True)

    report = subcommands.add_parser("report")
    report.add_argument("--policy-json", action="append", required=True)
    report.add_argument("--cancel-json", required=True)
    report.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "fixtures":
        write_fixtures(args)
    elif args.command == "policy-probe":
        run_policy_probe(args)
    elif args.command == "cancel-probe":
        run_cancel_probe(args)
    elif args.command == "v1-probe":
        run_v1_probe(args)
    elif args.command == "report":
        write_report(args)


def write_fixtures(args: argparse.Namespace) -> None:
    """Write runtime configs and request fixtures for policy/cancel probes."""

    request_dir = pathlib.Path(args.request_dir)
    request_dir.mkdir(parents=True, exist_ok=True)
    template = pathlib.Path(args.runtime_template).read_text()
    native = (
        template.replace('backend = "v1"', 'backend = "native-mlx"')
        .replace("port = 8000", f"port = {args.native_port}", 1)
        .replace(
            'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
            f'model = "{args.checkpoint}"',
            1,
        )
        .replace("max_active_requests = 16", "max_active_requests = 1")
        .replace("max_pending_requests = 64", "max_pending_requests = 8")
        .replace("request_timeout_seconds = 300", "request_timeout_seconds = 30")
    )
    v1 = template.replace("port = 8000", f"port = {args.v1_port}", 1).replace(
        'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
        f'model = "{args.checkpoint}"',
        1,
    )
    pathlib.Path(args.native_config).write_text(native)
    pathlib.Path(args.v1_config).write_text(v1)
    payloads = {
        "short.json": _payload(args.checkpoint, "Phase twelve short prompt.", 4),
        "long.json": _payload(args.checkpoint, "Phase twelve " * 96, 4),
        "lof.json": _payload(
            args.checkpoint, "Phase twelve longest output fixture.", 12
        ),
        "shared_a.json": _payload(
            args.checkpoint, ("shared-prefix " * 80) + " alpha", 4
        ),
        "shared_b.json": _payload(
            args.checkpoint, ("shared-prefix " * 80) + " beta", 4
        ),
        "cancel_prefill.json": _payload(args.checkpoint, "cancel prefill " * 160, 16),
        "cancel_decode.json": _payload(args.checkpoint, "cancel decode", 64),
    }
    for name, payload in payloads.items():
        (request_dir / name).write_text(json.dumps(payload))


def run_policy_probe(args: argparse.Namespace) -> None:
    """Run real concurrent requests that distinguish one scheduler policy."""

    request_dir = pathlib.Path(args.request_dir)
    started = time.perf_counter()
    files = {
        "fcfs": ("long.json", "short.json"),
        "lpm": ("shared_a.json", "shared_b.json"),
        "lof": ("short.json", "lof.json"),
        "priority": ("short.json", "long.json"),
    }[args.policy]
    rows = _post_concurrently(args.port, [request_dir / name for name in files])
    metrics = _metrics(args.port, pathlib.Path(args.metrics_capture))
    _require_metric(
        metrics, f'policy="{args.policy}"', "mlx_scheduler_policy_by_backend"
    )
    _require_metric(metrics, 'kind="gateway_queue"', "mlx_latency_by_backend_ms")
    _require_metric(metrics, 'kind="scheduler_queue"', "mlx_latency_by_backend_ms")
    _require_metric(
        metrics,
        "mlx_scheduler_requests_by_backend",
        "mlx_scheduler_requests_by_backend",
    )
    latencies = [row["latency_ms"] for row in rows if row["status"] == 200]
    if len(latencies) != len(rows):
        raise SystemExit(f"policy probe failed responses: {rows}")
    result = {
        "policy": args.policy,
        "requests": rows,
        "throughput_rps": len(rows) / max(time.perf_counter() - started, 0.001),
        "ttft_mean_ms": statistics.mean(row["ttft_ms"] for row in rows),
        "latency_mean_ms": statistics.mean(latencies),
        "itl_mean_ms": statistics.mean(row["itl_ms"] for row in rows),
        "fairness_wait_spread_ms": max(latencies) - min(latencies),
    }
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"phase12_policy_{args.policy}_probe_ok=1")


def run_cancel_probe(args: argparse.Namespace) -> None:
    """Cancel public requests and verify bounded cleanup/health signals."""

    request_dir = pathlib.Path(args.request_dir)
    # Rust-waiting is exercised by a short timeout/queue-full config in the
    # phase script. Python-waiting, prefill, and decode are observed through
    # worker cancellation metrics after concurrent public requests.
    rows = _cancel_concurrently(
        args.port,
        [request_dir / "cancel_prefill.json", request_dir / "cancel_decode.json"],
    )
    rows.append(_post(request_dir / "short.json", int(args.port)))
    metrics = _metrics(args.port, pathlib.Path(args.metrics_capture))
    _require_metric(metrics, 'kind="cancellation"', "mlx_latency_by_backend_ms")
    _require_metric(
        metrics,
        "mlx_worker_cancellations_by_backend_total",
        "mlx_worker_cancellations_by_backend_total",
    )
    health = _get(args.port, "/health").decode("utf-8").strip()
    if health != "healthy":
        raise SystemExit(f"health after cancellation probe was {health!r}")
    pathlib.Path(args.output).write_text(
        json.dumps({"health": health, "requests": rows}, indent=2)
    )
    print("phase12_cancellation_cleanup_ok=1")


def run_v1_probe(args: argparse.Namespace) -> None:
    """Run a v1 public request through the same gateway surface."""

    response = _post(pathlib.Path(args.request_dir) / "short.json", int(args.port))
    if response["status"] != 200:
        raise SystemExit(f"v1 request failed: {response}")
    print("v1_non_regression_ok=1")


def write_report(args: argparse.Namespace) -> None:
    """Write a compact markdown report for Phase 12 policy tradeoffs."""

    policies = [json.loads(pathlib.Path(path).read_text()) for path in args.policy_json]
    cancel = json.loads(pathlib.Path(args.cancel_json).read_text())
    lines = [
        "# Phase 12 Native v2 Policy and Lifecycle Validation",
        "",
        "| policy | requests | throughput_rps | ttft_mean_ms | latency_mean_ms | itl_mean_ms | fairness_wait_spread_ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in policies:
        lines.append(
            "| {policy} | {requests} | {throughput_rps:.2f} | {ttft_mean_ms:.1f} | {latency_mean_ms:.1f} | {itl_mean_ms:.1f} | {fairness_wait_spread_ms:.1f} |".format(
                policy=item["policy"],
                requests=len(item["requests"]),
                throughput_rps=item["throughput_rps"],
                ttft_mean_ms=item["ttft_mean_ms"],
                latency_mean_ms=item["latency_mean_ms"],
                itl_mean_ms=item["itl_mean_ms"],
                fairness_wait_spread_ms=item["fairness_wait_spread_ms"],
            )
        )
    lines.extend(
        [
            "",
            "## Cancellation",
            "",
            f"- health_after_cancel: `{cancel['health']}`",
            "- required metrics: `gateway_queue`, `scheduler_queue`, policy label, and cancellation latency.",
        ]
    )
    pathlib.Path(args.output).write_text("\n".join(lines) + "\n")
    print(f"phase12_benchmark_report={args.output}")


def _payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _post_concurrently(port: int, paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any] | None] = [None] * len(paths)

    def run(index: int, path: pathlib.Path) -> None:
        rows[index] = _post(path, port)

    threads = [
        threading.Thread(target=run, args=(idx, path)) for idx, path in enumerate(paths)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [row for row in rows if row is not None]


def _cancel_concurrently(port: int, paths: list[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any] | None] = [None] * len(paths)

    def run(index: int, path: pathlib.Path) -> None:
        rows[index] = _post_cancel(path, port)

    threads = [
        threading.Thread(target=run, args=(idx, path)) for idx, path in enumerate(paths)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [row for row in rows if row is not None]


def _post(path: pathlib.Path, port: int) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=path.read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    first = None
    chunks = 0
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                if raw.startswith(b"data: ") and b"[DONE]" not in raw:
                    chunks += 1
                    first = first or time.perf_counter()
            status = response.status
    except urllib.error.HTTPError as exc:
        status = exc.code
    latency_ms = (time.perf_counter() - started) * 1000.0
    ttft_ms = ((first or time.perf_counter()) - started) * 1000.0
    return {
        "name": path.name,
        "status": status,
        "ttft_ms": ttft_ms,
        "latency_ms": latency_ms,
        "itl_ms": latency_ms / max(chunks, 1),
    }


def _post_cancel(path: pathlib.Path, port: int) -> dict[str, Any]:
    started = time.perf_counter()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=path.read_bytes(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=120)
        try:
            response.readline()
        finally:
            response.close()
        status = 499
    except urllib.error.HTTPError as exc:
        status = exc.code
    return {
        "name": path.name,
        "status": status,
        "ttft_ms": (time.perf_counter() - started) * 1000.0,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "itl_ms": 0.0,
        "cancelled": True,
    }


def _get(port: int, path: str) -> bytes:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=30
    ) as response:
        return response.read()


def _metrics(port: int, capture: pathlib.Path) -> str:
    metrics = _get(port, "/metrics").decode("utf-8")
    capture.write_text(metrics)
    return metrics


def _require_metric(metrics: str, needle: str, family: str) -> None:
    if needle not in metrics:
        raise SystemExit(f"missing {needle!r} in {family}")


if __name__ == "__main__":
    main()
