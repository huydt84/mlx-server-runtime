"""Phase 10 host validation probes and benchmark helpers."""

from __future__ import annotations

import argparse
import http.client
import json
import math
import pathlib
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

from mlx_worker.native_mlx.bootstrap import (
    build_finalized_token_ids,
    resolve_model_path,
)


@dataclass(frozen=True)
class RequestResult:
    """Observed public gateway request timing."""

    name: str
    status: int
    ttft_ms: float
    latency_ms: float
    body: str


def main() -> None:
    """Run one Phase 10 helper subcommand."""

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

    probe = subcommands.add_parser("native-probes")
    probe.add_argument("--request-dir", required=True)
    probe.add_argument("--capture", required=True)
    probe.add_argument("--metrics-capture", required=True)
    probe.add_argument("--port", type=int, required=True)

    incompatible = subcommands.add_parser("incompatible-miss")
    incompatible.add_argument("--request-dir", required=True)
    incompatible.add_argument("--port", type=int, required=True)

    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("--request-dir", required=True)
    benchmark.add_argument("--port", type=int, required=True)
    benchmark.add_argument("--backend", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--metrics-capture")

    report = subcommands.add_parser("report")
    report.add_argument("--native-json", required=True)
    report.add_argument("--v1-json", required=True)
    report.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "fixtures":
        write_fixtures(args)
    elif args.command == "native-probes":
        run_native_probes(args)
    elif args.command == "incompatible-miss":
        run_incompatible_miss(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "report":
        write_report(args)


def write_fixtures(args: argparse.Namespace) -> None:
    """Write runtime configs and token-checked request fixtures."""

    source = pathlib.Path(args.runtime_template).read_text()
    native_target = pathlib.Path(args.native_config)
    v1_target = pathlib.Path(args.v1_config)
    request_dir = pathlib.Path(args.request_dir)
    request_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint

    native_target.write_text(
        source.replace("port = 8000", f"port = {args.native_port}")
        .replace('backend = "v1"', 'backend = "native-mlx"')
        .replace(
            'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
            f'model = "{checkpoint}"',
        )
        .replace(
            'ipc_path = "/tmp/mlx-runtime.sock"',
            'ipc_path = "/tmp/mlx-runtime-phase10-native.sock"',
        )
    )
    v1_target.write_text(
        source.replace("port = 8000", f"port = {args.v1_port}")
        .replace(
            'model = "mlx-community/Qwen2.5-7B-Instruct-4bit"',
            f'model = "{checkpoint}"',
        )
        .replace(
            'ipc_path = "/tmp/mlx-runtime.sock"',
            'ipc_path = "/tmp/mlx-runtime-phase10-v1.sock"',
        )
    )

    model_path = resolve_model_path(checkpoint)
    shared = _repeated_words(model_path, 112, "shared")
    partial_shared = _repeated_words(model_path, 64, "shared")
    print(f"phase10_shared_prompt_tokens={_token_count(model_path, shared)}")
    print(f"phase10_partial_prompt_tokens={_token_count(model_path, partial_shared)}")

    payloads = {
        "miss.json": _payload(checkpoint, shared + " first suffix", stream=True),
        "exact.json": _payload(checkpoint, shared + " second suffix", stream=True),
        "partial.json": _payload(
            checkpoint, partial_shared + " partial suffix", stream=True
        ),
        "tail.json": _payload(checkpoint, "one-token-tail-overlap", stream=True),
        "cancel.json": _payload(
            checkpoint, shared + " cancellation suffix", stream=True
        ),
        "failure.json": _payload(checkpoint, "", stream=True, max_tokens=-1),
        "v1.json": _payload(
            checkpoint,
            "Say hello in one short sentence.",
            stream=False,
            max_tokens=4,
        ),
        "bench_short.json": _payload(
            checkpoint,
            "Say hello in one short sentence.",
            stream=True,
            max_tokens=4,
        ),
        "bench_miss.json": _payload(
            checkpoint, shared + " benchmark miss", stream=True
        ),
        "bench_exact.json": _payload(
            checkpoint, shared + " benchmark exact", stream=True
        ),
        "bench_partial.json": _payload(
            checkpoint, partial_shared + " benchmark partial", stream=True
        ),
    }
    metadata: dict[str, dict[str, int]] = {}
    for name, payload in payloads.items():
        (request_dir / name).write_text(json.dumps(payload))
        content = str(payload["messages"][0]["content"])
        metadata[name] = {
            "prompt_tokens": _token_count(model_path, content),
            "max_tokens": int(payload["max_tokens"]),
        }
    (request_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run_native_probes(args: argparse.Namespace) -> None:
    """Run Phase 10 correctness probes against a native public gateway."""

    request_dir = pathlib.Path(args.request_dir)
    capture = pathlib.Path(args.capture)
    metrics_capture = pathlib.Path(args.metrics_capture)
    port = int(args.port)

    events: dict[str, Any] = {}
    events["miss"] = _post(request_dir / "miss.json", port)
    _require_status(events["miss"], "miss")
    print("phase10_public_miss_ok=1")

    events["exact"] = _post(request_dir / "exact.json", port)
    _require_status(events["exact"], "exact")
    after_exact = _metrics(port, metrics_capture)
    exact_reused = _metric_value(
        after_exact,
        'mlx_prefix_cache_reused_tokens_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    if exact_reused <= 0:
        raise SystemExit("exact full-page hit did not increase reused tokens")
    print("phase10_public_exact_hit_ok=1")

    events["partial"] = _post(request_dir / "partial.json", port)
    _require_status(events["partial"], "partial")
    after_partial = _metrics(port, metrics_capture)
    partial_reused = _metric_value(
        after_partial,
        'mlx_prefix_cache_reused_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    if partial_reused <= 0:
        raise SystemExit("partial full-page hit did not increase reused pages")
    print("phase10_public_partial_hit_ok=1")

    tail_before = _metric_value(
        after_partial,
        'mlx_prefix_cache_hits_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    tail_reused_before = _metric_value(
        after_partial,
        'mlx_prefix_cache_reused_tokens_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    events["tail"] = _post(request_dir / "tail.json", port)
    _require_status(events["tail"], "tail")
    tail_after_metrics = _metrics(port, metrics_capture)
    tail_after = _metric_value(
        tail_after_metrics,
        'mlx_prefix_cache_hits_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    tail_reused_after = _metric_value(
        tail_after_metrics,
        'mlx_prefix_cache_reused_tokens_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    if tail_after - tail_before > 1 or tail_reused_after - tail_reused_before > 16:
        raise SystemExit(
            "partial-tail-only overlap reused more than the stable template page"
        )
    print("phase10_partial_tail_miss_ok=1")

    concurrent_out: dict[str, Any] = {}
    threads = [
        threading.Thread(
            target=lambda key: concurrent_out.update(
                {key: _post(request_dir / "exact.json", port)}
            ),
            args=(f"concurrent-{index}",),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(value["status"] != 200 for value in concurrent_out.values()):
        raise SystemExit(f"concurrent sharing request failed: {concurrent_out}")
    sharing_metrics = _metrics(port, metrics_capture)
    if (
        'mlx_prefix_cache_pinned_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}'
        not in sharing_metrics
    ):
        raise SystemExit("missing pinned-page sharing metric")
    print("phase10_concurrent_sharing_ok=1")

    _assert_cancel_cleanup(request_dir, port, sharing_metrics, metrics_capture)
    print("phase10_cancellation_cleanup_ok=1")

    failure_before = _metric_value(
        _metrics(port, metrics_capture),
        'mlx_prefix_cache_entries_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    events["failure"] = _post(request_dir / "failure.json", port)
    failure_after_metrics = _metrics(port, metrics_capture)
    failure_after = _metric_value(
        failure_after_metrics,
        'mlx_prefix_cache_entries_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    if failure_after < failure_before:
        raise SystemExit("failure probe lost existing reusable pages")
    print("phase10_failure_non_publication_ok=1")

    if (
        _metric_value(
            failure_after_metrics,
            'mlx_prefix_cache_evictions_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        )
        < 0
    ):
        raise SystemExit("eviction metric missing")
    print("phase10_eviction_ok=1")

    for needle in _required_metric_names():
        if needle not in failure_after_metrics:
            raise SystemExit(f"missing metric {needle}")
    print("phase10_metrics_labels_ok=1")
    capture.write_text(json.dumps(events, indent=2))


def run_incompatible_miss(args: argparse.Namespace) -> None:
    """Run an incompatible page-size miss probe."""

    result = _post(pathlib.Path(args.request_dir) / "exact.json", int(args.port))
    _require_status(result, "incompatible-key")
    print("phase10_incompatible_key_miss_ok=1")


def run_benchmark(args: argparse.Namespace) -> None:
    """Run public gateway benchmark scenarios."""

    request_dir = pathlib.Path(args.request_dir)
    metadata = json.loads((request_dir / "metadata.json").read_text())
    metrics_capture = (
        pathlib.Path(args.metrics_capture) if args.metrics_capture else None
    )
    scenarios = [
        ("short_baseline", "bench_short.json", 3),
        ("shared_prefix_miss", "bench_miss.json", 2),
        ("shared_prefix_exact", "bench_exact.json", 3),
        ("shared_prefix_partial", "bench_partial.json", 3),
    ]
    results: dict[str, Any] = {"backend": args.backend, "scenarios": []}
    for scenario, filename, samples in scenarios:
        print(f"benchmark_backend={args.backend} scenario={scenario} samples={samples}")
        observed = [
            _timed_post(scenario, request_dir / filename, args.port)
            for _ in range(samples)
        ]
        metrics_text = _metrics(args.port, metrics_capture) if metrics_capture else ""
        results["scenarios"].append(
            _summarize(
                args.backend,
                scenario,
                observed,
                metrics_text,
                metadata[filename],
            )
        )
    pathlib.Path(args.output).write_text(json.dumps(results, indent=2))


def write_report(args: argparse.Namespace) -> None:
    """Write a markdown v1/v2 benchmark comparison."""

    native = json.loads(pathlib.Path(args.native_json).read_text())
    v1 = json.loads(pathlib.Path(args.v1_json).read_text())
    by_name = {item["scenario"]: item for item in v1["scenarios"]}
    lines = [
        "# Phase 10 Native v2 Benchmark",
        "",
        "Compared public gateway requests for `native-mlx` block-hash APC and v1.",
        "All scenarios use the same checkpoint, prompt fixtures, request parameters, and public `/v1/chat/completions` surface.",
        "",
        "| scenario | backend | samples | ttft_mean_ms | latency_mean_ms | prompt_tokens_mean | completion_tokens_mean | reused_tokens | reused_pages | scheduled_prefill_tokens | notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for native_item in native["scenarios"]:
        v1_item = by_name[native_item["scenario"]]
        for item in (v1_item, native_item):
            lines.append(
                "| {scenario} | {backend} | {samples} | {ttft_mean_ms:.1f} | {latency_mean_ms:.1f} | {prompt_tokens_mean:.1f} | {completion_tokens_mean:.1f} | {reused_tokens:.0f} | {reused_pages:.0f} | {scheduled_prefill_tokens:.0f} | {notes} |".format(
                    **item
                )
            )
    lines.extend(
        [
            "",
            "## Delta Summary",
            "",
        ]
    )
    for native_item in native["scenarios"]:
        v1_item = by_name[native_item["scenario"]]
        lines.append(
            "- {scenario}: native TTFT {native:.1f} ms vs v1 {v1:.1f} ms ({delta:+.1f} ms).".format(
                scenario=native_item["scenario"],
                native=native_item["ttft_mean_ms"],
                v1=v1_item["ttft_mean_ms"],
                delta=native_item["ttft_mean_ms"] - v1_item["ttft_mean_ms"],
            )
        )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"phase10_benchmark_report={args.output}")


def _payload(
    checkpoint: str,
    content: str,
    *,
    stream: bool,
    max_tokens: int = 1,
) -> dict[str, Any]:
    return {
        "model": checkpoint,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": stream,
    }


def _token_count(model_path: pathlib.Path, content: str) -> int:
    return len(
        build_finalized_token_ids(model_path, [{"role": "user", "content": content}])
    )


def _repeated_words(model_path: pathlib.Path, target_tokens: int, prefix: str) -> str:
    words: list[str] = []
    index = 0
    while _token_count(model_path, " ".join(words)) < target_tokens:
        words.append(f"{prefix}{index}")
        index += 1
    return " ".join(words)


def _post(path: pathlib.Path, port: int) -> dict[str, Any]:
    body = path.read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Accept", "text/event-stream")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    payload = response.read().decode("utf-8", errors="replace")
    return {"status": response.status, "body": payload}


def _timed_post(name: str, path: pathlib.Path, port: int) -> RequestResult:
    body = path.read_bytes()
    started = time.perf_counter()
    ttft_ms = math.nan
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Accept", "text/event-stream")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    response = conn.getresponse()
    chunks: list[str] = []
    while True:
        raw = response.fp.readline()
        if not raw:
            break
        if math.isnan(ttft_ms):
            ttft_ms = (time.perf_counter() - started) * 1000
        chunks.append(raw.decode("utf-8", errors="replace").rstrip())
    latency_ms = (time.perf_counter() - started) * 1000
    return RequestResult(
        name=name,
        status=response.status,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        body="\n".join(chunks),
    )


def _metrics(port: int, capture: pathlib.Path | None = None) -> str:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", "/metrics")
    response = conn.getresponse()
    body = response.read().decode("utf-8", errors="replace")
    if response.status != 200:
        raise SystemExit(f"metrics status {response.status}: {body}")
    if capture is not None:
        capture.write_text(body)
    return body


def _metric_value(text: str, needle: str) -> float:
    for line in text.splitlines():
        if line.startswith(needle + " "):
            return float(line.split()[-1])
    return 0.0


def _summarize(
    backend: str,
    scenario: str,
    observed: list[RequestResult],
    metrics_text: str,
    metadata: dict[str, int],
) -> dict[str, Any]:
    good = [item for item in observed if item.status == 200]
    if len(good) != len(observed):
        raise SystemExit(f"{backend} {scenario} had non-200 response")
    bodies = [_parse_stream_body(item.body) for item in good]
    return {
        "backend": backend,
        "scenario": scenario,
        "samples": len(good),
        "ttft_mean_ms": statistics.mean(item.ttft_ms for item in good),
        "latency_mean_ms": statistics.mean(item.latency_ms for item in good),
        "prompt_tokens_mean": _mean_or_metadata(
            [body["prompt_tokens"] for body in bodies],
            metadata["prompt_tokens"],
        ),
        "completion_tokens_mean": _mean_or_metadata(
            [body["completion_tokens"] for body in bodies],
            metadata["max_tokens"],
        ),
        "reused_tokens": _metric_value(
            metrics_text,
            'mlx_prefix_cache_reused_tokens_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        ),
        "reused_pages": _metric_value(
            metrics_text,
            'mlx_prefix_cache_reused_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        ),
        "scheduled_prefill_tokens": _metric_value(
            metrics_text,
            'mlx_scheduled_tokens_by_backend{backend="native-mlx",modality="text",phase="prefill"}',
        ),
        "notes": "-" if backend == "native-mlx" else "v1 baseline",
    }


def _mean_or_metadata(values: list[int], fallback: int) -> float:
    non_zero = [value for value in values if value > 0]
    if non_zero:
        return statistics.mean(non_zero)
    return float(fallback)


def _parse_stream_body(body: str) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for line in body.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line[6:])
        usage = payload.get("usage") or {}
        prompt_tokens = max(prompt_tokens, int(usage.get("prompt_tokens") or 0))
        completion_tokens = max(
            completion_tokens,
            int(usage.get("completion_tokens") or 0),
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def _require_status(result: dict[str, Any], name: str) -> None:
    if result["status"] != 200:
        raise SystemExit(f"{name} request failed: {result}")


def _assert_cancel_cleanup(
    request_dir: pathlib.Path,
    port: int,
    before_metrics: str,
    metrics_capture: pathlib.Path,
) -> None:
    active_before = _metric_value(before_metrics, "mlx_requests_active")
    pinned_before = _metric_value(
        before_metrics,
        'mlx_prefix_cache_pinned_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )
    body = (request_dir / "cancel.json").read_bytes()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.putrequest("POST", "/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Accept", "text/event-stream")
    conn.putheader("Content-Length", str(len(body)))
    conn.endheaders()
    conn.send(body)
    conn.close()
    for _ in range(60):
        time.sleep(0.5)
        current = _metrics(port, metrics_capture)
        active_after = _metric_value(current, "mlx_requests_active")
        pinned_after = _metric_value(
            current,
            'mlx_prefix_cache_pinned_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        )
        if active_after <= active_before and pinned_after <= pinned_before:
            return
    raise SystemExit("cancelled public stream did not release active request resources")


def _required_metric_names() -> tuple[str, ...]:
    return (
        'mlx_prefix_cache_queries_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_hits_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_misses_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_reused_tokens_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_reused_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_entries_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_bytes_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_pinned_pages_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_collisions_rejected_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
        'mlx_prefix_cache_evictions_by_backend{backend="native-mlx",modality="text",strategy="block-hash"}',
    )


if __name__ == "__main__":
    main()
