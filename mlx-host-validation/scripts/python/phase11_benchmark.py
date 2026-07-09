"""Phase 11 host validation probes and benchmark helpers."""

from __future__ import annotations

import argparse
import json
import pathlib
import threading
from typing import Any

import phase10_benchmark as phase10
from mlx_worker.native_mlx.bootstrap import resolve_model_path


def main() -> None:
    """Run one Phase 11 helper subcommand."""

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

    probes = subcommands.add_parser("strategy-probes")
    probes.add_argument("--request-dir", required=True)
    probes.add_argument("--port", type=int, required=True)
    probes.add_argument("--strategy", required=True, choices=("radix", "block-hash"))
    probes.add_argument("--metrics-capture", required=True)

    contract = subcommands.add_parser("cache-contract")
    contract.add_argument("--request-dir", required=True)
    contract.add_argument("--capture", required=True)
    contract.add_argument("--metrics-capture", required=True)
    contract.add_argument("--port", type=int, required=True)
    contract.add_argument("--strategy", required=True, choices=("radix", "block-hash"))

    incompatible = subcommands.add_parser("incompatible-miss")
    incompatible.add_argument("--request-dir", required=True)
    incompatible.add_argument("--port", type=int, required=True)

    benchmark = subcommands.add_parser("benchmark")
    benchmark.add_argument("--request-dir", required=True)
    benchmark.add_argument("--port", type=int, required=True)
    benchmark.add_argument("--backend", required=True)
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--metrics-capture")

    graph_profile = subcommands.add_parser("graph-profile")
    graph_profile.add_argument("--request-dir", required=True)
    graph_profile.add_argument("--port", type=int, required=True)
    graph_profile.add_argument("--output", required=True)
    graph_profile.add_argument("--metrics-capture", required=True)

    report = subcommands.add_parser("report")
    report.add_argument("--radix-json", required=True)
    report.add_argument("--block-hash-json", required=True)
    report.add_argument("--v1-json", required=True)
    report.add_argument("--graph-profile-json")
    report.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "fixtures":
        write_fixtures(args)
    elif args.command == "strategy-probes":
        run_strategy_probes(args)
    elif args.command == "cache-contract":
        phase10.run_native_probes(args)
    elif args.command == "incompatible-miss":
        phase10.run_incompatible_miss(args)
    elif args.command == "benchmark":
        phase10.run_benchmark(args)
    elif args.command == "graph-profile":
        phase10.run_graph_profile(args)
    elif args.command == "report":
        write_report(args)


def write_fixtures(args: argparse.Namespace) -> None:
    """Write shared Phase 10 fixtures plus Phase 11 branching prompts."""

    phase10.write_fixtures(args)
    request_dir = pathlib.Path(args.request_dir)
    model_path = resolve_model_path(args.checkpoint)
    branch = phase10._repeated_words(model_path, 96, "radix-shared")
    payloads = {
        "radix_branch_a.json": phase10._payload(
            args.checkpoint,
            branch + " alpha branch one",
            stream=True,
            max_tokens=2,
        ),
        "radix_branch_b.json": phase10._payload(
            args.checkpoint,
            branch + " beta branch two",
            stream=True,
            max_tokens=2,
        ),
        "radix_branch_c.json": phase10._payload(
            args.checkpoint,
            branch + " alpha branch three",
            stream=True,
            max_tokens=2,
        ),
    }
    metadata = json.loads((request_dir / "metadata.json").read_text())
    for name, payload in payloads.items():
        (request_dir / name).write_text(json.dumps(payload))
        metadata[name] = {
            "prompt_tokens": phase10._token_count(
                model_path,
                str(payload["messages"][0]["content"]),
            ),
            "max_tokens": int(payload["max_tokens"]),
        }
    (request_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def run_strategy_probes(args: argparse.Namespace) -> None:
    """Probe radix or block-hash strategy through the public gateway."""

    request_dir = pathlib.Path(args.request_dir)
    metrics_capture = pathlib.Path(args.metrics_capture)
    port = int(args.port)
    for filename in (
        "radix_branch_a.json",
        "radix_branch_b.json",
        "radix_branch_c.json",
    ):
        phase10._require_status(phase10._post(request_dir / filename, port), filename)
    concurrent: dict[str, Any] = {}
    threads = [
        threading.Thread(
            target=lambda key: concurrent.update(
                {key: phase10._post(request_dir / "radix_branch_a.json", port)}
            ),
            args=(f"concurrent-{index}",),
        )
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(value["status"] != 200 for value in concurrent.values()):
        raise SystemExit(f"concurrent strategy probe failed: {concurrent}")
    metrics = phase10._metrics(port, metrics_capture)
    labels = f'backend="native-mlx",modality="text",strategy="{args.strategy}"'
    reused = phase10._metric_value(
        metrics,
        f"mlx_prefix_cache_reused_tokens_by_backend{{{labels}}}",
    )
    if reused <= 0:
        raise SystemExit(f"{args.strategy} did not reuse prefix tokens")
    if args.strategy == "radix":
        if (
            phase10._metric_family_sum(
                metrics,
                "mlx_radix_cache_by_backend",
                {'backend="native-mlx"', 'modality="text"', 'strategy="radix"'},
            )
            <= 0
        ):
            raise SystemExit("radix metrics were not exported")
        print("phase11_radix_metrics_ok=1")
    print(f"phase11_{args.strategy.replace('-', '_')}_strategy_probe_ok=1")


def write_report(args: argparse.Namespace) -> None:
    """Write a markdown v1/block-hash/radix benchmark comparison."""

    radix = json.loads(pathlib.Path(args.radix_json).read_text())
    block_hash = json.loads(pathlib.Path(args.block_hash_json).read_text())
    v1 = json.loads(pathlib.Path(args.v1_json).read_text())
    graph_profile = (
        json.loads(pathlib.Path(args.graph_profile_json).read_text())
        if args.graph_profile_json
        else None
    )
    by_backend = {
        "v1": {item["scenario"]: item for item in v1["scenarios"]},
        "block-hash": {item["scenario"]: item for item in block_hash["scenarios"]},
        "radix": {item["scenario"]: item for item in radix["scenarios"]},
    }
    lines = [
        "# Phase 11 Native v2 Benchmark",
        "",
        "Compared public gateway requests for default `radix`, explicit `block-hash`, and v1.",
        "All scenarios use the same checkpoint, prompt fixtures, request parameters, and public `/v1/chat/completions` surface.",
        "",
        "| scenario | backend | mode | samples | ttft_mean_ms | latency_mean_ms | prompt_tokens_mean | completion_tokens_mean | prompt_tps | completion_tps | total_tps | reused_tokens | reused_pages | scheduled_prefill_tokens | scheduler_tick_ms | prefix_queries | prefix_hits | prefix_misses | radix_nodes | radix_splits | radix_shared_pages | radix_tree_depth | radix_leaf_evictions | notes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scenario in by_backend["radix"]:
        for backend in ("v1", "block-hash", "radix"):
            item = _with_metric_defaults(by_backend[backend][scenario])
            item["notes"] = _report_note(backend)
            lines.append(
                "| {scenario} | {backend} | {mode} | {samples} | {ttft_mean_ms:.1f} | {latency_mean_ms:.1f} | {prompt_tokens_mean:.1f} | {completion_tokens_mean:.1f} | {prompt_tokens_per_second:.1f} | {completion_tokens_per_second:.1f} | {total_tokens_per_second:.1f} | {reused_tokens:.0f} | {reused_pages:.0f} | {scheduled_prefill_tokens:.0f} | {scheduler_tick_ms:.0f} | {prefix_queries:.0f} | {prefix_hits:.0f} | {prefix_misses:.0f} | {radix_nodes:.0f} | {radix_splits:.0f} | {radix_shared_pages:.0f} | {radix_tree_depth:.0f} | {radix_leaf_evictions:.0f} | {notes} |".format(
                    **item,
                )
            )
    lines.extend(["", "## Delta Summary", ""])
    for scenario, radix_item in by_backend["radix"].items():
        block_item = by_backend["block-hash"][scenario]
        v1_item = by_backend["v1"][scenario]
        lines.append(
            "- {scenario}: radix latency {radix:.1f} ms vs block-hash {block:.1f} ms ({block_delta:+.1f} ms, {block_result}) and v1 {v1:.1f} ms ({v1_delta:+.1f} ms, {v1_result}).".format(
                scenario=scenario,
                radix=radix_item["latency_mean_ms"],
                block=block_item["latency_mean_ms"],
                block_delta=radix_item["latency_mean_ms"]
                - block_item["latency_mean_ms"],
                block_result=_classify_latency_delta(
                    radix_item["latency_mean_ms"] - block_item["latency_mean_ms"]
                ),
                v1=v1_item["latency_mean_ms"],
                v1_delta=radix_item["latency_mean_ms"] - v1_item["latency_mean_ms"],
                v1_result=_classify_latency_delta(
                    radix_item["latency_mean_ms"] - v1_item["latency_mean_ms"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "## Concurrent Request Detail",
            "",
            "| scenario | backend | request | ttft_ms | latency_ms |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for scenario in by_backend["radix"]:
        if "concurrent" not in scenario:
            continue
        for backend in ("v1", "block-hash", "radix"):
            item = by_backend[backend][scenario]
            for timing in item.get("request_timings", []):
                lines.append(
                    "| {scenario} | {backend} | {name} | {ttft_ms:.1f} | {latency_ms:.1f} |".format(
                        scenario=scenario,
                        backend=backend,
                        **timing,
                    )
                )
    if graph_profile is not None:
        lines.extend(["", "## Native Radix Graph Profile", ""])
        lines.append(
            "Graph profiling is collected in a separate native run with `MLX_RUNTIME_NATIVE_GRAPH_PROFILE=1`; it is diagnostic and excluded from fair latency/throughput rows."
        )
        lines.extend(
            [
                "",
                "| workload | samples | wall_ms | attention_ms | mlp_ms | projection_ms | norm_ms | layer_total_ms | worst_layer_ms | worst_layer_index | executor_eval_ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for item in graph_profile["workloads"]:
            lines.append(
                "| {workload} | {samples} | {wall_ms:.1f} | {model_graph_attention_ms:.0f} | {model_graph_mlp_ms:.0f} | {model_graph_projection_ms:.0f} | {model_graph_norm_ms:.0f} | {model_graph_layer_total_ms:.0f} | {model_graph_worst_layer_ms:.0f} | {model_graph_worst_layer_index:.0f} | {executor_eval_ms:.0f} |".format(
                    **item
                )
            )
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(f"phase11_benchmark_report={args.output}")


def _with_metric_defaults(item: dict[str, Any]) -> dict[str, Any]:
    with_defaults = dict(item)
    for key in (
        "prefix_queries",
        "prefix_hits",
        "prefix_misses",
        "radix_nodes",
        "radix_splits",
        "radix_shared_pages",
        "radix_tree_depth",
        "radix_leaf_evictions",
    ):
        with_defaults.setdefault(key, 0)
    return with_defaults


def _report_note(backend: str) -> str:
    if backend == "v1":
        return "v1 baseline"
    if backend == "block-hash":
        return "explicit block-hash"
    return "default radix"


def _classify_latency_delta(delta_ms: float) -> str:
    if delta_ms <= -5.0:
        return "improves"
    if delta_ms >= 5.0:
        return "regresses"
    return "neutral"


if __name__ == "__main__":
    main()
