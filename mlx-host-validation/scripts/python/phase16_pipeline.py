"""Run and join the public-gateway side of Phase 16 profiling."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import fields
from pathlib import Path

from mlx_worker.native_mlx.pipeline_profile import (
    PipelineEvent,
    write_pipeline_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    request_id, gateway_event, metrics = run_request(args.url, args.model, args.run_id)
    events = load_worker_events(args.output_dir)
    events.append(gateway_event)
    write_pipeline_artifacts(events, args.output_dir)
    request_events = [event for event in events if event.request_id == request_id]
    required = {
        "runtime",
        "transport",
        "scheduler",
        "executor",
        "cache",
        "model",
        "sampling",
        "mlx",
        "streaming",
        "gateway",
    }
    present = {event.component for event in request_events}
    missing = sorted(required - present)
    if missing:
        raise SystemExit(f"pipeline profile missing components: {', '.join(missing)}")
    print(f"phase16_request_id={request_id}")
    terminal = next(
        event
        for event in request_events
        if event.component == "runtime" and event.stage == "terminal"
    )
    details = terminal.details or {}
    stage_ms = {
        stage: sum(
            event.duration_us for event in request_events if event.stage == stage
        )
        // 1_000
        for stage in (
            "select_work",
            "batch_prepare",
            "reserve",
            "forward_dispatch",
            "sample",
            "detokenization",
            "response_assembly",
            "synchronize_eval",
        )
    }
    metrics.update(
        {
            "backend": "native-mlx",
            "ttft_ms": int(details.get("ttft_ms") or 0),
            "itl_ms": int(details.get("decode_time_ms") or 0)
            // max(1, terminal.completion_tokens or 1),
            "queue_ms": int(details.get("scheduler_queue_wait_ms") or 0),
            "scheduler_ms": stage_ms["select_work"],
            "executor_ms": stage_ms["batch_prepare"],
            "model_forward_ms": stage_ms["forward_dispatch"],
            "cache_ms": stage_ms["reserve"],
            "sampling_ms": stage_ms["sample"],
            "detokenization_ms": stage_ms["detokenization"],
            "streaming_overhead_ms": stage_ms["response_assembly"],
            "synchronization_ms": stage_ms["synchronize_eval"],
        }
    )
    for key, value in metrics.items():
        print(f"phase16_{key}={value}")
    print(f"phase16_pipeline_events={args.output_dir / 'pipeline-events.jsonl'}")
    print(f"phase16_pipeline_trace={args.output_dir / 'pipeline-trace.json'}")
    print(f"phase16_pipeline_report={args.output_dir / 'pipeline-report.md'}")
    print("phase_16_validation_ok=1")


def run_request(
    url: str, model: str, run_id: str
) -> tuple[str, PipelineEvent, dict[str, int]]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Count from one to four."}],
            "max_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
        }
    ).encode()
    started_ns = time.perf_counter_ns()
    with urllib.request.urlopen(
        urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        ),
        timeout=300,
    ) as response:
        payload = json.loads(response.read())
    finished_ns = time.perf_counter_ns()
    request_id = str(payload["id"]).removeprefix("chatcmpl-")
    usage = payload.get("usage", {})
    duration_us = (finished_ns - started_ns) // 1_000
    event = PipelineEvent(
        schema_version=1,
        run_id=run_id,
        request_id=request_id,
        backend="native-mlx",
        model=model,
        workload="phase16-public-gateway",
        component="gateway",
        stage="http_round_trip",
        monotonic_ns=started_ns,
        offset_us=0,
        duration_us=duration_us,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        state="completed",
        details={"stream": False, "status": 200},
    )
    return (
        request_id,
        event,
        {
            "total_latency_ms": max(1, duration_us // 1_000),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
        },
    )


def load_worker_events(output_dir: Path) -> list[PipelineEvent]:
    path = output_dir / "pipeline-events.jsonl"
    allowed = {field.name for field in fields(PipelineEvent)}
    return [
        PipelineEvent(
            **{key: value for key, value in json.loads(line).items() if key in allowed}
        )
        for line in path.read_text().splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
