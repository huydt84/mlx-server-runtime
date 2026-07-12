# Profile Inference

Use whole-pipeline profiling when you need to explain where request latency is
spent. The profiler correlates timing across the Rust gateway, worker
transport, Python runtime, scheduler, executor, cache, model, MLX
synchronization, detokenization, and response streaming.

Profiling is diagnostic and disabled by default. Do not compare profiled
latency with normal benchmark results.

## Requirements

- Apple Silicon with a working MLX environment
- `cargo` and the repository Python environment managed by `uv`
- A locally accessible model checkpoint
- Sufficient free space for trace artifacts
- Full Xcode when capturing a Metal `.gputrace`

The default validation checkpoint is
`mlx-community/Qwen2.5-7B-Instruct-4bit`. Override it when necessary:

```bash
export MLX_PROFILE_CHECKPOINT=/path/to/checkpoint
```

## Capture a Pipeline Profile

Run the bounded public-gateway workload from the repository root:

```bash
bash scripts/profile.sh
```

By default, artifacts are written below the system temporary directory. Choose
a durable location with:

```bash
export MLX_PROFILE_DIR="$PWD/profiles/native-mlx"
bash scripts/profile.sh
```

The command starts the gateway with profiling enabled, waits for readiness,
sends a bounded request through `/v1/chat/completions`, joins gateway and worker
events, validates the required pipeline components, and shuts the gateway down.

## Inspect the Artifacts

The trace directory contains:

- `pipeline-events.jsonl`: request-correlated stage events for scripts and
  detailed inspection.
- `pipeline-trace.json`: a Chrome Trace/Perfetto-compatible joined timeline.
- `pipeline-report.md`: stage totals ordered by their contribution to observed
  time.
- `gateway.log`: gateway and worker startup or failure output.

Every event includes the run ID, request ID, backend, model, workload,
component, stage, monotonic timestamp, duration, and relevant token or terminal
metadata.

Open `pipeline-trace.json` in Perfetto or another Chrome Trace-compatible
viewer. Start with long `mlx/synchronize_eval` spans and gaps between model
steps, then compare them with scheduler selection, cache work, model dispatch,
detokenization, and transport events.

## Capture a Metal Trace

Metal capture is heavier and must be enabled before the worker starts:

```bash
MTL_CAPTURE_ENABLED=1 \
MLX_PROFILE_METAL=1 \
bash scripts/profile.sh
```

The profiler brackets only the bounded public request and writes
`pipeline.gputrace` in the trace directory. Open it with Xcode to inspect Metal
kernels and buffer activity.

Captured wall-clock includes profiler overhead. Treat it as diagnostic evidence,
not benchmark latency.

For lower-overhead whole-process CPU/GPU utilization, stalls, thermals, and
overlap analysis, capture a Metal System Trace with Instruments or `xctrace`
around the same bounded workflow.

## Configure Profiling Manually

The host workflow sets these worker environment variables automatically:

| Variable | Purpose |
| --- | --- |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE=1` | Enables pipeline event collection. |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_DIR` | Selects the artifact directory. |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_RUN_ID` | Correlates gateway and worker events. |
| `MLX_RUNTIME_NATIVE_PIPELINE_PROFILE_WORKLOAD` | Labels the captured workload. |
| `MLX_RUNTIME_NATIVE_METAL_CAPTURE=1` | Enables bounded Metal capture. |

When profiling manually, set the variables before starting the Rust gateway so
the supervisor can forward them to the Python worker.

## Troubleshoot a Failed Capture

- If the script rejects the host architecture, run it on an Apple Silicon Mac.
- If Metal capture reports a missing startup flag, restart with
  `MTL_CAPTURE_ENABLED=1` in the gateway environment.
- If readiness times out, inspect `gateway.log` for checkpoint, tokenizer, or
  MLX startup errors.
- If a required component is missing, inspect `pipeline-events.jsonl` for the
  request ID printed by the script. Warmup events have a different request ID
  and do not satisfy public-request validation.
- If the trace is too large, reduce the workload instead of profiling a full
  benchmark suite.

After diagnosis, disable profiling and reproduce any performance comparison
with the normal benchmark workflow.
