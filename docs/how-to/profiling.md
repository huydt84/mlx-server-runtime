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

## Profile with `mlx-air profile`

`mlx-air profile` is the profiling entrypoint of the installed CLI;
`mlx-air profiler` is an alias. It runs in the same managed Python environment
as `mlx-air bench`.

### Capture

```bash
mlx-air profile run --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --requests 4 --concurrency 2 --max-tokens 64 --stream --open
```

`run` starts its own native-mlx gateway on a free loopback port with pipeline
profiling enabled, waits for readiness, sends the requested number of
greedy chat requests (`temperature` 0) through `/v1/chat/completions`, stops
the gateway, and joins the client-side HTTP round trip of each request with
the worker events. Profiling must be enabled before the worker starts, so
`run` cannot attach to a server that is already running.

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | required | Checkpoint or local model path. |
| `--prompt` | fixed prompt | User message sent with every request. |
| `--max-tokens` | 64 | Completion tokens per request (1–8192). |
| `--requests` | 1 | Profiled requests (1–1000). |
| `--concurrency` | 1 | Requests in flight at once (1–64, at most `--requests`). |
| `--stream` | off | Use streaming chat completions. |
| `--metal` | off | Capture `pipeline.gputrace` for the first request; needs `MTL_CAPTURE_ENABLED=1`. |
| `--output-dir` | `./profiles/<run id>` | New or empty artifact directory. |
| `--readiness-timeout` | 900 | Seconds to wait for readiness. |
| `--request-timeout` | 600 | Seconds allowed per request. |
| `--open` | off | Open `profile.html` when finished. |

With `--metal`, only the first request runs under Metal frame capture. That
request runs many times slower, and the `.gputrace` can take several GiB even
for a few tokens (about 9 GiB for 8 tokens of a 7B 4-bit model). When other
requests exist, the captured request stays in the timeline and request table
but is excluded from the means and stage totals; use `--requests 2` or more
so a measured request remains.

The command prints the path of `profile.html` on stdout and a short stage
summary on stderr. It exits 0 when every request succeeded, 50 when the
server failed to start or any request failed, and 128 plus the signal number
when interrupted; the gateway is stopped in every case.

### Visualize

`profile.html` loads nothing from the network. It shows:

- request count, mean worker latency, mean HTTP round trip, mean TTFT, and
  generated tokens;
- mean milliseconds per request for each pipeline stage, largest first;
- a zoomable timeline with one block per request and one row per component
  (scroll to zoom, drag to pan, hover for a span's duration, phase, and state);
- per-request and per-stage tables.

Rebuild the page for any profile directory, including ones captured with
`scripts/profile.sh` or `mlx-air bench diagnose`:

```bash
mlx-air profile view profiles/<run-id> --open
```

Worker warmup requests (request IDs starting with `warmup`) are excluded
unless you pass `--include-warmup`.

### Compare

```bash
mlx-air profile compare profiles/before profiles/after --output compare.md
```

`compare` prints a Markdown table of mean milliseconds per request for request
latency, TTFT, and every stage. Each delta is candidate minus baseline, with a
percentage relative to the baseline; lower time is better, so a negative delta
is labeled `better (lower)`. Capture both profiles with the same model,
prompt, token count, request count, and concurrency: a stage that runs once
per batch is counted once for each request in the batch, so per-request means
are only comparable at the same concurrency. A profile comparison points at
where to look; confirm any performance conclusion with `mlx-air bench run`.

## Profile from a Repository Checkout

The default validation checkpoint is
`mlx-community/Qwen2.5-7B-Instruct-4bit`. Override it when necessary:

```bash
bash scripts/profile.sh --checkpoint /path/to/checkpoint
```

Run `bash scripts/profile.sh --help` to list every option. Each flag also has
an `MLX_PROFILE_*` environment variable; a flag wins over its variable.

### Capture a Pipeline Profile

Run the bounded public-gateway workload from the repository root:

```bash
bash scripts/profile.sh
```

By default, artifacts are written below the system temporary directory and
the gateway listens on port 18016. Choose a durable location, or another port
when 18016 is taken, with:

```bash
bash scripts/profile.sh --artifacts-dir "$PWD/profiles/native-mlx" --port 18516
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

`mlx-air profile run` also writes `profile.html`, `profile-summary.json`
(per-request and per-stage totals), `runtime.toml`, and `worker.log`.

The worker writes these files incrementally while it serves requests: each
flush appends only the events recorded since the previous flush and re-renders
the report from running stage totals, so profiler overhead per request stays
constant over long profiled runs instead of growing with the stored event count.

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
MTL_CAPTURE_ENABLED=1 bash scripts/profile.sh --metal
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
