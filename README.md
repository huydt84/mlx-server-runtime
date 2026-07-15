# MLX Server Runtime

Rust control plane + Python MLX worker for local Apple Silicon inference.

> A serious local MLX inference service for Apple Silicon.
> This is **not** a proxy around `mlx_lm.server`.
> Rust owns serving. Python owns model execution.

## Documentation

Full documentation is in the [`docs/`](docs/) directory:

| Document | Description |
|----------|-------------|
| [Getting Started](docs/tutorial/getting-started.md) | Zero to first inference in 10 steps |
| [Architecture](docs/explanation/architecture.md) | Design rationale and system design |
| [HTTP API Reference](docs/reference/api.md) | All endpoints, schemas, error codes |
| [Configuration Reference](docs/reference/configuration.md) | All TOML fields and defaults |
| [IPC Protocol](docs/reference/protocol.md) | UDS handshake and frame format |
| [Profile Inference](docs/how-to/profiling.md) | Capture and interpret whole-pipeline timing |
| [Contributing](CONTRIBUTING.md) | Add model architectures, debug layer output, and profile model graphs |

## Quick Start

```bash
# Build
cargo build --workspace
cd python && uv sync && cd ..

# Run
cargo run --bin gateway

# In another terminal
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen2.5-7B-Instruct-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Endpoints

- `GET /live` checks whether HTTP process is alive.
- `GET /ready` checks whether current model can serve requests.
- `GET /startup` reports startup phase.
- `GET /health` remains backwards-compatible with liveness-style checks.
- `GET /version` reports the gateway package version.
- `GET /models` lists configured model state.
- `GET /models/{model}/status` returns detailed model lifecycle state.
- `GET /models/{model}/ready` returns model-specific readiness.
- `GET /metrics` exposes Prometheus metrics.

## Telemetry

Default config enables Prometheus metrics:

```toml
[telemetry]
enable_prometheus = true
metrics_path = "/metrics"
```

Request logs are structured JSON lines written by the Rust gateway and include:

- `request_id`
- `model`
- `prompt_tokens`
- `max_tokens`
- `stream`
- `queue_time_ms`
- `ttft_ms`
- `latency_ms`
- `completion_tokens`
- `finish_reason`
- `cancelled`
- `error`

## Benchmarks

The native-v2 ultimate benchmark is the required performance gate for
optimization work:

```bash
bash scripts/benchmark-v2.sh run
```

For hot-path optimization, this is the only accepted user-facing performance
gate. Keep `scripts/benchmark-v2.sh` and `benchmarks/v2_benchmark.py` unchanged
across optimization work: run the exact same benchmark files from isolated
baseline and candidate source worktrees. If the benchmark itself must change,
make that a separate change and rebaseline both snapshots with the new version;
results produced by different benchmark versions are not comparable.

It runs four supported model families across serial and overlap
configurations, rotates configuration order, and covers interactive streaming,
non-streaming latency, long prefill, sustained decode, shared-prefix reuse,
concurrency, and mixed prefill/decode pressure. The report leads with
user-visible TTFT, latency, and throughput. Lower TTFT/latency is explicitly
reported as better; higher throughput is explicitly reported as better.

Fair performance runs always keep profiling disabled. The same command then
starts separate diagnostic processes for whole-pipeline/system profiling and
model inference-graph profiling. Add `MTL_CAPTURE_ENABLED=1 --metal` for a
bounded Metal `.gputrace`; captured timings are diagnostic and never become
benchmark rows.

After every optimization, run the benchmark in the before and after source
snapshots, then compare their structured results:

```bash
bash scripts/benchmark-v2.sh compare \
  --baseline /path/to/before/results.json \
  --candidate /path/to/after/results.json
```

The comparison rejects mismatched models, prompts, token limits, sample counts,
configuration order, or cache budgets. It checks deterministic output/token
parity and writes absolute before/after values, 95% confidence intervals, and
an explicit pass/fail regression verdict. Do not claim an optimization works
from a one-model smoke run, a profiled timing, or two scripts run against the
same changed source tree.

Other benchmark entrypoints remain available for narrower work:

- Text-only benchmark: `scripts/benchmark.sh`
- VLM benchmark: `scripts/benchmark-vlm.sh`

Default reports land in `benchmarks/results/`.

For the full argument list, smoke commands, and full-suite commands, read:

- [`docs/how-to/run-benchmarks.md`](docs/how-to/run-benchmarks.md)

Fast wiring check (not sufficient for an optimization claim):

```bash
bash scripts/benchmark-v2.sh run --preset smoke
```

## Profiling

Whole-pipeline profiling helps explain latency across the Rust gateway, worker
transport, Python runtime, scheduler, executor, cache, model, MLX
synchronization, detokenization, and response streaming. It is disabled by
default and must remain off during fair benchmark runs.

Run the bounded profiling workflow on an Apple Silicon host:

```bash
bash scripts/profile.sh
```

The workflow writes request-correlated JSONL events, a Chrome Trace/Perfetto
timeline, and a Markdown summary. For configuration, optional Metal capture,
artifact interpretation, and failure handling, see
[Profile inference](docs/how-to/profiling.md).
