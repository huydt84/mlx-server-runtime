# How to Run Benchmarks

Compare inference performance across three backends: raw `mlx-lm`, `mlx_lm.server`, and this runtime.

## Prerequisites

- Working Rust and Python builds (see [Getting Started](../tutorial/getting-started.md))
- Three benchmark scripts under `benchmarks/`
- At least one model downloaded and cached

## Run the Suite

```bash
bash scripts/benchmark.sh
```

This runs the Python comparison tool at `benchmarks/compare.py`, which executes all benchmark cases and writes a report to `benchmarks/results/phase_6_report.md`.

## Benchmark Cases

| Case | Description |
|------|-------------|
| A | 1 request, 512 prompt tokens, 128 completion |
| B | 4 concurrent requests, 512 prompt, 128 completion |
| C | 8 concurrent requests, 512 prompt, 128 completion |
| D | 1 long prompt (8192 tokens), 256 completion |
| E | Mixed workload: 4 short + 2 medium + 1 long prompt |

## Metrics Collected

- Time to first token (TTFT)
- End-to-end latency
- Tokens/sec per request
- Aggregate tokens/sec
- Queue time
- Worker CPU and memory usage
- KV cache bytes
- IPC overhead
- Error rate

## Customizing Benchmarks

Edit `benchmarks/compare.py` to change model, token counts, concurrency levels, or output path.

```bash
# Run with a specific model
bash scripts/benchmark.sh --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

## Interpreting Results

The report compares:

- **Raw `mlx-lm`**: Best-case single-request latency, no serving overhead.
- **`mlx_lm.server`**: Official server, baseline for serving quality.
- **This runtime**: Target — match or beat `mlx_lm.server` on serving metrics while providing better telemetry, cancellation, and queue control.
