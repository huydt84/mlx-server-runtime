# How to Run Benchmarks

Compare inference performance across three backends: raw `mlx-lm`, `mlx_lm.server`, and this runtime.

## Native-v2 Ultimate Benchmark

Use this gate after every native-v2 optimization:

```bash
bash scripts/benchmark-v2.sh run
```

The default `optimization` preset uses all four supported text model families,
20 target samples per scenario, two rotated configuration-order rounds,
serial-radix and overlap-radix configurations, and separate whole-pipeline and
model-graph profiles. Performance rows are collected before the diagnostic
profiles with all profiling disabled.

Artifacts are written beneath `benchmarks/results/v2/<timestamp>-<label>/`:

- `results.json`: structured manifest, raw samples, aggregates, environment,
  configuration, and source commit.
- `report.md`: absolute TTFT, latency, and throughput by model, configuration,
  and scenario.
- `models/<model>/<configuration>/system-profile/`: request-correlated pipeline
  JSONL, Chrome Trace/Perfetto timeline, Markdown stage report, and optional
  `.gputrace`.
- `models/<model>/<configuration>/graph-profile/graph-profile.json`: diagnostic
  attention, MLP, projection, normalization, layer, worst-layer, and executor
  metrics exposed by the model-agnostic graph profiler.

For a fast wiring check:

```bash
bash scripts/benchmark-v2.sh run --preset smoke
```

The smoke preset is not comprehensive enough for an optimization claim.

To capture Metal work, export the required process-start environment variable:

```bash
MTL_CAPTURE_ENABLED=1 bash scripts/benchmark-v2.sh run --metal
```

To compare independently checked-out before/after source snapshots:

```bash
bash scripts/benchmark-v2.sh compare \
  --baseline /path/to/before/results.json \
  --candidate /path/to/after/results.json \
  --max-regression-pct 2
```

The comparison exits non-zero for output/token parity failures or statistically
supported regressions beyond the configured noise budget. Latency and TTFT are
reverse metrics, so the report says `lower` and `better` explicitly. Throughput
rows say `higher` and `better` explicitly.

Use `bash scripts/benchmark-v2.sh run --help` for model, configuration, preset,
profiling, output, and baseline options.

## Prerequisites

- Working Rust and Python builds (see [Getting Started](../tutorial/getting-started.md))
- Three benchmark scripts under `benchmarks/`
- At least one model downloaded and cached

## Text-Only Benchmark

```bash
bash scripts/benchmark.sh
```

This runs `benchmarks/compare.py` and writes its default report under
`benchmarks/results/`.

Use `bash scripts/benchmark.sh --help` for parser help from `benchmarks/compare.py`.

### Arguments

Defaults come from `benchmarks/compare.py`.

| Flag | Default | If you change this | Time impact | Result impact | When to use / pair with |
|------|---------|--------------------|-------------|---------------|--------------------------|
| `--model` | built-in list: `mlx-community/LFM2.5-8B-A1B-MLX-4bit`, `mlx-community/Qwen3-4B-Instruct-2507-4bit`, `mlx-community/gemma-3-270m-it-qat-8bit` | Runs only models you name. Repeat flag for multiple models. | More models scale runtime almost linearly. | Changes model quality, token speed, latency, memory footprint. | Use for focused regressions or one-model smoke runs. Pair with `--prompt`, `--trials`, `--report-path`. |
| `--prompt` | none; benchmark uses built-in suite when omitted | Replaces built-in prompt suite with exact prompts you provide. Repeat flag for multiple prompts. | Fewer prompts shorten run; more prompts lengthen run linearly. | Narrows workload coverage to your chosen prompt shapes. | Use for reproducing one workload. Pair with `--model`, `--max-tokens`, `--trials`. |
| `--prompt-limit` | `8` | Changes how many built-in prompts run. `0` means all built-in prompts. | Higher value increases runtime. `0` can be much slower. | More prompts improve coverage and stability across prompt types. | Use with built-in suite only. Pair with `--include-long-prompts` for broadest run. |
| `--include-long-prompts` | off | Adds long-prefill prompts to built-in suite. | Large increase. Long prefill dominates runtime. | Exposes prefill behavior, cache pressure, and long-context latency. | Use for serious evaluation, not newcomer smoke. Pair with `--prompt-limit 0`, maybe lower `--trials` if time constrained. |
| `--prefill-step-size` | `2048` | Changes target long-prompt size generation. Larger value means larger long prompts. | Higher values can increase runtime sharply. | Changes long-context stress level and comparability across runs. | Only matters with `--include-long-prompts`. Pair with `--long-prompt-multiplier`. |
| `--long-prompt-multiplier` | `2` | Changes long-prompt expansion target. Larger multiplier makes long prompts longer. | Higher multiplier increases runtime and memory pressure. | Changes prefill-heavy workload severity. | Only use when tuning long-context stress. Pair with `--include-long-prompts` and document value in report name. |
| `--max-tokens` | `256` | Caps generated completion length for each sample. | Higher value usually increases runtime. | Higher values weight decode performance more; lower values emphasize prompt/prefill and TTFT. | Lower for smoke, higher for throughput/decode study. Pair with `--prompt` or `--include-long-prompts` depending goal. |
| `--report-path` | script default | Writes the Markdown report elsewhere. | No material timing effect. | No metric effect. Only output location changes. | Use to keep multiple runs side by side. Pair with descriptive filenames. |
| `--project-port` | `8000` | Points project benchmark client at different gateway port. | No metric effect by itself. Wrong value causes failures. | No benchmark-shape effect if service is same. | Use when project server already running on non-default port. Pair with matching server launch config. |
| `--server-port` | `8001` | Points benchmark client at different `mlx_lm.server` port. | No metric effect by itself. Wrong value causes failures. | No benchmark-shape effect if service is same. | Use when `mlx_lm.server` runs on different port. Pair with matching server launch config. |
| `--warmup-trials` | `1` | Changes untimed-ish warmup passes per case. `0` removes warmup. | Extra warmups add runtime linearly. | More warmup can reduce first-run noise from model/server cold paths. | Use `0` for smoke, `1` for normal comparisons, higher only if cold-start noise is obvious. Pair with `--trials`. |
| `--trials` | `1` | Changes measured runs per case. | Increases runtime linearly. | More trials improve mean/percentile stability and reduce noise. | Use `1` for smoke, `3+` for personal analysis, higher if comparing close regressions. Pair with `--warmup-trials`. |

Help output:

```bash
bash scripts/benchmark.sh --help
```

### Examples

Pin one model:

```bash
bash scripts/benchmark.sh --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

Custom prompts:

```bash
bash scripts/benchmark.sh \
  --prompt "Write one sentence about Apple Silicon." \
  --prompt "Summarize Metal in 12 words." \
  --warmup-trials 0 \
  --trials 1
```

### Fast-Smoke

Newcomer check. Smallest useful run that confirms script wiring, backend startup, and report generation:

```bash
bash scripts/benchmark.sh \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --prompt "Say hello in one short sentence." \
  --warmup-trials 0 \
  --trials 1 \
  --max-tokens 8 \
  --report-path benchmarks/results/text_smoke.md
```

### Full-Suite

Most comprehensive fair run. Uses default model list, all built-in prompts, long prompts, and default backend comparison:

```bash
bash scripts/benchmark.sh \
  --prompt-limit 0 \
  --include-long-prompts \
  --warmup-trials 1 \
  --trials 1 \
  --report-path benchmarks/results/text_full.md
```

### Personal Run Suite

```bash
bash scripts/benchmark.sh \
  --prompt-limit 20 \
  --include-long-prompts \
  --warmup-trials 1 \
  --trials 3 \
  --max-tokens 256 \
  --report-path benchmarks/results/text_inference_bench.md
```

## VLM Benchmark

```bash
bash scripts/benchmark-vlm.sh
```

This runs `benchmarks/compare_vlm.py` and writes its default Markdown report
under `benchmarks/results/`.

Use `bash scripts/benchmark-vlm.sh --help` for parser help from `benchmarks/compare_vlm.py`.

### Arguments

Defaults come from `benchmarks/compare_vlm.py`.

| Flag | Default | If you change this | Time impact | Result impact | When to use / pair with |
|------|---------|--------------------|-------------|---------------|--------------------------|
| `--vlm` | off | No behavior change. Accepted only for CLI symmetry. | None. | None. | Ignore unless you want explicit text in scripts that branch between text-only and VLM modes. |
| `--model` | built-in list: `mlx-community/gemma-3-4b-it-4bit`, `mlx-community/Qwen3.5-2B-4bit`, `mlx-community/Qwen2-VL-2B-Instruct-4bit` | Runs only named VLMs. Repeat flag for multiple models. | More models increase runtime almost linearly. | Changes model quality, image understanding, token speed, and memory pressure. | Use one model for smoke or regression repro. Omit for broader comparison. Pair with `--output-md`/`--output-json`. |
| `--backend` | none; all backends used when omitted | Restricts compared backends to explicit set. Repeat flag for multiple values like `raw`, `server`, `project`. | Fewer backends shorten runs. | Removes comparison rows you skip. Can make fairness conclusions narrower. | Use for debugging one backend. Pair with `--skip-*` rarely; usually choose one approach or the other. |
| `--all-backends` | off, but effect is same as omitting `--backend` | Forces backend set to `raw`, `server`, `project`. | Same as normal all-backend run. | Restores full comparison surface. | Use for explicitness in scripts. Pair with `--backend-order`. |
| `--max-tokens` | `256` | Caps completion length for each VLM request. | Higher value usually increases runtime. | Higher values emphasize decode behavior; lower values emphasize image load, prompt build, and TTFT. | Lower for smoke, higher for decode-heavy comparison. Pair with scenario choice. |
| `--benchmark-mode` | `smoke` | Selects default run counts: `smoke=(0 warmup, 1 measured)`, `normal=(1, 3)`, `stable=(1, 5)`. | Major runtime control. | More samples improve stability. | Use `smoke` for newcomer check, `normal` for fast iteration, `stable` for real reporting. Pair with `--order-rounds`. |
| `--warmup-runs-per-fixture` | unset; inherits from `--benchmark-mode` | Overrides warmup count per fixture. | Higher value increases runtime linearly. | Can reduce cold-start noise. | Use when `--benchmark-mode` is right overall but warmup count is not. Pair with `--measured-runs-per-fixture`. |
| `--measured-runs-per-fixture` | unset; inherits from `--benchmark-mode` | Overrides measured count per fixture. | Higher value increases runtime linearly. | More stable averages and percentiles. | Use for more statistical confidence without changing other mode defaults. Pair with `--order-rounds`. |
| `--warmup-trials` | unset | Legacy alias that overrides warmup count after mode resolution. | Same as warmup-runs change. | Same as warmup-runs change. | Prefer `--warmup-runs-per-fixture` in new commands. |
| `--trials` | unset | Legacy alias that overrides measured count after mode resolution. | Same as measured-runs change. | Same as measured-runs change. | Prefer `--measured-runs-per-fixture` in new docs/scripts. |
| `--scenario` | `baseline` | Chooses what benchmark behavior runs: `baseline`, `streaming`, `cancellation`, `concurrency`, or `all`. `all` expands to four separate scenario sections. | `all` takes longest. Single-scenario runs are much faster. | Changes which metrics matter and which comparisons are fair. | Use `baseline` for end-to-end latency, `streaming` for TTFT/chunks, `cancellation` for lifecycle behavior, `concurrency` for pressure behavior, `all` for full evaluation. Pair with scenario-specific flags below. |
| `--fixtures` | none; all generated cases used when omitted | Restricts to exact fixture case names. | Fewer fixtures shorten run. | Narrows workload coverage. | Use to reproduce one problematic case. Pair with `--scenario` and `--model`. |
| `--fixture-category` | none; all categories used when omitted | Restricts fixture categories. Valid categories come from case builder: `single_image`, `prefix_single_image`, `two_image_compare`, `multi_image_summary`, `long_multi_image_analysis`. Repeat flag for multiple categories. | Fewer categories shorten run; multi-image and long-analysis cases tend to take longer. | Changes workload mix and fairness emphasis. | Use `single_image` for newcomer smoke, `multi_image_summary`/`long_multi_image_analysis` for harder multimodal stress. Pair with `--benchmark-mode` and `--max-tokens`. |
| `--concurrency-levels` | `1,2,4` | Changes concurrency sweep for concurrency scenario. Invalid or empty values collapse to positive unique integers; empty falls back to default tuple. | Higher levels and more levels increase runtime and system load. | Changes queueing, throughput, and contention picture. | Only matters with `--scenario concurrency` or `all`. Pair with `--measured-runs-per-fixture`. |
| `--backend-order` | `raw,server,project` | Changes execution order within each round. | Minimal direct runtime change. | Can change measured latency if order effects exist from thermal/cache/server state. | Use when investigating order bias. Pair with `--order-rounds` and optionally `--randomize-backend-order`. |
| `--randomize-backend-order` | off | Shuffles backend order between rounds. | Minimal direct runtime change. | Can reduce systematic order bias. | Best for serious fair comparisons. Pair with `--order-rounds 3` and `--backend-order-seed`. |
| `--backend-order-seed` | unset | Makes randomized backend order reproducible. | None. | No metric change except reproducibility of shuffled order. | Use whenever `--randomize-backend-order` is on and you want repeatable order. |
| `--order-rounds` | `1` | Repeats benchmark rounds with fresh backend orders. | Multiplies runtime approximately linearly. | Better exposes or averages out order effects. | Use `3` for fairness-focused reporting. Pair with `--randomize-backend-order` or deliberate `--backend-order`. |
| `--output-md` | script default | Writes the Markdown report elsewhere. | None. | No metric effect. | Use to keep scenario/model variants separate. Pair with `--output-json`. |
| `--output-json` | unset | Writes structured JSON report in addition to Markdown. | Small file-write overhead only. | No metric effect. | Use for later parsing, dashboards, or diffing. Pair with `--output-md`. |
| `--report-path` | unset; if set it overrides `--output-md` | Alias for Markdown output path. When both are provided, this value wins. | None. | No metric effect. | Use only for compatibility with older scripts. Prefer `--output-md` in new commands. |
| `--project-port` | `8000` | Points project backend client at different gateway port. | None unless wrong, then failures. | No benchmark-shape effect if service is same. | Use when gateway runs on non-default port. Pair with matching launch config. |
| `--server-port` | `8001` | Points `mlx_vlm.server` client at different port. | None unless wrong, then failures. | No benchmark-shape effect if service is same. | Use when server runs on non-default port. Pair with matching launch config. |
| `--launch-timeout` | `90` seconds | Changes how long benchmark waits for backend process startup. | Longer timeout only slows failure cases. | No metric effect on successful runs. | Raise for large models or cold hosts. Pair with `--readiness-timeout`. |
| `--readiness-timeout` | `90` seconds | Changes how long benchmark waits for readiness endpoint. | Longer timeout only slows failure cases. | No metric effect on successful runs. | Raise for large VLMs or slow first load. Pair with `--launch-timeout`. |
| `--timeout-seconds` | `120` seconds | Changes per-request timeout. | Higher value can prolong hung runs. | Too-low values create false failures on long prompts or slow models. | Raise for `long_multi_image_analysis`, lower-end machines, or high `--max-tokens`. |
| `--cancellation-delay-ms` | `300` | Changes how long cancellation scenario waits before sending cancel. | Small effect on cancellation scenario runtime only. | Lower values test early abort; higher values allow more generation before cancel. | Only matters with `--scenario cancellation` or `all`. Pair with `--max-tokens`. |
| `--skip-raw` | off | Removes raw direct-call backend after backend selection. | Shorter run. | Removes direct-call reference rows. | Use when you only care about serving-vs-serving fairness. Pair with `--scenario baseline` or `streaming`. |
| `--skip-server` | off | Removes `mlx_vlm.server` backend. | Shorter run. | Eliminates official-server comparison. | Use when server unavailable or debugging project-vs-raw only. |
| `--skip-project` | off | Removes this project backend. | Shorter run. | Eliminates primary target comparison. | Use when checking raw/server behavior only or isolating project startup issues. |

Help output:

```bash
bash scripts/benchmark-vlm.sh --help
```

### Examples

Pin one model:

```bash
bash scripts/benchmark-vlm.sh \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --benchmark-mode stable \
  --scenario baseline
```

Single fixture category:

```bash
bash scripts/benchmark-vlm.sh \
  --fixture-category single_image \
  --benchmark-mode normal \
  --scenario baseline
```

### Fast-Smoke

Newcomer check. Smallest useful VLM run that confirms model loading, one fixture path, and all three backends:

```bash
bash scripts/benchmark-vlm.sh \
  --model mlx-community/Qwen2-VL-2B-Instruct-4bit \
  --benchmark-mode smoke \
  --scenario baseline \
  --fixture-category single_image \
  --backend-order raw,server,project \
  --output-json benchmarks/results/vlm_smoke.json \
  --output-md benchmarks/results/vlm_smoke.md
```

### Full-Suite

Most comprehensive fair run. Uses default model list, all scenarios, rotated backend order, and multiple order rounds to reduce order bias:

```bash
bash scripts/benchmark-vlm.sh \
  --benchmark-mode stable \
  --scenario all \
  --backend-order raw,server,project \
  --randomize-backend-order \
  --backend-order-seed 42 \
  --order-rounds 3 \
  --concurrency-levels 1,2,4 \
  --measured-runs-per-fixture 7 \
  --output-json benchmarks/results/vlm_full.json \
  --output-md benchmarks/results/vlm_full.md
```

### Personal Run Suite

```bash
bash scripts/benchmark-vlm.sh \
  --benchmark-mode stable \
  --scenario all \
  --backend-order raw,server,project \
  --randomize-backend-order \
  --backend-order-seed 42 \
  --order-rounds 3 \
  --concurrency-levels 1,2,4 \
  --measured-runs-per-fixture 7 \
  --output-json benchmarks/results/vlm_full.json \
  --output-md benchmarks/results/vlm_full.md
```

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

For more exact backend-order and scenario guidance, see `scripts/benchmark.sh` and `scripts/benchmark-vlm.sh`.
