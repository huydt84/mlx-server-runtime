# How to Run Benchmarks

`mlx-air bench` is the benchmark entrypoint. A run measures every workload its
suite covers on each (model, runtime configuration) pair, and stores the
selected configuration next to its measurements.

## Run a Suite

```bash
mlx-air bench run --suite smoke
mlx-air bench run --suite optimization --focus scheduler
```

Suites, focuses, workloads, and models are defined in
[`benchmarks/config/default.toml`](../../benchmarks/config/default.toml):

| Suite | Rounds × trials | Coverage |
|-------|-----------------|----------|
| `smoke` | 2 × 1 | Four workloads on the primary model. A wiring check only; not enough for an optimization claim. |
| `optimization` | 2 × 2 | Each model role on the workloads it is meant to exercise. |
| `release` | 2 × 3 | Release-gate coverage. |
| `exhaustive` | 2 × 4 | Every workload on every model role. |
| `mlx_lm_comparison` | 2 × 2 | native-mlx `overlap` against the `mlx_lm` baseline on the Qwen3.5 hybrid checkpoints (`hybrid_binary`, `hybrid_ternary`). |

The `smoke`, `optimization`, `release`, and `exhaustive` suites run the native
`serial` and `overlap` runtime configurations. The `mlx_lm` runtime
configuration runs the v1 backend, which serves through `mlx-lm`'s
`BatchGenerator` behind the same gateway, so it measures `mlx-lm` under the
same load generator, prompts, and HTTP path as native-mlx. Its prompt cache
is off, because native-mlx keeps no prefix cache for hybrid models and the v1
backend does not clear its cache before cold-cache trials. Each
round starts one server per (model, runtime configuration) pair, and the
runtime order alternates between rounds and models to cancel order bias.
Before measuring, each workload runs once with distinct warmup prompts of the
same shape; warmup requests are recorded but never measured.

While it runs, the command prints one progress line per completed trial to
stderr, for example `[12/176] round 1 primary/serial decode_128_c1 trial 1: 4
ok, 0 errors`. `results.json` and `report.md` are rewritten after every trial.

Cold-cache workloads clear the prefix cache before each trial and fail
validation if a request reuses cached tokens. A model whose chat template puts
the same prefix before every prompt sets `cold_cache_allowance_tokens` to the
number of cached tokens that prefix produces.

`--focus` is optional. A focus (`gateway`, `prefill`, `decode`, `scheduler`,
`cache`) narrows the suite to the focus's models and workloads.

Use `--benchmark-config PATH` to load a different TOML file. Use
`--server-mode external --base-url URL` to measure an already running server
instead of letting the command launch one; the selection must then resolve to
exactly one model and runtime configuration.

## Metrics

Every workload reports each metric it can measure. Its `primary_metric` is
shown in bold in `report.md` and is the one calibration tracks.

| Metric | Unit | Better | Notes |
|--------|------|--------|-------|
| `ttft` | ms | lower | Streaming workloads only. |
| `tpot` | ms | lower | Streaming workloads only. |
| `end_to_end_latency` | ms | lower | |
| `aggregate_output_tokens_per_second` | tokens/s | higher | Per trial. |

Each trial's mean is one observation. A summary row gives the mean, median,
min–max, and coefficient of variation across trials, plus a request-level p95
that pools all requests in those trials.

## Artifacts

Results are written beneath `artifacts/benchmark/<timestamp>-<pid>/` in the
invocation directory, or under `--output-dir`. `results.json` holds the
configuration, raw request samples, summaries, environment, and runtime
counters. `report.md` is the readable summary.

## Profiling and Diagnostics

Performance rows are always collected with profiling disabled. Diagnostics run
afterwards in separate server processes:

```bash
mlx-air bench run --suite optimization --profile representative
```

`--profile representative` runs the first target of each diagnostic family the
suite lists; `--profile all` runs every target. With `--focus`, only the
diagnostic family of the same name runs. To collect diagnostics for an existing
result:

```bash
mlx-air bench diagnose --result artifacts/benchmark/<run>/results.json --workload-family decode
mlx-air bench diagnose --result artifacts/benchmark/<run>/results.json --all
```

Diagnostic timings never become benchmark rows.

## Calibrate Repeatability

```bash
mlx-air bench calibrate --suite optimization --focus scheduler --repetitions 5
```

Calibration repeats the same run and reports the coefficient of variation of
each primary metric's mean across repetitions. A row is unstable when it
exceeds `[calibration] max_cv_percent`.

## Compare Two Versions

Prepare each distribution independently, run the same suite, focus, and
configuration against both, and compare their `results.json` files. Require
matching workload inputs and output/token parity, and state the exact suite,
focus, models, and workloads covered by the verdict. Do not claim an
optimization from a smoke run or a profiled timing.

Run `mlx-air bench run --help` for the full argument list.
