# How to Run Benchmarks

`mlx-air bench` is the benchmark entrypoint. Each invocation measures one
runtime and stores the selected configuration next to its measurements.

## Run a Suite

```bash
mlx-air bench run --suite smoke
mlx-air bench run --suite optimization --focus scheduler
```

Suites and focuses are defined in
[`benchmarks/config/default.toml`](../../benchmarks/config/default.toml):

| Suite | Purpose |
|-------|---------|
| `smoke` | Fast wiring check on the primary model. Not sufficient for an optimization claim. |
| `optimization` | All four model roles across serial and overlap configurations. Requires `--focus`. |
| `release` | Release-gate coverage. |
| `exhaustive` | Every workload on every model role. |

Focuses (`gateway`, `prefill`, `decode`, `scheduler`, `cache`, `all`) narrow
the models and workloads a suite runs.

Use `--benchmark-config PATH` to load a different TOML file. Use
`--server-mode external --base-url URL` to measure an already running server
instead of letting the command launch one.

## Artifacts

Results are written beneath `artifacts/benchmark/<timestamp>-<pid>/` in the
invocation directory, or under `--output-dir`. `results.json` holds the
configuration, raw samples, aggregates, environment, and runtime counters.

## Profiling and Diagnostics

Performance rows are always collected with profiling disabled. To capture
diagnostics alongside a run:

```bash
mlx-air bench run --suite optimization --focus decode --profile representative
```

To collect diagnostics for an existing result:

```bash
mlx-air bench diagnose --result artifacts/benchmark/<run>/results.json --all
```

Diagnostic timings never become benchmark rows.

## Calibrate Repeatability

```bash
mlx-air bench calibrate --suite optimization --focus scheduler --repetitions 5
```

Calibration reports run-to-run coefficient of variation against the
`[calibration]` threshold in the benchmark configuration.

## Compare Two Versions

Prepare each distribution independently, run the same suite, focus, and
configuration against both, and compare their `results.json` files. Require
matching workload inputs and output/token parity, and state the exact suite,
focus, models, and workloads covered by the verdict. Do not claim an
optimization from a smoke run or a profiled timing.

Run `mlx-air bench run --help` for the full argument list.
