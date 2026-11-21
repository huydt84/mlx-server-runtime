# 0007. Worker startup failure handling

- Status: Accepted
- Date: 2026-11-22

## Context

`mlx-air serve`, detached launchd instances, and `mlx-air bench run` all wait
for a gateway to become ready, with readiness timeouts as long as 1800 s.
Some startup failures (for example a KV pool too small for the weights) can
never succeed, however long the caller waits.

## Decision

- **A failed worker startup is reported at once, not at the timeout.** Callers
  poll `/startup` while waiting for readiness and stop when it reports
  `failed`. A gateway whose worker exits before readiness also exits, so the
  caller sees the process end.
- **A new launchd pid before readiness counts as a failure.** launchd restarts
  a gateway that exits, so a changed pid means startup failed and will keep
  failing.
- **Failures name the setting to change**, for example
  `raise --memory-utilization` when the KV pool cannot hold the weights.
- **Detached instances render the request limit only when it was set
  explicitly**, so a detached worker derives it from its KV pool the same way
  a foreground worker does.
- **The managed environment rebuilds the worker package from the installed
  distribution** instead of reusing an older build, because it is a local,
  non-editable package.
- **`MLX_RUNTIME_TEXT_CACHE_BUDGET_BYTES` still pins the native KV pool when
  set to a non-zero value.** Zero only disables the v1 prompt cache, and
  native-mlx then sizes its pool from memory utilization ([#8](https://github.com/huydt84/mlx-server-runtime/issues/8)).
- **Unix socket paths are limited to 104 bytes on macOS**, so host scripts put
  worker sockets under `/tmp` rather than inside deep artifact directories.
- **`launchctl bootout` finishes asynchronously.** Callers poll until the
  service is gone instead of assuming it is.

## Code

- `rust/crates/gateway/src/instances.rs` — `wait_until_ready`,
  `render_runtime_config`.
- `rust/crates/gateway/src/environment.rs` — `ensure_runtime_environment`.
- `python/mlx_benchmark/runner.py` — `_wait_for_readiness`.
- `python/mlx_worker/config.py` — `load_config`.
