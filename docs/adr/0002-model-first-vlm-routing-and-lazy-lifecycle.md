# 0002. Model-first VLM routing and lazy VLM lifecycle

- Status: Accepted
- Date: 2026-11-22

## Context

The runtime can serve a text model and an optional VLM from the same worker.
Loading a VLM is slow and uses a lot of memory, and many deployments never
receive a VLM request. Requests may target the VLM without images, or target
the text model with images.

## Decision

- **Routing is by requested model name, not by whether the request contains
  images.** A text-only request that names the VLM goes to the VLM. A request
  with images that names the text model gets a 400 error, and so does any
  request with images when no VLM is configured. The gateway checks the
  image-count cap before queue admission.
- **The VLM engine is constructed and loaded lazily** on the first request
  that names it, not during worker bootstrap.
- **Cancellation is checked before the blocking `mlx_vlm.load` call**, inside
  the engine where the `should_cancel` closure is available, so a cancelled
  first request never waits on a cold model load. If cancellation arrives
  after the load finishes, the loaded model and processor are kept for the
  next request instead of being thrown away and loaded again.
- **VLM readiness is separate from text readiness.** The worker's generic
  bootstrap `STATUS` and `READY` messages update only the text model. The VLM
  becomes ready only through explicit warmup completion.
- **Only backend failures mark the VLM failed.** The first failed VLM request
  moves a `NotLoaded` VLM to `Failed` with `VLM_LOAD_FAILED`, unless the error
  is `GatewayError::InvalidRequest`, which means bad user input rather than a
  load failure. Failures do not change a VLM that is already `Ready`, and
  repeated transitions are no-ops that keep the first error message.

## Consequences

- The first VLM request pays the model load latency.
- `/health` can report the text model ready while the VLM is still
  `NotLoaded`.
- The text and VLM engines take different `complete_chat` arguments. Only the
  VLM engine accepts `should_cancel` ([#5](https://github.com/huydt84/mlx-server-runtime/issues/5)).

## Code

- `rust/crates/gateway/src/http.rs` — `handle_chat_completion`,
  `stream_chat_completion_with_disconnect`, `track_vlm_lifecycle_failure`.
- `rust/crates/gateway/src/supervisor.rs` — `bootstrap_worker` (`STATUS`
  updates only the text model), `snapshot_vlm` (`NotLoaded` fallback).
- `python/mlx_worker/main.py` — `main` (model-first dispatch, lazy engine
  construction).
- `python/mlx_worker/vlm_engine.py` — `MlxVlmEngine.initialize`.
