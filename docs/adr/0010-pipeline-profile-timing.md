# 0010. Pipeline profile timing model

- Status: Accepted
- Date: 2026-11-22

## Context

`mlx-air profile` correlates client, gateway, and worker events for each
request into a timeline and a per-stage breakdown. Events come from separate
processes, and some are recorded only after the work they describe is done.

## Decision

- **One clock.** The profiler client stamps each request with
  `time.perf_counter_ns()`, the same clock as the worker's event timestamps,
  so client and worker spans line up without clock translation.
- **Spans are anchored at their start, not at the moment they are recorded.**
  A prefill span is recorded when the step finishes, so its start is "now"
  minus the step time.
- **The runtime `terminal` span covers the whole request lifetime.** It is
  recorded at completion but starts at enqueue. Every later runtime event lies
  inside it. IPC decode and tokenization happen before enqueue, so they come
  before it.
- **Envelope spans stay out of the stage breakdown.** `runtime/terminal` and
  `gateway/http_round_trip` cover a whole request. They are reported as
  request latency and not as stages, so they do not dwarf the stages they
  contain.
- **Timeline color follows the component, never its rank.** A fixed component
  order picks the palette, so the same component has the same color in every
  profile.

## Consequences

- Spans derived from step timings are only as precise as the reported
  `step_time_ms`, which is a whole number of milliseconds.

## Code

- `python/mlx_benchmark/profiler.py` — request timing.
- `python/mlx_benchmark/profile_analysis.py` — `ENVELOPE_STAGES`,
  `COMPONENT_ORDER`.
- `python/mlx_worker/native_mlx/runtime.py` — profiler `record` calls for
  `prefill` and `terminal`.
- `mlx-host-validation/scripts/cli_profile.sh` — the ordering check for each
  request.
