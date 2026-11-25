# 0013. Speculative decoding

- Status: Accepted
- Date: 2026-11-25

## Context

A verified token costs nearly as much as a decoded one on a mixture of experts,
because each token uses its own experts. On the 5.4 GB pack a step that verified
five tokens took 83 ms against 35 ms for a plain step before tuning, and only one
or two of the proposals were kept, so a fixed or average-based draft length barely
paid.

## Decision

- **A proposal is kept only when it equals the target's own draw**, for greedy
  and sampled requests. The drafter never changes the output distribution, and
  short steps equal one-token steps exactly ([ADR-0011](0011-packed-weight-kernels.md)),
  so greedy text does not change either.
- **The drafter reports its probability of each proposal, and the step verifies
  the prefix with the most tokens per second.** Acceptance is tracked per range
  of probability, because drafters are overconfident, increasingly so for
  unlikely tokens. Costs are tracked per verified length; lengths that have not
  run use a prior (an empty proposal costs 1.2 plain steps, others 1.35 plus 0.24
  per token) scaled by the measured ones.
- **One stalled step must not look like a lasting slowdown**, so a measurement
  counts at most twice the current estimate. The first step of each length is
  ignored.
- **Drafted steps start on the next scheduler tick**, after the previous step's
  tokens are sent, so the first token is not delayed. Plain steps resolve about
  one step late while the next one is in flight, so an interval that changes that
  is not counted as a step time.
- **Speculation runs only while exactly one request decodes and none waits for
  prefill.** Batched steps are faster otherwise.
- **The drafter is off unless configured**, and its weights and contexts come out
  of the KV budget. Its block keys follow the context and the next append
  overwrites them.

## Consequences

- The proposals are read back before the step is built, which gives up overlap
  for drafted steps.
- A drafter for another model needs an entry in the registry and a target that
  returns the hidden states it reads.

## Code

- `python/mlx_worker/native_mlx/speculative.py` — `DraftLengthController`,
  `Speculator`.
- `python/mlx_worker/native_mlx/executor.py` — `_propose`, `InFlightStep`.
- `python/mlx_worker/native_mlx/scheduler.py` — `_drafts_next`.
- `python/mlx_worker/native_mlx/models/inkling2_drafter.py` — drafter.
