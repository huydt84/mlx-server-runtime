# Architecture Decision Records

An Architecture Decision Record (ADR) documents one important, non-obvious
decision about the code: the context that forced it, what was decided, and
what it costs. Code comments are banned in this repository (see
[ADR-0001](0001-ban-code-comments.md)), so an ADR is where a "why" lives.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-ban-code-comments.md) | Ban code comments; record decisions in ADRs | Accepted |
| [0002](0002-model-first-vlm-routing-and-lazy-lifecycle.md) | Model-first VLM routing and lazy VLM lifecycle | Accepted |
| [0003](0003-gateway-socket-and-sse-writes.md) | Gateway socket mode and SSE write strategy | Accepted |
| [0004](0004-native-scheduler-and-sampling.md) | Native scheduler admission, chaining, and sampling | Accepted |
| [0005](0005-paged-kv-cache-writes-and-buffer-lifetime.md) | Paged KV cache writes and buffer lifetime | Accepted |
| [0006](0006-paged-decode-attention-kernel.md) | Paged decode attention kernel | Accepted |
| [0007](0007-worker-startup-failure-handling.md) | Worker startup failure handling | Accepted |
| [0008](0008-checkpoint-loading-compatibility.md) | Checkpoint loading compatibility | Accepted |
| [0009](0009-benchmark-measurement-integrity.md) | Benchmark measurement integrity | Accepted |
| [0010](0010-pipeline-profile-timing.md) | Pipeline profile timing model | Accepted |

The [2026-11-22 comment audit](comment-audit-2026-11-22.md) lists every comment
that ADRs 0002–0010 replaced.

## Write a New ADR

1. Copy [template.md](template.md) to `NNNN-short-title.md` with the next free
   number.
2. Fill in every section. Name the code by file and symbol, never by line
   number, so the record survives edits.
3. Add a row to the index above.
4. Never rewrite an accepted ADR's decision. Supersede it with a new ADR and
   set the old one's status to `Superseded by NNNN`.
