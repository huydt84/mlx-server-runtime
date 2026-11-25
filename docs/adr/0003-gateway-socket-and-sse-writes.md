# 0003. Gateway socket mode and SSE write strategy

- Status: Accepted
- Date: 2026-11-22

## Context

The gateway serves HTTP on a hand-written `std::net` server. Streaming
responses are many small Server-Sent Events (SSE) frames where each token's
latency matters.

## Decision

- **Accepted sockets are switched to blocking mode.** On macOS an accepted
  socket inherits the listener's non-blocking flag. A request whose body
  arrives after its headers would then fail with `WouldBlock`.
- **`TCP_NODELAY` is set on every accepted connection.** Nagle's algorithm
  would hold small SSE frames until the previous segment is acknowledged.
  Non-streaming responses flush explicitly, so disabling Nagle is safe for all
  connections.
- **The SSE response headers and the first `data:` event go out in one
  write.** This saves a `write(2)` call and avoids splitting the first token
  across TCP segments.
- **Every SSE chunk carries the same `model` field.** The requested model name
  is copied before the worker request is moved into the request tracker.

## Consequences

- One blocked client holds one gateway thread until its I/O finishes. The
  admission limits in `docs/how-to/configure-backpressure.md` bound how many
  threads that can be.

## Code

- `rust/crates/gateway/src/http.rs` — `handle_connection`,
  `stream_chat_completion_with_disconnect`.
