# Bootstrap Protocol

Phase 0 uses a minimal line-based readiness handshake over a Unix domain socket.

## Worker -> Rust

- `READY` means the worker finished startup and is ready to accept future requests.
- `ERROR\t<message>` means the worker failed during bootstrap.

## Rust -> Worker

- `HEALTH` is reserved for a later ping and is not used in Phase 0.

This is intentionally small so the readiness path can be proven before the full IPC schema is introduced.

