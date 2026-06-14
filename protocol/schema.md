# Protocol

Phase 0 uses a minimal line-based readiness handshake over a Unix domain socket.

## Worker -> Rust

- `READY` means the worker finished startup and is ready to accept future requests.
- `ERROR\t<message>` means the worker failed during bootstrap.

## Rust -> Worker

- `HEALTH` is reserved for a later ping and is not used in Phase 0.

## Phase 1 Request / Response Frames

After the bootstrap handshake, Phase 1 uses newline-delimited JSON frames over the
same Unix domain socket connection.

### Rust -> Worker

```json
{
  "type": "chat_completion",
  "request": {
    "request_id": "req-1",
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello"}
    ],
    "max_tokens": 32,
    "temperature": 0.0,
    "top_p": 1.0
  }
}
```

### Worker -> Rust

Successful response:

```json
{
  "type": "chat_completion",
  "response": {
    "request_id": "req-1",
    "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "text": "Hello!",
    "finish_reason": "stop",
    "prompt_tokens": 14,
    "completion_tokens": 3
  }
}
```

Worker-side request error:

```json
{
  "type": "error",
  "request_id": "req-1",
  "message": "generation failed"
}
```

Phase 1 remains intentionally small:

- one request at a time
- one final non-streaming response
- no SSE token events yet
