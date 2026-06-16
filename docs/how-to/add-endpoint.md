# How to Add an HTTP Endpoint

Extend the Rust gateway to handle a new route.

## Where to Edit

All HTTP routing lives in `rust/crates/gateway/src/http.rs`. The routing logic is in `response_for_request()`:

```rust
fn response_for_request(request_line: &str, body: &[u8], state: &AppState) -> HttpResponse {
    let Some((method, path)) = parse_request_line(request_line) else {
        return not_found_response();
    };

    match (method, path.as_str()) {
        ("GET", "/live") => live_response(&state.runtime),
        // Add your route here
        _ => not_found_response(),
    }
}
```

## Steps

1. **Add a handler function** in `http.rs` that takes the relevant state and returns an `HttpResponse`.

   ```rust
   fn my_handler(state: &AppState) -> HttpResponse {
       HttpResponse {
           status: "200 OK".to_string(),
           content_type: "application/json",
           body: serde_json::json!({"message": "hello"}).to_string(),
       }
   }
   ```

2. **Register the route** in the `match` block in `response_for_request`.

   ```rust
   ("GET", "/my-endpoint") => my_handler(state),
   ```

3. **If the route needs worker interaction**, add a method to `ChatCompletionService` trait (if reusable) or access `state.runtime.worker_client` directly.

4. **If the route needs path parameters**, use `parse_request_line` plus manual path splitting (the gateway uses raw TCP, not a framework router). See `model_path_name()` for an example of path extraction.

5. **Add a response type** in `openai.rs` if the response has a structured JSON shape, with appropriate `Serialize` derives.

## Testing

Add test cases at the bottom of `http.rs` in the `#[cfg(test)] mod tests` block. Tests can construct an `AppState` with a `FakeService` backend to avoid needing a live worker.

```rust
#[test]
fn my_endpoint_returns_200() {
    let response = response_for_request(
        "GET /my-endpoint HTTP/1.1\r\n",
        &[],
        &test_state(ModelState::Ready, Arc::new(FakeService::default())),
    );
    assert_eq!(response.status, "200 OK");
}
```

Run tests:

```bash
cargo test -p gateway
```
