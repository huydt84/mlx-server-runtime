use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::openai::{ChatCompletionHttpRequest, ChatCompletionHttpResponse};
use crate::supervisor::RuntimeState;
use mlx_runtime_protocol::{
    ChatCompletionRequest, ChatCompletionResponse, ModelState, ModelStatus,
};
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant};

/// Service used by the HTTP layer to fulfill completions.
pub trait ChatCompletionService: Send + Sync {
    /// Execute a non-streaming chat completion request.
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError>;

    /// Stream a chat completion and return the final response.
    fn stream(
        &self,
        request: ChatCompletionRequest,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        let _ = on_delta;
        self.complete(request)
    }

    /// Cancel an in-flight completion, if supported.
    fn cancel(&self, _request_id: &str) -> Result<(), GatewayError> {
        Ok(())
    }
}

impl<T: ChatCompletionService + ?Sized> ChatCompletionService for Arc<T> {
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        (**self).complete(request)
    }

    fn stream(
        &self,
        request: ChatCompletionRequest,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        (**self).stream(request, on_delta)
    }

    fn cancel(&self, request_id: &str) -> Result<(), GatewayError> {
        (**self).cancel(request_id)
    }
}

impl ChatCompletionService for crate::ipc::WorkerClient {
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        self.complete_chat(request)
    }

    fn stream(
        &self,
        request: ChatCompletionRequest,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        self.stream_chat(request, on_delta)
    }

    fn cancel(&self, request_id: &str) -> Result<(), GatewayError> {
        self.cancel_chat(request_id)
    }
}

struct AppState {
    runtime: RuntimeState,
    generation: crate::config::GenerationConfig,
    limits: crate::config::RequestLimits,
    admission: Arc<RequestAdmission>,
    model: String,
    test_backend: Option<Arc<dyn ChatCompletionService>>,
}

struct RequestAdmission {
    max_pending_requests: usize,
    max_active_requests: usize,
    request_timeout: Duration,
    state: Mutex<AdmissionState>,
    wakeup: Condvar,
}

struct AdmissionState {
    active: usize,
    waiting: usize,
}

struct RequestPermit {
    admission: Arc<RequestAdmission>,
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        if let Ok(mut state) = self.admission.state.lock() {
            if state.active > 0 {
                state.active -= 1;
            }
            self.admission.wakeup.notify_one();
        }
    }
}

impl RequestAdmission {
    fn new(
        max_pending_requests: usize,
        max_active_requests: usize,
        request_timeout: Duration,
    ) -> Self {
        Self {
            max_pending_requests,
            max_active_requests,
            request_timeout,
            state: Mutex::new(AdmissionState {
                active: 0,
                waiting: 0,
            }),
            wakeup: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>) -> Result<RequestPermit, HttpResponse> {
        match self.acquire_until(None)? {
            Some(permit) => Ok(permit),
            None => Err(json_error_response(
                "500 Internal Server Error",
                "BACKPRESSURE_CANCELLED",
                "request admission cancelled unexpectedly",
            )),
        }
    }

    fn acquire_until(
        self: &Arc<Self>,
        disconnected: Option<&AtomicBool>,
    ) -> Result<Option<RequestPermit>, HttpResponse> {
        let deadline = Instant::now() + self.request_timeout;
        let mut guard = self.state.lock().map_err(|_| {
            json_error_response(
                "500 Internal Server Error",
                "BACKPRESSURE_LOCK_POISONED",
                "request admission lock poisoned",
            )
        })?;

        loop {
            if disconnected.is_some_and(|flag| flag.load(Ordering::Relaxed)) {
                return Ok(None);
            }
            if guard.active < self.max_active_requests {
                guard.active += 1;
                return Ok(Some(RequestPermit {
                    admission: self.clone(),
                }));
            }

            if guard.waiting >= self.max_pending_requests {
                return Err(json_error_response(
                    "429 Too Many Requests",
                    "QUEUE_FULL",
                    "request queue is full",
                ));
            }

            guard.waiting += 1;
            let now = Instant::now();
            let wait_for = deadline.saturating_duration_since(now);
            let wait_for = if disconnected.is_some() {
                wait_for.min(Duration::from_millis(50))
            } else {
                wait_for
            };
            let (next_guard, wait_result) =
                self.wakeup.wait_timeout(guard, wait_for).map_err(|_| {
                    json_error_response(
                        "500 Internal Server Error",
                        "BACKPRESSURE_LOCK_POISONED",
                        "request admission lock poisoned",
                    )
                })?;
            guard = next_guard;
            if guard.waiting > 0 {
                guard.waiting -= 1;
            }

            if wait_result.timed_out() && Instant::now() >= deadline {
                return Err(json_error_response(
                    "429 Too Many Requests",
                    "QUEUE_TIMEOUT",
                    "request queue wait timed out",
                ));
            }
        }
    }
}

/// Serves the Phase 1 HTTP surface.
pub fn serve(config: RuntimeConfig, runtime: RuntimeState) -> Result<(), GatewayError> {
    let listener = TcpListener::bind(format!("{}:{}", config.server.host, config.server.port))?;
    let admission = Arc::new(RequestAdmission::new(
        config.limits.max_pending_requests,
        config.limits.max_active_requests,
        Duration::from_secs(config.limits.request_timeout_seconds),
    ));

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = AppState {
                    runtime: runtime.clone(),
                    generation: config.generation.clone(),
                    limits: config.limits.clone(),
                    admission: admission.clone(),
                    model: config.worker.model.clone(),
                    test_backend: None,
                };
                thread::spawn(move || {
                    let _ = handle_connection(stream, state);
                });
            }
            Err(err) => return Err(GatewayError::Io(err)),
        }
    }

    Ok(())
}

fn handle_connection(mut stream: TcpStream, state: AppState) -> Result<(), GatewayError> {
    let mut reader = BufReader::new(&stream);
    let mut request_line = String::new();
    let _ = reader.read_line(&mut request_line)?;

    let headers = read_headers(&mut reader)?;
    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let mut body = vec![0; content_length];
    if content_length > 0 {
        reader.read_exact(&mut body)?;
    }

    let is_stream = request_line.starts_with("POST /v1/chat/completions ")
        && request_streams(&body).unwrap_or(false);
    if is_stream {
        stream_chat_completion(&mut stream, &body, &state)?;
        return Ok(());
    }

    let response = response_for_request(&request_line, &body, &state);
    write_response(
        &mut stream,
        &response.status,
        response.content_type,
        &response.body,
    )?;
    Ok(())
}

fn read_headers(
    reader: &mut BufReader<&TcpStream>,
) -> Result<HashMap<String, String>, GatewayError> {
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 || line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }
    Ok(headers)
}

struct HttpResponse {
    status: String,
    content_type: &'static str,
    body: String,
}

fn response_for_request(request_line: &str, body: &[u8], state: &AppState) -> HttpResponse {
    let Some((method, path)) = parse_request_line(request_line) else {
        return not_found_response();
    };

    match (method, path.as_str()) {
        ("GET", "/live") => live_response(&state.runtime),
        ("GET", "/startup") => startup_response(&state.runtime),
        ("GET", "/ready") => readiness_response(&state.runtime),
        ("GET", "/health") => health_response(&state.runtime),
        ("GET", "/models") => models_response(&state.runtime),
        ("POST", "/v1/chat/completions") => handle_chat_completion(body, state),
        _ if method == "GET" && path.starts_with("/models/") && path.ends_with("/status") => {
            model_status_response(&state.runtime, &path)
        }
        _ if method == "GET" && path.starts_with("/models/") && path.ends_with("/ready") => {
            model_ready_response(&state.runtime, &path)
        }
        _ => not_found_response(),
    }
}

fn request_streams(body: &[u8]) -> Option<bool> {
    serde_json::from_slice::<ChatCompletionHttpRequest>(body)
        .ok()
        .map(|request| request.stream.unwrap_or(false))
}

fn parse_request_line(request_line: &str) -> Option<(&str, String)> {
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?;
    let path = parts.next()?;
    Some((method, path.to_string()))
}

fn live_response(runtime: &RuntimeState) -> HttpResponse {
    let body = json!({
        "status": "live",
        "uptime_seconds": runtime.started_at.elapsed().as_secs(),
        "pid": runtime.pid,
    })
    .to_string();
    HttpResponse {
        status: "200 OK".to_string(),
        content_type: "application/json",
        body,
    }
}

fn startup_response(runtime: &RuntimeState) -> HttpResponse {
    match runtime.snapshot() {
        Ok(status) if status.ready => HttpResponse {
            status: "200 OK".to_string(),
            content_type: "application/json",
            body: json!({ "status": "started" }).to_string(),
        },
        Ok(status) if status.state == ModelState::Failed => HttpResponse {
            status: "503 Service Unavailable".to_string(),
            content_type: "application/json",
            body: json!({
                "status": "failed",
                "phase": status.state,
                "elapsed_seconds": runtime.started_at.elapsed().as_secs(),
                "error": status.last_error,
            })
            .to_string(),
        },
        Ok(status) => HttpResponse {
            status: "200 OK".to_string(),
            content_type: "application/json",
            body: json!({
                "status": "starting",
                "phase": status.state,
                "elapsed_seconds": runtime.started_at.elapsed().as_secs(),
            })
            .to_string(),
        },
        Err(err) => internal_error_response(&err.to_string()),
    }
}

fn readiness_response(runtime: &RuntimeState) -> HttpResponse {
    match runtime.snapshot() {
        Ok(status) if status.ready => HttpResponse {
            status: "200 OK".to_string(),
            content_type: "application/json",
            body: json!({
                "status": "ready",
                "ready": true,
                "model": status.model,
                "revision": status.revision,
                "loaded_at": status.loaded_at,
                "device": status.device,
                "dtype": status.dtype,
                "warmup_passed": status.warmup_passed,
            })
            .to_string(),
        },
        Ok(status) => not_ready_response(&status),
        Err(err) => internal_error_response(&err.to_string()),
    }
}

fn health_response(runtime: &RuntimeState) -> HttpResponse {
    match runtime.snapshot() {
        Ok(status) if status.ready => HttpResponse {
            status: "200 OK".to_string(),
            content_type: "text/plain; charset=utf-8",
            body: "healthy".to_string(),
        },
        Ok(_) => HttpResponse {
            status: "503 Service Unavailable".to_string(),
            content_type: "text/plain; charset=utf-8",
            body: "unhealthy".to_string(),
        },
        Err(err) => internal_error_response(&err.to_string()),
    }
}

fn models_response(runtime: &RuntimeState) -> HttpResponse {
    match runtime.snapshot() {
        Ok(status) => HttpResponse {
            status: "200 OK".to_string(),
            content_type: "application/json",
            body: json!({
                "models": [status.summary()],
            })
            .to_string(),
        },
        Err(err) => internal_error_response(&err.to_string()),
    }
}

fn model_status_response(runtime: &RuntimeState, path: &str) -> HttpResponse {
    let model_name = runtime_model_name(runtime);
    match model_path_name(path) {
        Some(name) if name == model_name => match runtime.snapshot() {
            Ok(status) => HttpResponse {
                status: "200 OK".to_string(),
                content_type: "application/json",
                body: serde_json::to_string(&status).unwrap_or_else(|_| {
                    "{\"error\":{\"message\":\"status serialization failed\"}}".to_string()
                }),
            },
            Err(err) => internal_error_response(&err.to_string()),
        },
        _ => not_found_response(),
    }
}

fn model_ready_response(runtime: &RuntimeState, path: &str) -> HttpResponse {
    let model_name = runtime_model_name(runtime);
    match model_path_name(path) {
        Some(name) if name == model_name => match runtime.snapshot() {
            Ok(status) if status.ready => HttpResponse {
                status: "200 OK".to_string(),
                content_type: "application/json",
                body: json!({
                    "model": status.model,
                    "ready": true,
                    "state": status.state,
                })
                .to_string(),
            },
            Ok(status) => HttpResponse {
                status: "503 Service Unavailable".to_string(),
                content_type: "application/json",
                body: json!({
                    "model": status.model,
                    "ready": false,
                    "state": status.state,
                    "reason": readiness_reason(&status),
                })
                .to_string(),
            },
            Err(err) => internal_error_response(&err.to_string()),
        },
        _ => not_found_response(),
    }
}

fn handle_chat_completion(body: &[u8], state: &AppState) -> HttpResponse {
    let status = match state.runtime.snapshot() {
        Ok(status) => status,
        Err(err) => return internal_error_response(&err.to_string()),
    };

    if !status.ready {
        return not_ready_error(&status);
    }

    let request = match serde_json::from_slice::<ChatCompletionHttpRequest>(body) {
        Ok(request) => request,
        Err(err) => {
            return json_error_response(
                "400 Bad Request",
                "INVALID_REQUEST",
                &format!("invalid JSON body: {err}"),
            );
        }
    };

    let worker_request =
        match request.into_worker_request(&state.generation, &state.limits, &state.model) {
            Ok(request) => request,
            Err(message) => {
                return json_error_response("400 Bad Request", "INVALID_REQUEST", &message)
            }
        };

    let _permit = match state.admission.acquire() {
        Ok(permit) => permit,
        Err(response) => return response,
    };

    let backend = if let Some(backend) = &state.test_backend {
        Some(backend.clone())
    } else {
        match state.runtime.worker_client.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|client| client.clone() as Arc<dyn ChatCompletionService>),
            Err(_) => {
                return json_error_response(
                    "500 Internal Server Error",
                    "WORKER_CLIENT_LOCK_POISONED",
                    "worker client lock poisoned",
                );
            }
        }
    };

    let Some(backend) = backend else {
        return not_ready_error(&status);
    };

    match backend.complete(worker_request) {
        Ok(response) => {
            let body = serde_json::to_string(&ChatCompletionHttpResponse::from(response))
                .unwrap_or_else(|_| {
                    "{\"error\":{\"message\":\"response serialization failed\"}}".to_string()
                });
            HttpResponse {
                status: "200 OK".to_string(),
                content_type: "application/json",
                body,
            }
        }
        Err(err) => gateway_error_response(&err),
    }
}

fn stream_chat_completion(
    stream: &mut TcpStream,
    body: &[u8],
    state: &AppState,
) -> Result<(), GatewayError> {
    let disconnect_monitor = DisconnectMonitor::start(stream.try_clone()?)?;
    let result = stream_chat_completion_with_disconnect(
        stream,
        body,
        state,
        Some(disconnect_monitor.disconnected()),
    );
    drop(disconnect_monitor);
    result
}

fn stream_chat_completion_with_disconnect<W: Write>(
    writer: &mut W,
    body: &[u8],
    state: &AppState,
    disconnected: Option<Arc<AtomicBool>>,
) -> Result<(), GatewayError> {
    let status = state
        .runtime
        .snapshot()
        .map_err(|err| GatewayError::Protocol(err.to_string()))?;
    if !status.ready {
        let response = not_ready_error(&status);
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    }

    let request = match serde_json::from_slice::<ChatCompletionHttpRequest>(body) {
        Ok(request) => request,
        Err(err) => {
            let response = json_error_response(
                "400 Bad Request",
                "INVALID_REQUEST",
                &format!("invalid JSON body: {err}"),
            );
            return write_response(
                writer,
                &response.status,
                response.content_type,
                &response.body,
            );
        }
    };

    let worker_request =
        match request.into_worker_request(&state.generation, &state.limits, &state.model) {
            Ok(request) => request,
            Err(message) => {
                let response = json_error_response("400 Bad Request", "INVALID_REQUEST", &message);
                return write_response(
                    writer,
                    &response.status,
                    response.content_type,
                    &response.body,
                );
            }
        };

    let _permit = match state.admission.acquire_until(disconnected.as_deref()) {
        Ok(Some(permit)) => permit,
        Ok(None) => {
            return Ok(());
        }
        Err(response) => {
            return write_response(
                writer,
                &response.status,
                response.content_type,
                &response.body,
            )
        }
    };

    let backend = if let Some(backend) = &state.test_backend {
        Some(backend.clone())
    } else {
        match state.runtime.worker_client.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|client| client.clone() as Arc<dyn ChatCompletionService>),
            Err(_) => {
                let response = json_error_response(
                    "500 Internal Server Error",
                    "WORKER_CLIENT_LOCK_POISONED",
                    "worker client lock poisoned",
                );
                return write_response(
                    writer,
                    &response.status,
                    response.content_type,
                    &response.body,
                );
            }
        }
    };

    let Some(backend) = backend else {
        let response = not_ready_error(&status);
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    };

    let created = stream_completion_created();
    let model = state.model.clone();
    let request_id = worker_request.request_id.clone();
    let mut stream_started = false;
    let cancel_on_disconnect =
        DisconnectCancellation::start(disconnected.clone(), backend.clone(), request_id.clone());

    let mut on_delta = |delta: String| -> Result<(), GatewayError> {
        ensure_sse_stream_started(writer, &mut stream_started)?;
        let event = json!({
            "id": format!("chatcmpl-{}", request_id),
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": delta,
                },
                "finish_reason": null,
            }],
        });
        write!(writer, "data: {}\n\n", event)?;
        writer.flush()?;
        Ok(())
    };

    let response = match backend.stream(worker_request, &mut on_delta) {
        Ok(response) => response,
        Err(err) => {
            drop(cancel_on_disconnect);
            let _ = backend.cancel(&request_id);
            if disconnected
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                return Ok(());
            }
            return match err {
                GatewayError::InvalidRequest(message) if !stream_started => {
                    let response =
                        json_error_response("400 Bad Request", "INVALID_REQUEST", &message);
                    write_response(
                        writer,
                        &response.status,
                        response.content_type,
                        &response.body,
                    )
                }
                other if !stream_started => {
                    let response = gateway_error_response(&other);
                    write_response(
                        writer,
                        &response.status,
                        response.content_type,
                        &response.body,
                    )
                }
                GatewayError::InvalidRequest(message) => Err(GatewayError::Protocol(message)),
                other => Err(other),
            };
        }
    };
    drop(cancel_on_disconnect);
    if disconnected
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
    {
        return Ok(());
    }
    ensure_sse_stream_started(writer, &mut stream_started)?;
    let done_event = json!({
        "id": format!("chatcmpl-{}", response.request_id),
        "object": "chat.completion.chunk",
        "created": created,
        "model": response.model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": response.finish_reason,
        }],
    });
    write!(writer, "data: {}\n\n", done_event)?;
    write!(writer, "data: [DONE]\n\n")?;
    writer.flush()?;
    Ok(())
}

fn ensure_sse_stream_started<W: Write>(
    writer: &mut W,
    stream_started: &mut bool,
) -> Result<(), GatewayError> {
    if *stream_started {
        return Ok(());
    }
    write!(
        writer,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    writer.flush()?;
    *stream_started = true;
    Ok(())
}

fn gateway_error_response(err: &GatewayError) -> HttpResponse {
    match err {
        GatewayError::InvalidRequest(message) => {
            json_error_response("400 Bad Request", "INVALID_REQUEST", message)
        }
        GatewayError::WorkerStartup(message) => {
            json_error_response("503 Service Unavailable", "WORKER_UNAVAILABLE", message)
        }
        GatewayError::Protocol(message) => {
            json_error_response("500 Internal Server Error", "PROTOCOL_ERROR", message)
        }
        GatewayError::Io(err) => {
            json_error_response("500 Internal Server Error", "IO_ERROR", &err.to_string())
        }
    }
}

struct DisconnectMonitor {
    disconnected: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DisconnectMonitor {
    fn start(stream: TcpStream) -> Result<Self, GatewayError> {
        stream.set_read_timeout(Some(Duration::from_millis(50)))?;
        let disconnected = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let watch_disconnected = disconnected.clone();
        let watch_stop = stop.clone();
        let handle = thread::spawn(move || {
            let mut probe = [0_u8; 1];
            while !watch_stop.load(Ordering::Relaxed) {
                match stream.peek(&mut probe) {
                    Ok(0) => {
                        watch_disconnected.store(true, Ordering::Relaxed);
                        break;
                    }
                    Ok(_) => thread::sleep(Duration::from_millis(50)),
                    Err(err)
                        if matches!(
                            err.kind(),
                            std::io::ErrorKind::WouldBlock
                                | std::io::ErrorKind::TimedOut
                                | std::io::ErrorKind::Interrupted
                        ) => {}
                    Err(_) => {
                        watch_disconnected.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        });
        Ok(Self {
            disconnected,
            stop,
            handle: Some(handle),
        })
    }

    fn disconnected(&self) -> Arc<AtomicBool> {
        self.disconnected.clone()
    }
}

impl Drop for DisconnectMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct DisconnectCancellation {
    stop: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl DisconnectCancellation {
    fn start(
        disconnected: Option<Arc<AtomicBool>>,
        backend: Arc<dyn ChatCompletionService>,
        request_id: String,
    ) -> Option<Self> {
        let disconnected = disconnected?;
        let stop = Arc::new(AtomicBool::new(false));
        let cancel_stop = stop.clone();
        let handle = thread::spawn(move || {
            while !cancel_stop.load(Ordering::Relaxed) {
                if disconnected.load(Ordering::Relaxed) {
                    let _ = backend.cancel(&request_id);
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        });
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }
}

impl Drop for DisconnectCancellation {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn not_ready_response(status: &ModelStatus) -> HttpResponse {
    HttpResponse {
        status: "503 Service Unavailable".to_string(),
        content_type: "application/json",
        body: json!({
            "status": "not_ready",
            "ready": false,
            "reason": readiness_reason(status),
            "model": status.model,
            "state": status.state,
            "last_error": status.last_error,
        })
        .to_string(),
    }
}

fn not_ready_error(status: &ModelStatus) -> HttpResponse {
    let code = if status.state == ModelState::Failed {
        "MODEL_LOAD_FAILED"
    } else {
        "MODEL_NOT_READY"
    };
    let message = if status.state == ModelState::Failed {
        "model failed to load"
    } else {
        "model is not ready"
    };
    let mut payload = json!({
        "error": {
            "code": code,
            "message": message,
            "model": status.model,
            "state": status.state,
        }
    });

    if let Some(error) = &status.last_error {
        payload["error"]["last_error"] = json!(error);
    }

    HttpResponse {
        status: "503 Service Unavailable".to_string(),
        content_type: "application/json",
        body: payload.to_string(),
    }
}

fn readiness_reason(status: &ModelStatus) -> &'static str {
    match status.state {
        ModelState::NotLoaded => "model_not_loaded",
        ModelState::Downloading => "model_downloading",
        ModelState::Verifying => "model_verifying",
        ModelState::LoadingWeights => "model_loading",
        ModelState::InitializingRuntime => "runtime_initializing",
        ModelState::WarmingUp => "warmup_not_finished",
        ModelState::Ready => "ready",
        ModelState::Degraded => "model_degraded",
        ModelState::Failed => "model_load_failed",
        ModelState::Unloading => "model_unloading",
    }
}

fn json_error_response(status: &str, code: &str, message: &str) -> HttpResponse {
    HttpResponse {
        status: status.to_string(),
        content_type: "application/json",
        body: json!({ "error": { "code": code, "message": message } }).to_string(),
    }
}

fn internal_error_response(message: &str) -> HttpResponse {
    json_error_response("500 Internal Server Error", "INTERNAL_ERROR", message)
}

fn runtime_model_name(runtime: &RuntimeState) -> String {
    runtime
        .snapshot()
        .map(|status| status.model)
        .unwrap_or_else(|_| "unknown".to_string())
}

fn stream_completion_created() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn model_path_name(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();
    match segments.as_slice() {
        ["models", model, "status"] | ["models", model, "ready"] => percent_decode(model),
        _ => None,
    }
}

fn percent_decode(segment: &str) -> Option<String> {
    let mut bytes = Vec::with_capacity(segment.len());
    let mut index = 0;
    let raw = segment.as_bytes();

    while index < raw.len() {
        match raw[index] {
            b'%' if index + 2 < raw.len() => {
                let hex = &segment[index + 1..index + 3];
                let value = u8::from_str_radix(hex, 16).ok()?;
                bytes.push(value);
                index += 3;
            }
            value => {
                bytes.push(value);
                index += 1;
            }
        }
    }

    String::from_utf8(bytes).ok()
}

fn not_found_response() -> HttpResponse {
    HttpResponse {
        status: "404 Not Found".to_string(),
        content_type: "text/plain; charset=utf-8",
        body: "not found".to_string(),
    }
}

fn write_response<W: Write>(
    stream: &mut W,
    status: &str,
    content_type: &str,
    body: &str,
) -> Result<(), GatewayError> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: {content_type}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::GenerationConfig;
    use crate::supervisor::RuntimeState;
    use mlx_runtime_protocol::{ModelLoadProgress, ModelState, ModelStatus};
    use std::io::Cursor;
    use std::sync::Mutex;

    #[test]
    fn live_endpoint_returns_200_even_when_not_ready() {
        assert_eq!(
            response_for_request(
                "GET /live HTTP/1.1\r\n",
                &[],
                &test_state(ModelState::LoadingWeights, Arc::new(FakeService::default()))
            )
            .status,
            "200 OK"
        );
    }

    #[test]
    fn ready_endpoint_returns_503_when_model_is_not_loaded() {
        assert_eq!(
            response_for_request(
                "GET /ready HTTP/1.1\r\n",
                &[],
                &test_state(ModelState::NotLoaded, Arc::new(FakeService::default()))
            )
            .status,
            "503 Service Unavailable"
        );
    }

    #[test]
    fn ready_endpoint_returns_200_when_model_is_ready() {
        let response = response_for_request(
            "GET /ready HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::Ready, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"ready\":true"));
    }

    #[test]
    fn ready_endpoint_returns_503_when_model_has_failed() {
        let runtime = test_runtime(ModelState::Failed);
        runtime
            .mark_failed("MODEL_LOAD_FAILED", "failed to load model weights")
            .unwrap();

        let response = response_for_request(
            "GET /ready HTTP/1.1\r\n",
            &[],
            &AppState {
                runtime,
                generation: GenerationConfig {
                    temperature: 0.7,
                    top_p: 0.9,
                    max_tokens: 32,
                },
                limits: crate::config::RequestLimits {
                    max_pending_requests: 64,
                    max_active_requests: 16,
                    max_prompt_tokens: 32_768,
                    max_completion_tokens: 4_096,
                    max_total_tokens_per_request: 65_536,
                    request_timeout_seconds: 300,
                },
                admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
                model: "test-model".to_string(),
                test_backend: Some(Arc::new(FakeService::default())),
            },
        );

        assert_eq!(response.status, "503 Service Unavailable");
        assert!(response.body.contains("\"MODEL_LOAD_FAILED\""));
    }

    #[test]
    fn models_endpoint_lists_the_configured_model() {
        let response = response_for_request(
            "GET /models HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::Ready, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"models\""));
        assert!(response.body.contains("\"test-model\""));
    }

    #[test]
    fn model_ready_endpoint_returns_200_for_ready_model() {
        let response = response_for_request(
            "GET /models/test-model/ready HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::Ready, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"ready\":true"));
    }

    #[test]
    fn startup_endpoint_reports_starting_state() {
        let response = response_for_request(
            "GET /startup HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::LoadingWeights, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"status\":\"starting\""));
        assert!(response.body.contains("\"loading_weights\""));
    }

    #[test]
    fn health_endpoint_remains_plain_text_healthy_when_ready() {
        let response = response_for_request(
            "GET /health HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::Ready, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "200 OK");
        assert_eq!(response.content_type, "text/plain; charset=utf-8");
        assert_eq!(response.body, "healthy");
    }

    #[test]
    fn health_endpoint_remains_plain_text_unhealthy_when_loading() {
        let response = response_for_request(
            "GET /health HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::LoadingWeights, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "503 Service Unavailable");
        assert_eq!(response.content_type, "text/plain; charset=utf-8");
        assert_eq!(response.body, "unhealthy");
    }

    #[test]
    fn model_status_endpoint_returns_detailed_state() {
        let runtime = test_runtime(ModelState::LoadingWeights);
        runtime
            .set_status(ModelStatus {
                model: "test-model".to_string(),
                revision: Some("rev-1".to_string()),
                state: ModelState::LoadingWeights,
                ready: false,
                servable: false,
                progress: Some(ModelLoadProgress {
                    downloaded_bytes: Some(8),
                    total_bytes: Some(16),
                    loaded_tensors: Some(1),
                    total_tensors: Some(2),
                    current_phase: Some("loading_weights".to_string()),
                }),
                device: Some("mps".to_string()),
                dtype: Some("float16".to_string()),
                loaded_at: None,
                started_loading_at: Some(1),
                last_transition_at: 2,
                last_error: None,
                warmup_passed: false,
                last_warmup_at: None,
                last_warmup_latency_ms: None,
            })
            .unwrap();

        let response = response_for_request(
            "GET /models/test-model/status HTTP/1.1\r\n",
            &[],
            &AppState {
                runtime,
                generation: GenerationConfig {
                    temperature: 0.7,
                    top_p: 0.9,
                    max_tokens: 32,
                },
                limits: crate::config::RequestLimits {
                    max_pending_requests: 64,
                    max_active_requests: 16,
                    max_prompt_tokens: 32_768,
                    max_completion_tokens: 4_096,
                    max_total_tokens_per_request: 65_536,
                    request_timeout_seconds: 300,
                },
                admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
                model: "test-model".to_string(),
                test_backend: Some(Arc::new(FakeService::default())),
            },
        );

        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"loading_weights\""));
        assert!(response.body.contains("\"loaded_tensors\":1"));
    }

    #[test]
    fn model_ready_endpoint_returns_404_for_unknown_model() {
        let response = response_for_request(
            "GET /models/other-model/ready HTTP/1.1\r\n",
            &[],
            &test_state(ModelState::Ready, Arc::new(FakeService::default())),
        );
        assert_eq!(response.status, "404 Not Found");
    }

    #[test]
    fn chat_completion_rejects_when_model_is_not_ready() {
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let response = response_for_request(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            &body,
            &test_state(ModelState::LoadingWeights, Arc::new(FakeService::default())),
        );

        assert_eq!(response.status, "503 Service Unavailable");
        assert!(response.body.contains("\"MODEL_NOT_READY\""));
    }

    #[test]
    fn chat_completion_returns_429_when_queue_full() {
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();
        let state = AppState {
            runtime: test_runtime(ModelState::Ready),
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 32,
            },
            limits: crate::config::RequestLimits {
                max_pending_requests: 0,
                max_active_requests: 1,
                max_prompt_tokens: 32_768,
                max_completion_tokens: 4_096,
                max_total_tokens_per_request: 65_536,
                request_timeout_seconds: 300,
            },
            admission: Arc::new(RequestAdmission::new(0, 1, Duration::from_secs(300))),
            model: "test-model".to_string(),
            test_backend: Some(Arc::new(FakeService::default())),
        };
        let _permit = match state.admission.acquire() {
            Ok(permit) => permit,
            Err(response) => panic!("unexpected admission error: {}", response.status),
        };
        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);

        assert_eq!(response.status, "429 Too Many Requests");
        assert!(response.body.contains("\"QUEUE_FULL\""));
    }

    #[test]
    fn chat_completion_returns_openai_style_response() {
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-1".to_string(),
                model: "test-model".to_string(),
                text: "hello back".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 10,
                completion_tokens: 3,
            })),
        });
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        let response = response_for_request(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            &body,
            &test_state(ModelState::Ready, service),
        );

        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"chat.completion\""));
        assert!(response.body.contains("\"hello back\""));
    }

    #[test]
    fn streaming_chat_completion_writes_sse_chunks() {
        let service = Arc::new(StreamingService);
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": true
        }))
        .unwrap();
        let state = test_state(ModelState::Ready, service);
        let mut buffer = Cursor::new(Vec::new());

        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let body = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(body.contains("text/event-stream"));
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("data: [DONE]"));
    }

    #[test]
    fn streaming_chat_completion_returns_http_400_before_stream_starts() {
        let service = Arc::new(InvalidStreamingService);
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": true
        }))
        .unwrap();
        let state = test_state(ModelState::Ready, service);
        let mut buffer = Cursor::new(Vec::new());

        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let body = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(body.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(body.contains("\"INVALID_REQUEST\""));
        assert!(!body.contains("text/event-stream"));
    }

    #[test]
    fn streaming_chat_completion_cancels_when_disconnected_before_first_delta() {
        let service = Arc::new(CancellableStreamingService::default());
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": true
        }))
        .unwrap();
        let state = test_state(ModelState::Ready, service.clone());
        let mut buffer = Cursor::new(Vec::new());
        let disconnected = Arc::new(AtomicBool::new(false));
        let trigger = disconnected.clone();

        let flip = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            trigger.store(true, Ordering::Relaxed);
        });

        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, Some(disconnected))
            .unwrap();
        flip.join().unwrap();

        assert_eq!(service.cancel_count(), 2);
        assert_eq!(String::from_utf8(buffer.into_inner()).unwrap(), "");
    }

    fn test_state(state: ModelState, backend: Arc<dyn ChatCompletionService>) -> AppState {
        let runtime = test_runtime(state);
        AppState {
            runtime,
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 32,
            },
            limits: crate::config::RequestLimits {
                max_pending_requests: 64,
                max_active_requests: 16,
                max_prompt_tokens: 32_768,
                max_completion_tokens: 4_096,
                max_total_tokens_per_request: 65_536,
                request_timeout_seconds: 300,
            },
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "test-model".to_string(),
            test_backend: Some(backend),
        }
    }

    fn test_runtime(state: ModelState) -> RuntimeState {
        let runtime = RuntimeState::new("test-model");
        let mut status = ModelStatus::new("test-model");
        status.set_state(state);
        runtime.set_status(status).unwrap();
        runtime
    }

    #[derive(Default)]
    struct FakeService {
        response: Mutex<Option<ChatCompletionResponse>>,
    }

    impl ChatCompletionService for FakeService {
        fn complete(
            &self,
            _request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Ok(self
                .response
                .lock()
                .unwrap()
                .clone()
                .unwrap_or(ChatCompletionResponse {
                    request_id: "req-default".to_string(),
                    model: "test-model".to_string(),
                    text: "default".to_string(),
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                }))
        }

        fn stream(
            &self,
            request: ChatCompletionRequest,
            on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            let response = self.complete(request.clone())?;
            on_delta("hel".to_string())?;
            on_delta("lo".to_string())?;
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: response.model,
                text: "hello".to_string(),
                finish_reason: response.finish_reason,
                prompt_tokens: response.prompt_tokens,
                completion_tokens: response.completion_tokens,
            })
        }
    }

    #[derive(Default)]
    struct StreamingService;

    impl ChatCompletionService for StreamingService {
        fn complete(
            &self,
            request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: "hello".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }

        fn stream(
            &self,
            request: ChatCompletionRequest,
            on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            on_delta("hel".to_string())?;
            on_delta("lo".to_string())?;
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: "hello".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }
    }

    #[derive(Default)]
    struct InvalidStreamingService;

    impl ChatCompletionService for InvalidStreamingService {
        fn complete(
            &self,
            request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: "hello".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }

        fn stream(
            &self,
            _request: ChatCompletionRequest,
            _on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Err(GatewayError::InvalidRequest("prompt too long".to_string()))
        }
    }

    #[derive(Default, Clone)]
    struct CancellableStreamingService {
        cancelled: Arc<AtomicBool>,
        cancel_calls: Arc<Mutex<usize>>,
    }

    impl CancellableStreamingService {
        fn cancel_count(&self) -> usize {
            *self.cancel_calls.lock().unwrap()
        }
    }

    impl ChatCompletionService for CancellableStreamingService {
        fn complete(
            &self,
            request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: "hello".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: 1,
            })
        }

        fn stream(
            &self,
            _request: ChatCompletionRequest,
            _on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            while !self.cancelled.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(10));
            }
            Err(GatewayError::Protocol("cancelled".to_string()))
        }

        fn cancel(&self, _request_id: &str) -> Result<(), GatewayError> {
            self.cancelled.store(true, Ordering::Relaxed);
            *self.cancel_calls.lock().unwrap() += 1;
            Ok(())
        }
    }
}
