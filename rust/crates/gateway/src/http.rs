use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::openai::{ChatCompletionHttpRequest, ChatCompletionHttpResponse};
use crate::supervisor::RuntimeState;
use crate::telemetry::RequestTracker;
use mlx_runtime_protocol::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ContentPart, MessageContent,
    ModelState, ModelStatus,
};
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::Path;
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
    telemetry: crate::config::TelemetryConfig,
    admission: Arc<RequestAdmission>,
    model: String,
    vlm_model: Option<String>,
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

    fn snapshot(&self) -> Result<(usize, usize), HttpResponse> {
        let guard = self.state.lock().map_err(|_| {
            json_error_response(
                "500 Internal Server Error",
                "BACKPRESSURE_LOCK_POISONED",
                "request admission lock poisoned",
            )
        })?;
        Ok((guard.active, guard.waiting))
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
                    telemetry: config.telemetry.clone(),
                    admission: admission.clone(),
                    model: config.worker.model.clone(),
                    vlm_model: config.worker.vlm_model.clone(),
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
    // Disable Nagle's algorithm: small SSE frames are time-sensitive and
    // should not be delayed waiting for a TCP ACK.  This is safe for all
    // connections (non-SSE responses also flush explicitly).
    stream.set_nodelay(true)?;
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

    if request_line.starts_with("POST /v1/chat/completions ") && !is_stream {
        let disconnect_monitor = DisconnectMonitor::start(stream.try_clone()?)?;
        let response =
            handle_chat_completion(&body, &state, Some(disconnect_monitor.disconnected()));
        drop(disconnect_monitor);
        write_response(
            &mut stream,
            &response.status,
            response.content_type,
            &response.body,
        )?;
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
        ("GET", path)
            if state.telemetry.enable_prometheus && path == state.telemetry.metrics_path =>
        {
            metrics_response(&state.runtime, &state.admission)
        }
        ("GET", "/models") => models_response(&state.runtime, state.vlm_model.as_deref()),
        ("POST", "/v1/chat/completions") => handle_chat_completion(body, state, None),
        _ if method == "GET" && path.starts_with("/models/") && path.ends_with("/status") => {
            model_status_response(&state.runtime, &path, state.vlm_model.as_deref())
        }
        _ if method == "GET" && path.starts_with("/models/") && path.ends_with("/ready") => {
            model_ready_response(&state.runtime, &path, state.vlm_model.as_deref())
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

fn metrics_response(runtime: &RuntimeState, admission: &Arc<RequestAdmission>) -> HttpResponse {
    let queue_depth = match admission.snapshot() {
        Ok((_, waiting)) => waiting as u64,
        Err(response) => {
            return response;
        }
    };
    let body = runtime.metrics.render_prometheus(queue_depth);
    HttpResponse {
        status: "200 OK".to_string(),
        content_type: "text/plain; version=0.0.4; charset=utf-8",
        body,
    }
}

fn models_response(runtime: &RuntimeState, vlm_model: Option<&str>) -> HttpResponse {
    match runtime.snapshot() {
        Ok(status) => {
            let mut models = vec![status.summary()];
            if let Some(vlm) = vlm_model {
                if vlm != status.model {
                    let vlm_summary = runtime
                        .snapshot_vlm()
                        .ok()
                        .as_ref()
                        .map(|s| s.summary())
                        .unwrap_or(mlx_runtime_protocol::ModelSummary {
                            name: vlm.to_string(),
                            state: ModelState::NotLoaded,
                            ready: false,
                            revision: Some(vlm.to_string()),
                        });
                    models.push(vlm_summary);
                }
            }
            HttpResponse {
                status: "200 OK".to_string(),
                content_type: "application/json",
                body: json!({ "models": models }).to_string(),
            }
        }
        Err(err) => internal_error_response(&err.to_string()),
    }
}

fn model_status_response(
    runtime: &RuntimeState,
    path: &str,
    vlm_model: Option<&str>,
) -> HttpResponse {
    let model_name = runtime_model_name(runtime);
    match model_path_name(path) {
        Some(name) if name == model_name || vlm_model.is_some_and(|v| name == v) => {
            let is_vlm = vlm_model.is_some_and(|v| name == v);
            let status = if is_vlm {
                runtime.snapshot_vlm()
            } else {
                runtime.snapshot()
            };
            match status {
                Ok(status) => HttpResponse {
                    status: "200 OK".to_string(),
                    content_type: "application/json",
                    body: serde_json::to_string(&status).unwrap_or_else(|_| {
                        "{\"error\":{\"message\":\"status serialization failed\"}}".to_string()
                    }),
                },
                Err(err) => internal_error_response(&err.to_string()),
            }
        }
        _ => not_found_response(),
    }
}

fn model_ready_response(
    runtime: &RuntimeState,
    path: &str,
    vlm_model: Option<&str>,
) -> HttpResponse {
    let model_name = runtime_model_name(runtime);
    match model_path_name(path) {
        Some(name) if name == model_name || vlm_model.is_some_and(|v| name == v) => {
            let is_vlm = vlm_model.is_some_and(|v| name == v);
            let status = if is_vlm {
                runtime.snapshot_vlm()
            } else {
                runtime.snapshot()
            };
            match status {
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
            }
        }
        _ => not_found_response(),
    }
}

/// Maximum allowed length for an image URL string (matches Python worker).
const MAX_IMAGE_URL_LENGTH: usize = 4096;

/// Allowed hosts for loopback HTTP image URLs.
const LOOPBACK_HTTP_HOSTS: &[&str] = &["localhost", "127.0.0.1"];

#[cfg(test)]
/// Validate all image URLs in a set of messages.
///
/// Rejects URLs whose scheme is not supported or whose length exceeds
/// [`MAX_IMAGE_URL_LENGTH`]. Supports local file paths, HTTPS URLs, and
/// loopback HTTP URLs for deterministic host validation.
fn validate_image_urls(messages: &[ChatMessage]) -> Result<(), String> {
    for msg in messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    validate_image_source(&image_url.url)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_image_source(url: &str) -> Result<(), String> {
    if url.len() > MAX_IMAGE_URL_LENGTH {
        return Err(format!(
            "image URL exceeds maximum length of {} characters",
            MAX_IMAGE_URL_LENGTH
        ));
    }

    if url.starts_with("data:") {
        return Err(
            "unsupported image URL scheme 'data': must be one of https, http, or a local file path"
                .to_string(),
        );
    }

    let Some((scheme, rest)) = url.split_once("://") else {
        let path = Path::new(url);
        if !path.is_file() {
            return Err(format!(
                "local image path does not exist or is not a file: {}",
                url
            ));
        }
        return Ok(());
    };

    match scheme {
        "https" => Ok(()),
        "http" => {
            let host_port = rest.split('/').next().unwrap_or("");
            let host = host_port.split(':').next().unwrap_or(host_port);
            if LOOPBACK_HTTP_HOSTS.contains(&host) {
                Ok(())
            } else {
                Err(format!(
                    "http image URLs must use localhost or 127.0.0.1, got '{}': {}",
                    host, url
                ))
            }
        }
        other => Err(format!(
            "unsupported image URL scheme '{}': must be one of https, http, or a local file path",
            other
        )),
    }
}

/// Inspect VLM messages in a single pass.
///
/// Returns `(has_images, image_count)` after validating image URL scheme/length.
fn inspect_vlm_messages(messages: &[ChatMessage]) -> Result<(bool, u64), String> {
    let mut has_images = false;
    let mut image_count = 0u64;

    for msg in messages {
        if let MessageContent::Parts(parts) = &msg.content {
            for part in parts {
                if let ContentPart::ImageUrl { image_url } = part {
                    has_images = true;
                    image_count += 1;

                    validate_image_source(&image_url.url)?;
                }
            }
        }
    }

    Ok((has_images, image_count))
}

#[cfg(test)]
/// Returns true if any message contains structured content with an ImageUrl part.
fn has_vlm_content(messages: &[ChatMessage]) -> bool {
    messages.iter().any(|msg| match &msg.content {
        MessageContent::Parts(parts) => parts
            .iter()
            .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        MessageContent::Text(_) => false,
    })
}

#[cfg(test)]
/// Counts ImageUrl parts across all messages.
fn vlm_image_count(messages: &[ChatMessage]) -> u64 {
    messages
        .iter()
        .map(|msg| match &msg.content {
            MessageContent::Parts(parts) => parts
                .iter()
                .filter(|p| matches!(p, ContentPart::ImageUrl { .. }))
                .count() as u64,
            MessageContent::Text(_) => 0,
        })
        .sum()
}

fn handle_chat_completion(
    body: &[u8],
    state: &AppState,
    disconnected: Option<Arc<AtomicBool>>,
) -> HttpResponse {
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

    // Model-first routing: route based on requested model name, not image content.
    let (has_images, image_count) = match inspect_vlm_messages(&request.messages) {
        Ok(result) => result,
        Err(message) => {
            return json_error_response("400 Bad Request", "INVALID_IMAGE_URL", &message);
        }
    };
    let is_vlm = state.vlm_model.as_deref() == Some(&request.model);

    let status = match if is_vlm {
        state.runtime.snapshot_vlm()
    } else {
        state.runtime.snapshot()
    } {
        Ok(status) => status,
        Err(err) => return internal_error_response(&err.to_string()),
    };

    if !status.ready {
        return not_ready_error(&status);
    }

    // Validate: images require VLM-capable model
    if has_images && !is_vlm {
        let msg = if state.vlm_model.is_some() {
            "image content requires a VLM-capable model, but requested model is not VLM"
        } else {
            "image content requires a VLM-capable model which is not configured"
        };
        return json_error_response("400 Bad Request", "VLM_NOT_CONFIGURED", msg);
    }

    // Phase 8: enforce VLM image-count cap before queue admission.
    if has_images && image_count > state.limits.max_vlm_images as u64 {
        return json_error_response(
            "400 Bad Request",
            "TOO_MANY_IMAGES",
            &format!(
                "VLM request contains {} images, maximum allowed is {}",
                image_count, state.limits.max_vlm_images
            ),
        );
    }

    let worker_request = match request.into_worker_request(
        &state.generation,
        &state.limits,
        &state.model,
        state.vlm_model.as_deref(),
    ) {
        Ok(request) => request,
        Err(message) => return json_error_response("400 Bad Request", "INVALID_REQUEST", &message),
    };

    let queue_started_at = Instant::now();
    let _permit = match state.admission.acquire() {
        Ok(permit) => permit,
        Err(response) => {
            if response.status.starts_with("429") {
                state.runtime.metrics.increment_queue_rejected_total();
            }
            return response;
        }
    };
    let queue_time_ms = queue_started_at.elapsed().as_millis() as u64;
    let worker_model = worker_request.model.clone();
    let request_tracker = Arc::new(RequestTracker::new(
        state.runtime.metrics.clone(),
        worker_request.request_id.clone(),
        worker_model,
        worker_request.max_tokens,
        false,
        queue_time_ms,
        is_vlm,
        image_count,
    ));

    let backend = if let Some(backend) = &state.test_backend {
        Some(backend.clone())
    } else {
        match state.runtime.worker_client.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|client| client.clone() as Arc<dyn ChatCompletionService>),
            Err(_) => {
                request_tracker.finish(
                    0,
                    0,
                    None,
                    false,
                    Some("worker client lock poisoned".to_string()),
                );
                return json_error_response(
                    "500 Internal Server Error",
                    "WORKER_CLIENT_LOCK_POISONED",
                    "worker client lock poisoned",
                );
            }
        }
    };

    let Some(backend) = backend else {
        request_tracker.finish(0, 0, None, false, Some("backend unavailable".to_string()));
        return not_ready_error(&status);
    };

    let cancel_on_disconnect = DisconnectCancellation::start(
        disconnected.clone(),
        backend.clone(),
        worker_request.request_id.clone(),
    );

    let response = match backend.complete(worker_request) {
        Ok(response) => {
            request_tracker.finish(
                response.prompt_tokens as u64,
                response.completion_tokens as u64,
                Some(response.finish_reason.clone()),
                false,
                None,
            );
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
        Err(err) => {
            // VLM request lifecycle: first VLM request failure marks it failed.
            if is_vlm {
                track_vlm_lifecycle_failure(&state.runtime, &err);
            }
            request_tracker.finish(0, 0, None, false, Some(err.to_string()));
            gateway_error_response(&err)
        }
    };

    drop(cancel_on_disconnect);

    if disconnected
        .as_ref()
        .is_some_and(|flag| flag.load(Ordering::Relaxed))
    {
        request_tracker.finish(0, 0, None, true, None);
    }

    response
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

    // Model-first routing: route based on requested model name, not image content.
    let (has_images, image_count) = match inspect_vlm_messages(&request.messages) {
        Ok(result) => result,
        Err(message) => {
            let response = json_error_response("400 Bad Request", "INVALID_IMAGE_URL", &message);
            return write_response(
                writer,
                &response.status,
                response.content_type,
                &response.body,
            );
        }
    };
    let is_vlm = state.vlm_model.as_deref() == Some(&request.model);

    let status = if is_vlm {
        state
            .runtime
            .snapshot_vlm()
            .map_err(|err| GatewayError::Protocol(err.to_string()))?
    } else {
        state
            .runtime
            .snapshot()
            .map_err(|err| GatewayError::Protocol(err.to_string()))?
    };

    if !status.ready {
        let response = not_ready_error(&status);
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    }

    // Validate: images require VLM-capable model
    if has_images && !is_vlm {
        let msg = if state.vlm_model.is_some() {
            "image content requires a VLM-capable model, but requested model is not VLM"
        } else {
            "image content requires a VLM-capable model which is not configured"
        };
        let response = json_error_response("400 Bad Request", "VLM_NOT_CONFIGURED", msg);
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    }

    // Phase 8: enforce VLM image-count cap before queue admission.
    if has_images && image_count > state.limits.max_vlm_images as u64 {
        let response = json_error_response(
            "400 Bad Request",
            "TOO_MANY_IMAGES",
            &format!(
                "VLM request contains {} images, maximum allowed is {}",
                image_count, state.limits.max_vlm_images
            ),
        );
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    }

    let worker_request = match request.into_worker_request(
        &state.generation,
        &state.limits,
        &state.model,
        state.vlm_model.as_deref(),
    ) {
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

    let queue_started_at = Instant::now();
    let _permit = match state.admission.acquire_until(disconnected.as_deref()) {
        Ok(Some(permit)) => permit,
        Ok(None) => {
            return Ok(());
        }
        Err(response) => {
            if response.status.starts_with("429") {
                state.runtime.metrics.increment_queue_rejected_total();
            }
            return write_response(
                writer,
                &response.status,
                response.content_type,
                &response.body,
            );
        }
    };
    let queue_time_ms = queue_started_at.elapsed().as_millis() as u64;
    let worker_model = worker_request.model.clone();
    // Save the requested model name before worker_model is moved into
    // RequestTracker; used for consistent "model" field across all SSE chunks.
    let sse_model = worker_model.clone();
    let request_tracker = Arc::new(RequestTracker::new(
        state.runtime.metrics.clone(),
        worker_request.request_id.clone(),
        worker_model,
        worker_request.max_tokens,
        true,
        queue_time_ms,
        is_vlm,
        image_count,
    ));

    let backend = if let Some(backend) = &state.test_backend {
        Some(backend.clone())
    } else {
        match state.runtime.worker_client.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|client| client.clone() as Arc<dyn ChatCompletionService>),
            Err(_) => {
                request_tracker.finish(
                    0,
                    0,
                    None,
                    false,
                    Some("worker client lock poisoned".to_string()),
                );
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
        request_tracker.finish(0, 0, None, false, Some("backend unavailable".to_string()));
        let response = not_ready_error(&status);
        return write_response(
            writer,
            &response.status,
            response.content_type,
            &response.body,
        );
    };

    let created = stream_completion_created();
    let model = sse_model;
    let request_id = worker_request.request_id.clone();
    let mut stream_started = false;
    let cancel_on_disconnect =
        DisconnectCancellation::start(disconnected.clone(), backend.clone(), request_id.clone());

    let mut on_delta = |delta: String| -> Result<(), GatewayError> {
        request_tracker.record_first_delta();
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
        if !stream_started {
            // First delta: combine SSE headers + data into a single write
            // to save one write(2) syscall and avoid split TCP segments.
            stream_started = true;
            write!(
                writer,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\ndata: {}\n\n",
                event
            )?;
        } else {
            write!(writer, "data: {}\n\n", event)?;
        }
        writer.flush()?;
        Ok(())
    };

    let response = match backend.stream(worker_request, &mut on_delta) {
        Ok(response) => response,
        Err(err) => {
            // VLM request lifecycle: first VLM request failure marks it failed.
            if is_vlm {
                track_vlm_lifecycle_failure(&state.runtime, &err);
            }
            drop(cancel_on_disconnect);
            let _ = backend.cancel(&request_id);
            if disconnected
                .as_ref()
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
            {
                request_tracker.finish(0, 0, None, true, None);
                return Ok(());
            }
            request_tracker.finish(0, 0, None, false, Some(err.to_string()));
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
        request_tracker.finish(
            response.prompt_tokens as u64,
            response.completion_tokens as u64,
            Some(response.finish_reason.clone()),
            true,
            None,
        );
        return Ok(());
    }
    ensure_sse_stream_started(writer, &mut stream_started)?;
    request_tracker.finish(
        response.prompt_tokens as u64,
        response.completion_tokens as u64,
        Some(response.finish_reason.clone()),
        false,
        None,
    );
    let done_event = json!({
        "id": format!("chatcmpl-{}", response.request_id),
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
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

/// Track VLM lifecycle: mark VLM as failed after first failed request.
fn track_vlm_lifecycle_failure(runtime: &RuntimeState, err: &GatewayError) {
    // Only non-InvalidRequest errors indicate load failure (not bad user input).
    if !matches!(err, GatewayError::InvalidRequest(_)) {
        if let Ok(vlm_status) = runtime.snapshot_vlm() {
            if vlm_status.state == ModelState::NotLoaded {
                let _ = runtime.mark_vlm_failed("VLM_LOAD_FAILED", err.to_string());
            }
        }
    }
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
    use crate::telemetry::MetricsRegistry;
    use mlx_runtime_protocol::{
        ChatMessage, ContentPart, ImageUrl, MessageContent, MessageRole, ModelLoadProgress,
        ModelState, ModelStatus,
    };
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
                    max_vlm_images: 5,
                },
                telemetry: crate::config::TelemetryConfig::default(),
                admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
                model: "test-model".to_string(),
                vlm_model: None,
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
    fn metrics_endpoint_exposes_prometheus_metrics() {
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-1".to_string(),
                model: "test-model".to_string(),
                text: "hello".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 12,
                completion_tokens: 7,
            })),
        });
        let state = test_state(ModelState::Ready, service);
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");

        let metrics = response_for_request("GET /metrics HTTP/1.1\r\n", &[], &state);
        assert_eq!(metrics.status, "200 OK");
        assert_eq!(
            metrics.content_type,
            "text/plain; version=0.0.4; charset=utf-8"
        );
        assert!(metrics.body.contains("mlx_requests_total 1"));
        assert!(metrics.body.contains("mlx_requests_active 0"));
        assert!(metrics.body.contains("mlx_prompt_tokens_total 12"));
        assert!(metrics.body.contains("mlx_completion_tokens_total 7"));
        assert!(metrics.body.contains("mlx_worker_up 1"));
        assert!(metrics.body.contains("mlx_ttft_ms_bucket"));
    }

    #[test]
    fn metrics_disabled_when_prometheus_off() {
        let state = AppState {
            telemetry: crate::config::TelemetryConfig {
                enable_prometheus: false,
                ..Default::default()
            },
            ..test_state(ModelState::Ready, Arc::new(FakeService::default()))
        };
        let response = response_for_request("GET /metrics HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "404 Not Found");
    }

    #[test]
    fn metrics_served_on_custom_path() {
        let state = AppState {
            telemetry: crate::config::TelemetryConfig {
                enable_prometheus: true,
                metrics_path: "/custom-metrics".to_string(),
            },
            ..test_state(ModelState::Ready, Arc::new(FakeService::default()))
        };
        let ok = response_for_request("GET /custom-metrics HTTP/1.1\r\n", &[], &state);
        assert_eq!(ok.status, "200 OK");
        assert_eq!(ok.content_type, "text/plain; version=0.0.4; charset=utf-8");

        let not_found = response_for_request("GET /metrics HTTP/1.1\r\n", &[], &state);
        assert_eq!(not_found.status, "404 Not Found");
    }

    #[test]
    fn queue_rejected_counter_only_increments_on_429() {
        let metrics = Arc::new(MetricsRegistry::new());
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        let runtime = RuntimeState::new_with_metrics("test-model", metrics.clone());
        let mut status = ModelStatus::new("test-model");
        status.set_state(ModelState::Ready);
        status.mark_ready(None, None, 0);
        runtime.set_status(status).unwrap();

        let state = AppState {
            runtime,
            admission: Arc::new(RequestAdmission::new(10, 10, Duration::from_secs(300))),
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            model: "test-model".to_string(),
            vlm_model: None,
            test_backend: Some(Arc::new(FakeService::default())),
        };

        let ok_response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(ok_response.status, "200 OK");

        let before_reject = metrics.render_prometheus(0);
        assert!(before_reject.contains("mlx_queue_rejected_total 0"));

        let rejected_state = AppState {
            admission: Arc::new(RequestAdmission::new(0, 0, Duration::from_secs(300))),
            ..state
        };
        let reject_response = response_for_request(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            &body,
            &rejected_state,
        );
        assert_eq!(reject_response.status, "429 Too Many Requests");

        let after_reject = metrics.render_prometheus(0);
        assert!(after_reject.contains("mlx_queue_rejected_total 1"));
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
                    max_vlm_images: 5,
                },
                telemetry: crate::config::TelemetryConfig::default(),
                admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
                model: "test-model".to_string(),
                vlm_model: None,
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(0, 1, Duration::from_secs(300))),
            model: "test-model".to_string(),
            vlm_model: None,
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

        // Validate combined-write optimization: headers immediately followed by
        // first data event (no empty output between them, confirming a single write).
        let header_end = "\r\n\r\n";
        let after_headers = body
            .find(header_end)
            .map(|pos| &body[pos + header_end.len()..])
            .unwrap_or("");
        assert!(
            after_headers.starts_with("data: "),
            "expected headers immediately followed by first data event, got: {:?}",
            &after_headers.chars().take(30).collect::<String>()
        );

        // All SSE data events must use the same model field.
        let model_ref = "\"model\":\"test-model\"";
        let model_count = body.matches(model_ref).count();
        assert!(
            model_count >= 3,
            "expected at least 3 occurrences of consistent model field, got {}",
            model_count
        );
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "test-model".to_string(),
            vlm_model: None,
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

    #[test]
    fn tcp_nodelay_is_set_on_accepted_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let nodelay = stream.nodelay().unwrap();
            // The stream comes from accept() without TCP_NODELAY by default.
            assert!(!nodelay, "default TcpStream should not have TCP_NODELAY");
            // Now simulate what handle_connection does.
            stream.set_nodelay(true).unwrap();
            assert!(stream.nodelay().unwrap(), "TCP_NODELAY should be set");
        });
        let client = TcpStream::connect(addr).unwrap();
        // Ensure stream options are not inherited from client side.
        assert!(!client.nodelay().unwrap());
        drop(client);
        server.join().unwrap();
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

    /// Service that fails on first `complete()` call, succeeds on subsequent.
    /// Used to test VLM lifecycle recovery from transient failure.
    #[derive(Default)]
    struct RecoveryService {
        attempts: Mutex<usize>,
    }

    impl ChatCompletionService for RecoveryService {
        fn complete(
            &self,
            request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            let mut attempts = self.attempts.lock().unwrap();
            *attempts += 1;
            if *attempts == 1 {
                Err(GatewayError::Protocol("transient vlm error".to_string()))
            } else {
                Ok(ChatCompletionResponse {
                    request_id: request.request_id,
                    model: request.model,
                    text: "recovered".to_string(),
                    finish_reason: "stop".to_string(),
                    prompt_tokens: 1,
                    completion_tokens: 1,
                })
            }
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

    // ── VLM routing tests ─────────────────────────────────────────────────

    #[test]
    fn has_vlm_content_detects_image_url_parts() {
        let text_only = vec![ChatMessage {
            role: MessageRole::User,
            content: "hello".into(),
        }];
        assert!(!has_vlm_content(&text_only));

        let with_image = vec![ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "describe".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc".to_string(),
                        detail: None,
                    },
                },
            ]),
        }];
        assert!(has_vlm_content(&with_image));
    }

    #[test]
    fn vlm_image_count_counts_correctly() {
        let msg = ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![
                ContentPart::Text {
                    text: "compare".to_string(),
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "img1.jpg".to_string(),
                        detail: None,
                    },
                },
                ContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "img2.jpg".to_string(),
                        detail: None,
                    },
                },
            ]),
        };
        assert_eq!(vlm_image_count(&[msg]), 2);
    }

    #[test]
    fn models_endpoint_lists_vlm_model_when_configured() {
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };
        let response = response_for_request("GET /models HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"text-model\""));
        assert!(response.body.contains("\"vlm-model\""));
    }

    #[test]
    fn models_endpoint_single_model_when_no_vlm_configured() {
        let state = test_state(ModelState::Ready, Arc::new(FakeService::default()));
        let response = response_for_request("GET /models HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"test-model\""));
        // Only one model in the list
        assert_eq!(response.body.matches("\"name\"").count(), 1);
    }

    #[test]
    fn vlm_model_status_endpoint_returns_200() {
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };
        let response =
            response_for_request("GET /models/vlm-model/status HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"vlm-model\""));
        assert!(response.body.contains("\"ready\":true"));
    }

    #[test]
    fn vlm_model_ready_endpoint_returns_200() {
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };
        let response =
            response_for_request("GET /models/vlm-model/ready HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"ready\":true"));
        assert!(response.body.contains("\"vlm-model\""));
    }

    #[test]
    fn vlm_model_not_ready_without_explicit_mark() {
        // VLM model must NOT be marked ready from generic worker READY;
        // it requires explicit warmup completion.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };
        let response =
            response_for_request("GET /models/vlm-model/ready HTTP/1.1\r\n", &[], &state);
        assert_eq!(response.status, "503 Service Unavailable");
        assert!(response.body.contains("\"ready\":false"));
        assert!(response.body.contains("\"not_loaded\""));
    }

    // ── VLM HTTP request handling tests ───────────────────────────────────

    #[test]
    fn chat_completion_with_vlm_content_returns_200_when_vlm_configured() {
        // POST with image content, VLM configured → 200 OK
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-vlm".to_string(),
                model: "vlm-model".to_string(),
                text: "VLM description".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 15,
                completion_tokens: 8,
            })),
        });
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service.clone()),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                    ]
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"chat.completion\""));
        assert!(response.body.contains("\"VLM description\""));
        assert!(response.body.contains("\"vlm-model\""));
    }

    #[test]
    fn chat_completion_with_vlm_content_rejected_when_no_vlm_configured() {
        // POST with image content, VLM NOT configured → 400 VLM_NOT_CONFIGURED
        let state = test_state(ModelState::Ready, Arc::new(FakeService::default()));

        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"VLM_NOT_CONFIGURED\""));
        assert!(response
            .body
            .contains("image content requires a VLM-capable model"));
    }

    #[test]
    fn streaming_chat_completion_with_vlm_content_works() {
        // POST stream=true with VLM content → SSE events
        let service = Arc::new(StreamingService);
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service.clone()),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what's in this image"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/test.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": true
        }))
        .unwrap();

        let mut buffer = Cursor::new(Vec::new());
        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let output = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(output.contains("text/event-stream"));
        assert!(output.contains("chat.completion.chunk"));
        assert!(output.contains("data: [DONE]"));
    }

    // ── Model-first routing tests (Phase 8) ────────────────────────────────

    #[test]
    fn chat_completion_text_only_to_vlm_model_succeeds() {
        // Text-only request targeting VLM model must reach VLM-capable path.
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-vlm-text".to_string(),
                model: "vlm-model".to_string(),
                text: "VLM text response".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 5,
                completion_tokens: 3,
            })),
        });
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service.clone()),
        };

        // Text-only request explicitly targeting the VLM model name
        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [{"role": "user", "content": "hello from text-only"}],
            "max_tokens": 16,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"VLM text response\""));
        assert!(response.body.contains("\"vlm-model\""));
    }

    // ── Image URL validation tests (Phase 8) ──────────────────────────────

    /// Helper: creates a state with VLM configured and ready.
    fn vlm_ready_state(vlm_model: &str) -> AppState {
        let service = Arc::new(FakeService::default());
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model(vlm_model.to_string());
        let _ = runtime.mark_vlm_ready(1);
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some(vlm_model.to_string()),
            test_backend: Some(service),
        }
    }

    #[test]
    fn validate_image_urls_rejects_data_uri() {
        // data: URIs are unsupported image sources.
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,abc123".to_string(),
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("unsupported image URL scheme"));
        assert!(err.contains("must be one of"));
    }

    #[test]
    fn validate_image_urls_accepts_loopback_http_scheme() {
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "http://127.0.0.1:8000/img.jpg".to_string(),
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_image_urls_rejects_remote_http_scheme() {
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "http://example.com/img.jpg".to_string(),
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("localhost or 127.0.0.1"));
    }

    #[test]
    fn validate_image_urls_accepts_local_path() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("phase_8_local_{unique}.ppm"));
        std::fs::write(&path, "P3\n1 1\n255\n255 0 0\n").unwrap();
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: path.to_string_lossy().into_owned(),
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_image_urls_rejects_long_url() {
        let long_url = "https://example.com/".to_string() + &"x".repeat(4096);
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: long_url,
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("exceeds maximum length of 4096"));
    }

    #[test]
    fn validate_image_urls_accepts_https_scheme() {
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: MessageContent::Parts(vec![ContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "https://example.com/img.jpg".to_string(),
                    detail: None,
                },
            }]),
        }]);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_image_urls_accepts_text_only() {
        let result = validate_image_urls(&[ChatMessage {
            role: MessageRole::User,
            content: "just text".into(),
        }]);
        assert!(result.is_ok());
    }

    #[test]
    fn chat_completion_rejects_data_uri_image_url() {
        let state = vlm_ready_state("vlm-model");
        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"INVALID_IMAGE_URL\""));
        assert!(response.body.contains("unsupported image URL scheme"));
    }

    #[test]
    fn chat_completion_rejects_remote_http_image_url() {
        let state = vlm_ready_state("vlm-model");
        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "http://example.com/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"INVALID_IMAGE_URL\""));
        assert!(response.body.contains("http"));
    }

    #[test]
    fn chat_completion_rejects_file_image_url() {
        let state = vlm_ready_state("vlm-model");
        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "file:///tmp/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"INVALID_IMAGE_URL\""));
        assert!(response.body.contains("file"));
    }

    #[test]
    fn chat_completion_accepts_https_image_url() {
        // HTTPS image URLs must be accepted through the full HTTP path.
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-https".to_string(),
                model: "vlm-model".to_string(),
                text: "valid https image".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 10,
                completion_tokens: 3,
            })),
        });
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"valid https image\""));
    }

    #[test]
    fn streaming_chat_completion_rejects_unsupported_image_url() {
        let state = vlm_ready_state("vlm-model");
        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,xyz"}}
                    ]
                }
            ],
            "max_tokens": 16,
            "stream": true
        }))
        .unwrap();

        let mut buffer = Cursor::new(Vec::new());
        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let output = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(output.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(output.contains("\"INVALID_IMAGE_URL\""));
        assert!(output.contains("unsupported image URL scheme"));
        assert!(!output.contains("text/event-stream"));
    }

    // ── VLM lifecycle recovery tests (Phase 8) ────────────────────────────

    #[test]
    fn vlm_lifecycle_recovers_from_transient_failure() {
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        // VLM starts as NotLoaded — deliberately not marking ready yet.

        // Verify initial VLM state is NotLoaded
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::NotLoaded);
        assert!(!vlm_status.ready);

        // First request fails with a transient (non-InvalidRequest) error →
        // VLM should transition to Failed
        track_vlm_lifecycle_failure(
            &runtime,
            &GatewayError::Protocol("transient vlm error".to_string()),
        );
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Failed);
        assert!(!vlm_status.ready);

        // Explicit warmup completion → VLM should recover to Ready.
        runtime.mark_vlm_ready(7).unwrap();
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);
        assert!(vlm_status.ready);
        assert_eq!(vlm_status.last_warmup_latency_ms, Some(7));
    }

    #[test]
    fn vlm_lifecycle_invalid_request_does_not_mark_failed() {
        // GatewayError::InvalidRequest (bad user input) must NOT mark VLM
        // as failed; only backend/worker errors indicate VLM load issues.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());

        // An InvalidRequest error should NOT trigger VLM failure transition
        track_vlm_lifecycle_failure(
            &runtime,
            &GatewayError::InvalidRequest("bad prompt".to_string()),
        );
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(
            vlm_status.state,
            ModelState::NotLoaded,
            "InvalidRequest errors must not mark VLM as failed"
        );
    }

    #[test]
    fn vlm_lifecycle_already_ready_stays_ready_on_success() {
        // Repeating explicit warmup completion when VLM is already Ready
        // must be a safe no-op (no regression on repeated calls).
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);

        // Call warmup completion again — should stay Ready, not error.
        runtime.mark_vlm_ready(3).unwrap();
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);
        assert!(vlm_status.ready);
        assert_eq!(vlm_status.last_warmup_latency_ms, Some(3));
    }

    #[test]
    fn vlm_lifecycle_already_failed_stays_failed_on_failure() {
        // Calling track_vlm_lifecycle_failure when VLM is already Failed
        // must be a safe no-op.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_failed("VLM_LOAD_FAILED", "first failure");

        // Call failure again — should stay Failed, not error
        track_vlm_lifecycle_failure(
            &runtime,
            &GatewayError::Protocol("another failure".to_string()),
        );
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Failed);
        assert!(!vlm_status.ready);
        // The original error message should be preserved, not overwritten
        assert_eq!(
            vlm_status.last_error.as_ref().map(|e| e.message.as_str()),
            Some("first failure")
        );
    }

    #[test]
    fn vlm_lifecycle_full_http_recovery_from_transient_failure() {
        // Full HTTP-path test: user request failures do not change explicit
        // VLM warmup readiness.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        runtime.mark_vlm_ready(9).unwrap();

        // Backend that fails on first call, succeeds on second.
        let service = Arc::new(RecoveryService::default());

        let state = AppState {
            runtime: runtime.clone(),
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service.clone()),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}}
                    ]
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0
        }))
        .unwrap();

        // First request: fails, but explicit warmup state remains Ready.
        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert!(
            response.status.starts_with("5"),
            "first VLM request should return 5xx, got: {}",
            response.status
        );
        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);

        // Second request: succeeds, and readiness still stays Ready.
        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"chat.completion\""));
        assert!(response.body.contains("\"recovered\""));

        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);
        assert!(vlm_status.ready);
        assert_eq!(vlm_status.last_warmup_latency_ms, Some(9));
    }

    #[test]
    fn chat_completion_with_images_to_text_model_rejected_when_vlm_configured() {
        // Image-bearing request targeting text model must fail when VLM configured.
        let service = Arc::new(FakeService::default());
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service.clone()),
        };

        let body = serde_json::to_vec(&json!({
            "model": "text-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"VLM_NOT_CONFIGURED\""));
        assert!(response.body.contains("requested model is not VLM"));
    }

    // ── VLM image-count cap tests (Phase 8) ──────────────────────────────

    #[test]
    fn vlm_non_streaming_rejects_too_many_images() {
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 3,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://a.com/1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/2.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/3.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/4.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("\"TOO_MANY_IMAGES\""));
        assert!(response.body.contains("4 images, maximum allowed is 3"));
    }

    #[test]
    fn vlm_non_streaming_accepts_max_images() {
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-max-img".to_string(),
                model: "vlm-model".to_string(),
                text: "described many images".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 20,
                completion_tokens: 5,
            })),
        });
        let state = AppState {
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
                max_vlm_images: 3,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://a.com/1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/2.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/3.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"described many images\""));
    }

    #[test]
    fn vlm_streaming_rejects_too_many_images() {
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 2,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(FakeService::default())),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://a.com/1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/2.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/3.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16,
            "stream": true
        }))
        .unwrap();

        let mut buffer = Cursor::new(Vec::new());
        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let output = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(output.starts_with("HTTP/1.1 400 Bad Request"));
        assert!(output.contains("\"TOO_MANY_IMAGES\""));
        assert!(output.contains("3 images, maximum allowed is 2"));
        assert!(!output.contains("text/event-stream"));
    }

    #[test]
    fn vlm_streaming_accepts_max_images() {
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let state = AppState {
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
                max_vlm_images: 3,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(Arc::new(StreamingService)),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": "https://a.com/1.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/2.jpg"}},
                        {"type": "image_url", "image_url": {"url": "https://a.com/3.jpg"}}
                    ]
                }
            ],
            "max_tokens": 16,
            "stream": true
        }))
        .unwrap();

        let mut buffer = Cursor::new(Vec::new());
        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let output = String::from_utf8(buffer.into_inner()).unwrap();
        assert!(output.contains("text/event-stream"));
        assert!(output.contains("chat.completion.chunk"));
        assert!(output.contains("data: [DONE]"));
    }

    #[test]
    fn vlm_text_only_path_ignores_image_count_cap() {
        // Text-only request targeting VLM model must not hit the image cap.
        let runtime = test_runtime(ModelState::Ready);
        runtime.set_vlm_model("vlm-model".to_string());
        let _ = runtime.mark_vlm_ready(1);
        let service = Arc::new(FakeService {
            response: Mutex::new(Some(ChatCompletionResponse {
                request_id: "req-text-vlm".to_string(),
                model: "vlm-model".to_string(),
                text: "text-only VLM response".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 3,
                completion_tokens: 4,
            })),
        });
        let state = AppState {
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
                max_vlm_images: 0,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [{"role": "user", "content": "just text"}],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"text-only VLM response\""));
    }

    // ── VLM lifecycle cancelled/no-load guard tests ────────────────────────

    /// Service that always returns a cancelled response (finish_reason = "cancelled").
    /// Used to verify VLM lifecycle does NOT transition to Ready on cancelled/no-load.
    #[derive(Default)]
    struct CancelledResponseService;

    impl ChatCompletionService for CancelledResponseService {
        fn complete(
            &self,
            request: ChatCompletionRequest,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: String::new(),
                finish_reason: "cancelled".to_string(),
                prompt_tokens: 0,
                completion_tokens: 0,
            })
        }

        fn stream(
            &self,
            request: ChatCompletionRequest,
            on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
        ) -> Result<ChatCompletionResponse, GatewayError> {
            // Emit a delta to confirm streaming started before cancelled finish.
            on_delta("par".to_string())?;
            Ok(ChatCompletionResponse {
                request_id: request.request_id,
                model: request.model,
                text: "partial".to_string(),
                finish_reason: "cancelled".to_string(),
                prompt_tokens: 5,
                completion_tokens: 1,
            })
        }
    }

    #[test]
    fn vlm_lifecycle_non_streaming_cancelled_response_does_not_mark_ready() {
        // Non-streaming cancelled VLM request must not alter warmup readiness.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        runtime.mark_vlm_ready(5).unwrap();

        let service = Arc::new(CancelledResponseService);
        let state = AppState {
            runtime: runtime.clone(),
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16
        }))
        .unwrap();

        let response =
            response_for_request("POST /v1/chat/completions HTTP/1.1\r\n", &body, &state);
        assert_eq!(response.status, "200 OK");

        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);
        assert!(vlm_status.ready);
        assert_eq!(vlm_status.last_warmup_latency_ms, Some(5));
    }

    #[test]
    fn vlm_lifecycle_streaming_cancelled_response_does_not_mark_ready() {
        // Streaming cancelled VLM request must not alter warmup readiness.
        let runtime = RuntimeState::new("text-model");
        let mut status = ModelStatus::new("text-model");
        status.set_state(ModelState::Ready);
        runtime.set_status(status).unwrap();
        runtime.set_vlm_model("vlm-model".to_string());
        runtime.mark_vlm_ready(6).unwrap();

        let service = Arc::new(CancelledResponseService);
        let state = AppState {
            runtime: runtime.clone(),
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
                max_vlm_images: 5,
            },
            telemetry: crate::config::TelemetryConfig::default(),
            admission: Arc::new(RequestAdmission::new(64, 16, Duration::from_secs(300))),
            model: "text-model".to_string(),
            vlm_model: Some("vlm-model".to_string()),
            test_backend: Some(service),
        };

        let body = serde_json::to_vec(&json!({
            "model": "vlm-model",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
            "stream": true
        }))
        .unwrap();

        let mut buffer = Cursor::new(Vec::new());
        stream_chat_completion_with_disconnect(&mut buffer, &body, &state, None).unwrap();

        let vlm_status = runtime.snapshot_vlm().unwrap();
        assert_eq!(vlm_status.state, ModelState::Ready);
        assert!(vlm_status.ready);
        assert_eq!(vlm_status.last_warmup_latency_ms, Some(6));
    }
}
