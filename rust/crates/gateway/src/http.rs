use crate::config::RuntimeConfig;
use crate::errors::GatewayError;
use crate::ipc::WorkerClient;
use crate::openai::{ChatCompletionHttpRequest, ChatCompletionHttpResponse};
use crate::supervisor::RuntimeState;
use mlx_runtime_protocol::{ChatCompletionRequest, ChatCompletionResponse};
use serde_json::json;
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;

/// Service used by the HTTP layer to fulfill completions.
pub trait ChatCompletionService: Send + Sync {
    /// Execute a non-streaming chat completion request.
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError>;
}

impl<T: ChatCompletionService + ?Sized> ChatCompletionService for Arc<T> {
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        (**self).complete(request)
    }
}

impl ChatCompletionService for crate::ipc::WorkerClient {
    fn complete(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        self.complete_chat(request)
    }
}

struct AppState {
    healthy: Arc<AtomicBool>,
    worker_client: Arc<Mutex<Option<Arc<WorkerClient>>>>,
    generation: crate::config::GenerationConfig,
    model: String,
    test_backend: Option<Arc<dyn ChatCompletionService>>,
}

/// Serves the Phase 1 HTTP surface.
pub fn serve(config: RuntimeConfig, runtime: RuntimeState) -> Result<(), GatewayError> {
    let listener = TcpListener::bind(format!("{}:{}", config.server.host, config.server.port))?;

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let state = AppState {
                    healthy: runtime.healthy.clone(),
                    worker_client: runtime.worker_client.clone(),
                    generation: config.generation.clone(),
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
    if request_line.starts_with("GET /health ") {
        return if state.healthy.load(Ordering::SeqCst) {
            HttpResponse {
                status: "200 OK".to_string(),
                content_type: "text/plain; charset=utf-8",
                body: "healthy".to_string(),
            }
        } else {
            HttpResponse {
                status: "503 Service Unavailable".to_string(),
                content_type: "text/plain; charset=utf-8",
                body: "unhealthy".to_string(),
            }
        };
    }

    if request_line.starts_with("POST /v1/chat/completions ") {
        return handle_chat_completion(body, state);
    }

    HttpResponse {
        status: "404 Not Found".to_string(),
        content_type: "text/plain; charset=utf-8",
        body: "not found".to_string(),
    }
}

fn handle_chat_completion(body: &[u8], state: &AppState) -> HttpResponse {
    if !state.healthy.load(Ordering::SeqCst) {
        return json_error_response("503 Service Unavailable", "worker is not ready");
    }

    let request = match serde_json::from_slice::<ChatCompletionHttpRequest>(body) {
        Ok(request) => request,
        Err(err) => {
            return json_error_response("400 Bad Request", &format!("invalid JSON body: {err}"));
        }
    };

    let worker_request = match request.into_worker_request(&state.generation, &state.model) {
        Ok(request) => request,
        Err(message) => return json_error_response("400 Bad Request", &message),
    };

    let backend = if let Some(backend) = &state.test_backend {
        Some(backend.clone())
    } else {
        match state.worker_client.lock() {
            Ok(guard) => guard
                .as_ref()
                .map(|client| client.clone() as Arc<dyn ChatCompletionService>),
            Err(_) => {
                return json_error_response(
                    "500 Internal Server Error",
                    "worker client lock poisoned",
                );
            }
        }
    };

    let Some(backend) = backend else {
        return json_error_response("503 Service Unavailable", "worker is not ready");
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
        Err(GatewayError::WorkerStartup(message)) => {
            json_error_response("503 Service Unavailable", &message)
        }
        Err(GatewayError::Protocol(message)) => {
            json_error_response("500 Internal Server Error", &message)
        }
        Err(GatewayError::Io(err)) => {
            json_error_response("500 Internal Server Error", &err.to_string())
        }
    }
}

fn json_error_response(status: &str, message: &str) -> HttpResponse {
    HttpResponse {
        status: status.to_string(),
        content_type: "application/json",
        body: json!({ "error": { "message": message } }).to_string(),
    }
}

fn write_response(
    stream: &mut TcpStream,
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
    use std::sync::Mutex;

    #[test]
    fn health_endpoint_returns_503_when_worker_is_not_ready() {
        assert_eq!(
            response_for_request(
                "GET /health HTTP/1.1\r\n",
                &[],
                &test_state(false, Arc::new(FakeService::default()))
            )
            .status,
            "503 Service Unavailable"
        );
    }

    #[test]
    fn health_endpoint_returns_200_when_worker_is_ready() {
        assert_eq!(
            response_for_request(
                "GET /health HTTP/1.1\r\n",
                &[],
                &test_state(true, Arc::new(FakeService::default()))
            )
            .status,
            "200 OK"
        );
    }

    #[test]
    fn chat_completion_rejects_streaming_requests() {
        let body = serde_json::to_vec(&json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true
        }))
        .unwrap();

        let response = response_for_request(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            &body,
            &test_state(true, Arc::new(FakeService::default())),
        );

        assert_eq!(response.status, "400 Bad Request");
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
            &test_state(true, service),
        );

        assert_eq!(response.status, "200 OK");
        assert!(response.body.contains("\"chat.completion\""));
        assert!(response.body.contains("\"hello back\""));
    }

    #[test]
    fn chat_completion_rejects_wrong_model() {
        let body = serde_json::to_vec(&json!({
            "model": "wrong-model",
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .unwrap();

        let response = response_for_request(
            "POST /v1/chat/completions HTTP/1.1\r\n",
            &body,
            &test_state(true, Arc::new(FakeService::default())),
        );

        assert_eq!(response.status, "400 Bad Request");
        assert!(response.body.contains("configured model"));
    }

    fn test_state(healthy: bool, backend: Arc<dyn ChatCompletionService>) -> AppState {
        AppState {
            healthy: Arc::new(AtomicBool::new(healthy)),
            worker_client: Arc::new(Mutex::new(None)),
            generation: GenerationConfig {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: 32,
            },
            model: "test-model".to_string(),
            test_backend: Some(backend),
        }
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
    }
}
