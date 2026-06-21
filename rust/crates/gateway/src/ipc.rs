use crate::errors::GatewayError;
use crate::telemetry::MetricsRegistry;
use mlx_runtime_protocol::{
    decode_worker_event, encode_gateway_command, ChatCompletionRequest, ChatCompletionResponse,
    GatewayCommand, WorkerEvent,
};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

enum RoutedWorkerEvent {
    Event(Box<WorkerEvent>),
    Failure(String),
}

/// A request/response client for Python worker IPC.
pub struct WorkerClient {
    writer: Mutex<UnixStream>,
    inflight: Arc<Mutex<HashMap<String, mpsc::Sender<RoutedWorkerEvent>>>>,
    closed: Arc<AtomicBool>,
    metrics: Arc<MetricsRegistry>,
}

impl WorkerClient {
    /// Creates client from established worker connection.
    pub fn new(stream: UnixStream, metrics: Arc<MetricsRegistry>) -> Result<Self, GatewayError> {
        let reader_stream = stream.try_clone()?;
        let inflight = Arc::new(Mutex::new(HashMap::new()));
        let closed = Arc::new(AtomicBool::new(false));
        Self::spawn_reader(
            reader_stream,
            Arc::clone(&inflight),
            Arc::clone(&closed),
            Arc::clone(&metrics),
        );

        Ok(Self {
            writer: Mutex::new(stream),
            inflight,
            closed,
            metrics,
        })
    }

    /// Sends non-streaming chat completion request.
    pub fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        self.execute_chat(request, false, &mut |_| Ok(()))
    }

    /// Streams chat completion and invokes callback for each delta.
    pub fn stream_chat(
        &self,
        mut request: ChatCompletionRequest,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        request.stream = true;
        self.execute_chat(request, true, on_delta)
    }

    /// Sends cancellation request for in-flight completion.
    pub fn cancel_chat(&self, request_id: &str) -> Result<(), GatewayError> {
        self.send_command(GatewayCommand::CancelRequest {
            request_id: request_id.to_string(),
        })
    }

    fn execute_chat(
        &self,
        request: ChatCompletionRequest,
        stream: bool,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        let request_id = request.request_id.clone();
        let (sender, receiver) = mpsc::channel();
        self.register_request(&request_id, sender)?;
        self.send_command(GatewayCommand::ChatCompletion { request })?;
        let roundtrip_started = std::time::Instant::now();

        loop {
            match receiver.recv() {
                Ok(RoutedWorkerEvent::Failure(message)) => {
                    self.unregister_request(&request_id);
                    return Err(GatewayError::Protocol(message));
                }
                Ok(RoutedWorkerEvent::Event(event)) => match *event {
                    WorkerEvent::ChatCompletionDelta { delta } => {
                        if !stream {
                            self.unregister_request(&request_id);
                            return Err(GatewayError::Protocol(
                                "received unexpected stream delta".to_string(),
                            ));
                        }
                        if let Err(err) = on_delta(delta.delta) {
                            self.unregister_request(&request_id);
                            return Err(err);
                        }
                    }
                    WorkerEvent::ChatCompletion {
                        response,
                        image_count: _image_count,
                        image_preprocess_latency_ms,
                        prompt_template_latency_ms,
                        prompt_cache_hit,
                        cached_tokens,
                        prompt_cache_bytes,
                        active_batch_cache_bytes,
                        prompt_batch_size,
                        decode_batch_size,
                    } => {
                        if response.request_id != request_id {
                            continue;
                        }
                        self.metrics.record_ipc_roundtrip_latency_ms(
                            roundtrip_started.elapsed().as_millis() as u64,
                        );
                        if let Some(value_ms) = image_preprocess_latency_ms {
                            self.metrics
                                .record_vlm_image_preprocess_latency_ms(value_ms as u64);
                        }
                        if let Some(value_ms) = prompt_template_latency_ms {
                            self.metrics
                                .record_vlm_prompt_template_latency_ms(value_ms as u64);
                        }
                        if let Some(hit) = prompt_cache_hit {
                            self.metrics.record_prompt_cache_hit(hit);
                        }
                        if let Some(tokens) = cached_tokens {
                            self.metrics.add_prompt_cache_cached_tokens(tokens as u64);
                        }
                        if let Some(bytes) = prompt_cache_bytes {
                            self.metrics.set_prompt_cache_bytes(bytes);
                        }
                        if let Some(bytes) = active_batch_cache_bytes {
                            self.metrics.set_active_batch_cache_bytes(bytes);
                        }
                        if let Some(size) = prompt_batch_size {
                            self.metrics.set_prompt_batch_size(size as u64);
                        }
                        if let Some(size) = decode_batch_size {
                            self.metrics.set_decode_batch_size(size as u64);
                        }
                        self.unregister_request(&request_id);
                        return Ok(response);
                    }
                    WorkerEvent::Error {
                        code,
                        request_id: rid,
                        message,
                    } => {
                        if rid != request_id {
                            continue;
                        }
                        self.unregister_request(&request_id);
                        if code == "INVALID_REQUEST" {
                            return Err(GatewayError::InvalidRequest(message));
                        }
                        return Err(GatewayError::Protocol(message));
                    }
                },
                Err(_) => {
                    self.unregister_request(&request_id);
                    return Err(GatewayError::Protocol(
                        "worker closed the inference socket".to_string(),
                    ));
                }
            }
        }
    }

    fn send_command(&self, command: GatewayCommand) -> Result<(), GatewayError> {
        let encoded = encode_gateway_command(&command)
            .map_err(|err| GatewayError::Protocol(format!("encode command failed: {err}")))?;
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| GatewayError::Protocol("worker writer lock poisoned".to_string()))?;
        writeln!(writer, "{encoded}")?;
        writer.flush()?;
        self.metrics.increment_ipc_messages_sent_total();
        Ok(())
    }

    fn register_request(
        &self,
        request_id: &str,
        sender: mpsc::Sender<RoutedWorkerEvent>,
    ) -> Result<(), GatewayError> {
        if self.closed.load(Ordering::Relaxed) {
            return Err(GatewayError::Protocol(
                "worker closed the inference socket".to_string(),
            ));
        }

        let mut guard = self
            .inflight
            .lock()
            .map_err(|_| GatewayError::Protocol("worker inflight lock poisoned".to_string()))?;
        guard.insert(request_id.to_string(), sender);
        Ok(())
    }

    fn unregister_request(&self, request_id: &str) {
        if let Ok(mut guard) = self.inflight.lock() {
            guard.remove(request_id);
        }
    }

    fn spawn_reader(
        reader_stream: UnixStream,
        inflight: Arc<Mutex<HashMap<String, mpsc::Sender<RoutedWorkerEvent>>>>,
        closed: Arc<AtomicBool>,
        metrics: Arc<MetricsRegistry>,
    ) {
        thread::spawn(move || {
            let mut reader = BufReader::new(reader_stream);
            loop {
                let mut line = String::new();
                let bytes = match reader.read_line(&mut line) {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        Self::fail_all(&inflight, format!("worker read failed: {err}"));
                        closed.store(true, Ordering::Relaxed);
                        return;
                    }
                };

                if bytes == 0 {
                    Self::fail_all(&inflight, "worker closed the inference socket".to_string());
                    closed.store(true, Ordering::Relaxed);
                    return;
                }

                metrics.increment_ipc_messages_received_total();

                let event = match decode_worker_event(&line) {
                    Ok(event) => event,
                    Err(err) => {
                        Self::fail_all(&inflight, format!("decode worker event failed: {err}"));
                        closed.store(true, Ordering::Relaxed);
                        return;
                    }
                };

                let request_id = match &event {
                    WorkerEvent::ChatCompletionDelta { delta } => delta.request_id.clone(),
                    WorkerEvent::ChatCompletion { response, .. } => response.request_id.clone(),
                    WorkerEvent::Error { request_id, .. } => request_id.clone(),
                };

                let sender = match inflight.lock() {
                    Ok(guard) => guard.get(&request_id).cloned(),
                    Err(_) => None,
                };

                if let Some(sender) = sender {
                    let _ = sender.send(RoutedWorkerEvent::Event(Box::new(event)));
                }
            }
        });
    }

    fn fail_all(
        inflight: &Arc<Mutex<HashMap<String, mpsc::Sender<RoutedWorkerEvent>>>>,
        message: String,
    ) {
        if let Ok(mut guard) = inflight.lock() {
            for (_, sender) in guard.drain() {
                let _ = sender.send(RoutedWorkerEvent::Failure(message.clone()));
            }
        }
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Relaxed);
        if let Ok(writer) = self.writer.lock() {
            let _ = writer.shutdown(std::net::Shutdown::Both);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_runtime_protocol::{
        decode_gateway_command, encode_worker_event, ChatCompletionRequest, ChatCompletionResponse,
        ChatMessage, GatewayCommand, MessageRole, WorkerEvent,
    };
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;

    fn request(request_id: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            request_id: request_id.to_string(),
            model: "test-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: 4,
            temperature: 0.0,
            top_p: 1.0,
            max_prompt_tokens: 16,
            max_completion_tokens: 16,
            max_total_tokens_per_request: 32,
            stream: false,
        }
    }

    #[test]
    fn routes_responses_by_request_id_under_concurrent_load() {
        let (client, server) = UnixStream::pair().unwrap();
        let metrics = Arc::new(MetricsRegistry::new());
        let worker = Arc::new(WorkerClient::new(client, metrics).unwrap());

        let server_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(server.try_clone().unwrap());
            let mut writer = server;
            let mut request_ids = Vec::new();

            while request_ids.len() < 2 {
                let mut line = String::new();
                let bytes = reader.read_line(&mut line).unwrap();
                assert!(bytes > 0);
                let command = decode_gateway_command(&line).unwrap();
                if let GatewayCommand::ChatCompletion { request } = command {
                    request_ids.push(request.request_id);
                }
            }

            let req_1 = request_ids
                .iter()
                .find(|request_id| request_id.as_str() == "req-1")
                .cloned()
                .unwrap();
            let req_2 = request_ids
                .iter()
                .find(|request_id| request_id.as_str() == "req-2")
                .cloned()
                .unwrap();

            let events = [
                WorkerEvent::ChatCompletion {
                    response: ChatCompletionResponse {
                        request_id: req_2,
                        model: "test-model".to_string(),
                        text: "second".to_string(),
                        finish_reason: "stop".to_string(),
                        prompt_tokens: 4,
                        completion_tokens: 1,
                        ..Default::default()
                    },
                    image_count: None,
                    image_preprocess_latency_ms: None,
                    prompt_template_latency_ms: None,
                    prompt_cache_hit: None,
                    cached_tokens: None,
                    prompt_cache_bytes: None,
                    active_batch_cache_bytes: None,
                    prompt_batch_size: None,
                    decode_batch_size: None,
                },
                WorkerEvent::ChatCompletion {
                    response: ChatCompletionResponse {
                        request_id: req_1,
                        model: "test-model".to_string(),
                        text: "first".to_string(),
                        finish_reason: "stop".to_string(),
                        prompt_tokens: 4,
                        completion_tokens: 1,
                        ..Default::default()
                    },
                    image_count: None,
                    image_preprocess_latency_ms: None,
                    prompt_template_latency_ms: None,
                    prompt_cache_hit: None,
                    cached_tokens: None,
                    prompt_cache_bytes: None,
                    active_batch_cache_bytes: None,
                    prompt_batch_size: None,
                    decode_batch_size: None,
                },
            ];

            for event in events {
                let encoded = encode_worker_event(&event).unwrap();
                writeln!(writer, "{encoded}").unwrap();
                writer.flush().unwrap();
            }
        });

        let worker_a = Arc::clone(&worker);
        let first = std::thread::spawn(move || worker_a.complete_chat(request("req-1")));
        let worker_b = Arc::clone(&worker);
        let second = std::thread::spawn(move || worker_b.complete_chat(request("req-2")));

        let first = first.join().unwrap().unwrap();
        let second = second.join().unwrap().unwrap();
        server_thread.join().unwrap();

        assert_eq!(first.request_id, "req-1");
        assert_eq!(first.text, "first");
        assert_eq!(second.request_id, "req-2");
        assert_eq!(second.text, "second");
    }

    #[test]
    fn disconnect_fails_all_inflight_requests() {
        let (client, server) = UnixStream::pair().unwrap();
        let metrics = Arc::new(MetricsRegistry::new());
        let worker = Arc::new(WorkerClient::new(client, metrics).unwrap());

        let server_thread = std::thread::spawn(move || {
            let mut reader = BufReader::new(server);
            let mut line = String::new();
            let bytes = reader.read_line(&mut line).unwrap();
            assert!(bytes > 0);
        });

        let result = worker.complete_chat(request("req-1"));
        server_thread.join().unwrap();

        assert!(
            matches!(result, Err(GatewayError::Protocol(message)) if message.contains("closed the inference socket"))
        );
    }
}
