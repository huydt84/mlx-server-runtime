use crate::errors::GatewayError;
use mlx_runtime_protocol::{
    decode_worker_event, encode_gateway_command, ChatCompletionRequest, ChatCompletionResponse,
    GatewayCommand, WorkerEvent,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::Mutex;

/// A serialized request/response client for the Python worker.
pub struct WorkerClient {
    request_lock: Mutex<()>,
    writer: Mutex<UnixStream>,
    reader: Mutex<BufReader<UnixStream>>,
}

impl WorkerClient {
    /// Creates a client from an established worker connection.
    pub fn new(stream: UnixStream) -> Result<Self, GatewayError> {
        let reader_stream = stream.try_clone()?;
        Ok(Self {
            request_lock: Mutex::new(()),
            writer: Mutex::new(stream),
            reader: Mutex::new(BufReader::new(reader_stream)),
        })
    }

    /// Sends a non-streaming chat completion request and waits for one final response.
    pub fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        self.execute_chat(request, false, &mut |_| Ok(()))
    }

    /// Streams a chat completion and invokes the callback for each delta.
    pub fn stream_chat(
        &self,
        mut request: ChatCompletionRequest,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        request.stream = true;
        self.execute_chat(request, true, on_delta)
    }

    /// Sends a cancellation request for an in-flight completion.
    pub fn cancel_chat(&self, request_id: &str) -> Result<(), GatewayError> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|_| GatewayError::Protocol("worker writer lock poisoned".to_string()))?;
        let command = GatewayCommand::CancelRequest {
            request_id: request_id.to_string(),
        };
        let encoded = encode_gateway_command(&command)
            .map_err(|err| GatewayError::Protocol(format!("encode command failed: {err}")))?;
        writeln!(writer, "{encoded}")?;
        writer.flush()?;
        Ok(())
    }

    fn execute_chat(
        &self,
        request: ChatCompletionRequest,
        stream: bool,
        on_delta: &mut dyn FnMut(String) -> Result<(), GatewayError>,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        let _request_guard = self
            .request_lock
            .lock()
            .map_err(|_| GatewayError::Protocol("worker request lock poisoned".to_string()))?;

        let request_id = request.request_id.clone();
        let command = GatewayCommand::ChatCompletion { request };
        let encoded = encode_gateway_command(&command)
            .map_err(|err| GatewayError::Protocol(format!("encode command failed: {err}")))?;
        {
            let mut writer = self
                .writer
                .lock()
                .map_err(|_| GatewayError::Protocol("worker writer lock poisoned".to_string()))?;
            writeln!(writer, "{encoded}")?;
            writer.flush()?;
        }

        let mut stale_count = 0u32;
        const MAX_STALE_EVENTS: u32 = 10000;

        loop {
            let mut line = String::new();
            let bytes = {
                let mut reader = self.reader.lock().map_err(|_| {
                    GatewayError::Protocol("worker reader lock poisoned".to_string())
                })?;
                reader.read_line(&mut line)?
            };
            if bytes == 0 {
                return Err(GatewayError::Protocol(
                    "worker closed the inference socket".to_string(),
                ));
            }

            match decode_worker_event(&line).map_err(|err| {
                GatewayError::Protocol(format!("decode worker event failed: {err}"))
            })? {
                WorkerEvent::ChatCompletionDelta { delta } => {
                    if delta.request_id != request_id {
                        stale_count += 1;
                        if stale_count > MAX_STALE_EVENTS {
                            return Err(GatewayError::Protocol(
                                "exceeded stale event limit from worker".to_string(),
                            ));
                        }
                        continue;
                    }
                    if !stream {
                        return Err(GatewayError::Protocol(
                            "received unexpected stream delta".to_string(),
                        ));
                    }
                    on_delta(delta.delta)?;
                }
                WorkerEvent::ChatCompletion { response } => {
                    if response.request_id != request_id {
                        stale_count += 1;
                        if stale_count > MAX_STALE_EVENTS {
                            return Err(GatewayError::Protocol(
                                "exceeded stale event limit from worker".to_string(),
                            ));
                        }
                        continue;
                    }
                    return Ok(response);
                }
                WorkerEvent::Error {
                    code,
                    request_id: rid,
                    message,
                } => {
                    if rid != request_id {
                        stale_count += 1;
                        if stale_count > MAX_STALE_EVENTS {
                            return Err(GatewayError::Protocol(
                                "exceeded stale event limit from worker".to_string(),
                            ));
                        }
                        continue;
                    }
                    if code == "INVALID_REQUEST" {
                        return Err(GatewayError::InvalidRequest(message));
                    }
                    return Err(GatewayError::Protocol(message));
                }
            }
        }
    }
}
