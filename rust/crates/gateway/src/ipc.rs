use crate::errors::GatewayError;
use mlx_runtime_protocol::{
    decode_worker_event, encode_gateway_command, ChatCompletionRequest, ChatCompletionResponse,
    GatewayCommand, WorkerEvent,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::sync::Mutex;

struct WorkerConnection {
    writer: UnixStream,
    reader: BufReader<UnixStream>,
}

/// A serialized request/response client for the Python worker.
pub struct WorkerClient {
    inner: Mutex<WorkerConnection>,
}

impl WorkerClient {
    /// Creates a client from an established worker connection.
    pub fn new(stream: UnixStream) -> Result<Self, GatewayError> {
        let reader_stream = stream.try_clone()?;
        Ok(Self {
            inner: Mutex::new(WorkerConnection {
                writer: stream,
                reader: BufReader::new(reader_stream),
            }),
        })
    }

    /// Sends a non-streaming chat completion request and waits for one final response.
    pub fn complete_chat(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, GatewayError> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| GatewayError::Protocol("worker client lock poisoned".to_string()))?;

        let command = GatewayCommand::ChatCompletion { request };
        let encoded = encode_gateway_command(&command)
            .map_err(|err| GatewayError::Protocol(format!("encode command failed: {err}")))?;
        writeln!(guard.writer, "{encoded}")?;
        guard.writer.flush()?;

        let mut line = String::new();
        let bytes = guard.reader.read_line(&mut line)?;
        if bytes == 0 {
            return Err(GatewayError::Protocol(
                "worker closed the inference socket".to_string(),
            ));
        }

        match decode_worker_event(&line)
            .map_err(|err| GatewayError::Protocol(format!("decode worker event failed: {err}")))?
        {
            WorkerEvent::ChatCompletion { response } => Ok(response),
            WorkerEvent::Error {
                request_id: _,
                message,
            } => Err(GatewayError::Protocol(message)),
        }
    }
}
