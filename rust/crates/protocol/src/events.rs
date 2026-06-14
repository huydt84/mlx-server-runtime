use super::request::ChatCompletionRequest;
use super::response::{ChatCompletionResponse, WorkerError, WorkerReady};
use super::status::ModelStatus;
use core::fmt;
use serde::{Deserialize, Serialize};

/// Messages sent from the worker to the gateway during bootstrap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerMessage {
    /// The worker reported a lifecycle update.
    Status(Box<ModelStatus>),
    /// The worker is ready to serve requests.
    Ready(WorkerReady),
    /// The worker failed to start.
    Error(WorkerError),
}

/// Encodes a worker message as a single line.
pub fn encode_worker_message(message: &WorkerMessage) -> String {
    match message {
        WorkerMessage::Status(status) => match serde_json::to_string(status) {
            Ok(payload) => format!("STATUS\t{payload}"),
            Err(_) => "STATUS\t{}".to_string(),
        },
        WorkerMessage::Ready(_) => "READY".to_string(),
        WorkerMessage::Error(error) => format!("ERROR\t{}", error.message.replace('\n', " ")),
    }
}

/// Decodes a worker message from a single line.
pub fn decode_worker_message(line: &str) -> Option<WorkerMessage> {
    let trimmed = line.trim();
    if trimmed == "READY" {
        return Some(WorkerMessage::Ready(WorkerReady));
    }

    let (prefix, payload) = trimmed.split_once('\t')?;
    match prefix {
        "STATUS" => serde_json::from_str::<ModelStatus>(payload)
            .ok()
            .map(|status| WorkerMessage::Status(Box::new(status))),
        "ERROR" => Some(WorkerMessage::Error(WorkerError {
            message: payload.to_string(),
        })),
        _ => None,
    }
}

impl fmt::Display for WorkerMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&encode_worker_message(self))
    }
}

/// Commands sent from the gateway to the worker after bootstrap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GatewayCommand {
    /// A non-streaming chat completion request.
    ChatCompletion { request: ChatCompletionRequest },
}

/// Events sent from the worker to the gateway after bootstrap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerEvent {
    /// A non-streaming chat completion result.
    ChatCompletion { response: ChatCompletionResponse },
    /// A worker-side request failure.
    Error {
        /// The request id associated with the error.
        request_id: String,
        /// Human-readable failure reason.
        message: String,
    },
}

/// Encodes a gateway command as JSON.
pub fn encode_gateway_command(command: &GatewayCommand) -> Result<String, serde_json::Error> {
    serde_json::to_string(command)
}

/// Decodes a gateway command from JSON.
pub fn decode_gateway_command(line: &str) -> Result<GatewayCommand, serde_json::Error> {
    serde_json::from_str(line)
}

/// Encodes a worker event as JSON.
pub fn encode_worker_event(event: &WorkerEvent) -> Result<String, serde_json::Error> {
    serde_json::to_string(event)
}

/// Decodes a worker event from JSON.
pub fn decode_worker_event(line: &str) -> Result<WorkerEvent, serde_json::Error> {
    serde_json::from_str(line)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{ChatMessage, MessageRole};
    use crate::response::ChatCompletionResponse;

    #[test]
    fn encode_and_decode_ready_round_trip() {
        let message = WorkerMessage::Ready(WorkerReady);
        let encoded = encode_worker_message(&message);
        assert_eq!(encoded, "READY");
        assert_eq!(decode_worker_message(&encoded), Some(message));
    }

    #[test]
    fn encode_and_decode_error_normalizes_newlines() {
        let message = WorkerMessage::Error(WorkerError {
            message: "boom\nmore".to_string(),
        });
        let encoded = encode_worker_message(&message);
        assert_eq!(encoded, "ERROR\tboom more");
        assert_eq!(
            decode_worker_message(&encoded),
            Some(WorkerMessage::Error(WorkerError {
                message: "boom more".to_string(),
            }))
        );
    }

    #[test]
    fn encode_and_decode_status_round_trip() {
        let status = ModelStatus::new("test-model");
        let message = WorkerMessage::Status(Box::new(status.clone()));
        let encoded = encode_worker_message(&message);
        assert!(encoded.starts_with("STATUS\t"));
        assert_eq!(decode_worker_message(&encoded), Some(message));
    }

    #[test]
    fn gateway_command_round_trip() {
        let command = GatewayCommand::ChatCompletion {
            request: ChatCompletionRequest {
                request_id: "req-1".to_string(),
                model: "test-model".to_string(),
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: "hello".to_string(),
                }],
                max_tokens: 16,
                temperature: 0.2,
                top_p: 0.9,
            },
        };

        let encoded = encode_gateway_command(&command).unwrap();
        let decoded = decode_gateway_command(&encoded).unwrap();
        assert_eq!(decoded, command);
    }

    #[test]
    fn worker_event_round_trip() {
        let event = WorkerEvent::ChatCompletion {
            response: ChatCompletionResponse {
                request_id: "req-1".to_string(),
                model: "test-model".to_string(),
                text: "hello back".to_string(),
                finish_reason: "stop".to_string(),
                prompt_tokens: 12,
                completion_tokens: 3,
            },
        };

        let encoded = encode_worker_event(&event).unwrap();
        let decoded = decode_worker_event(&encoded).unwrap();
        assert_eq!(decoded, event);
    }
}
