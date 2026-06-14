use super::response::{WorkerError, WorkerReady};
use core::fmt;

/// Messages sent from the worker to the gateway during bootstrap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerMessage {
    /// The worker is ready to serve requests.
    Ready(WorkerReady),
    /// The worker failed to start.
    Error(WorkerError),
}

/// Encodes a worker message as a single line.
pub fn encode_worker_message(message: &WorkerMessage) -> String {
    match message {
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
