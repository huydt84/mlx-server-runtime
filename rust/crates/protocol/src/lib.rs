//! Shared bootstrap protocol types used by Rust and Python.

mod events;
mod request;
mod response;
mod status;

pub use events::{
    decode_gateway_command, decode_worker_event, decode_worker_message, encode_gateway_command,
    encode_worker_event, encode_worker_message, GatewayCommand, WorkerEvent, WorkerMessage,
};
pub use request::{ChatCompletionRequest, ChatMessage, HealthCheck, MessageRole};
pub use response::{ChatCompletionResponse, WorkerError, WorkerReady};
pub use status::{ModelError, ModelLoadProgress, ModelState, ModelStatus, ModelSummary};
