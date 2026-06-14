use crate::config::{GenerationConfig, RequestLimits};
use mlx_runtime_protocol::{
    ChatCompletionRequest as WorkerChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    MessageRole,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Incoming `/v1/chat/completions` request body.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionHttpRequest {
    /// Requested model.
    pub model: String,
    /// Chat message history.
    pub messages: Vec<ChatMessage>,
    /// Optional override for maximum generated tokens.
    pub max_tokens: Option<u32>,
    /// Optional override for sampling temperature.
    pub temperature: Option<f32>,
    /// Optional override for top-p.
    pub top_p: Option<f32>,
    /// Whether the client requested streaming.
    pub stream: Option<bool>,
}

/// Outgoing non-streaming OpenAI-compatible response.
#[derive(Debug, Serialize)]
pub struct ChatCompletionHttpResponse {
    /// Response id.
    pub id: String,
    /// Object type.
    pub object: &'static str,
    /// Unix creation timestamp.
    pub created: u64,
    /// Model used for completion.
    pub model: String,
    /// Response choices.
    pub choices: Vec<ChatCompletionChoice>,
    /// Token usage stats.
    pub usage: Usage,
}

/// A single response choice.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    /// Choice index.
    pub index: u32,
    /// Assistant message.
    pub message: AssistantMessage,
    /// Stop reason.
    pub finish_reason: String,
}

/// Assistant response message.
#[derive(Debug, Serialize)]
pub struct AssistantMessage {
    /// Assistant role.
    pub role: MessageRole,
    /// Final text content.
    pub content: String,
}

/// Token usage payload.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Completion tokens.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

impl ChatCompletionHttpRequest {
    /// Converts an incoming HTTP request into a worker request using gateway defaults.
    pub fn into_worker_request(
        self,
        generation: &GenerationConfig,
        limits: &RequestLimits,
        configured_model: &str,
    ) -> Result<WorkerChatCompletionRequest, String> {
        if self.model.trim().is_empty() {
            return Err("model must not be empty".to_string());
        }
        if self.model != configured_model {
            return Err(format!(
                "requested model '{}' does not match configured model '{}'",
                self.model, configured_model
            ));
        }
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        if self
            .messages
            .iter()
            .any(|message| message.content.trim().is_empty())
        {
            return Err("message content must not be empty".to_string());
        }
        if self.max_tokens == Some(0) {
            return Err("max_tokens must be positive".to_string());
        }

        Ok(WorkerChatCompletionRequest {
            request_id: format!("req-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)),
            model: configured_model.to_string(),
            messages: self.messages,
            max_tokens: self.max_tokens.unwrap_or(generation.max_tokens),
            temperature: self.temperature.unwrap_or(generation.temperature),
            top_p: self.top_p.unwrap_or(generation.top_p),
            max_prompt_tokens: u32::try_from(limits.max_prompt_tokens)
                .map_err(|_| "max_prompt_tokens out of range".to_string())?,
            max_completion_tokens: u32::try_from(limits.max_completion_tokens)
                .map_err(|_| "max_completion_tokens out of range".to_string())?,
            max_total_tokens_per_request: u32::try_from(limits.max_total_tokens_per_request)
                .map_err(|_| "max_total_tokens_per_request out of range".to_string())?,
            stream: self.stream.unwrap_or(false),
        })
    }
}

impl From<ChatCompletionResponse> for ChatCompletionHttpResponse {
    fn from(value: ChatCompletionResponse) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let total_tokens = value.prompt_tokens.saturating_add(value.completion_tokens);

        Self {
            id: format!("chatcmpl-{}", value.request_id),
            object: "chat.completion",
            created,
            model: value.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: MessageRole::Assistant,
                    content: value.text,
                },
                finish_reason: value.finish_reason,
            }],
            usage: Usage {
                prompt_tokens: value.prompt_tokens,
                completion_tokens: value.completion_tokens,
                total_tokens,
            },
        }
    }
}
