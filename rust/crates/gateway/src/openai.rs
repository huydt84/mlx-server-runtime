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
    /// Optional streaming controls.
    pub stream_options: Option<StreamOptions>,
}

/// OpenAI-compatible streaming options.
#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    /// Whether to emit a final usage-only chunk.
    pub include_usage: bool,
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
    /// Accepts either the configured text model or an optional VLM model.
    pub fn into_worker_request(
        self,
        generation: &GenerationConfig,
        limits: &RequestLimits,
        configured_model: &str,
        vlm_model: Option<&str>,
    ) -> Result<WorkerChatCompletionRequest, String> {
        if self.model.trim().is_empty() {
            return Err("model must not be empty".to_string());
        }
        let model_match =
            self.model == configured_model || vlm_model.is_some_and(|vlm| self.model == vlm);
        if !model_match {
            let allowed = if let Some(vlm) = vlm_model {
                format!("'{}' or '{}'", configured_model, vlm)
            } else {
                format!("'{}'", configured_model)
            };
            return Err(format!(
                "requested model '{}' does not match configured model(s): {}",
                self.model, allowed
            ));
        }
        if self.messages.is_empty() {
            return Err("messages must not be empty".to_string());
        }
        if self
            .messages
            .iter()
            .any(|message| !message.content.has_content())
        {
            return Err("message content must not be empty".to_string());
        }
        if self.max_tokens == Some(0) {
            return Err("max_tokens must be positive".to_string());
        }

        Ok(WorkerChatCompletionRequest {
            request_id: format!("req-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)),
            model: self.model.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GenerationConfig, RequestLimits};

    fn default_generation() -> GenerationConfig {
        GenerationConfig {
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 32,
        }
    }

    fn default_limits() -> RequestLimits {
        RequestLimits {
            max_pending_requests: 64,
            max_active_requests: 16,
            max_prompt_tokens: 32_768,
            max_completion_tokens: 4_096,
            max_total_tokens_per_request: 65_536,
            request_timeout_seconds: 300,
            max_vlm_images: 5,
        }
    }

    #[test]
    fn into_worker_request_accepts_text_model() {
        let request = ChatCompletionHttpRequest {
            model: "text-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stream: Some(false),
            stream_options: None,
        };
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            None,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model, "text-model");
    }

    #[test]
    fn into_worker_request_accepts_vlm_model_when_configured() {
        let request = ChatCompletionHttpRequest {
            model: "vlm-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stream: Some(false),
            stream_options: None,
        };
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            Some("vlm-model"),
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().model, "vlm-model");
    }

    #[test]
    fn into_worker_request_rejects_unknown_model() {
        let request = ChatCompletionHttpRequest {
            model: "unknown-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stream: Some(false),
            stream_options: None,
        };
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            Some("vlm-model"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }

    #[test]
    fn into_worker_request_preserves_user_requested_model() {
        let request = ChatCompletionHttpRequest {
            model: "vlm-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "describe this image".into(),
            }],
            max_tokens: Some(64),
            temperature: Some(0.5),
            top_p: Some(0.8),
            stream: Some(true),
            stream_options: None,
        };
        let worker = request
            .into_worker_request(
                &default_generation(),
                &default_limits(),
                "text-model",
                Some("vlm-model"),
            )
            .unwrap();
        assert_eq!(worker.model, "vlm-model");
        assert_eq!(worker.max_tokens, 64);
        assert!(worker.stream);
    }

    #[test]
    fn into_worker_request_rejects_empty_model() {
        let request = ChatCompletionHttpRequest {
            model: "".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stream_options: None,
        };
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("empty"));
    }

    #[test]
    fn into_worker_request_rejects_vlm_without_config() {
        let request = ChatCompletionHttpRequest {
            model: "vlm-model".to_string(),
            messages: vec![ChatMessage {
                role: MessageRole::User,
                content: "hello".into(),
            }],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stream: None,
            stream_options: None,
        };
        // No VLM model configured, so "vlm-model" should be rejected
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }
}
