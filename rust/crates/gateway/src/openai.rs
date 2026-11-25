use crate::config::{GenerationConfig, RequestLimits};
use mlx_runtime_protocol::{
    ChatCompletionRequest as WorkerChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    MessageRole, ToolCall,
};
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static REQUEST_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Incoming `/v1/chat/completions` request body.
#[derive(Debug, Default, Deserialize)]
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
    /// Optional stop sequence or sequences.
    pub stop: Option<StopSequences>,
    /// Whether the client requested streaming.
    pub stream: Option<bool>,
    /// Optional streaming controls.
    pub stream_options: Option<StreamOptions>,
    /// Optional tool definitions, kept as the text the client sent.
    pub tools: Option<Box<RawValue>>,
    /// Optional control over tool use.
    pub tool_choice: Option<Value>,
    /// Whether one reply may hold several tool calls.
    pub parallel_tool_calls: Option<bool>,
    /// Deprecated OpenAI field that `tools` replaces; always rejected.
    pub functions: Option<Value>,
    /// Deprecated OpenAI field that `tool_choice` replaces; always rejected.
    pub function_call: Option<Value>,
}

/// OpenAI-compatible streaming options.
#[derive(Debug, Deserialize)]
pub struct StreamOptions {
    /// Whether to emit a final usage-only chunk.
    pub include_usage: bool,
}

/// OpenAI-compatible stop sequence input.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    Single(String),
    Multiple(Vec<String>),
}

impl StopSequences {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(value) => vec![value],
            Self::Multiple(values) => values,
        }
    }
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
    /// Whether prompt cache reused prompt tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_hit: Option<bool>,
    /// Cached prompt token count reused for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Prompt cache bytes referenced by this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_cache_bytes: Option<u64>,
    /// Active batch cache bytes observed for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_batch_cache_bytes: Option<u64>,
    /// Prompt batch size observed for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_batch_size: Option<u32>,
    /// Decode batch size observed for this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_batch_size: Option<u32>,
    /// Number of source images in a VLM request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_count: Option<u32>,
    /// Image preprocessing latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_preprocess_latency_ms: Option<u32>,
    /// Prompt-template latency in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_template_latency_ms: Option<u32>,
    /// Whether the VLM vision-feature cache was reused.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_feature_cache_hit: Option<bool>,
    /// Vision-feature cache bytes referenced by this request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_feature_cache_bytes: Option<u64>,
    /// Vision-feature cache entry count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_feature_cache_entries: Option<u32>,
    /// Vision-feature cache evictions count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_feature_cache_evictions: Option<u32>,
    /// Generator peak memory in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_memory_bytes: Option<u64>,
    /// Source image width.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_width: Option<u32>,
    /// Source image height.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_height: Option<u32>,
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
    /// Final text content, or null when the reply only calls tools.
    pub content: Option<String>,
    /// Thinking text, split from the content when the request sends tools.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Tool calls made in this reply.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
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
    /// Optional prompt-cache token details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokenDetails>,
}

/// Prompt token usage details.
#[derive(Debug, Serialize)]
pub struct PromptTokenDetails {
    /// Cached prompt token count reused for this request.
    pub cached_tokens: u32,
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
        let messages = self
            .messages
            .into_iter()
            .map(validated_message)
            .collect::<Result<Vec<_>, _>>()?;
        if self.functions.is_some() || self.function_call.is_some() {
            return Err(
                "functions and function_call are not supported; use tools and tool_choice"
                    .to_string(),
            );
        }
        let tools_enabled = tool_choice_enables_tools(self.tool_choice.as_ref())?;
        let tools = match self.tools.as_deref() {
            Some(tools) => validated_tools(tools)?.filter(|_| tools_enabled),
            None => None,
        };
        if self.max_tokens == Some(0) {
            return Err("max_tokens must be positive".to_string());
        }
        let stop = self.stop.map(StopSequences::into_vec).unwrap_or_default();
        if stop.iter().any(|value| value.is_empty()) {
            return Err("stop sequences must not be empty".to_string());
        }

        Ok(WorkerChatCompletionRequest {
            request_id: format!("req-{}", REQUEST_COUNTER.fetch_add(1, Ordering::Relaxed)),
            model: self.model.clone(),
            messages,
            max_tokens: self.max_tokens.unwrap_or(generation.max_tokens),
            temperature: self.temperature.unwrap_or(generation.temperature),
            top_p: self.top_p.unwrap_or(generation.top_p),
            max_prompt_tokens: u32::try_from(limits.max_prompt_tokens)
                .map_err(|_| "max_prompt_tokens out of range".to_string())?,
            max_completion_tokens: u32::try_from(limits.max_completion_tokens)
                .map_err(|_| "max_completion_tokens out of range".to_string())?,
            max_total_tokens_per_request: u32::try_from(limits.max_total_tokens_per_request)
                .map_err(|_| "max_total_tokens_per_request out of range".to_string())?,
            stop,
            stream: self.stream.unwrap_or(false),
            parallel_tool_calls: tools
                .is_some()
                .then(|| self.parallel_tool_calls.unwrap_or(true)),
            tools,
        })
    }
}

/// Checks one history message and drops fields its role does not use.
fn validated_message(mut message: ChatMessage) -> Result<ChatMessage, String> {
    let is_assistant = message.role == MessageRole::Assistant;
    let is_tool = message.role == MessageRole::Tool;

    if !is_assistant {
        if !message.tool_calls.is_empty() {
            return Err("tool_calls are allowed only on assistant messages".to_string());
        }
        message.reasoning_content = None;
    }
    for call in &message.tool_calls {
        if call.id.is_empty() {
            return Err("tool call id must not be empty".to_string());
        }
        if call.function.name.trim().is_empty() {
            return Err("tool call function name must not be empty".to_string());
        }
        if serde_json::from_str::<Map<String, Value>>(&call.function.arguments).is_err() {
            return Err(format!(
                "arguments of tool call '{}' must be JSON text of an object",
                call.id
            ));
        }
    }

    if is_tool {
        if message.tool_call_id.as_deref().unwrap_or("").is_empty() {
            return Err("tool messages require a non-empty tool_call_id".to_string());
        }
    } else {
        message.tool_call_id = None;
        message.name = None;
    }

    let may_be_empty = is_tool
        || (is_assistant
            && (!message.tool_calls.is_empty() || message.reasoning_content.is_some()));
    if !may_be_empty && !message.content.has_content() {
        return Err("message content must not be empty".to_string());
    }
    Ok(message)
}

/// Checks tool definitions and returns their JSON text, or `None` for an empty list.
fn validated_tools(tools: &RawValue) -> Result<Option<String>, String> {
    let definitions = serde_json::from_str::<Vec<Value>>(tools.get())
        .map_err(|_| "tools must be an array".to_string())?;
    for definition in &definitions {
        if definition.get("type").and_then(Value::as_str) != Some("function") {
            return Err("each tool must have type \"function\"".to_string());
        }
        let function = definition
            .get("function")
            .and_then(Value::as_object)
            .ok_or("each tool must have a function object")?;
        if function
            .get("name")
            .and_then(Value::as_str)
            .is_none_or(|name| name.trim().is_empty())
        {
            return Err("each tool function must have a non-empty name".to_string());
        }
        if function
            .get("parameters")
            .is_some_and(|parameters| !parameters.is_object())
        {
            return Err("tool function parameters must be an object".to_string());
        }
    }
    Ok((!definitions.is_empty()).then(|| tools.get().to_string()))
}

/// Returns whether the model may see the tools, or an error for choices that
/// would need constrained decoding.
fn tool_choice_enables_tools(choice: Option<&Value>) -> Result<bool, String> {
    match choice {
        None => Ok(true),
        Some(Value::String(choice)) if choice == "auto" => Ok(true),
        Some(Value::String(choice)) if choice == "none" => Ok(false),
        Some(Value::String(choice)) if choice == "required" => {
            Err("tool_choice \"required\" is not supported; use \"auto\" or \"none\"".to_string())
        }
        Some(Value::Object(_)) => Err(
            "tool_choice that names a function is not supported; use \"auto\" or \"none\""
                .to_string(),
        ),
        Some(_) => Err("tool_choice must be \"auto\" or \"none\"".to_string()),
    }
}

impl From<ChatCompletionResponse> for ChatCompletionHttpResponse {
    fn from(value: ChatCompletionResponse) -> Self {
        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let total_tokens = value.prompt_tokens.saturating_add(value.completion_tokens);
        let only_calls_tools = value.text.is_empty() && !value.tool_calls.is_empty();

        Self {
            id: format!("chatcmpl-{}", value.request_id),
            object: "chat.completion",
            created,
            model: value.model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: AssistantMessage {
                    role: MessageRole::Assistant,
                    content: (!only_calls_tools).then_some(value.text),
                    reasoning_content: value.reasoning,
                    tool_calls: value.tool_calls,
                },
                finish_reason: value.finish_reason,
            }],
            usage: Usage {
                prompt_tokens: value.prompt_tokens,
                completion_tokens: value.completion_tokens,
                total_tokens,
                prompt_tokens_details: Some(PromptTokenDetails {
                    cached_tokens: value.cached_tokens.unwrap_or(0),
                }),
            },
            prompt_cache_hit: value.prompt_cache_hit,
            cached_tokens: value.cached_tokens,
            prompt_cache_bytes: value.prompt_cache_bytes,
            active_batch_cache_bytes: value.active_batch_cache_bytes,
            prompt_batch_size: value.prompt_batch_size,
            decode_batch_size: value.decode_batch_size,
            image_count: value.image_count,
            image_preprocess_latency_ms: value.image_preprocess_latency_ms,
            prompt_template_latency_ms: value.prompt_template_latency_ms,
            vision_feature_cache_hit: value.vision_feature_cache_hit,
            vision_feature_cache_bytes: value.vision_feature_cache_bytes,
            vision_feature_cache_entries: value.vision_feature_cache_entries,
            vision_feature_cache_evictions: value.vision_feature_cache_evictions,
            peak_memory_bytes: value.peak_memory_bytes,
            image_width: value.image_width,
            image_height: value.image_height,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{GenerationConfig, RequestLimits};
    use serde_json::json;

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
            max_audio_bytes: crate::config::DEFAULT_MAX_AUDIO_BYTES,
        }
    }

    #[test]
    fn into_worker_request_accepts_text_model() {
        let request = ChatCompletionHttpRequest {
            model: "text-model".to_string(),
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            ..Default::default()
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
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            ..Default::default()
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
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: Some(16),
            temperature: None,
            top_p: None,
            stop: None,
            stream: Some(false),
            stream_options: None,
            ..Default::default()
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
            messages: vec![ChatMessage::new(MessageRole::User, "describe this image")],
            max_tokens: Some(64),
            temperature: Some(0.5),
            top_p: Some(0.8),
            stop: None,
            stream: Some(true),
            stream_options: None,
            ..Default::default()
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
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            stream_options: None,
            ..Default::default()
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
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop: None,
            stream: None,
            stream_options: None,
            ..Default::default()
        };
        let result = request.into_worker_request(
            &default_generation(),
            &default_limits(),
            "text-model",
            None,
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("does not match"));
    }

    #[test]
    fn into_worker_request_normalizes_stop_sequences() {
        let request = ChatCompletionHttpRequest {
            model: "text-model".to_string(),
            messages: vec![ChatMessage::new(MessageRole::User, "hello")],
            max_tokens: Some(16),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stop: Some(StopSequences::Single("END".to_string())),
            stream: Some(false),
            stream_options: None,
            ..Default::default()
        };

        let worker = request
            .into_worker_request(&default_generation(), &default_limits(), "text-model", None)
            .unwrap();

        assert_eq!(worker.stop, vec!["END".to_string()]);
    }

    fn parse(body: Value) -> Result<WorkerChatCompletionRequest, String> {
        serde_json::from_str::<ChatCompletionHttpRequest>(&body.to_string())
            .map_err(|err| err.to_string())?
            .into_worker_request(&default_generation(), &default_limits(), "text-model", None)
    }

    fn with_messages(messages: Value) -> Result<WorkerChatCompletionRequest, String> {
        parse(json!({"model": "text-model", "messages": messages}))
    }

    fn with_fields(fields: Value) -> Result<WorkerChatCompletionRequest, String> {
        let mut body = json!({
            "model": "text-model",
            "messages": [{"role": "user", "content": "hello"}],
        });
        for (key, value) in fields.as_object().unwrap() {
            body[key] = value.clone();
        }
        parse(body)
    }

    fn weather_tool() -> Value {
        json!({"type": "function", "function": {"name": "weather"}})
    }

    #[test]
    fn tools_keep_the_text_the_client_sent() {
        let tools = r#"[{"type": "function", "function": {"name": "weather", "description": "d", "parameters": {"type": "object", "properties": {"b": {}, "a": {}}}}}]"#;
        let body = format!(
            r#"{{"model":"text-model","messages":[{{"role":"user","content":"hi"}}],"tools":{tools}}}"#
        );

        let request = serde_json::from_str::<ChatCompletionHttpRequest>(&body)
            .unwrap()
            .into_worker_request(&default_generation(), &default_limits(), "text-model", None)
            .unwrap();

        assert_eq!(request.tools.as_deref(), Some(tools));
        assert_eq!(request.parallel_tool_calls, Some(true));
    }

    #[test]
    fn requests_without_usable_tools_send_none() {
        for fields in [
            json!({}),
            json!({"tools": null}),
            json!({"tools": []}),
            json!({"tools": [], "parallel_tool_calls": false}),
            json!({"tools": [weather_tool()], "tool_choice": "none"}),
        ] {
            let request = with_fields(fields.clone()).unwrap();
            assert_eq!(request.tools, None, "{fields}");
            assert_eq!(request.parallel_tool_calls, None, "{fields}");
        }
    }

    #[test]
    fn tool_options_reach_the_worker() {
        for choice in [json!(null), json!("auto")] {
            let request = with_fields(json!({
                "tools": [weather_tool()],
                "tool_choice": choice,
                "parallel_tool_calls": false,
            }))
            .unwrap();
            assert!(request.tools.is_some());
            assert_eq!(request.parallel_tool_calls, Some(false));
        }
    }

    #[test]
    fn unsupported_tool_options_are_rejected() {
        let named = json!({"type": "function", "function": {"name": "weather"}});
        for (fields, expected) in [
            (json!({"tool_choice": "required"}), "required"),
            (json!({"tool_choice": named}), "names a function"),
            (json!({"tool_choice": "sometimes"}), "tool_choice must be"),
            (json!({"tool_choice": 1}), "tool_choice must be"),
            (json!({"functions": []}), "use tools"),
            (json!({"function_call": "auto"}), "use tools"),
        ] {
            let mut fields = fields;
            fields["tools"] = json!([weather_tool()]);
            let error = with_fields(fields.clone()).unwrap_err();
            assert!(error.contains(expected), "{fields}: {error}");
        }
    }

    #[test]
    fn malformed_tools_are_rejected() {
        for (tools, expected) in [
            (json!({}), "must be an array"),
            (json!("weather"), "must be an array"),
            (json!([1]), "type"),
            (json!([{"function": {"name": "f"}}]), "type"),
            (json!([{"type": "custom", "custom": {"name": "f"}}]), "type"),
            (json!([{"type": "function"}]), "function object"),
            (json!([{"type": "function", "function": {}}]), "name"),
            (
                json!([{"type": "function", "function": {"name": " "}}]),
                "name",
            ),
            (
                json!([{"type": "function", "function": {"name": "f", "parameters": []}}]),
                "parameters",
            ),
        ] {
            let error = with_fields(json!({"tools": tools})).unwrap_err();
            assert!(error.contains(expected), "{tools}: {error}");
            let hidden = with_fields(json!({"tools": tools, "tool_choice": "none"}));
            assert!(hidden.is_err(), "{tools}");
        }
    }

    #[test]
    fn tool_history_is_forwarded() {
        let request = with_messages(json!([
            {"role": "user", "content": "weather?", "name": "dropped"},
            {
                "role": "assistant",
                "content": null,
                "reasoning_content": "look it up",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{\"city\": \"Hanoi\"}"},
                }],
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "weather", "content": ""},
            {"role": "user", "content": "thanks", "reasoning_content": "dropped"},
        ]))
        .unwrap();

        let [user, assistant, tool, last] = request.messages.as_slice() else {
            panic!("expected four messages");
        };
        assert_eq!(user.name, None);
        assert_eq!(assistant.content, "".into());
        assert_eq!(assistant.reasoning_content.as_deref(), Some("look it up"));
        assert_eq!(assistant.tool_calls[0].id, "call_1");
        assert_eq!(
            assistant.tool_calls[0].function.arguments,
            r#"{"city": "Hanoi"}"#
        );
        assert_eq!(tool.tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(tool.name.as_deref(), Some("weather"));
        assert_eq!(last.reasoning_content, None);
    }

    #[test]
    fn malformed_tool_history_is_rejected() {
        let call = |id: &str, name: &str, arguments: &str| {
            json!([{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }])
        };
        for (message, expected) in [
            (json!({"role": "tool", "content": "x"}), "tool_call_id"),
            (
                json!({"role": "tool", "content": "x", "tool_call_id": ""}),
                "tool_call_id",
            ),
            (
                json!({"role": "user", "content": "x", "tool_calls": call("a", "f", "{}")}),
                "only on assistant",
            ),
            (
                json!({"role": "assistant", "tool_calls": call("", "f", "{}")}),
                "id must not be empty",
            ),
            (
                json!({"role": "assistant", "tool_calls": call("a", "", "{}")}),
                "name must not be empty",
            ),
            (
                json!({"role": "assistant", "tool_calls": call("a", "f", "{")}),
                "JSON text of an object",
            ),
            (
                json!({"role": "assistant", "tool_calls": call("a", "f", "[]")}),
                "JSON text of an object",
            ),
            (json!({"role": "assistant", "content": null}), "content"),
            (json!({"role": "user"}), "content"),
        ] {
            let error = with_messages(json!([message])).unwrap_err();
            assert!(error.contains(expected), "{message}: {error}");
        }
    }

    fn http_response(response: ChatCompletionResponse) -> Value {
        serde_json::to_value(ChatCompletionHttpResponse::from(response)).unwrap()
    }

    fn weather_call() -> ToolCall {
        serde_json::from_value(json!({
            "id": "call_1",
            "type": "function",
            "function": {"name": "weather", "arguments": "{}"},
        }))
        .unwrap()
    }

    #[test]
    fn response_with_only_tool_calls_has_null_content() {
        let body = http_response(ChatCompletionResponse {
            finish_reason: "tool_calls".to_string(),
            reasoning: Some("why".to_string()),
            tool_calls: vec![weather_call()],
            ..Default::default()
        });

        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            body["choices"][0]["message"],
            json!({
                "role": "assistant",
                "content": null,
                "reasoning_content": "why",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{}"},
                }],
            })
        );
    }

    #[test]
    fn response_keeps_text_next_to_tool_calls() {
        let body = http_response(ChatCompletionResponse {
            text: "Checking.".to_string(),
            tool_calls: vec![weather_call()],
            ..Default::default()
        });

        assert_eq!(body["choices"][0]["message"]["content"], "Checking.");
    }

    #[test]
    fn response_without_tool_output_keeps_its_shape() {
        let body = http_response(ChatCompletionResponse::default());

        assert_eq!(
            body["choices"][0]["message"],
            json!({"role": "assistant", "content": ""})
        );
    }
}
