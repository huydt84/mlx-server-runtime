use serde::{Deserialize, Deserializer, Serialize};

/// Health check marker reserved for future worker ping support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthCheck;

/// Content of a chat message — plain text string or structured parts with images.
///
/// Text-only messages are serialized as `"string"` for backward compatibility.
/// VLM messages are serialized as `[{ "type": "text", "text": "..." }, ...]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content (standard OpenAI format).
    Text(String),
    /// Structured content parts (text + images for VLM).
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Returns true if the content is non-empty.
    pub fn has_content(&self) -> bool {
        match self {
            MessageContent::Text(s) => !s.trim().is_empty(),
            MessageContent::Parts(parts) => {
                !parts.is_empty()
                    && parts.iter().any(|p| match p {
                        ContentPart::Text { text } => !text.trim().is_empty(),
                        ContentPart::ImageUrl { .. } => true,
                    })
            }
        }
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        MessageContent::Text(String::new())
    }
}

impl From<String> for MessageContent {
    fn from(s: String) -> Self {
        MessageContent::Text(s)
    }
}

impl From<&str> for MessageContent {
    fn from(s: &str) -> Self {
        MessageContent::Text(s.to_string())
    }
}

/// A single content part in a structured VLM message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text content block.
    Text {
        /// The text payload.
        text: String,
    },
    /// An image URL or data URI content block.
    ImageUrl {
        /// Image location metadata.
        image_url: ImageUrl,
    },
}

/// Image source for a VLM content part.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL, local path, or base64 data URI for the image.
    pub url: String,
    /// Optional detail hint ("auto", "low", "high").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A chat message in OpenAI-compatible format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The message role.
    pub role: MessageRole,
    /// The message content (text or structured parts with images). A missing
    /// or null content is empty text.
    #[serde(default, deserialize_with = "null_as_default")]
    pub content: MessageContent,
    /// Tool calls the assistant made in this message.
    #[serde(
        default,
        deserialize_with = "null_as_default",
        skip_serializing_if = "Vec::is_empty"
    )]
    pub tool_calls: Vec<ToolCall>,
    /// Id of the tool call a tool message answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Name of the function a tool message answers for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Thinking text of an earlier assistant message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// Creates a message that holds only content.
    pub fn new(role: MessageRole, content: impl Into<MessageContent>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }
}

/// A function call made by the assistant, in OpenAI-compatible format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Call identifier that tool messages refer to.
    pub id: String,
    /// The kind of tool; only functions exist.
    #[serde(rename = "type")]
    pub kind: ToolKind,
    /// The called function.
    pub function: FunctionCall,
}

/// The supported tool kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    /// A function tool.
    Function,
}

/// Name and arguments of a called function.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Function name.
    pub name: String,
    /// JSON text of the arguments object.
    pub arguments: String,
}

fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Default + Deserialize<'de>,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// The supported chat roles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    /// System message.
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Result of a tool call.
    Tool,
}

/// A single non-streaming chat completion request sent to the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Request identifier generated by the gateway.
    pub request_id: String,
    /// Model id requested by the client.
    pub model: String,
    /// Chat history.
    pub messages: Vec<ChatMessage>,
    /// Maximum number of generated tokens.
    pub max_tokens: u32,
    /// Sampling temperature.
    pub temperature: f32,
    /// Top-p sampling cutoff.
    pub top_p: f32,
    /// Maximum prompt tokens allowed for this request.
    pub max_prompt_tokens: u32,
    /// Maximum completion tokens allowed for this request.
    pub max_completion_tokens: u32,
    /// Maximum total tokens allowed for this request.
    pub max_total_tokens_per_request: u32,
    /// User-visible stop strings enforced by runtime.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Whether the request should stream token deltas.
    #[serde(default)]
    pub stream: bool,
    /// JSON text of the offered tool definitions. Text keeps the key order the
    /// client sent, which chat templates render into the prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<String>,
    /// Whether one reply may hold several tool calls; sent only with tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_message_serializes_without_tool_fields() {
        let message = ChatMessage::new(MessageRole::User, "hello");

        assert_eq!(
            serde_json::to_string(&message).unwrap(),
            r#"{"role":"user","content":"hello"}"#
        );
    }

    #[test]
    fn null_content_and_null_tool_calls_are_empty() {
        let message: ChatMessage = serde_json::from_str(
            r#"{"role":"assistant","content":null,"tool_calls":null,"refusal":null}"#,
        )
        .unwrap();

        assert_eq!(message, ChatMessage::new(MessageRole::Assistant, ""));
    }

    #[test]
    fn tool_history_round_trips() {
        let encoded = concat!(
            r#"{"role":"assistant","content":"","tool_calls":[{"id":"call_1","type":"function","#,
            r#""function":{"name":"weather","arguments":"{}"}}],"reasoning_content":"why"}"#
        );

        let message: ChatMessage = serde_json::from_str(encoded).unwrap();

        assert_eq!(message.tool_calls[0].function.name, "weather");
        assert_eq!(serde_json::to_string(&message).unwrap(), encoded);
    }

    #[test]
    fn tool_call_without_id_or_with_another_type_is_rejected() {
        for call in [
            r#"{"type":"function","function":{"name":"f","arguments":"{}"}}"#,
            r#"{"id":"call_1","type":"custom","function":{"name":"f","arguments":"{}"}}"#,
            r#"{"id":"call_1","type":"function","function":{"name":"f","arguments":{}}}"#,
        ] {
            assert!(serde_json::from_str::<ToolCall>(call).is_err(), "{call}");
        }
    }
}

/// Whisper task requested for an audio upload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AudioTask {
    /// Transcribe speech in its spoken language (`/v1/audio/transcriptions`).
    Transcribe,
    /// Translate speech into English text (`/v1/audio/translations`).
    Translate,
}

impl AudioTask {
    /// Returns the wire name of the task.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transcribe => "transcribe",
            Self::Translate => "translate",
        }
    }
}

/// An audio transcription or translation request sent to the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptionRequest {
    /// Request identifier generated by the gateway.
    pub request_id: String,
    /// Speech-to-text model id requested by the client.
    pub model: String,
    /// Whether to transcribe or translate to English.
    pub task: AudioTask,
    /// The uploaded audio file bytes, standard base64 encoded.
    pub audio_base64: String,
    /// Uploaded file name, used only as a format hint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// ISO-639-1 language of the input audio; detected when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Text that conditions the first decoding window.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    /// Sampling temperature; `0` enables Whisper's temperature fallback.
    pub temperature: f32,
    /// Whether word-level timestamps are requested.
    #[serde(default)]
    pub word_timestamps: bool,
}
