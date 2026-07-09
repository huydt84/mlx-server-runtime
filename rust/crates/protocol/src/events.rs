use super::request::ChatCompletionRequest;
use super::response::{
    ChatCompletionDelta, ChatCompletionResponse, SchedulerMetricsEvent, WorkerError, WorkerReady,
};
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
        WorkerMessage::Error(error) => {
            if error.error.is_some() {
                match serde_json::to_string(error) {
                    Ok(payload) => format!("ERROR\t{payload}"),
                    Err(_) => format!("ERROR\t{}", error.message.replace('\n', " ")),
                }
            } else {
                format!("ERROR\t{}", error.message.replace('\n', " "))
            }
        }
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
        "ERROR" => match serde_json::from_str::<WorkerError>(payload) {
            Ok(error) => Some(WorkerMessage::Error(error)),
            Err(_) => Some(WorkerMessage::Error(WorkerError {
                message: payload.to_string(),
                error: None,
            })),
        },
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
    /// Cancel an in-flight chat completion request.
    CancelRequest { request_id: String },
}

/// Events sent from the worker to the gateway after bootstrap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerEvent {
    /// Per-step scheduler metrics not tied to one request id.
    SchedulerMetrics { metrics: SchedulerMetricsEvent },
    /// A streamed chat completion delta.
    ChatCompletionDelta { delta: ChatCompletionDelta },
    /// A non-streaming chat completion result.
    ChatCompletion {
        response: ChatCompletionResponse,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_preprocess_latency_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_template_latency_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_cache_hit: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cached_tokens: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_cache_bytes: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        active_batch_cache_bytes: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_batch_size: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decode_batch_size: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        configured_prompt_batch_size: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        configured_decode_batch_size: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        backend: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        modality: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        apc_mode: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scheduler_stage: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cancellation_stage: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        queue_time_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prefill_time_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        ttft_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        decode_time_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        completion_time_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        scheduler_tick_latency_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        arbitration_delay_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worker_cancellation_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        worker_error_count: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vision_feature_cache_hit: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vision_feature_cache_bytes: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vision_feature_cache_entries: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vision_feature_cache_evictions: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        vision_encoder_latency_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        embedding_latency_ms: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_cache_entries: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        prompt_cache_evictions: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        peak_memory_bytes: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_width: Option<u32>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image_height: Option<u32>,
    },
    /// A worker-side request failure.
    Error {
        /// Machine-readable failure code.
        code: String,
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
    use crate::response::{ChatCompletionDelta, ChatCompletionResponse};

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
            error: None,
        });
        let encoded = encode_worker_message(&message);
        assert_eq!(encoded, "ERROR\tboom more");
        assert_eq!(
            decode_worker_message(&encoded),
            Some(WorkerMessage::Error(WorkerError {
                message: "boom more".to_string(),
                error: None,
            }))
        );
    }

    #[test]
    fn encode_and_decode_structured_error_round_trip() {
        let message = WorkerMessage::Error(WorkerError {
            message: "boom".to_string(),
            error: Some(crate::status::ModelError {
                code: "UNSUPPORTED_ARCHITECTURE_CLASS".to_string(),
                message: "boom".to_string(),
                at: 7,
                backend: Some("native-mlx".to_string()),
                stage: Some("architecture_detection".to_string()),
                category: Some("unsupported_class".to_string()),
                detail: Some("LlamaForCausalLM".to_string()),
            }),
        });

        let encoded = encode_worker_message(&message);
        assert!(encoded.starts_with("ERROR\t{"));
        assert_eq!(decode_worker_message(&encoded), Some(message));
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
                    content: "hello".into(),
                }],
                max_tokens: 16,
                temperature: 0.2,
                top_p: 0.9,
                max_prompt_tokens: 10,
                max_completion_tokens: 10,
                max_total_tokens_per_request: 20,
                stop: vec!["END".to_string()],
                stream: false,
            },
        };

        let encoded = encode_gateway_command(&command).unwrap();
        let decoded = decode_gateway_command(&encoded).unwrap();
        assert_eq!(decoded, command);
    }

    #[test]
    fn scheduler_metrics_event_round_trip() {
        let event = WorkerEvent::SchedulerMetrics {
            metrics: SchedulerMetricsEvent {
                backend: "native-mlx".to_string(),
                modality: "text".to_string(),
                phase: "decode".to_string(),
                scheduled_tokens: 2,
                batch_size: 2,
                waiting_requests: 1,
                running_requests: 2,
                scheduler_tick_latency_ms: 3,
                forward_mode: Some("mixed".to_string()),
                physical_batch_size: Some(2),
                model_forward_count: Some(1),
                cache_backend: Some("paged-mlx".to_string()),
                attention_backend: Some("native-metal-paged-sdpa".to_string()),
                attention_mode: Some("mixed".to_string()),
                attention_time_ms: Some(4),
                executor_prepare_ms: Some(5),
                executor_reserve_ms: Some(6),
                executor_forward_ms: Some(7),
                executor_sample_ms: Some(8),
                executor_eval_ms: Some(9),
                executor_commit_ms: Some(10),
                model_graph_embedding_ms: Some(11),
                model_graph_projection_total_ms: Some(12),
                model_graph_attention_ms: Some(13),
                model_graph_mlp_total_ms: Some(14),
                model_graph_norm_ms: Some(15),
                model_graph_lm_head_ms: Some(16),
                model_graph_layer_total_ms: Some(17),
                model_graph_worst_layer_ms: Some(18),
                model_graph_worst_layer_index: Some(1),
                total_pages: Some(16),
                used_pages: Some(2),
                free_pages: Some(14),
                pinned_pages: Some(0),
                internal_fragmentation_tokens: Some(3),
                active_kv_bytes: Some(1024),
                allocation_failures: Some(0),
                page_size: Some(16),
                prefix_strategy: Some("block-hash".to_string()),
                prefix_queries: Some(3),
                prefix_hits: Some(1),
                prefix_misses: Some(2),
                prefix_reused_tokens: Some(4),
                prefix_reused_pages: Some(1),
                prefix_entries: Some(2),
                prefix_bytes: Some(1024),
                prefix_pinned_pages: Some(1),
                prefix_collisions_rejected: Some(0),
                prefix_evictions: Some(1),
                radix_nodes: Some(5),
                radix_splits: Some(2),
                radix_shared_pages: Some(3),
                radix_protected_pages: Some(1),
                radix_evictable_pages: Some(2),
                radix_tree_depth: Some(8),
                radix_leaf_evictions: Some(1),
            },
        };

        let encoded = encode_worker_event(&event).unwrap();
        let decoded = decode_worker_event(&encoded).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn cancel_request_round_trip() {
        let command = GatewayCommand::CancelRequest {
            request_id: "req-1".to_string(),
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
                prompt_cache_hit: Some(true),
                cached_tokens: Some(8),
                prompt_cache_bytes: Some(128),
                active_batch_cache_bytes: Some(256),
                prompt_batch_size: Some(2),
                decode_batch_size: Some(2),
                configured_prompt_batch_size: Some(2),
                configured_decode_batch_size: Some(2),
                backend: Some("vlm".to_string()),
                modality: Some("vlm".to_string()),
                apc_mode: Some("apc_manager".to_string()),
                scheduler_stage: Some("completed".to_string()),
                cancellation_stage: None,
                queue_time_ms: Some(3),
                prefill_time_ms: Some(5),
                ttft_ms: Some(7),
                decode_time_ms: Some(11),
                completion_time_ms: Some(18),
                scheduler_tick_latency_ms: Some(1),
                arbitration_delay_ms: Some(2),
                worker_cancellation_count: Some(3),
                worker_error_count: Some(1),
                vision_feature_cache_hit: Some(true),
                vision_feature_cache_bytes: Some(64),
                vision_feature_cache_entries: Some(1),
                vision_feature_cache_evictions: Some(0),
                vision_encoder_latency_ms: Some(9),
                embedding_latency_ms: Some(9),
                prompt_cache_entries: Some(4),
                prompt_cache_evictions: Some(2),
                peak_memory_bytes: Some(1024),
                image_width: Some(640),
                image_height: Some(480),
            },
            image_count: Some(1),
            image_preprocess_latency_ms: Some(8),
            prompt_template_latency_ms: Some(4),
            prompt_cache_hit: Some(true),
            cached_tokens: Some(8),
            prompt_cache_bytes: Some(128),
            active_batch_cache_bytes: Some(256),
            prompt_batch_size: Some(2),
            decode_batch_size: Some(2),
            configured_prompt_batch_size: Some(2),
            configured_decode_batch_size: Some(2),
            backend: Some("vlm".to_string()),
            modality: Some("vlm".to_string()),
            apc_mode: Some("apc_manager".to_string()),
            scheduler_stage: Some("completed".to_string()),
            cancellation_stage: None,
            queue_time_ms: Some(3),
            prefill_time_ms: Some(5),
            ttft_ms: Some(7),
            decode_time_ms: Some(11),
            completion_time_ms: Some(18),
            scheduler_tick_latency_ms: Some(1),
            arbitration_delay_ms: Some(2),
            worker_cancellation_count: Some(3),
            worker_error_count: Some(1),
            vision_feature_cache_hit: Some(true),
            vision_feature_cache_bytes: Some(64),
            vision_feature_cache_entries: Some(1),
            vision_feature_cache_evictions: Some(0),
            vision_encoder_latency_ms: Some(9),
            embedding_latency_ms: Some(9),
            prompt_cache_entries: Some(4),
            prompt_cache_evictions: Some(2),
            peak_memory_bytes: Some(1024),
            image_width: Some(640),
            image_height: Some(480),
        };

        let encoded = encode_worker_event(&event).unwrap();
        let decoded = decode_worker_event(&encoded).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn worker_error_round_trip() {
        let event = WorkerEvent::Error {
            code: "INVALID_REQUEST".to_string(),
            request_id: "req-1".to_string(),
            message: "prompt too long".to_string(),
        };

        let encoded = encode_worker_event(&event).unwrap();
        let decoded = decode_worker_event(&encoded).unwrap();
        assert_eq!(decoded, event);
    }

    #[test]
    fn vlm_gateway_command_round_trip_with_image_parts() {
        let command = GatewayCommand::ChatCompletion {
            request: ChatCompletionRequest {
                request_id: "vlm-req".to_string(),
                model: "vlm-model".to_string(),
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: crate::request::MessageContent::Parts(vec![
                        crate::request::ContentPart::Text {
                            text: "describe".to_string(),
                        },
                        crate::request::ContentPart::ImageUrl {
                            image_url: crate::request::ImageUrl {
                                url: "data:image/png;base64,abc".to_string(),
                                detail: None,
                            },
                        },
                    ]),
                }],
                max_tokens: 32,
                temperature: 0.2,
                top_p: 0.9,
                max_prompt_tokens: 64,
                max_completion_tokens: 64,
                max_total_tokens_per_request: 128,
                stop: vec![],
                stream: true,
            },
        };

        let encoded = encode_gateway_command(&command).unwrap();
        let decoded = decode_gateway_command(&encoded).unwrap();
        assert_eq!(decoded, command);

        // Verify the JSON structure has the expected content parts
        assert!(encoded.contains("\"type\":\"text\""));
        assert!(encoded.contains("\"type\":\"image_url\""));
        assert!(encoded.contains("\"image_url\":{\"url\":"));
    }

    #[test]
    fn image_url_detail_is_preserved_through_json_round_trip() {
        let command = GatewayCommand::ChatCompletion {
            request: ChatCompletionRequest {
                request_id: "vlm-detail".to_string(),
                model: "vlm-model".to_string(),
                messages: vec![ChatMessage {
                    role: MessageRole::User,
                    content: crate::request::MessageContent::Parts(vec![
                        crate::request::ContentPart::ImageUrl {
                            image_url: crate::request::ImageUrl {
                                url: "http://example.com/img.jpg".to_string(),
                                detail: Some("high".to_string()),
                            },
                        },
                    ]),
                }],
                max_tokens: 32,
                temperature: 0.2,
                top_p: 0.9,
                max_prompt_tokens: 64,
                max_completion_tokens: 64,
                max_total_tokens_per_request: 128,
                stop: vec![],
                stream: false,
            },
        };

        let encoded = encode_gateway_command(&command).unwrap();
        let decoded = decode_gateway_command(&encoded).unwrap();
        assert_eq!(decoded, command);
        assert!(encoded.contains("\"detail\":\"high\""));
    }

    #[test]
    fn image_url_detail_is_optional_and_defaults_to_none() {
        let json = r#"{
            \"type\": \"chat_completion\",
            \"request\": {
                \"request_id\": \"vlm-nodetail\",
                \"model\": \"vlm-model\",
                \"messages\": [{
                    \"role\": \"user\",
                    \"content\": [{
                        \"type\": \"image_url\",
                        \"image_url\": { \"url\": \"http://example.com/img.jpg\" }
                    }]
                }],
                \"max_tokens\": 32,
                \"temperature\": 0.2,
                \"top_p\": 0.9,
                \"max_prompt_tokens\": 64,
                \"max_completion_tokens\": 64,
                \"max_total_tokens_per_request\": 128,
                \"stream\": false
            }
        }"#;
        // Remove the escaped quotes we inserted for readability
        let clean = json.replace("\\\"", "\"");
        let decoded = decode_gateway_command(&clean).unwrap();
        if let GatewayCommand::ChatCompletion { request } = &decoded {
            if let crate::request::MessageContent::Parts(parts) = &request.messages[0].content {
                if let crate::request::ContentPart::ImageUrl { image_url } = &parts[0] {
                    assert_eq!(image_url.detail, None);
                } else {
                    panic!("expected ImageUrl content part");
                }
            } else {
                panic!("expected Parts content");
            }
        } else {
            panic!("expected ChatCompletion command");
        }
    }

    #[test]
    fn worker_delta_round_trip() {
        let event = WorkerEvent::ChatCompletionDelta {
            delta: ChatCompletionDelta {
                request_id: "req-1".to_string(),
                delta: "hello".to_string(),
            },
        };

        let encoded = encode_worker_event(&event).unwrap();
        let decoded = decode_worker_event(&encoded).unwrap();
        assert_eq!(decoded, event);
    }
}
