from __future__ import annotations

import json
import unittest

from mlx_worker.ipc import (
    ChatCompletionRequest,
    ChatCompletionDelta,
    ChatCompletionResponse,
    ChatMessage,
    CancelRequest,
    ImageContent,
    ModelError,
    ModelLoadProgress,
    ModelStatus,
    SchedulerMetricsEvent,
    TextContent,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_bootstrap_message,
    decode_command,
    decode_event,
    encode_bootstrap_message,
    encode_command,
    encode_event,
    has_image_content,
    request_has_images,
)


class IpcEncodingTests(unittest.TestCase):
    def test_encode_ready_round_trip(self) -> None:
        message = WorkerReady()
        encoded = encode_bootstrap_message(message)
        self.assertEqual(encoded, b"READY\n")
        self.assertEqual(decode_bootstrap_message(encoded), message)

    def test_encode_error_round_trip(self) -> None:
        message = WorkerError("boom\nmore")
        encoded = encode_bootstrap_message(message)
        self.assertEqual(encoded, b"ERROR\tboom more\n")
        self.assertEqual(decode_bootstrap_message(encoded), WorkerError("boom more"))

    def test_encode_structured_error_round_trip(self) -> None:
        message = WorkerError(
            "boom",
            error=ModelError(
                code="UNSUPPORTED_ARCHITECTURE_CLASS",
                message="boom",
                at=3,
                backend="native-mlx",
                stage="architecture_detection",
                category="unsupported_class",
                detail="LlamaForCausalLM",
            ),
        )

        encoded = encode_bootstrap_message(message)
        self.assertEqual(decode_bootstrap_message(encoded), message)

    def test_encode_status_round_trip(self) -> None:
        message = ModelStatus(
            model="test-model",
            revision="rev-1",
            state="warming_up",
            ready=False,
            servable=False,
            progress=ModelLoadProgress(
                downloaded_bytes=8,
                total_bytes=16,
                loaded_tensors=1,
                total_tensors=2,
                current_phase="warming_up",
            ),
            device="mps",
            dtype="float16",
            loaded_at=None,
            started_loading_at=1,
            last_transition_at=2,
            last_error=ModelError(code="MODEL_LOAD_FAILED", message="boom", at=3),
            warmup_passed=False,
            last_warmup_at=None,
            last_warmup_latency_ms=None,
        )

        encoded = encode_bootstrap_message(message)
        self.assertEqual(decode_bootstrap_message(encoded), message)

    def test_chat_completion_command_round_trip(self) -> None:
        request = ChatCompletionRequest(
            request_id="req-1",
            model="test-model",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=12,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
            stop=("END",),
            stream=True,
        )

        encoded = encode_command(request)
        self.assertEqual(decode_command(encoded), request)

    def test_chat_completion_event_round_trip(self) -> None:
        response = ChatCompletionResponse(
            request_id="req-1",
            model="test-model",
            text="hello back",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=3,
            gateway_queue_wait_ms=2,
            scheduler_queue_wait_ms=4,
            cancellation_latency_ms=0,
        )

        encoded = encode_event(response)
        self.assertEqual(decode_event(encoded), response)

    def test_chat_completion_delta_round_trip(self) -> None:
        delta = ChatCompletionDelta(request_id="req-1", delta="hel")
        encoded = encode_event(delta)
        self.assertEqual(decode_event(encoded), delta)

    def test_worker_error_event_round_trip(self) -> None:
        event = WorkerCommandError(
            code="WORKER_ERROR", request_id="req-1", message="boom\nmore"
        )

        encoded = encode_event(event)
        self.assertEqual(
            decode_event(encoded),
            WorkerCommandError(
                code="WORKER_ERROR", request_id="req-1", message="boom more"
            ),
        )

    def test_scheduler_metrics_event_round_trip(self) -> None:
        event = SchedulerMetricsEvent(
            backend="native-mlx",
            modality="text",
            phase="decode",
            scheduled_tokens=2,
            batch_size=2,
            waiting_requests=1,
            running_requests=2,
            scheduler_tick_latency_ms=3,
            scheduler_select_ms=4,
            scheduler_cache_probe_ms=5,
            scheduler_cache_acquire_ms=6,
            scheduler_cache_publish_ms=7,
            scheduler_apply_ms=8,
            scheduler_queue_wait_ms=9,
            cancellation_latency_ms=10,
            scheduling_policy="lpm",
            forward_mode="mixed",
            physical_batch_size=2,
            model_forward_count=1,
            cache_backend="paged-mlx",
            attention_backend="native-metal-paged",
            attention_mode="mixed",
            attention_time_ms=4,
            executor_prepare_ms=5,
            executor_reserve_ms=6,
            executor_forward_ms=7,
            executor_sample_ms=8,
            executor_eval_ms=9,
            executor_commit_ms=10,
            model_graph_embedding_ms=11,
            model_graph_projection_total_ms=12,
            model_graph_attention_ms=13,
            model_graph_mlp_total_ms=14,
            model_graph_norm_ms=15,
            model_graph_lm_head_ms=16,
            model_graph_layer_total_ms=17,
            model_graph_worst_layer_ms=18,
            model_graph_worst_layer_index=1,
            total_pages=16,
            used_pages=2,
            free_pages=14,
            pinned_pages=0,
            internal_fragmentation_tokens=3,
            active_kv_bytes=1024,
            allocation_failures=0,
            page_size=16,
            prefix_strategy="block-hash",
            prefix_queries=3,
            prefix_hits=1,
            prefix_misses=2,
            prefix_reused_tokens=4,
            prefix_reused_pages=1,
            prefix_entries=2,
            prefix_bytes=1024,
            prefix_pinned_pages=1,
            prefix_collisions_rejected=0,
            prefix_evictions=1,
            radix_nodes=5,
            radix_splits=2,
            radix_shared_pages=3,
            radix_protected_pages=1,
            radix_evictable_pages=2,
            radix_tree_depth=8,
            radix_leaf_evictions=1,
        )

        encoded = encode_event(event)
        self.assertEqual(decode_event(encoded), event)

    def test_cancel_command_round_trip(self) -> None:
        encoded = b'{"type":"cancel_request","request_id":"req-1"}\n'
        self.assertEqual(decode_command(encoded), CancelRequest(request_id="req-1"))


class VlmIpcTests(unittest.TestCase):
    """Phase 8 VLM IPC encoding/decoding."""

    def test_image_content_dataclass(self) -> None:
        img = ImageContent(url="path/to/image.jpg", detail="high")
        self.assertEqual(img.url, "path/to/image.jpg")
        self.assertEqual(img.detail, "high")

    def test_text_content_dataclass(self) -> None:
        txt = TextContent(text="hello")
        self.assertEqual(txt.text, "hello")

    def test_has_image_content_text_only(self) -> None:
        self.assertFalse(has_image_content("plain text"))

    def test_has_image_content_with_image(self) -> None:
        content = (TextContent(text="desc"), ImageContent(url="img.png"))
        self.assertTrue(has_image_content(content))

    def test_has_image_content_text_only_tuple(self) -> None:
        content = (TextContent(text="only text"),)
        self.assertFalse(has_image_content(content))

    def test_request_has_images_true(self) -> None:
        request = ChatCompletionRequest(
            request_id="vlm-req",
            model="vlm-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=(TextContent(text="what"), ImageContent(url="img.jpg")),
                )
            ],
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        )
        self.assertTrue(request_has_images(request))

    def test_request_has_images_false(self) -> None:
        request = ChatCompletionRequest(
            request_id="text-req",
            model="text-model",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=16,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=32,
            max_completion_tokens=32,
            max_total_tokens_per_request=64,
        )
        self.assertFalse(request_has_images(request))

    def test_vlm_command_round_trip(self) -> None:
        """Encode and decode a VLM request with image content."""
        request = ChatCompletionRequest(
            request_id="vlm-1",
            model="vlm-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        TextContent(text="What is this?"),
                        ImageContent(url="https://example.com/img.jpg"),
                    ),
                )
            ],
            max_tokens=32,
            temperature=0.2,
            top_p=0.9,
            max_prompt_tokens=64,
            max_completion_tokens=64,
            max_total_tokens_per_request=128,
            stream=True,
        )

        encoded = encode_command(request)
        decoded = decode_command(encoded)

        assert decoded is not None
        assert not isinstance(decoded, CancelRequest)
        self.assertEqual(decoded.request_id, "vlm-1")
        self.assertEqual(decoded.model, "vlm-model")
        self.assertEqual(len(decoded.messages), 1)
        msg = decoded.messages[0]
        self.assertEqual(msg.role, "user")
        self.assertIsInstance(msg.content, tuple)
        if isinstance(msg.content, tuple):
            self.assertEqual(len(msg.content), 2)
            self.assertIsInstance(msg.content[0], TextContent)
            self.assertEqual(msg.content[0].text, "What is this?")
            self.assertIsInstance(msg.content[1], ImageContent)
            self.assertEqual(msg.content[1].url, "https://example.com/img.jpg")

    def test_vlm_response_round_trip_preserves_timing_fields(self) -> None:
        """Worker responses preserve optional VLM timing fields."""
        response = ChatCompletionResponse(
            request_id="vlm-resp",
            model="vlm-model",
            text="done",
            finish_reason="stop",
            prompt_tokens=11,
            completion_tokens=3,
            image_count=1,
            image_preprocess_latency_ms=8,
            prompt_template_latency_ms=4,
            prompt_cache_hit=True,
            cached_tokens=7,
            prompt_cache_bytes=96,
            active_batch_cache_bytes=128,
            prompt_batch_size=2,
            decode_batch_size=2,
            gateway_queue_wait_ms=3,
            scheduler_queue_wait_ms=5,
            cancellation_latency_ms=0,
        )

        encoded = encode_event(response)
        decoded = decode_event(encoded)

        self.assertEqual(decoded, response)

    def test_vlm_command_json_structure(self) -> None:
        """Verify the JSON structure of an encoded VLM command."""
        request = ChatCompletionRequest(
            request_id="vlm-2",
            model="vlm-model",
            messages=[
                ChatMessage(
                    role="user",
                    content=(
                        TextContent(text="desc"),
                        ImageContent(url="img.png"),
                    ),
                )
            ],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=16,
            max_completion_tokens=16,
            max_total_tokens_per_request=32,
        )

        encoded = encode_command(request)
        raw = json.loads(encoded.decode("utf-8").strip())

        self.assertEqual(raw["type"], "chat_completion")
        messages = raw["request"]["messages"]
        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertIsInstance(content, list)
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0], {"type": "text", "text": "desc"})
        self.assertEqual(
            content[1],
            {"type": "image_url", "image_url": {"url": "img.png", "detail": "auto"}},
        )

    def test_text_command_json_structure_unchanged(self) -> None:
        """Text-only commands still use plain string content for backward compat."""
        request = ChatCompletionRequest(
            request_id="text-1",
            model="text-model",
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            max_prompt_tokens=16,
            max_completion_tokens=16,
            max_total_tokens_per_request=32,
        )

        encoded = encode_command(request)
        raw = json.loads(encoded.decode("utf-8").strip())

        content = raw["request"]["messages"][0]["content"]
        self.assertIsInstance(content, str)
        self.assertEqual(content, "hello")


if __name__ == "__main__":
    unittest.main()
