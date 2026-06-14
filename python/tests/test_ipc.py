from __future__ import annotations

import unittest

from mlx_worker.ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ModelError,
    ModelLoadProgress,
    ModelStatus,
    WorkerCommandError,
    WorkerError,
    WorkerReady,
    decode_bootstrap_message,
    decode_command,
    decode_event,
    encode_bootstrap_message,
    encode_command,
    encode_event,
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
        )

        encoded = encode_event(response)
        self.assertEqual(decode_event(encoded), response)

    def test_worker_error_event_round_trip(self) -> None:
        event = WorkerCommandError(request_id="req-1", message="boom\nmore")

        encoded = encode_event(event)
        self.assertEqual(
            decode_event(encoded),
            WorkerCommandError(request_id="req-1", message="boom more"),
        )


if __name__ == "__main__":
    unittest.main()
