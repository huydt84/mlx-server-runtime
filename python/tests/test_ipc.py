from __future__ import annotations

import unittest

from mlx_worker.ipc import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
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
