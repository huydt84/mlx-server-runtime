from __future__ import annotations

import unittest

from mlx_worker.ipc import WorkerError, WorkerReady, decode_message, encode_message


class IpcEncodingTests(unittest.TestCase):
    def test_encode_ready_round_trip(self) -> None:
        message = WorkerReady()
        encoded = encode_message(message)
        self.assertEqual(encoded, b"READY\n")
        self.assertEqual(decode_message(encoded), message)

    def test_encode_error_round_trip(self) -> None:
        message = WorkerError("boom\nmore")
        encoded = encode_message(message)
        self.assertEqual(encoded, b"ERROR\tboom more\n")
        self.assertEqual(decode_message(encoded), WorkerError("boom more"))


if __name__ == "__main__":
    unittest.main()
