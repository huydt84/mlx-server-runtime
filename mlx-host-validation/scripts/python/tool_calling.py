"""Tool-calling host validation requests and assertions."""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
from typing import Any, NoReturn

TOOL_NAME = "get_weather"
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name."},
                },
                "required": ["city"],
            },
        },
    }
]
QUESTION = {
    "role": "user",
    "content": "What is the weather in Hanoi right now? Use the get_weather tool.",
}
TWO_CITIES = {
    "role": "user",
    "content": "Use the get_weather tool for both Hanoi and Paris.",
}
TOOL_RESULT = '{"temperature_celsius": 31, "condition": "sunny"}'


def main() -> None:
    """Run one tool-calling check."""

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("supported", "unsupported"))
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    client = Client(args.port, args.model, args.timeout)
    try:
        if args.command == "supported":
            check_supported(client)
        else:
            check_unsupported(client)
    finally:
        args.output.write_text(json.dumps(client.exchanges, indent=2))


class Client:
    """Chat completion client that records every exchange."""

    def __init__(self, port: int, model: str, timeout: float) -> None:
        self._port = port
        self._model = model
        self._timeout = timeout
        self.exchanges: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
        """Return the only choice of a successful request."""

        status, text = self.post(messages, **fields)
        if status != 200:
            fail(f"HTTP {status}: {text}")
        return json.loads(text)["choices"][0]

    def stream(self, messages: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
        """Return a streamed choice assembled into a non-streaming one."""

        status, text = self.post(messages, stream=True, **fields)
        if status != 200:
            fail(f"HTTP {status}: {text}")
        if not text.rstrip().endswith("data: [DONE]"):
            fail("stream did not end with [DONE]")
        message: dict[str, Any] = {"content": ""}
        finish_reason = None
        for line in text.splitlines():
            if not line.startswith("data: {"):
                continue
            for choice in json.loads(line.removeprefix("data: "))["choices"]:
                delta = choice["delta"]
                message["content"] += delta.get("content") or ""
                if "reasoning_content" in delta:
                    message.setdefault("reasoning_content", "")
                    message["reasoning_content"] += delta["reasoning_content"]
                for call in delta.get("tool_calls", []):
                    calls = message.setdefault("tool_calls", [])
                    if call["index"] != len(calls):
                        fail(f"tool call index {call['index']} is out of order")
                    calls.append({key: call[key] for key in ("id", "type", "function")})
                finish_reason = choice["finish_reason"] or finish_reason
        return {"message": message, "finish_reason": finish_reason}

    def rejected(
        self, code: str, messages: list[dict[str, Any]], **fields: Any
    ) -> None:
        """Require a 400 response with an error code, plain and streamed."""

        for stream in (False, True):
            status, text = self.post(messages, stream=stream, **fields)
            if status != 400 or json.loads(text)["error"]["code"] != code:
                fail(f"expected 400 {code}, got HTTP {status}: {text}")

    def disconnect_mid_stream(self, messages: list[dict[str, Any]]) -> None:
        """Close the connection after the first streamed event."""

        connection = self._send(messages, {"stream": True, "tools": TOOLS})
        response = connection.getresponse()
        if response.status != 200:
            fail(f"HTTP {response.status} for the disconnecting request")
        response.readline()
        connection.close()

    def metrics(self) -> str:
        """Return the Prometheus metrics text."""

        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout=self._timeout
        )
        connection.request("GET", "/metrics")
        return connection.getresponse().read().decode("utf-8")

    def post(self, messages: list[dict[str, Any]], **fields: Any) -> tuple[int, str]:
        """Send one request and return its status and body."""

        response = self._send(messages, fields).getresponse()
        text = response.read().decode("utf-8")
        self.exchanges.append(
            {
                "messages": messages,
                "fields": fields,
                "status": response.status,
                "body": text,
            }
        )
        return response.status, text

    def _send(
        self, messages: list[dict[str, Any]], fields: dict[str, Any]
    ) -> http.client.HTTPConnection:
        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0,
            **fields,
        }
        connection = http.client.HTTPConnection(
            "127.0.0.1", self._port, timeout=self._timeout
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        return connection


def check_supported(client: Client) -> None:
    """Check a model that can call tools."""

    first = client.chat([QUESTION], tools=TOOLS)
    require_weather_call(first, "Hanoi")
    print("call_ok=1")

    streamed = client.stream([QUESTION], tools=TOOLS)
    require_weather_call(streamed, "Hanoi")
    if functions(streamed) != functions(first):
        fail("streamed calls differ from the non-streamed ones")
    print("stream_call_ok=1")

    answer = client.chat([QUESTION, *tool_turn(first)], tools=TOOLS)
    if answer["finish_reason"] != "stop" or "31" not in answer["message"]["content"]:
        fail(f"reply to the tool result does not use it: {answer}")
    print("tool_result_ok=1")

    single = client.chat([TWO_CITIES], tools=TOOLS, parallel_tool_calls=False)
    require_weather_call(single, "")
    if len(functions(single)) != 1:
        fail(f"parallel_tool_calls false returned several calls: {single}")
    print("single_call_ok=1")

    hidden = client.chat([QUESTION], tools=TOOLS, tool_choice="none", max_tokens=32)
    if {"tool_calls", "reasoning_content"} & hidden["message"].keys():
        fail(f"tool_choice none returned tool output: {hidden}")
    print("tool_choice_none_ok=1")

    client.rejected("INVALID_REQUEST", [QUESTION], tools=TOOLS, tool_choice="required")
    print("tool_choice_required_rejected_ok=1")

    client.disconnect_mid_stream([QUESTION])
    require_weather_call(client.chat([QUESTION], tools=TOOLS), "Hanoi")
    print("disconnect_ok=1")

    parsers = [
        line
        for line in client.metrics().splitlines()
        if line.startswith("mlx_tool_calls_total{")
    ]
    if not parsers:
        fail("mlx_tool_calls_total is missing from the metrics")
    print(f"tool_calls_metric={parsers[0]}")


def check_unsupported(client: Client) -> None:
    """Check a model without a tool-call parser."""

    client.rejected("TOOLS_NOT_SUPPORTED", [QUESTION], tools=TOOLS)
    for fields in ({"tools": []}, {"tools": TOOLS, "tool_choice": "none"}, {}):
        choice = client.chat([QUESTION], max_tokens=8, **fields)
        if "tool_calls" in choice["message"]:
            fail(f"unexpected tool calls: {choice}")


def require_weather_call(choice: dict[str, Any], city: str) -> None:
    """Require valid calls to the weather tool and clean surrounding text."""

    message = choice["message"]
    if choice["finish_reason"] != "tool_calls" or not functions(choice):
        fail(f"expected a tool call: {choice}")
    for call in message["tool_calls"]:
        if call["type"] != "function" or not call["id"].startswith("call_"):
            fail(f"malformed tool call: {call}")
        arguments = json.loads(call["function"]["arguments"])
        if call["function"]["name"] != TOOL_NAME or not isinstance(arguments, dict):
            fail(f"malformed tool call: {call}")
        if city.lower() not in str(arguments.get("city", "")).lower():
            fail(f"expected a call for {city}: {call}")
    visible = (message["content"] or "") + (message.get("reasoning_content") or "")
    for marker in ("<tool_call>", "<|tool_call_start|>", "<think>", "</think>"):
        if marker in visible:
            fail(f"{marker} leaked into the reply: {choice}")


def functions(choice: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the called functions without their random ids."""

    return [call["function"] for call in choice["message"].get("tool_calls", [])]


def tool_turn(choice: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the assistant message and one result per call."""

    message = choice["message"]
    assistant = {"role": "assistant", "content": message["content"]}
    for key in ("reasoning_content", "tool_calls"):
        if key in message:
            assistant[key] = message[key]
    results = [
        {"role": "tool", "tool_call_id": call["id"], "content": TOOL_RESULT}
        for call in message["tool_calls"]
    ]
    return [assistant, *results]


def fail(reason: str) -> NoReturn:
    """Exit with a failure reason."""

    raise SystemExit(f"tool_calling_failed={reason}")


if __name__ == "__main__":
    main()
