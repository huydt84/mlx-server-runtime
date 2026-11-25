"""Conversion of chat requests into the inputs of a model's chat template."""

from __future__ import annotations

import json
from typing import Any

from .ipc import ChatCompletionRequest, ChatMessage, TextContent, ToolCall


def text_content(message: ChatMessage) -> str:
    """Return message text, joining text parts.

    Raises:
        ValueError: If the message holds image content.
    """

    if isinstance(message.content, str):
        return message.content
    if not all(isinstance(part, TextContent) for part in message.content):
        raise ValueError("text model does not support image content")
    return "".join(
        part.text for part in message.content if isinstance(part, TextContent)
    )


def template_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    """Return the messages in the shape chat templates read.

    Raises:
        ValueError: If a message holds images or malformed call arguments.
    """

    return [_template_message(message) for message in request.messages]


def render_prompt_tokens(tokenizer: Any, request: ChatCompletionRequest) -> list[int]:
    """Render a request through the tokenizer's chat template.

    Raises:
        ValueError: If the template rejects the conversation.
    """

    messages = template_messages(request)
    kwargs: dict[str, Any] = {}
    if request.tools:
        kwargs["tools"] = [_template_tool(tool) for tool in request.tools]
    try:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )
    except Exception as exc:
        raise ValueError(f"chat template rejected the messages: {exc}") from exc
    if hasattr(tokens, "tolist"):
        tokens = tokens.tolist()
    return [int(token) for token in tokens]


def _template_message(message: ChatMessage) -> dict[str, Any]:
    """Return one message in the shape chat templates read.

    LFM2.5 templates read earlier thinking from ``thinking``, the other
    families from ``reasoning_content``.
    """

    rendered: dict[str, Any] = {"role": message.role, "content": text_content(message)}
    if message.reasoning_content:
        rendered["reasoning_content"] = message.reasoning_content
        rendered["thinking"] = message.reasoning_content
    if message.tool_calls:
        rendered["tool_calls"] = [
            _template_tool_call(call) for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        rendered["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        rendered["name"] = message.name
    return rendered


def _template_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Return a tool definition without ``strict``, which is not enforced."""

    function = {k: v for k, v in tool["function"].items() if k != "strict"}
    return {**tool, "function": function}


def _template_tool_call(call: ToolCall) -> dict[str, Any]:
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"arguments of tool call '{call.id}' are not valid JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"arguments of tool call '{call.id}' are not a JSON object")
    return {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": arguments},
    }


__all__ = ["render_prompt_tokens", "template_messages", "text_content"]
