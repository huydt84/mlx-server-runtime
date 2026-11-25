"""Streaming parsers that split generated text into content, thinking and calls.

A :class:`ToolCallFormat` is chosen once per loaded model and creates one
:class:`ToolOutputParser` per request that sends tools. The parser takes text
as it is generated and returns what can be shown so far; a tool call appears
only once it is complete and valid, so nothing sent to a client has to be
taken back. ``docs/explanation/tool-calling.md`` describes the rules.
"""

from __future__ import annotations

import importlib
import json
import logging
import secrets
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .ipc import ToolCall, ToolCallDelta
from .tool_call_parsers import (
    AUTO_TOOL_CALL_PARSER,
    DISABLED_TOOL_CALL_PARSER,
    INKLING2_TOOL_CALL_PARSER,
    MLX_LM_TOOL_CALL_PARSERS,
    TOOL_CALL_PARSERS,
)

logger = logging.getLogger(__name__)

_PROMPT_TAIL_TOKENS = 8

_INKLING2_HEADER = "<|message_model|>"
_INKLING2_THINKING = "<|content_thinking|>"
_INKLING2_CALL = "<|content_invoke_tool_json|>"
_INKLING2_TEXT_MARKERS = (
    "<|content_text|>",
    "<|content_tool_error|>",
    "<|content_invoke_tool_text|>",
)
_INKLING2_END = "<|end_message|>"
_INKLING2_OPENERS = (_INKLING2_THINKING, _INKLING2_CALL, *_INKLING2_TEXT_MARKERS)


class ToolsNotSupportedError(ValueError):
    """A request offers tools to a model that cannot call them."""

    code = "TOOLS_NOT_SUPPORTED"

    def __init__(self, model: str) -> None:
        super().__init__(
            f"model '{model}' does not support tool calling: no tool-call parser "
            "is selected for it (see worker.tool_call_parser)"
        )


def request_error_code(error: Exception) -> str:
    """Return the IPC error code for a rejected request."""

    if isinstance(error, ToolsNotSupportedError):
        return error.code
    return "INVALID_REQUEST"


@dataclass(frozen=True)
class OutputDelta:
    """Output that became final since the previous step."""

    content: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCallDelta, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.content or self.reasoning or self.tool_calls)


class ToolOutputParser(ABC):
    """Split one request's generated text; subclasses know the model's markers.

    Whitespace around thinking and calls is formatting rather than content, so
    it is held back until non-blank text of the same kind follows.

    Attributes:
        name: Parser name reported in metrics.
        parse_failures: Calls returned as text because they could not be parsed.
        done: Whether generation should stop because no more calls are allowed.
    """

    def __init__(self, name: str, *, parallel: bool) -> None:
        self.name = name
        self.parse_failures = 0
        self.done = False
        self._parallel = parallel
        self._calls: list[ToolCall] = []
        self._content = ""
        self._reasoning = ""
        self._held_content = ""
        self._held_reasoning = ""
        self._step_content = ""
        self._step_reasoning = ""
        self._step_calls: list[ToolCallDelta] = []

    @property
    def content(self) -> str:
        """Return all visible text so far."""

        return self._content

    @property
    def reasoning(self) -> str | None:
        """Return all thinking text so far, or ``None`` without any."""

        return self._reasoning or None

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return the completed calls so far."""

        return tuple(self._calls)

    def push(self, text: str) -> OutputDelta:
        """Consume generated text."""

        if text and not self.done:
            self._feed(text)
        return self._take_step()

    def finish(self, *, turn_ended: bool) -> OutputDelta:
        """Consume the end of generation.

        Args:
            turn_ended: Whether the model ended its turn. When false, the text
                was cut short, so an open call may be missing arguments and
                must not be returned as a call even if it parses.
        """

        if not self.done:
            self._flush(turn_ended)
            if self._held_content and self._keeps_trailing_whitespace():
                self._emit_content(self._held_content)
        self._held_content = ""
        return self._take_step()

    def finish_reason(self, reason: str) -> str:
        """Return the reported finish reason for a generation that ended."""

        ended_by_model = reason == "stop" or (reason == "length" and self.done)
        return "tool_calls" if ended_by_model and self._calls else reason

    def response_fields(self) -> dict[str, Any]:
        """Return the response fields this parser owns."""

        return {
            "text": self.content,
            "reasoning": self.reasoning,
            "tool_calls": self.tool_calls,
            "tool_parser": self.name,
            "tool_parse_failures": self.parse_failures,
        }

    @abstractmethod
    def _feed(self, text: str) -> None:
        """Consume text, reporting output through the ``_add_*`` methods."""

    @abstractmethod
    def _flush(self, turn_ended: bool) -> None:
        """Report whatever is still buffered."""

    def _add_content(self, text: str) -> None:
        if self.done:
            return
        text = self._held_content + text
        visible = text.rstrip()
        self._held_content = text[len(visible) :]
        if not self._content and (self._reasoning or self._calls):
            visible = visible.lstrip()
        if visible:
            self._emit_content(visible)

    def _add_reasoning(self, text: str) -> None:
        if self.done:
            return
        text = self._held_reasoning + text
        visible = text.rstrip()
        self._held_reasoning = text[len(visible) :]
        if not self._reasoning:
            visible = visible.lstrip()
        self._reasoning += visible
        self._step_reasoning += visible

    def _add_call(self, body: str, raw: str, parse: Callable[[str], Any]) -> None:
        """Report one call block, or its raw text when it does not parse.

        Arguments may hold private data, so only the error type is logged.
        """

        if self.done:
            return
        try:
            calls = _normalize_calls(parse(body))
        except Exception as exc:
            self.parse_failures += 1
            logger.warning(
                "tool call from parser %s could not be parsed (%s)",
                self.name,
                type(exc).__name__,
            )
            self._add_content(raw)
            return
        for name, arguments in calls:
            call = ToolCall(
                id=f"call_{secrets.token_hex(12)}", name=name, arguments=arguments
            )
            self._step_calls.append(ToolCallDelta(index=len(self._calls), call=call))
            self._calls.append(call)
            if not self._parallel:
                self.done = True
                return

    def _keeps_trailing_whitespace(self) -> bool:
        return not self._calls and bool(self._content or not self._reasoning)

    def _emit_content(self, text: str) -> None:
        self._content += text
        self._step_content += text

    def _take_step(self) -> OutputDelta:
        delta = OutputDelta(
            content=self._step_content,
            reasoning=self._step_reasoning,
            tool_calls=tuple(self._step_calls),
        )
        self._step_content = ""
        self._step_reasoning = ""
        self._step_calls = []
        return delta


class DelimitedToolParser(ToolOutputParser):
    """Parse calls written between a fixed start and a fixed end marker."""

    def __init__(
        self,
        name: str,
        *,
        call_start: str,
        call_end: str,
        parse: Callable[[str], Any],
        think_start: str | None,
        think_end: str | None,
        starts_in_thinking: bool,
        parallel: bool,
    ) -> None:
        super().__init__(name, parallel=parallel)
        self._call_start = call_start
        self._call_end = call_end
        self._parse = parse
        self._think_start = think_start
        self._think_end = think_end
        self._state = "thinking" if starts_in_thinking else "text"
        self._buffer = ""
        self._searched = 0

    def _feed(self, text: str) -> None:
        self._buffer += text
        while not self.done and self._advance():
            pass

    def _flush(self, turn_ended: bool) -> None:
        buffer, self._buffer = self._buffer, ""
        if self._state == "thinking":
            self._add_reasoning(buffer)
        elif self._state == "text":
            self._add_content(buffer)
        else:
            raw = self._call_start + buffer
            if turn_ended:
                self._add_call(buffer, raw, self._parse)
            else:
                self._add_content(raw)

    def _advance(self) -> bool:
        """Consume up to the next marker; return whether one was found."""

        if self._state == "thinking":
            return self._advance_text(
                {str(self._think_end): "text"}, self._add_reasoning
            )
        if self._state == "text":
            markers = {self._call_start: "call"}
            if self._think_start is not None:
                markers[self._think_start] = "thinking"
            return self._advance_text(markers, self._add_content)
        end = self._buffer.find(self._call_end, self._searched)
        if end < 0:
            self._searched = max(0, len(self._buffer) - len(self._call_end) + 1)
            return False
        body = self._buffer[:end]
        self._buffer = self._buffer[end + len(self._call_end) :]
        self._state = "text"
        self._add_call(body, self._call_start + body + self._call_end, self._parse)
        return True

    def _advance_text(
        self, markers: dict[str, str], report: Callable[[str], None]
    ) -> bool:
        text, marker, rest = _split_at_marker(self._buffer, tuple(markers))
        self._buffer = rest
        if text:
            report(text)
        if marker is None:
            return False
        self._state = markers[marker]
        self._searched = 0
        return True


class Inkling2ToolParser(ToolOutputParser):
    """Parse Inkling 2 output, which frames every message it generates.

    A message is ``<|message_model|>NAME``, a content marker that says whether
    thinking, text or a call follows, the body, and ``<|end_message|>``. Text
    outside a frame is shown rather than held forever.
    """

    def __init__(self, *, parallel: bool) -> None:
        super().__init__(INKLING2_TOOL_CALL_PARSER, parallel=parallel)
        self._buffer = ""
        self._frame_start = ""
        self._opener: str | None = None
        self._searched = 0

    def _feed(self, text: str) -> None:
        self._buffer += text
        while not self.done and self._advance():
            pass

    def _flush(self, turn_ended: bool) -> None:
        buffer, self._buffer = self._buffer, ""
        if self._opener is None:
            self._add_content(buffer)
        elif self._opener == _INKLING2_CALL:
            raw = self._frame_start + buffer
            if turn_ended:
                self._add_call(buffer, raw, _parse_inkling2_call)
            else:
                self._add_content(raw)
        else:
            self._report(buffer)

    def _advance(self) -> bool:
        if self._opener is None:
            return self._advance_header()
        if self._opener == _INKLING2_CALL:
            return self._advance_call()
        text, marker, rest = _split_at_marker(self._buffer, (_INKLING2_END,))
        self._buffer = rest
        if text:
            self._report(text)
        if marker is None:
            return False
        self._opener = None
        return True

    def _advance_header(self) -> bool:
        if self._buffer.startswith(_INKLING2_HEADER):
            text, marker, rest = _split_at_marker(self._buffer, _INKLING2_OPENERS)
            if marker is None:
                return False
            self._frame_start = text + marker
        else:
            text, marker, rest = _split_at_marker(
                self._buffer, (_INKLING2_HEADER, *_INKLING2_OPENERS, _INKLING2_END)
            )
            if text:
                self._add_content(text)
            if marker == _INKLING2_HEADER:
                self._buffer = marker + rest
                return True
            self._frame_start = marker or ""
        self._buffer = rest
        if marker in _INKLING2_OPENERS:
            self._opener = marker
            self._searched = 0
        return marker is not None

    def _advance_call(self) -> bool:
        end = self._buffer.find(_INKLING2_END, self._searched)
        if end < 0:
            self._searched = max(0, len(self._buffer) - len(_INKLING2_END) + 1)
            return False
        body = self._buffer[:end]
        self._buffer = self._buffer[end + len(_INKLING2_END) :]
        self._opener = None
        self._add_call(
            body, self._frame_start + body + _INKLING2_END, _parse_inkling2_call
        )
        return True

    def _report(self, text: str) -> None:
        if self._opener == _INKLING2_THINKING:
            self._add_reasoning(text)
        else:
            self._add_content(text)


class ToolCallFormat(ABC):
    """The tool-call format of one loaded model."""

    name: str

    @abstractmethod
    def create_parser(
        self,
        tools: Sequence[dict[str, Any]],
        prompt_token_ids: Sequence[int],
        *,
        parallel: bool,
    ) -> ToolOutputParser:
        """Return a parser for one request.

        Args:
            tools: Tool definitions offered by the request.
            prompt_token_ids: Prompt the generated text continues.
            parallel: Whether the reply may contain more than one call.
        """


class DelimitedToolFormat(ToolCallFormat):
    """A format whose call bodies are decoded by an ``mlx_lm`` tool parser."""

    def __init__(self, name: str, tokenizer: Any) -> None:
        module = importlib.import_module(f"mlx_lm.tool_parsers.{name}")
        self.name = name
        self._call_start = str(module.tool_call_start)
        self._call_end = str(module.tool_call_end)
        self._parse_call = module.parse_tool_call
        self._tokenizer = tokenizer
        self._think_start: str | None = getattr(tokenizer, "think_start", None)
        self._think_end: str | None = getattr(tokenizer, "think_end", None)

    def create_parser(
        self,
        tools: Sequence[dict[str, Any]],
        prompt_token_ids: Sequence[int],
        *,
        parallel: bool,
    ) -> ToolOutputParser:
        offered = list(tools)
        return DelimitedToolParser(
            self.name,
            call_start=self._call_start,
            call_end=self._call_end,
            parse=lambda body: self._parse_call(body.strip(), offered),
            think_start=self._think_start,
            think_end=self._think_end,
            starts_in_thinking=self._opens_thinking(prompt_token_ids),
            parallel=parallel,
        )

    def _opens_thinking(self, prompt_token_ids: Sequence[int]) -> bool:
        """Return whether the prompt ends inside a thinking block."""

        if self._think_start is None:
            return False
        tail = self._tokenizer.decode(list(prompt_token_ids[-_PROMPT_TAIL_TOKENS:]))
        return str(tail).rstrip().endswith(self._think_start)


class Inkling2ToolFormat(ToolCallFormat):
    """The Inkling 2 message-frame format."""

    name = INKLING2_TOOL_CALL_PARSER

    def create_parser(
        self,
        tools: Sequence[dict[str, Any]],
        prompt_token_ids: Sequence[int],
        *,
        parallel: bool,
    ) -> ToolOutputParser:
        del tools, prompt_token_ids
        return Inkling2ToolParser(parallel=parallel)


def resolve_tool_format(
    tokenizer: Any, setting: str = AUTO_TOOL_CALL_PARSER
) -> ToolCallFormat | None:
    """Return the tool-call format for a loaded model, or ``None`` without one.

    Args:
        tokenizer: Tokenizer loaded through ``mlx_lm``.
        setting: ``auto`` to infer the format, ``none`` to disable tool calling,
            or a parser name.

    Raises:
        ValueError: If ``setting`` names no known parser.
    """

    if setting == DISABLED_TOOL_CALL_PARSER:
        return None
    name: str | None = setting
    if setting == AUTO_TOOL_CALL_PARSER:
        name = _infer_parser_name(tokenizer)
        if name is None:
            return None
    if name not in TOOL_CALL_PARSERS:
        raise ValueError(
            f"unknown tool_call_parser {name!r}; expected one of: "
            + ", ".join(TOOL_CALL_PARSERS)
        )
    if name == INKLING2_TOOL_CALL_PARSER:
        return Inkling2ToolFormat()
    return DelimitedToolFormat(name, tokenizer)


def _infer_parser_name(tokenizer: Any) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if isinstance(template, dict):
        template = template.get("tool_use") or template.get("default")
    if isinstance(template, str) and _INKLING2_CALL in template:
        return INKLING2_TOOL_CALL_PARSER
    module = getattr(getattr(tokenizer, "tool_parser", None), "__module__", "") or ""
    name = module.rpartition(".")[2]
    return name if name in MLX_LM_TOOL_CALL_PARSERS else None


def _parse_inkling2_call(body: str) -> dict[str, Any]:
    call = json.loads(body.strip())
    return {"name": call["name"], "arguments": call.get("args")}


def _normalize_calls(parsed: Any) -> list[tuple[str, str]]:
    """Validate parsed calls and encode their arguments as JSON text.

    Text with an unpaired surrogate cannot cross the IPC boundary, so it is
    rejected in names and escaped in arguments.
    """

    items = parsed if isinstance(parsed, list) else [parsed]
    if not items:
        raise ValueError("call block holds no calls")
    return [_normalize_call(item) for item in items]


def _normalize_call(item: Any) -> tuple[str, str]:
    if not isinstance(item, dict):
        raise ValueError("call is not an object")
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("call has no function name")
    name = name.strip()
    name.encode("utf-8")
    arguments = item.get("arguments")
    if arguments is None:
        arguments = {}
    elif isinstance(arguments, str):
        arguments = json.loads(arguments)
    if not isinstance(arguments, dict):
        raise ValueError("call arguments are not an object")
    encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    try:
        encoded.encode("utf-8")
    except UnicodeEncodeError:
        encoded = json.dumps(arguments, allow_nan=False)
    return name, encoded


def _split_at_marker(
    buffer: str, markers: Sequence[str]
) -> tuple[str, str | None, str]:
    """Split ``buffer`` at its first marker.

    Returns the text before the marker, the marker, and the text after it.
    Without a marker, the last element is the tail that may still become one.
    """

    found: tuple[int, str] | None = None
    for marker in markers:
        index = buffer.find(marker)
        if index >= 0 and (found is None or index < found[0]):
            found = (index, marker)
    if found is not None:
        index, marker = found
        return buffer[:index], marker, buffer[index + len(marker) :]
    held = _partial_marker_length(buffer, markers)
    return buffer[: len(buffer) - held], None, buffer[len(buffer) - held :]


def _partial_marker_length(buffer: str, markers: Sequence[str]) -> int:
    longest = min(len(buffer), max(len(marker) for marker in markers) - 1)
    for size in range(longest, 0, -1):
        if any(marker.startswith(buffer[-size:]) for marker in markers):
            return size
    return 0


__all__ = [
    "DelimitedToolFormat",
    "DelimitedToolParser",
    "Inkling2ToolFormat",
    "Inkling2ToolParser",
    "OutputDelta",
    "ToolCallFormat",
    "ToolOutputParser",
    "ToolsNotSupportedError",
    "request_error_code",
    "resolve_tool_format",
]
