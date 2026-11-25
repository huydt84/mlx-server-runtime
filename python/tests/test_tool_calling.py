from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from mlx_worker.chat_template import render_prompt_tokens, template_messages
from mlx_worker.ipc import (
    ChatCompletionRequest,
    ChatMessage,
    ImageContent,
    TextContent,
    ToolCall,
)
from mlx_worker.tool_calling import (
    ToolCallFormat,
    ToolOutputParser,
    ToolsNotSupportedError,
    request_error_code,
    resolve_tool_format,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
        },
    },
}
JSON_CALL = (
    '<tool_call>\n{"name": "weather", "arguments": {"city": "Hà Nội"}}\n</tool_call>'
)
XML_CALL = (
    "<tool_call>\n<function=weather>\n<parameter=city>\nHà Nội\n</parameter>\n"
    "<parameter=days>\n2\n</parameter>\n</function>\n</tool_call>"
)
PYTHONIC_CALL = '<|tool_call_start|>[weather(city="Hà Nội", days=2)]<|tool_call_end|>'
INKLING_CALL = (
    "<|message_model|>weather<|content_invoke_tool_json|>"
    '{"name": "weather", "args": {"city": "Hà Nội"}}<|end_message|>'
)
INKLING_REPLY = (
    "<|message_model|><|content_thinking|>hm <b><|end_message|>"
    "<|message_model|>user<|content_text|>On it.<|end_message|>"
    f"{INKLING_CALL}{INKLING_CALL}"
)
PAIR_CALL = '<|tool_call_start|>[a(x=1), b(y="z")]<|tool_call_end|>'


class _Tokenizer:
    """Tokenizer whose token ids are code points."""

    def __init__(self, think: bool = False) -> None:
        self.think_start = "<think>" if think else None
        self.think_end = "</think>" if think else None

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _parser(
    name: str,
    *,
    think: bool = False,
    prompt: str = "",
    parallel: bool = True,
    tools: tuple[dict[str, Any], ...] = (WEATHER_TOOL,),
) -> ToolOutputParser:
    tool_format = resolve_tool_format(_Tokenizer(think), name)
    assert tool_format is not None
    return tool_format.create_parser(
        tools, [ord(char) for char in prompt], parallel=parallel
    )


def _run(
    parser: ToolOutputParser, pieces: list[str], *, turn_ended: bool = True
) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    """Feed pieces and return what a streaming client would have assembled."""

    deltas = [parser.push(piece) for piece in pieces]
    deltas.append(parser.finish(turn_ended=turn_ended))
    content = "".join(delta.content for delta in deltas)
    reasoning = "".join(delta.reasoning for delta in deltas)
    calls = [item for delta in deltas for item in delta.tool_calls]
    assert [item.index for item in calls] == list(range(len(calls)))
    assert content == parser.content
    assert (reasoning or None) == parser.reasoning
    assert tuple(item.call for item in calls) == parser.tool_calls
    return (
        content,
        reasoning,
        [(item.call.name, json.loads(item.call.arguments)) for item in calls],
    )


CASES = [
    ("json_tools", False, "", "Hello there.", "Hello there.", "", []),
    (
        "json_tools",
        False,
        "",
        JSON_CALL,
        "",
        "",
        [("weather", {"city": "Hà Nội"})],
    ),
    (
        "json_tools",
        False,
        "",
        f"Let me look.\n\n{JSON_CALL}\n{JSON_CALL}\n",
        "Let me look.",
        "",
        [("weather", {"city": "Hà Nội"})] * 2,
    ),
    (
        "json_tools",
        True,
        "",
        f"<think>\nCall <tool_call>{{}}</tool_call>?\n</think>\n\n{JSON_CALL}",
        "",
        "Call <tool_call>{}</tool_call>?",
        [("weather", {"city": "Hà Nội"})],
    ),
    (
        "qwen3_coder",
        True,
        "assistant\n<think>\n",
        f"I should look.\n</think>\n\nOn it.\n\n{XML_CALL}",
        "On it.",
        "I should look.",
        [("weather", {"city": "Hà Nội", "days": 2})],
    ),
    (
        "qwen3_coder",
        True,
        "assistant\n<think>\n\n</think>\n\n",
        f"Hi.\n{XML_CALL}",
        "Hi.",
        "",
        [("weather", {"city": "Hà Nội", "days": 2})],
    ),
    (
        "qwen3_coder",
        True,
        "assistant\n<think>\n",
        "Never closed",
        "",
        "Never closed",
        [],
    ),
    (
        "pythonic",
        True,
        "",
        f"<think>hm</think>{PYTHONIC_CALL}Done.",
        "Done.",
        "hm",
        [("weather", {"city": "Hà Nội", "days": 2})],
    ),
    (
        "pythonic",
        False,
        "",
        PAIR_CALL,
        "",
        "",
        [("a", {"x": 1}), ("b", {"y": "z"})],
    ),
    (
        "inkling2",
        False,
        "",
        INKLING_REPLY,
        "On it.",
        "hm <b>",
        [("weather", {"city": "Hà Nội"})] * 2,
    ),
    ("inkling2", False, "", "bare text", "bare text", "", []),
    (
        "inkling2",
        False,
        "",
        "<|message_model|><|content_text|>unfinished",
        "unfinished",
        "",
        [],
    ),
]


@pytest.mark.parametrize(
    ("name", "think", "prompt", "text", "content", "reasoning", "calls"), CASES
)
def test_output_is_split_the_same_however_it_is_streamed(
    name: str,
    think: bool,
    prompt: str,
    text: str,
    content: str,
    reasoning: str,
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    expected = (content, reasoning, calls)

    for pieces in (
        [text],
        list(text),
        *([text[:cut], text[cut:]] for cut in range(1, len(text))),
    ):
        assert _run(_parser(name, think=think, prompt=prompt), pieces) == expected


def test_text_before_a_possible_marker_is_shown_at_once() -> None:
    parser = _parser("json_tools")

    assert parser.push("Hello <").content == "Hello"
    assert parser.push("b> x").content == " <b> x"
    assert not parser.push(" <tool_c")


def test_a_call_is_streamed_only_once_complete() -> None:
    parser = _parser("json_tools")

    deltas = [parser.push(char) for char in JSON_CALL]

    assert not any(deltas[:-1])
    (item,) = deltas[-1].tool_calls
    assert item.index == 0
    assert item.call.id.startswith("call_")
    assert item.call.arguments == '{"city": "Hà Nội"}'


def test_call_ids_are_unique() -> None:
    parser = _parser("json_tools")
    parser.push(JSON_CALL * 20)

    assert len({call.id for call in parser.tool_calls}) == 20


MALFORMED = [
    ("json_tools", "<tool_call>{not json}</tool_call>"),
    ("json_tools", "<tool_call></tool_call>"),
    ("json_tools", "<tool_call>[]</tool_call>"),
    ("json_tools", '<tool_call>{"arguments": {}}</tool_call>'),
    ("json_tools", '<tool_call>{"name": "", "arguments": {}}</tool_call>'),
    ("json_tools", '<tool_call>{"name": "f", "arguments": [1]}</tool_call>'),
    ("json_tools", '<tool_call>{"name": "f", "arguments": "x"}</tool_call>'),
    ("json_tools", '<tool_call>{"name": "f", "arguments": {"a": NaN}}</tool_call>'),
    ("json_tools", '<tool_call>{"name": "\\ud800", "arguments": {}}</tool_call>'),
    ("qwen3_coder", "<tool_call>\nweather()\n</tool_call>"),
    ("pythonic", "<|tool_call_start|>weather<|tool_call_end|>"),
    (
        "inkling2",
        "<|message_model|>f<|content_invoke_tool_json|>{oops<|end_message|>",
    ),
    (
        "inkling2",
        '<|message_model|>f<|content_invoke_tool_json|>{"args": {}}<|end_message|>',
    ),
]


@pytest.mark.parametrize(("name", "text"), MALFORMED)
def test_malformed_call_is_returned_as_text(name: str, text: str) -> None:
    for pieces in ([text], list(text)):
        parser = _parser(name)

        assert _run(parser, pieces) == (text, "", [])
        assert parser.parse_failures == 1
        assert parser.finish_reason("stop") == "stop"


def test_malformed_call_next_to_a_valid_one_keeps_both() -> None:
    parser = _parser("json_tools")
    bad = "<tool_call>{oops}</tool_call>"

    content, _, calls = _run(parser, [f"{bad}\n{JSON_CALL}"])

    assert content == bad
    assert calls == [("weather", {"city": "Hà Nội"})]
    assert parser.finish_reason("stop") == "tool_calls"
    assert parser.response_fields()["tool_parse_failures"] == 1


def test_malformed_call_log_leaves_out_the_arguments(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        _run(_parser("json_tools"), ["<tool_call>{secret-value}</tool_call>"])

    assert "json_tools" in caplog.text
    assert "secret-value" not in caplog.text


def test_unknown_function_names_are_returned() -> None:
    _, _, calls = _run(
        _parser("json_tools"), ['<tool_call>{"name": "other"}</tool_call>']
    )

    assert calls == [("other", {})]


def test_string_arguments_are_decoded() -> None:
    call = json.dumps({"name": "f", "arguments": '{"a": 1}'})

    _, _, calls = _run(_parser("json_tools"), [f"<tool_call>{call}</tool_call>"])

    assert calls == [("f", {"a": 1})]


def test_unpaired_surrogates_in_arguments_can_be_sent() -> None:
    parser = _parser("json_tools")

    parser.push('<tool_call>{"name": "f", "arguments": {"a": "\\ud800"}}</tool_call>')

    assert parser.tool_calls[0].arguments.encode("utf-8")
    assert json.loads(parser.tool_calls[0].arguments) == {"a": "\ud800"}


@pytest.mark.parametrize(
    ("name", "text", "call"),
    [
        ("json_tools", '<tool_call>{"name": "f", "arguments": {"a": 1}}', ("f", 1)),
        (
            "inkling2",
            '<|message_model|>f<|content_invoke_tool_json|>{"name": "f", "args": {"a": 1}}',
            ("f", 1),
        ),
        ("pythonic", "<|tool_call_start|>[f(a=1)]", ("f", 1)),
    ],
)
def test_call_without_an_end_marker_depends_on_how_the_turn_ended(
    name: str, text: str, call: tuple[str, int]
) -> None:
    ended = _parser(name)
    cut_short = _parser(name)

    assert _run(ended, [text]) == ("", "", [(call[0], {"a": call[1]})])
    assert ended.finish_reason("stop") == "tool_calls"
    assert _run(cut_short, [text], turn_ended=False) == (text, "", [])
    assert cut_short.finish_reason("length") == "length"
    assert cut_short.parse_failures == 0


def test_unparseable_call_at_the_end_of_the_turn_is_a_failure() -> None:
    parser = _parser("json_tools")

    assert _run(parser, ['<tool_call>{"name": "f"']) == (
        '<tool_call>{"name": "f"',
        "",
        [],
    )
    assert parser.parse_failures == 1


def test_cut_short_reply_keeps_its_finish_reason_next_to_calls() -> None:
    parser = _parser("json_tools")
    parser.push(JSON_CALL)

    assert parser.finish_reason("length") == "length"
    assert parser.finish_reason("cancelled") == "cancelled"


def test_partial_marker_at_the_end_is_text() -> None:
    assert _run(_parser("json_tools"), ["a <tool"]) == ("a <tool", "", [])


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("json_tools", f"Hi. {JSON_CALL} more {JSON_CALL}"),
        ("inkling2", f"{INKLING_CALL}{INKLING_CALL}"),
        ("pythonic", f"{PAIR_CALL} more"),
    ],
)
def test_only_the_first_call_is_kept_without_parallel_calls(
    name: str, text: str
) -> None:
    for pieces in ([text], list(text)):
        parser = _parser(name, parallel=False)
        content, _, calls = _run(parser, pieces)

        assert len(calls) == 1
        assert "more" not in content
        assert parser.done
        assert parser.finish_reason("stop") == "tool_calls"
        assert parser.finish_reason("length") == "tool_calls"


def test_malformed_first_call_does_not_end_a_single_call_reply() -> None:
    parser = _parser("json_tools", parallel=False)

    _, _, calls = _run(parser, [f"<tool_call>oops</tool_call>{JSON_CALL}"])

    assert calls == [("weather", {"city": "Hà Nội"})]


def test_whitespace_without_thinking_or_calls_is_kept() -> None:
    assert _run(_parser("json_tools"), ["\n a \n", "b \n"]) == ("\n a \nb \n", "", [])


def test_xml_arguments_without_a_schema_stay_text() -> None:
    tool = {"type": "function", "function": {"name": "weather"}}

    _, _, calls = _run(_parser("qwen3_coder", tools=(tool,)), [XML_CALL])

    assert calls == [("weather", {"city": "Hà Nội", "days": "2"})]


def _inferred(tokenizer: Any) -> ToolCallFormat | None:
    return resolve_tool_format(tokenizer, "auto")


@pytest.mark.parametrize("name", ["json_tools", "qwen3_coder", "pythonic"])
def test_format_is_inferred_from_the_mlx_lm_tool_parser(name: str) -> None:
    def parse(text: str, tools: Any) -> None: ...

    parse.__module__ = f"mlx_lm.tool_parsers.{name}"
    tool_format = _inferred(SimpleNamespace(tool_parser=parse, chat_template="{{ x }}"))

    assert tool_format is not None
    assert tool_format.name == name


@pytest.mark.parametrize(
    "template",
    [
        "a <|content_invoke_tool_json|> b",
        {"default": "x", "tool_use": "a <|content_invoke_tool_json|> b"},
    ],
)
def test_inkling_format_is_inferred_from_its_template(template: Any) -> None:
    tool_format = _inferred(SimpleNamespace(tool_parser=None, chat_template=template))

    assert tool_format is not None
    assert tool_format.name == "inkling2"


def test_models_outside_the_supported_families_have_no_format() -> None:
    def parse(text: str, tools: Any) -> None: ...

    parse.__module__ = "mlx_lm.tool_parsers.mistral"

    assert _inferred(SimpleNamespace(tool_parser=parse, chat_template="x")) is None
    assert _inferred(SimpleNamespace(tool_parser=None, chat_template=None)) is None
    assert _inferred(SimpleNamespace()) is None


def test_setting_overrides_inference() -> None:
    def parse(text: str, tools: Any) -> None: ...

    parse.__module__ = "mlx_lm.tool_parsers.json_tools"
    tokenizer = SimpleNamespace(tool_parser=parse)

    assert resolve_tool_format(tokenizer, "none") is None
    overridden = resolve_tool_format(tokenizer, "pythonic")
    assert overridden is not None
    assert overridden.name == "pythonic"
    with pytest.raises(ValueError, match="unknown tool_call_parser"):
        resolve_tool_format(tokenizer, "mistral")


def test_error_codes() -> None:
    error = ToolsNotSupportedError("gemma")

    assert "gemma" in str(error)
    assert request_error_code(error) == "TOOLS_NOT_SUPPORTED"
    assert request_error_code(ValueError("x")) == "INVALID_REQUEST"


def _request(
    messages: list[ChatMessage], tools: tuple[dict[str, Any], ...] = ()
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        request_id="req-1",
        model="m",
        messages=messages,
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        max_prompt_tokens=8,
        max_completion_tokens=8,
        max_total_tokens_per_request=16,
        tools=tools,
    )


def test_history_is_rendered_in_the_shape_templates_read() -> None:
    call = ToolCall(id="call_1", name="weather", arguments='{"b": 1, "a": 2}')
    messages = template_messages(
        _request(
            [
                ChatMessage(role="user", content=(TextContent("a"), TextContent("b"))),
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=(call,),
                    reasoning_content="why",
                ),
                ChatMessage(
                    role="tool", content="30", tool_call_id="call_1", name="weather"
                ),
            ]
        )
    )

    assert messages == [
        {"role": "user", "content": "ab"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "why",
            "thinking": "why",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": {"b": 1, "a": 2}},
                }
            ],
        },
        {
            "role": "tool",
            "content": "30",
            "tool_call_id": "call_1",
            "name": "weather",
        },
    ]
    arguments = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert list(arguments) == ["b", "a"]


def test_images_and_malformed_arguments_are_rejected() -> None:
    image = ImageContent(url="data:image/png;base64,x")
    with pytest.raises(ValueError, match="image content"):
        template_messages(_request([ChatMessage(role="user", content=(image,))]))
    for arguments in ("{", "[]"):
        call = ToolCall(id="call_1", name="f", arguments=arguments)
        message = ChatMessage(role="assistant", content="", tool_calls=(call,))
        with pytest.raises(ValueError, match="call_1"):
            template_messages(_request([message]))


class _TemplateTokenizer:
    def __init__(self, error: Exception | None = None) -> None:
        self.kwargs: list[dict[str, Any]] = []
        self._error = error

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[int]:
        del messages
        self.kwargs.append(kwargs)
        if self._error is not None:
            raise self._error
        return [1, 2]


def test_tools_reach_the_template_only_when_present() -> None:
    tokenizer = _TemplateTokenizer()
    message = ChatMessage(role="user", content="hi")
    strict = {
        "type": "function",
        "function": {**WEATHER_TOOL["function"], "strict": True},
    }

    render_prompt_tokens(tokenizer, _request([message]))
    render_prompt_tokens(tokenizer, _request([message], (strict,)))

    assert "tools" not in tokenizer.kwargs[0]
    assert tokenizer.kwargs[1]["tools"] == [WEATHER_TOOL]


def test_template_errors_reject_the_request() -> None:
    tokenizer = _TemplateTokenizer(RuntimeError("roles must alternate"))

    with pytest.raises(ValueError, match="roles must alternate"):
        render_prompt_tokens(
            tokenizer, _request([ChatMessage(role="user", content="hi")])
        )
