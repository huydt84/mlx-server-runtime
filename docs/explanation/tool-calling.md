# Tool Calling

`POST /v1/chat/completions` accepts OpenAI tool definitions, shows them to the
model, and returns the calls the model writes. This page records how the
feature behaves and why. [HTTP API](../reference/api.md#tool-calling) lists the
fields.

## Who does what

1. The gateway validates the tool fields and the history, and forwards the tool
   definitions to the worker as the JSON text the client sent. Text keeps the
   key order, which chat templates render into the prompt.
2. The worker passes the tools and the history to the checkpoint's chat
   template ([chat_template.py](../../python/mlx_worker/chat_template.py)).
3. While the model generates, an output parser splits the text into visible
   content, thinking and calls
   ([tool_calling.py](../../python/mlx_worker/tool_calling.py)).
4. The gateway renders the OpenAI response and counts calls in the metrics.

Parsing happens in the worker because the format depends on the tokenizer and
because the worker can stop generating once no more calls are allowed.

Tool calling works on native-mlx and on every v1 text path (batch, continuous
batching, streaming). Requests to a VLM cannot use tools.

## Choosing the parser

One parser is chosen when the model loads. `worker.tool_call_parser`
(`mlx-air serve --tool-call-parser`) decides how:

| Setting | Effect |
|---------|--------|
| `auto` (default) | Infer the parser from the checkpoint |
| `none` | Reject requests with tools |
| parser name | Use that parser for a checkpoint that cannot be inferred |

An unknown name stops the worker at startup. native-mlx logs the choice as
`native-mlx tool_call_parser=...`.

| Parser | Models | Format |
|--------|--------|--------|
| `json_tools` | Qwen2.5, Qwen3 | `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` |
| `qwen3_coder` | Qwen3.5, Bonsai | `<tool_call><function=...><parameter=...>` XML |
| `pythonic` | LFM2.5 | `<\|tool_call_start\|>[f(a=1)]<\|tool_call_end\|>` |
| `inkling2` | Inkling 2 | message frames with JSON calls |

The first three use the parser of the same name in `mlx_lm.tool_parsers` for
the call body, inferred from what `mlx_lm` selects for the tokenizer. `inkling2`
is inferred from the checkpoint's chat template; it is tested against recorded
output shapes and has not been run on a real model.

Any other model has no parser. A request with tools then fails with HTTP 400
`TOOLS_NOT_SUPPORTED`, because silently dropping the tools would make the model
answer as if they had never been offered. `tool_choice: "none"` and an empty
`tools` list still work.

## Output rules

These apply when the request offers tools. Requests without tools, and requests
with `tool_choice: "none"`, return the generated text unchanged.

- **Thinking** goes to `reasoning_content`, in messages and in stream deltas. A
  call marker inside thinking is not a call. `reasoning_content` on an assistant
  message in the history is passed back to the template.
- **Whitespace** around thinking and calls is dropped.
- **A complete call** is returned with a generated `call_...` id. The name and
  arguments are not checked against the offered tools. The finish reason becomes
  `tool_calls` when the model ended its turn or a stop string ended it.
- **Streaming** sends one delta per complete call, with index, id, name and the
  full arguments. A call is sent only once it is known to be valid, so nothing
  has to be taken back.
- **A malformed call** is returned as text in `content`, markers included. The
  finish reason is `stop` when no valid call was made. The worker logs the
  parser name and the error type, never the arguments.
- **An unfinished call** counts as a call when the model ended its turn and the
  body parses, because some models end the turn instead of writing the end
  marker. When `max_tokens` or a stop string cut it short, it is returned as
  text and the finish reason stays `length` or `stop`: a cut call can parse and
  still miss arguments.
- **Cancellation** never emits a call that was still being written.
- **`parallel_tool_calls: false`** returns the first valid call and ends
  generation there.
- **`stop` strings** match all generated text on native-mlx, including thinking
  and calls. v1 does not enforce stop strings.

## Not supported

| Field | Behavior | Reason |
|-------|----------|--------|
| `tool_choice: "required"` or a named function | HTTP 400 | Needs constrained decoding to be honest |
| `functions`, `function_call` | HTTP 400 | Deprecated; use `tools` and `tool_choice` |
| `strict` on a function | Accepted and dropped | Needs constrained decoding |
| `reasoning_effort` | Accepted and dropped | No template mapping yet |
| Tools on a VLM | HTTP 400 `TOOLS_NOT_SUPPORTED` | No parser runs on VLM output |

The `pythonic` format cannot express function names that are not Python
identifiers (dots are allowed). A call to a name such as `get-weather` is
returned as text.

## Conversation history

A `tool` message needs a non-empty `tool_call_id`. An assistant call needs `id`,
`type` and a function name, and its `arguments` must be JSON text of an object,
which templates receive as an object. Ids are not checked against each other.
The Inkling template uses the id, or `name` on the tool message, to label a
result with its function; a result it cannot link gets an unnamed header.

## Adding a parser

- When `mlx_lm.tool_parsers` has a module for the format and calls have fixed
  start and end markers, add its name to `MLX_LM_TOOL_CALL_PARSERS` in
  `tool_call_parsers.py`.
- Otherwise subclass `ToolOutputParser` and `ToolCallFormat`, and extend
  `TOOL_CALL_PARSERS`, `resolve_tool_format` and, for `auto`,
  `_infer_parser_name`.
- Add cases to `python/tests/test_tool_calling.py`, which feeds every case split
  at every character, and a checkpoint to
  `mlx-host-validation/scripts/native_tool_calling.sh`.
