"""Names accepted by the ``tool_call_parser`` setting, free of model imports.

``MLX_LM_TOOL_CALL_PARSERS`` are decoded by the module of the same name in
``mlx_lm.tool_parsers``.
"""

AUTO_TOOL_CALL_PARSER = "auto"
DISABLED_TOOL_CALL_PARSER = "none"
INKLING2_TOOL_CALL_PARSER = "inkling2"
MLX_LM_TOOL_CALL_PARSERS = ("json_tools", "qwen3_coder", "pythonic")
TOOL_CALL_PARSERS = (*MLX_LM_TOOL_CALL_PARSERS, INKLING2_TOOL_CALL_PARSER)
TOOL_CALL_PARSER_SETTINGS = (
    AUTO_TOOL_CALL_PARSER,
    DISABLED_TOOL_CALL_PARSER,
    *TOOL_CALL_PARSERS,
)
