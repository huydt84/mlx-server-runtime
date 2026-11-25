"""Incremental text for tokenizers that decode the whole output every step."""

from __future__ import annotations

from typing import Any

_REPLACEMENT_CHARACTER = "�"


def next_delta(decoded: str, emitted: str, *, final: bool = False) -> str:
    """Return the part of the decoded text that has not been emitted yet.

    A character split across tokens decodes to trailing replacement characters
    until its last token arrives, so those are held back unless ``final``.

    Args:
        decoded: Decoded text of all tokens generated so far.
        emitted: Text returned by earlier calls, joined.
        final: Whether no more tokens follow.

    Returns:
        The text to append to ``emitted``, which may be empty.
    """

    stable = decoded if final else decoded.rstrip(_REPLACEMENT_CHARACTER)
    return stable[len(emitted) :]


class RedecodingDetokenizer:
    """Streaming detokenizer for tokenizers that do not provide one."""

    def __init__(self, tokenizer: Any) -> None:
        self.last_segment = ""
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._decoded = ""
        self._emitted = ""

    def add_token(self, token_id: int) -> None:
        """Decode one more token and set ``last_segment`` to its new text."""

        self._token_ids.append(token_id)
        self._decoded = str(
            self._tokenizer.decode(self._token_ids, skip_special_tokens=False)
        )
        self._advance(final=False)

    def finalize(self) -> None:
        """Set ``last_segment`` to the text held back so far."""

        self._advance(final=True)

    def _advance(self, *, final: bool) -> None:
        self.last_segment = next_delta(self._decoded, self._emitted, final=final)
        self._emitted += self.last_segment
