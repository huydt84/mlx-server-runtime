"""Load Metal kernel sources kept as ``.metal`` files under ``kernels/``.

Kernels live in real ``.metal`` files so they can be edited, composed with
``#include``, and compiled on their own (``xcrun -sdk macosx metal -c``).
``mx.fast.metal_kernel`` compiles from a source string and cannot resolve
local includes, so this module inlines each quoted ``#include "name.metal"``
once and marks every inlined file with ``#line`` directives: compiler errors
and GPU captures then name the ``.metal`` file and line, not the generated
string.
"""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path

KERNEL_DIR = Path(__file__).resolve().with_name("kernels")

_LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"\s*$')
_PRAGMA_ONCE = re.compile(r"^\s*#\s*pragma\s+once\s*$")


@cache
def metal_header(name: str) -> str:
    """Return a kernel file with its local includes inlined.

    Args:
        name: File name relative to the ``kernels`` directory, such as
            ``"paged_decode.metal"``.

    Returns:
        Source suitable for the ``header`` argument of
        ``mx.fast.metal_kernel``.

    Raises:
        FileNotFoundError: If the file or one of its local includes is missing.
        ValueError: If an include escapes the ``kernels`` directory.
    """

    return "".join(_expand(KERNEL_DIR / name, set()))


def _expand(path: Path, seen: set[Path]) -> list[str]:
    path = path.resolve()
    if not path.is_relative_to(KERNEL_DIR):
        raise ValueError(f"Metal include escapes the kernel directory: {path}")
    if path in seen:
        return []
    seen.add(path)
    label = path.relative_to(KERNEL_DIR).as_posix()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    parts = [f'#line 1 "{label}"\n']
    for number, line in enumerate(lines, start=1):
        include = _LOCAL_INCLUDE.match(line)
        if include:
            parts.extend(_expand(path.parent / include.group(1), seen))
            parts.append(f'#line {number + 1} "{label}"\n')
        elif _PRAGMA_ONCE.match(line):
            # Pragma once is meaningless in a single string and warns there.
            parts.append("\n")
        else:
            parts.append(line)
    if lines and not lines[-1].endswith("\n"):
        parts.append("\n")
    return parts
