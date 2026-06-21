"""Phase 9 VLM benchmark fixture set.

Provides synthetic image generators and checked-in image paths for VLM
benchmarking.  Synthetic images are small PPM files generated on-the-fly
so no external dependencies (PIL, OpenCV) are required.

Checked-in images under ``benchmarks/images/`` are used when available,
falling back to synthetic generation for fixture categories without a
matching checked-in image.

Each fixture pairs a prompt string with an image path so the benchmark
runner can pass the path to all backends consistently.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Callable


@dataclass(frozen=True)
class VlmFixture:
    """One VLM benchmark case: prompt text + image generator + metadata.

    Attributes:
        name: Short label for the benchmark case (e.g. ``"natural"``).
        prompt_text: User message text to accompany the image.
        image_path: Absolute path to the generated fixture image.
            Set by :func:`prepare_fixtures`.
        tags: Tuple of categorical tags for filtering or grouping.
    """

    name: str
    prompt_text: str
    image_path: Path | None = None
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageMetadata:
    """Basic local image metadata for VLM benchmark normalization."""

    path: Path
    format: str | None
    width: int | None
    height: int | None
    file_size_bytes: int

    @property
    def pixels(self) -> int | None:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height

    @property
    def megapixels(self) -> float | None:
        pixels = self.pixels
        if pixels is None:
            return None
        return pixels / 1_000_000.0


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------


def _gradient_ppm(dest: Path, width: int = 64, height: int = 64) -> Path:
    """Write a small vertical-gradient PPM image to *dest*.

    The gradient goes from red at the top to blue at the bottom,
    simulating a natural-sky colour transition at tiny scale.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            r = int(255 * (1 - y / height))
            g = int(128 * (1 - y / height))
            b = int(255 * (y / height))
            row = " ".join(f"{r} {g} {b}" for _ in range(width))
            f.write(row + "\n")
    return dest


def _chart_pattern_ppm(dest: Path, width: int = 64, height: int = 64) -> Path:
    """Write a small synthetic chart-like PPM image to *dest*.

    Draws horizontal bars at varying heights to simulate a simple
    bar chart.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        bar_heights = [height // 4, height // 2, 3 * height // 4, height // 3]
        for y in range(height):
            row_parts: list[str] = []
            for x in range(width):
                # Map x to one of 4 bar regions
                bar_idx = min(x * 4 // width, 3)
                if y >= height - bar_heights[bar_idx]:
                    # Bar region — white
                    row_parts.append("255 255 255")
                elif y < height // 8 and x % 4 == 0:
                    # Grid line — light grey
                    row_parts.append("200 200 200")
                else:
                    # Background — dark grey
                    row_parts.append("40 40 40")
            f.write(" ".join(row_parts) + "\n")
    return dest


def _text_pattern_ppm(dest: Path, width: int = 64, height: int = 64) -> Path:
    """Write a small synthetic OCR-like PPM image to *dest*.

    Draws horizontal stripe patterns that vaguely resemble lines of
    text (no actual character rendering — purely for consistent
    pre-processing measurement).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for y in range(height):
            row_parts: list[str] = []
            for x in range(width):
                text_line = (y // 8) % 2 == 0
                char_stroke = (x % 6) < 3 and text_line
                if char_stroke:
                    # Text foreground — off-white
                    row_parts.append("220 220 220")
                elif text_line:
                    # Line background — dark
                    row_parts.append("20 20 20")
                else:
                    # Empty space — black
                    row_parts.append("0 0 0")
            f.write(" ".join(row_parts) + "\n")
    return dest


# ---------------------------------------------------------------------------
# Checked-in images
# ---------------------------------------------------------------------------

_CHECKED_IN_IMAGES_DIR = Path(__file__).resolve().parent / "images"
"""Absolute path to the ``benchmarks/images/`` directory with checked-in images."""


# ---------------------------------------------------------------------------
# Fixture definitions
# ---------------------------------------------------------------------------

_NATURAL_IMAGE_PROMPT = (
    "Describe this natural scene in detail. What colours and shapes do you see?"
)
_CHART_SCREENSHOT_PROMPT = (
    "Describe the data or structure shown in this chart or table. "
    "What trends or patterns can you identify?"
)
_OCR_PROMPT = "Read any text displayed in the image and describe what you see. What does the text say?"


def _build_fixture(
    name: str,
    prompt_text: str,
    generator: Callable[[Path], Path],
    image_dir: Path,
    *,
    tags: tuple[str, ...] = (),
) -> VlmFixture:
    """Create a single VlmFixture with a generated image."""
    image_path = generator(image_dir / f"vlm_fixture_{name}.ppm")
    return VlmFixture(
        name=name,
        prompt_text=prompt_text,
        image_path=image_path,
        tags=tags,
    )


def _copy_checked_in_images(image_dir: Path) -> dict[str, Path]:
    """Copy checked-in images to *image_dir* and return name→path map.

    Returns ``{stem: dest_path}`` for every file under the checked-in
    images directory.  Supported extensions: ``.png``, ``.jpg``,
    ``.jpeg``, ``.webp``.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    if not _CHECKED_IN_IMAGES_DIR.is_dir():
        return result
    for src in sorted(_CHECKED_IN_IMAGES_DIR.iterdir()):
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            dest = image_dir / src.name
            shutil.copy2(str(src), str(dest))
            result[src.stem] = dest
    return result


def collect_image_metadata(path: Path) -> ImageMetadata:
    """Return lightweight metadata for supported fixture image formats."""

    file_size_bytes = path.stat().st_size
    suffix = path.suffix.lower()

    width: int | None = None
    height: int | None = None
    image_format: str | None = None

    if suffix == ".png":
        image_format = "png"
        width, height = _read_png_size(path)
    elif suffix in {".jpg", ".jpeg"}:
        image_format = "jpeg"
        width, height = _read_jpeg_size(path)
    elif suffix == ".webp":
        image_format = "webp"
        width, height = _read_webp_size(path)
    elif suffix == ".ppm":
        image_format = "ppm"
        width, height = _read_ppm_size(path)

    return ImageMetadata(
        path=path,
        format=image_format,
        width=width,
        height=height,
        file_size_bytes=file_size_bytes,
    )


def collect_many_image_metadata(
    paths: list[Path] | tuple[Path, ...],
) -> tuple[ImageMetadata, ...]:
    """Return metadata for each image path in order."""

    return tuple(collect_image_metadata(path) for path in paths)


def _read_png_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return (None, None)
    return struct.unpack(">II", header[16:24])


def _read_ppm_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as f:
        tokens: list[bytes] = []
        while len(tokens) < 4:
            line = f.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped or stripped.startswith(b"#"):
                continue
            tokens.extend(stripped.split())
    if len(tokens) < 4 or tokens[0] not in {b"P3", b"P6"}:
        return (None, None)
    return (int(tokens[1]), int(tokens[2]))


def _read_jpeg_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            return (None, None)
        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                return (None, None)
            if marker_prefix != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker:
                return (None, None)
            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9}:
                continue
            segment_length_bytes = f.read(2)
            if len(segment_length_bytes) != 2:
                return (None, None)
            segment_length = struct.unpack(">H", segment_length_bytes)[0]
            if segment_length < 2:
                return (None, None)
            if marker_value in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                data = f.read(5)
                if len(data) != 5:
                    return (None, None)
                height, width = struct.unpack(">HH", data[1:5])
                return (width, height)
            f.seek(segment_length - 2, 1)


def _read_webp_size(path: Path) -> tuple[int | None, int | None]:
    with path.open("rb") as f:
        header = f.read(30)
    if len(header) < 16 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        return (None, None)
    chunk = header[12:16]
    if chunk == b"VP8X" and len(header) >= 30:
        width = 1 + int.from_bytes(header[24:27], "little")
        height = 1 + int.from_bytes(header[27:30], "little")
        return (width, height)
    if chunk == b"VP8L" and len(header) >= 25:
        b0, b1, b2, b3 = header[21:25]
        width = 1 + (((b1 & 0x3F) << 8) | b0)
        height = 1 + (((b3 & 0x0F) << 10) | (b2 << 2) | ((b1 & 0xC0) >> 6))
        return (width, height)
    if chunk == b"VP8 " and len(header) >= 30:
        return (
            int.from_bytes(header[26:28], "little"),
            int.from_bytes(header[28:30], "little"),
        )
    return (None, None)


def prepare_fixtures(image_dir: Path, use_checked_in: bool = True) -> list[VlmFixture]:
    """Generate the standard VLM benchmark fixture set under *image_dir*.

    When ``use_checked_in=True`` and checked-in images exist under
    ``benchmarks/images/``, those images are used instead of generating
    synthetic fixtures.

    Returns a list of :class:`VlmFixture` instances.  The synthetic
    fallback set has three entries (``natural``, ``chart``, ``ocr``).
    When checked-in images are available the list includes one fixture
    per checked-in image.

    The caller is responsible for cleaning up *image_dir* if desired.
    """
    image_dir.mkdir(parents=True, exist_ok=True)

    if use_checked_in:
        checked = _copy_checked_in_images(image_dir)
        if checked:
            fixtures: list[VlmFixture] = []
            for stem, path in checked.items():
                fixtures.append(
                    VlmFixture(
                        name=stem,
                        prompt_text=f"Describe this image ({stem}) in detail.",
                        image_path=path,
                        tags=("vlm", "checked-in"),
                    )
                )
            return fixtures

    # Fallback: synthetic PPM fixtures.
    return [
        _build_fixture(
            "natural",
            _NATURAL_IMAGE_PROMPT,
            _gradient_ppm,
            image_dir,
            tags=("vlm", "natural"),
        ),
        _build_fixture(
            "chart",
            _CHART_SCREENSHOT_PROMPT,
            _chart_pattern_ppm,
            image_dir,
            tags=("vlm", "chart"),
        ),
        _build_fixture(
            "ocr",
            _OCR_PROMPT,
            _text_pattern_ppm,
            image_dir,
            tags=("vlm", "ocr"),
        ),
    ]
