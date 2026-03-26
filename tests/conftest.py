"""Shared test fixtures."""

import base64
import io

import pytest
from PIL import Image, ImageDraw


def make_image(
    width: int = 800,
    height: int = 600,
    color: str = "red",
    fmt: str = "JPEG",
    mode: str = "RGB",
) -> tuple[bytes, str]:
    """Create a solid color test image. Returns (raw_bytes, base64_data_uri)."""
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and mode == "RGBA":
        img = img.convert("RGB")
    img.save(buf, format=fmt)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    if fmt.upper() == "JPEG":
        mime = "image/jpeg"
    data_uri = f"data:{mime};base64,{b64}"
    return raw, data_uri


def make_text_image(
    width: int = 800,
    height: int = 600,
    lines: int = 20,
    fmt: str = "PNG",
) -> tuple[bytes, str]:
    """Create a text-heavy image (simulating a document/screenshot).
    Draws horizontal lines of 'text-like' content using PIL.
    """
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)

    # Draw many horizontal lines of varying length to simulate text
    y = 20
    line_height = max(12, height // (lines + 2))
    for i in range(lines):
        # Alternate line widths to simulate paragraphs
        line_width = int(width * (0.6 + 0.35 * (i % 3 == 0)))
        x_start = 40
        # Draw a thin dark rectangle to simulate a line of text
        draw.rectangle([x_start, y, x_start + line_width, y + 8], fill="black")
        # Add some smaller blocks to simulate words
        word_x = x_start
        while word_x < x_start + line_width - 20:
            word_len = 20 + (i * 7 + word_x) % 40
            draw.rectangle([word_x, y + 1, word_x + word_len, y + 7], fill="black")
            word_x += word_len + 6
        y += line_height

    buf = io.BytesIO()
    img.save(buf, format=fmt)
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode("utf-8")
    mime = f"image/{fmt.lower()}"
    data_uri = f"data:{mime};base64,{b64}"
    return raw, data_uri


@pytest.fixture
def small_jpeg():
    """200x200 red JPEG image."""
    return make_image(200, 200, "red", "JPEG")


@pytest.fixture
def large_jpeg():
    """4000x3000 blue JPEG image."""
    return make_image(4000, 3000, "blue", "JPEG")


@pytest.fixture
def medium_png():
    """1024x1024 green PNG image."""
    return make_image(1024, 1024, "green", "PNG")


@pytest.fixture
def png_with_alpha():
    """800x600 RGBA PNG image."""
    return make_image(800, 600, "blue", "PNG", mode="RGBA")


@pytest.fixture
def text_heavy_png():
    """800x600 PNG that looks like a document page."""
    return make_text_image(800, 600, lines=25, fmt="PNG")


@pytest.fixture
def small_text_image():
    """400x300 text-heavy image."""
    return make_text_image(400, 300, lines=15, fmt="PNG")
