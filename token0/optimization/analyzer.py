"""Image analysis — determines dimensions, content type, and text density."""

import base64
import io
import math
import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageAnalysis:
    width: int
    height: int
    size_bytes: int
    format: str  # jpeg, png, webp, etc.
    text_density: float  # 0.0 to 1.0 — proportion of image that is text-like
    is_mostly_text: bool  # True if text_density > threshold
    estimated_tokens_openai_high: int
    estimated_tokens_openai_low: int
    estimated_tokens_anthropic: int
    has_transparency: bool
    dominant_colors: list[str] = field(default_factory=list)


def decode_image(image_input: str) -> tuple[bytes, Image.Image]:
    """Decode a base64 data URI or raw base64 string into bytes + PIL Image."""
    if image_input.startswith("data:"):
        # data:image/jpeg;base64,/9j/4AAQ...
        header, b64_data = image_input.split(",", 1)
    else:
        b64_data = image_input

    raw_bytes = base64.b64decode(b64_data)
    pil_image = Image.open(io.BytesIO(raw_bytes))
    return raw_bytes, pil_image


def estimate_openai_tokens(width: int, height: int, detail: str = "high") -> int:
    """Estimate token cost for OpenAI GPT-4o vision.

    Low detail: flat 85 tokens.
    High detail: scale to fit 2048px, then tile into 512px squares.
    Each tile = 170 tokens + 85 base.
    """
    if detail == "low":
        return 85

    # Scale to fit within 2048x2048
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)

    # Scale shortest side to 768px
    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)

    # Count 512x512 tiles
    tiles_x = math.ceil(width / 512)
    tiles_y = math.ceil(height / 512)
    total_tiles = tiles_x * tiles_y

    return 85 + (170 * total_tiles)


# Claude image limits per resolution tier: (max long edge px, max visual tokens).
# https://platform.claude.com/docs/en/build-with-claude/vision#evaluate-image-size
CLAUDE_STANDARD_LIMITS = (1568, 1568)
CLAUDE_HIGH_RES_LIMITS = (2576, 4784)  # Claude 4.7 and later

_CLAUDE_VERSION_RE = re.compile(r"claude-(?:opus|sonnet|haiku)-(\d+)(?:[-.](\d{1,2}))?(?!\d)")


def claude_image_limits(model: str | None = None) -> tuple[int, int]:
    """Edge and visual-token limits for a Claude model; standard tier if unknown."""
    if not model:
        return CLAUDE_STANDARD_LIMITS
    name = model.lower()
    if "claude-fable" in name or "claude-mythos" in name:
        return CLAUDE_HIGH_RES_LIMITS
    m = _CLAUDE_VERSION_RE.search(name)
    if m and (int(m.group(1)), int(m.group(2) or 0)) >= (4, 7):
        return CLAUDE_HIGH_RES_LIMITS
    return CLAUDE_STANDARD_LIMITS


def claude_visual_tokens(width: int, height: int) -> int:
    """One visual token per 28x28 patch."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def claude_resized_size(
    width: int, height: int, max_edge: int = 1568, max_tokens: int = 1568
) -> tuple[int, int]:
    """The size Claude downscales an image to, per Anthropic's reference implementation."""

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / 28) * 28 <= max_edge
            and math.ceil(h / 28) * 28 <= max_edge
            and claude_visual_tokens(w, h) <= max_tokens
        )

    if fits(width, height):
        return width, height
    if height > width:
        h, w = claude_resized_size(height, width, max_edge, max_tokens)
        return w, h

    aspect_ratio = width / height
    lo, hi = 1, width  # lo always fits; hi never fits
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect_ratio), 1)


def estimate_anthropic_tokens(width: int, height: int, model: str | None = None) -> int:
    """Estimate token cost for Claude Vision: ceil(w/28) x ceil(h/28) after downscaling."""
    w, h = claude_resized_size(width, height, *claude_image_limits(model))
    return claude_visual_tokens(w, h)


def detect_text_density(pil_image: Image.Image) -> float:
    """Estimate what fraction of the image contains text using multiple signals.

    Combines:
    1. Background uniformity — documents have large uniform regions (white/light background)
    2. Color variance — photos have high color variance, documents have low
    3. Horizontal line structure — text forms regular horizontal patterns
    4. Edge density profile — documents have moderate edges, photos have chaotic edges

    Returns 0.0 (definitely not text) to 1.0 (definitely text/document).
    """
    gray = np.array(pil_image.convert("L"))

    # Resize for speed
    target_width = 640
    scale = target_width / gray.shape[1] if gray.shape[1] > target_width else 1.0
    if scale < 1.0:
        new_h = int(gray.shape[0] * scale)
        gray = cv2.resize(gray, (target_width, new_h))

    total_area = gray.shape[0] * gray.shape[1]
    scores = []

    # Signal 1: Background uniformity
    # Documents have large areas of near-white or near-uniform background.
    # Photos have varied pixel intensities throughout.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    # What % of pixels are in the top 20% brightness (near-white)?
    bright_pixels = hist[200:].sum() / total_area
    # What % of pixels are in a single dominant 30-intensity-wide band?
    max_band_ratio = max(hist[i : i + 30].sum() for i in range(0, 226)) / total_area
    bg_score = min(1.0, bright_pixels * 1.2 + max_band_ratio * 0.5)
    scores.append(bg_score)

    # Signal 2: Color variance
    # Documents: low standard deviation (mostly white + black text)
    # Photos: high standard deviation (varied colors)
    color_arr = np.array(pil_image.convert("RGB"))
    if scale < 1.0:
        new_h = int(color_arr.shape[0] * scale)
        new_w = int(color_arr.shape[1] * scale)
        color_arr = cv2.resize(color_arr, (new_w, new_h))
    global_std = np.std(color_arr)
    # Low std = likely document (threshold: photos typically > 50, docs < 40)
    color_score = max(0.0, min(1.0, (55 - global_std) / 35))
    scores.append(color_score)

    # Signal 3: Horizontal structure regularity
    # Text creates regular horizontal patterns. Use horizontal projection profile.
    edges = cv2.Canny(gray, 80, 200)
    # Horizontal projection: sum edge pixels per row
    h_proj = edges.sum(axis=1).astype(float)
    if h_proj.max() > 0:
        h_proj_normalized = h_proj / h_proj.max()
        # Text documents have periodic peaks in horizontal projection
        # Count zero-crossings of (projection - mean) as proxy for line count
        mean_proj = h_proj_normalized.mean()
        crossings = np.sum(np.diff(np.sign(h_proj_normalized - mean_proj)) != 0)
        # Documents: many regular crossings (text lines). Photos: fewer, irregular.
        # A typical document page has 30-80 line-like crossings
        line_score = min(1.0, crossings / 60)
    else:
        line_score = 0.0
    scores.append(line_score)

    # Signal 4: Edge density — too many edges = photo, moderate = document
    edge_density = edges.sum() / (255.0 * total_area)
    # Sweet spot for documents: 0.02 - 0.15 edge density
    # Photos: > 0.15 (too many edges from textures, objects)
    # Blank: < 0.01
    if edge_density < 0.01:
        edge_score = 0.0  # too few edges — blank or near-blank
    elif edge_density <= 0.15:
        edge_score = min(1.0, edge_density / 0.08)  # peaks around 0.08
    else:
        edge_score = max(0.0, 1.0 - (edge_density - 0.15) / 0.15)  # penalize photo-like density
    scores.append(edge_score)

    # Combine signals with weights
    # Background uniformity and color variance are strongest indicators
    weights = [0.30, 0.35, 0.15, 0.20]
    combined = sum(s * w for s, w in zip(scores, weights))

    return min(1.0, max(0.0, combined))


def analyze_image(image_input: str) -> tuple[ImageAnalysis, bytes, Image.Image]:
    """Full analysis of an image from base64 input.

    Returns (analysis, raw_bytes, pil_image) so downstream steps
    don't need to re-decode.
    """
    raw_bytes, pil_image = decode_image(image_input)
    width, height = pil_image.size

    text_density = detect_text_density(pil_image)

    from token0.config import settings

    analysis = ImageAnalysis(
        width=width,
        height=height,
        size_bytes=len(raw_bytes),
        format=pil_image.format.lower() if pil_image.format else "unknown",
        text_density=text_density,
        is_mostly_text=text_density > settings.text_density_threshold,
        estimated_tokens_openai_high=estimate_openai_tokens(width, height, "high"),
        estimated_tokens_openai_low=estimate_openai_tokens(width, height, "low"),
        estimated_tokens_anthropic=estimate_anthropic_tokens(width, height),
        has_transparency=pil_image.mode in ("RGBA", "LA", "PA"),
    )

    return analysis, raw_bytes, pil_image
