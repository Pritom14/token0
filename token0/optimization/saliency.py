"""Saliency-based ROI cropping — crops images to the region the prompt asks about.

Phase 1: Rule-based spatial keyword matching (zero ML deps).
Maps prompt keywords to crop boxes (fractions of image dimensions).

Examples:
  "What's the total on this invoice?"  → bottom 40% of image
  "Read the header"                    → top 25% of image
  "What's in the top-right corner?"   → top-right quadrant
  "What does the signature say?"       → bottom-right quadrant
"""

import re
from dataclasses import dataclass

from PIL import Image

# ---------------------------------------------------------------------------
# Spatial keyword → crop box mapping
# crop_box = (left, top, right, bottom) as fractions of (width, height)
# ---------------------------------------------------------------------------

_REGION_RULES: list[tuple[list[str], tuple[float, float, float, float]]] = [
    # Full top strip
    (
        ["header", "title", "heading", "logo", "top of", "top section", "letterhead", "subject"],
        (0.0, 0.0, 1.0, 0.30),
    ),
    # Full bottom strip
    (
        ["footer", "total", "amount due", "grand total", "subtotal", "bottom of",
         "bottom section", "signature", "sign", "terms", "footnote", "fine print"],
        (0.0, 0.60, 1.0, 1.0),
    ),
    # Top-left quadrant
    (
        ["top left", "top-left", "upper left", "upper-left"],
        (0.0, 0.0, 0.55, 0.55),
    ),
    # Top-right quadrant
    (
        ["top right", "top-right", "upper right", "upper-right", "date", "invoice number",
         "reference number", "ref no", "order number"],
        (0.45, 0.0, 1.0, 0.55),
    ),
    # Bottom-left quadrant
    (
        ["bottom left", "bottom-left", "lower left", "lower-left"],
        (0.0, 0.45, 0.55, 1.0),
    ),
    # Bottom-right quadrant
    (
        ["bottom right", "bottom-right", "lower right", "lower-right", "total amount",
         "balance due", "net total"],
        (0.45, 0.45, 1.0, 1.0),
    ),
    # Center region
    (
        ["center", "centre", "middle", "central"],
        (0.2, 0.2, 0.8, 0.8),
    ),
    # Left half
    (
        ["left side", "left half", "left column", "left panel"],
        (0.0, 0.0, 0.55, 1.0),
    ),
    # Right half
    (
        ["right side", "right half", "right column", "right panel"],
        (0.45, 0.0, 1.0, 1.0),
    ),
]

# Minimum image size (px) to bother cropping — tiny images not worth it
_MIN_DIMENSION_PX = 200
# Minimum savings ratio to apply crop — skip if crop is >80% of original
_MIN_SAVINGS_RATIO = 0.20


@dataclass
class SaliencyResult:
    cropped: bool
    crop_box: tuple[int, int, int, int] | None  # pixel coords (left, top, right, bottom)
    matched_keyword: str | None
    savings_pct: float  # 0.0–1.0, fraction of pixels removed


def detect_roi(prompt: str, image: Image.Image) -> SaliencyResult:
    """Detect region of interest from prompt keywords.

    Returns a SaliencyResult. If no region detected or savings too small,
    cropped=False and the original image should be used.
    """
    if not prompt or image is None:
        return SaliencyResult(cropped=False, crop_box=None, matched_keyword=None, savings_pct=0.0)

    w, h = image.size
    if w < _MIN_DIMENSION_PX or h < _MIN_DIMENSION_PX:
        return SaliencyResult(cropped=False, crop_box=None, matched_keyword=None, savings_pct=0.0)

    prompt_lower = prompt.lower()

    for keywords, (fl, ft, fr, fb) in _REGION_RULES:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", prompt_lower):
                left = int(fl * w)
                top = int(ft * h)
                right = int(fr * w)
                bottom = int(fb * h)

                crop_area = (right - left) * (bottom - top)
                original_area = w * h
                savings = 1.0 - (crop_area / original_area)

                if savings < _MIN_SAVINGS_RATIO:
                    continue

                return SaliencyResult(
                    cropped=True,
                    crop_box=(left, top, right, bottom),
                    matched_keyword=kw,
                    savings_pct=savings,
                )

    return SaliencyResult(cropped=False, crop_box=None, matched_keyword=None, savings_pct=0.0)


def apply_saliency_crop(image: Image.Image, result: SaliencyResult) -> Image.Image:
    """Crop the image to the detected ROI box."""
    if not result.cropped or result.crop_box is None:
        return image
    return image.crop(result.crop_box)
