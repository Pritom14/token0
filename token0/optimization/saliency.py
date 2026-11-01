"""Saliency-based ROI cropping — crops images to the region the prompt asks about.

Phase 1: Rule-based spatial keyword matching (zero ML deps).
Maps prompt keywords to crop boxes, as fractions of the image's *content box*: the area
that actually holds something, so on a half-empty page "the bottom" is where the writing
ends, not the blank paper below it. A crop that would keep almost none of the content is
never applied; the whole image is sent instead.

Examples:
  "What's the total on this invoice?"  → bottom 40% of the content
  "Read the header"                    → top 30% of the content
  "What's in the top-right corner?"   → top-right quadrant
  "What does the signature say?"       → bottom strip
"""

import math
import re
from dataclasses import dataclass

import numpy as np
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
        [
            "footer",
            "total",
            "amount due",
            "grand total",
            "subtotal",
            "bottom of",
            "bottom section",
            "signature",
            "sign",
            "terms",
            "footnote",
            "fine print",
        ],
        (0.0, 0.60, 1.0, 1.0),
    ),
    # Top-left quadrant
    (
        ["top left", "top-left", "upper left", "upper-left"],
        (0.0, 0.0, 0.55, 0.55),
    ),
    # Top-right quadrant
    (
        [
            "top right",
            "top-right",
            "upper right",
            "upper-right",
            "date",
            "invoice number",
            "reference number",
            "ref no",
            "order number",
        ],
        (0.45, 0.0, 1.0, 0.55),
    ),
    # Bottom-left quadrant
    (
        ["bottom left", "bottom-left", "lower left", "lower-left"],
        (0.0, 0.45, 0.55, 1.0),
    ),
    # Bottom-right quadrant
    (
        [
            "bottom right",
            "bottom-right",
            "lower right",
            "lower-right",
            "total amount",
            "balance due",
            "net total",
        ],
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
# A pixel is content when its grey level differs from the background by more than this
_INK_THRESHOLD = 40
# A crop must keep at least this share of the image's content pixels, or it isn't applied
_MIN_INK_SHARE = 0.05
# Margin kept around the content box, as a fraction of the image size
_CONTENT_MARGIN = 0.02
# Content is measured on a copy no larger than this (px, longest side)
_MASK_MAX_PX = 400


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
    no_crop = SaliencyResult(cropped=False, crop_box=None, matched_keyword=None, savings_pct=0.0)

    mask, scale = _content_mask(image)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return no_crop  # blank image: no region holds anything to read
    mx, my = int(_CONTENT_MARGIN * w), int(_CONTENT_MARGIN * h)
    cl = max(0, int(xs.min() / scale) - mx)
    ct = max(0, int(ys.min() / scale) - my)
    cr = min(w, math.ceil((xs.max() + 1) / scale) + mx)
    cb = min(h, math.ceil((ys.max() + 1) / scale) + my)
    cw, ch = cr - cl, cb - ct
    total_ink = int(mask.sum())

    for keywords, (fl, ft, fr, fb) in _REGION_RULES:
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", prompt_lower):
                left = cl + int(fl * cw)
                top = ct + int(ft * ch)
                right = cl + int(fr * cw)
                bottom = ct + int(fb * ch)

                crop_area = (right - left) * (bottom - top)
                original_area = w * h
                savings = 1.0 - (crop_area / original_area)

                if savings < _MIN_SAVINGS_RATIO:
                    continue

                # The region the prompt names is (almost) empty: send the whole image.
                sl, st = int(left * scale), int(top * scale)
                sr, sb = math.ceil(right * scale), math.ceil(bottom * scale)
                if mask[st:sb, sl:sr].sum() < _MIN_INK_SHARE * total_ink:
                    return no_crop

                return SaliencyResult(
                    cropped=True,
                    crop_box=(left, top, right, bottom),
                    matched_keyword=kw,
                    savings_pct=savings,
                )

    return no_crop


def _content_mask(image: Image.Image) -> tuple[np.ndarray, float]:
    """Content pixels on a downscaled greyscale copy, and the downscale factor.

    Content is anything clearly different from the background, taken as the median of the
    border pixels (the paper of a scan, the backdrop of a screenshot).
    """
    grey = image.convert("L")
    scale = min(1.0, _MASK_MAX_PX / max(grey.size))
    if scale < 1.0:
        grey = grey.resize((max(1, round(grey.width * scale)), max(1, round(grey.height * scale))))
    a = np.asarray(grey, dtype=np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return np.abs(a - int(np.median(border))) > _INK_THRESHOLD, scale


def apply_saliency_crop(image: Image.Image, result: SaliencyResult) -> Image.Image:
    """Crop the image to the detected ROI box."""
    if not result.cropped or result.crop_box is None:
        return image
    return image.crop(result.crop_box)
