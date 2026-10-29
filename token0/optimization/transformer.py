"""Image transformer — applies the optimization plan to produce an optimized image or text."""

import base64
import io
import math

import easyocr
from PIL import Image

from token0.optimization.analyzer import ImageAnalysis
from token0.optimization.router import OptimizationPlan

# Lazy-loaded EasyOCR reader
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def transform_image(
    plan: OptimizationPlan,
    analysis: ImageAnalysis,
    raw_bytes: bytes,
    pil_image: Image.Image,
) -> dict:
    """Apply optimization plan and return the transformed content.

    Returns a dict with either:
    - {"type": "image", "base64": "...", "media_type": "image/jpeg"}
    - {"type": "text", "content": "extracted text..."}
    """
    # OCR Route — extract text, skip image entirely
    if plan.use_ocr_route:
        return _ocr_extract(raw_bytes)

    # Apply transforms to image
    img = pil_image.copy()

    # Resize
    if plan.resize and plan.target_width and plan.target_height:
        img = img.resize(
            (plan.target_width, plan.target_height),
            Image.LANCZOS,
        )

    # Convert to JPEG if needed
    if plan.recompress_jpeg:
        if img.mode in ("RGBA", "LA", "PA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=plan.jpeg_quality, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "type": "image",
            "base64": encoded,
            "media_type": "image/jpeg",
            "size_bytes": buf.tell(),
        }

    # Re-encode in original format (after resize)
    buf = io.BytesIO()
    fmt = pil_image.format or "JPEG"
    if img.mode in ("RGBA", "LA", "PA") and fmt.upper() == "JPEG":
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=plan.jpeg_quality if fmt.upper() == "JPEG" else None)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    media_type = f"image/{fmt.lower()}"
    if fmt.upper() == "JPEG":
        media_type = "image/jpeg"

    return {
        "type": "image",
        "base64": encoded,
        "media_type": media_type,
        "size_bytes": buf.tell(),
    }


def estimate_text_tokens(text: str) -> int:
    """Conservative token count for English text (~3.5 chars/token) plus prompt prefix."""
    return math.ceil(len(text) / 3.5) + 10


def resolve_ocr(
    plan: OptimizationPlan, raw_bytes: bytes, replan
) -> tuple[OptimizationPlan, dict | None]:
    """Run OCR for an OCR-route plan and keep it only if the text is actually cheaper.

    `replan()` must return the plan computed with allow_ocr=False. Returns the plan to
    use and, if OCR won, the extracted-text result (so callers don't OCR twice).
    """
    if not plan.use_ocr_route:
        return plan, None
    result = _ocr_extract(raw_bytes)
    text = result.get("content", "")
    fallback = replan()
    if not result.get("detected"):
        fallback.reasons.append("OCR found no text — sent image instead")
        return fallback, None
    ocr_tokens = estimate_text_tokens(text)
    if ocr_tokens >= fallback.estimated_tokens_after:
        image_tokens = fallback.estimated_tokens_after
        fallback.reasons.append(
            f"OCR text ~{ocr_tokens} tokens ≥ image {image_tokens} — sent image instead"
        )
        return fallback, None
    plan.estimated_tokens_after = ocr_tokens
    plan.reasons.append(f"OCR text ~{ocr_tokens} tokens (verified)")
    return plan, result


def _ocr_extract(raw_bytes: bytes) -> dict:
    """Extract text from image using EasyOCR."""
    reader = _get_ocr_reader()
    results = reader.readtext(raw_bytes, detail=0, paragraph=True)
    text = "\n".join(results) if results else "[No text detected in image]"
    return {
        "type": "text",
        "content": text,
        "detected": bool(results),
    }
