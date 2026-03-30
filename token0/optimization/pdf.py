"""PDF text layer extraction — route PDFs to text instead of vision tokens.

When a PDF has a text layer (the vast majority of typed/digital PDFs),
extracting it costs ~10-50x fewer tokens than rendering as an image.
Scanned PDFs (no text layer) fall back to passthrough.
"""

import base64
import io
import logging

logger = logging.getLogger("token0.pdf")


def is_pdf_data_uri(url: str) -> bool:
    return url.startswith("data:application/pdf;")


def decode_pdf(url: str) -> bytes:
    """Decode PDF from base64 data URI into raw bytes."""
    _, b64_data = url.split(",", 1)
    return base64.b64decode(b64_data)


def extract_pdf_text(pdf_bytes: bytes) -> str | None:
    """Extract text layer from PDF bytes using pypdf.

    Returns extracted text string, or None if no usable text layer found
    (e.g. scanned PDF — caller should fall back to passthrough).
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("pypdf not installed — PDF text extraction unavailable. pip install pypdf")
        return None

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())

        combined = "\n\n".join(pages_text)
        if len(combined) < 20:
            return None  # too little text — probably a scanned PDF
        return combined

    except Exception as e:
        logger.warning("PDF text extraction failed: %s", e)
        return None


def estimate_pdf_tokens(text: str) -> int:
    """Rough token estimate for extracted PDF text (4 chars ≈ 1 token)."""
    return max(10, len(text) // 4)
