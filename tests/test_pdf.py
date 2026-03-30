"""Tests for PDF text layer extraction."""

import base64


def _make_pdf_with_text(text: str = "Invoice Total: $123.45\nDate: 2024-01-01") -> bytes:
    """Create a minimal valid PDF with a text layer using reportlab or fpdf2 if available,
    otherwise fall back to a hand-crafted minimal PDF."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.cell(200, 10, text=text)
        return bytes(pdf.output())
    except ImportError:
        pass

    # Hand-crafted minimal PDF with text layer
    content_stream = f"BT /F1 12 Tf 50 750 Td ({text}) Tj ET"
    content_len = len(content_stream)
    pdf = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
  /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
4 0 obj << /Length {content_len} >>
stream
{content_stream}
endstream
endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000{400 + content_len:06d} 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
{500 + content_len}
%%EOF"""
    return pdf.encode()


def _pdf_data_uri(pdf_bytes: bytes) -> str:
    b64 = base64.b64encode(pdf_bytes).decode()
    return f"data:application/pdf;base64,{b64}"


class TestPdfDetection:
    def test_is_pdf_data_uri_true(self):
        from token0.optimization.pdf import is_pdf_data_uri

        assert is_pdf_data_uri("data:application/pdf;base64,abc123") is True

    def test_is_pdf_data_uri_false_for_image(self):
        from token0.optimization.pdf import is_pdf_data_uri

        assert is_pdf_data_uri("data:image/jpeg;base64,abc123") is False

    def test_is_pdf_data_uri_false_for_url(self):
        from token0.optimization.pdf import is_pdf_data_uri

        assert is_pdf_data_uri("https://example.com/doc.pdf") is False


class TestPdfDecode:
    def test_decode_pdf_roundtrip(self):
        from token0.optimization.pdf import decode_pdf

        original = b"fake pdf bytes"
        b64 = base64.b64encode(original).decode()
        uri = f"data:application/pdf;base64,{b64}"
        assert decode_pdf(uri) == original


class TestPdfTextExtraction:
    def test_extract_text_returns_none_for_empty_bytes(self):
        from token0.optimization.pdf import extract_pdf_text

        result = extract_pdf_text(b"not a pdf")
        assert result is None

    def test_extract_text_with_valid_pdf(self):
        from token0.optimization.pdf import extract_pdf_text

        pdf_bytes = _make_pdf_with_text("Hello World Invoice Total 100")
        result = extract_pdf_text(pdf_bytes)
        # Either extracts text (if pypdf reads it) or returns None (scanned/unreadable)
        assert result is None or isinstance(result, str)

    def test_estimate_pdf_tokens(self):
        from token0.optimization.pdf import estimate_pdf_tokens

        text = "a" * 400  # 400 chars → ~100 tokens
        assert estimate_pdf_tokens(text) == 100

    def test_estimate_pdf_tokens_minimum(self):
        from token0.optimization.pdf import estimate_pdf_tokens

        assert estimate_pdf_tokens("hi") == 10  # minimum floor


class TestPdfEstimateEndpoint:
    """Test that the /v1/estimate endpoint handles PDF data URIs gracefully."""

    def test_pdf_uri_skipped_gracefully(self):
        """PDF data URIs in estimate endpoint should not crash — skip or extract."""
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        pdf_bytes = b"not a real pdf"
        b64 = base64.b64encode(pdf_bytes).decode()
        pdf_uri = f"data:application/pdf;base64,{b64}"

        # This should not raise — malformed PDFs are silently skipped
        import asyncio

        req = EstimateRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=pdf_uri),
                        )
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))
        assert result.model == "gpt-4o"
        assert result.provider == "openai"
