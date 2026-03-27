"""Tests for src/optimization/transformer.py"""

import base64
import io

from PIL import Image

from tests.conftest import make_image, make_text_image
from token0.optimization.analyzer import ImageAnalysis, analyze_image
from token0.optimization.router import OptimizationPlan
from token0.optimization.transformer import transform_image


def _make_analysis_and_image(width=800, height=600, fmt="JPEG", mode="RGB", color="red"):
    """Helper to create analysis + raw bytes + PIL image for transformer tests."""
    _, data_uri = make_image(width, height, color, fmt, mode)
    return analyze_image(data_uri)


class TestResize:
    def test_resize_reduces_dimensions(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(4000, 3000)
        plan = OptimizationPlan(
            resize=True,
            target_width=1568,
            target_height=1176,
            reasons=["resize test"],
            estimated_tokens_before=5000,
            estimated_tokens_after=2000,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "image"
        assert "base64" in result

        # Decode and verify dimensions
        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (1568, 1176)

    def test_resize_output_is_valid_image(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(2000, 1500)
        plan = OptimizationPlan(
            resize=True,
            target_width=1000,
            target_height=750,
            reasons=["resize"],
            estimated_tokens_before=3000,
            estimated_tokens_after=1500,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)
        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (1000, 750)


class TestJPEGRecompression:
    def test_png_to_jpeg_conversion(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(800, 600, fmt="PNG")
        plan = OptimizationPlan(
            recompress_jpeg=True,
            jpeg_quality=85,
            reasons=["png → jpeg"],
            estimated_tokens_before=1000,
            estimated_tokens_after=1000,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "image"
        assert result["media_type"] == "image/jpeg"

        # Verify it's actually a JPEG
        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.format == "JPEG"

    def test_rgba_png_to_jpeg_drops_alpha(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(
            800, 600, fmt="PNG", mode="RGBA", color="blue"
        )
        plan = OptimizationPlan(
            recompress_jpeg=True,
            jpeg_quality=85,
            reasons=["rgba png → jpeg"],
            estimated_tokens_before=1000,
            estimated_tokens_after=1000,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.mode == "RGB"  # alpha removed

    def test_jpeg_recompression_reduces_size(self):
        # Use a noisy/complex image where JPEG compression is effective
        # Solid-color PNGs are already tiny, so use a photo-like image
        import numpy as np

        arr = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw_bytes = buf.getvalue()
        original_size = len(raw_bytes)

        analysis = ImageAnalysis(
            width=800,
            height=600,
            size_bytes=original_size,
            format="png",
            text_density=0.0,
            is_mostly_text=False,
            estimated_tokens_openai_high=1000,
            estimated_tokens_openai_low=85,
            estimated_tokens_anthropic=640,
            has_transparency=False,
        )

        plan = OptimizationPlan(
            recompress_jpeg=True,
            jpeg_quality=70,
            reasons=["recompress"],
            estimated_tokens_before=1000,
            estimated_tokens_after=1000,
        )

        result = transform_image(plan, analysis, raw_bytes, img)
        assert result["size_bytes"] < original_size


class TestResizeAndRecompress:
    def test_resize_plus_jpeg_conversion(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(4000, 3000, fmt="PNG")
        plan = OptimizationPlan(
            resize=True,
            target_width=1568,
            target_height=1176,
            recompress_jpeg=True,
            jpeg_quality=85,
            reasons=["resize", "png → jpeg"],
            estimated_tokens_before=5000,
            estimated_tokens_after=2000,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "image"
        assert result["media_type"] == "image/jpeg"

        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (1568, 1176)
        assert img.format == "JPEG"


class TestOCRRoute:
    def test_ocr_returns_text_type(self):
        _, data_uri = make_text_image(800, 600, lines=20)
        analysis, raw_bytes, pil_img = analyze_image(data_uri)

        plan = OptimizationPlan(
            use_ocr_route=True,
            reasons=["OCR route"],
            estimated_tokens_before=1500,
            estimated_tokens_after=200,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "text"
        assert "content" in result
        assert isinstance(result["content"], str)

    def test_ocr_on_blank_image_returns_placeholder(self):
        _, data_uri = make_image(800, 600, "white", "JPEG")
        analysis, raw_bytes, pil_img = analyze_image(data_uri)

        plan = OptimizationPlan(
            use_ocr_route=True,
            reasons=["OCR route"],
            estimated_tokens_before=1000,
            estimated_tokens_after=200,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "text"
        # Should have some content (even if "No text detected")
        assert len(result["content"]) > 0


class TestPassthrough:
    def test_no_optimization_returns_original(self):
        analysis, raw_bytes, pil_img = _make_analysis_and_image(800, 600)
        plan = OptimizationPlan(
            reasons=[],
            estimated_tokens_before=1000,
            estimated_tokens_after=1000,
        )

        result = transform_image(plan, analysis, raw_bytes, pil_img)

        assert result["type"] == "image"
        assert "base64" in result
        # Should still be a valid image
        decoded = base64.b64decode(result["base64"])
        img = Image.open(io.BytesIO(decoded))
        assert img.size == (800, 600)
