"""Tests for src/optimization/router.py"""

from token0.optimization.analyzer import ImageAnalysis
from token0.optimization.router import (
    get_provider_from_model,
    plan_optimization,
)


def _make_analysis(
    width=800,
    height=600,
    text_density=0.1,
    is_mostly_text=False,
    fmt="jpeg",
    has_transparency=False,
) -> ImageAnalysis:
    """Helper to create an ImageAnalysis with sensible defaults."""
    from token0.optimization.analyzer import estimate_anthropic_tokens, estimate_openai_tokens

    return ImageAnalysis(
        width=width,
        height=height,
        size_bytes=width * height * 3,  # rough estimate
        format=fmt,
        text_density=text_density,
        is_mostly_text=is_mostly_text,
        estimated_tokens_openai_high=estimate_openai_tokens(width, height, "high"),
        estimated_tokens_openai_low=estimate_openai_tokens(width, height, "low"),
        estimated_tokens_anthropic=estimate_anthropic_tokens(width, height),
        has_transparency=has_transparency,
    )


class TestGetProviderFromModel:
    def test_openai_models(self):
        assert get_provider_from_model("gpt-4o") == "openai"
        assert get_provider_from_model("gpt-4o-mini") == "openai"
        assert get_provider_from_model("gpt-4.1") == "openai"
        assert get_provider_from_model("o1-preview") == "openai"
        assert get_provider_from_model("o3-mini") == "openai"
        assert get_provider_from_model("o4-mini") == "openai"

    def test_anthropic_models(self):
        assert get_provider_from_model("claude-sonnet-4-6") == "anthropic"
        assert get_provider_from_model("claude-opus-4-6") == "anthropic"
        assert get_provider_from_model("claude-haiku-4-5-20251001") == "anthropic"

    def test_google_models(self):
        assert get_provider_from_model("gemini-2.5-flash") == "google"
        assert get_provider_from_model("gemini-2.5-pro") == "google"

    def test_unknown_defaults_to_openai(self):
        assert get_provider_from_model("some-unknown-model") == "openai"


class TestOCRRoute:
    def test_text_heavy_image_gets_ocr_route(self):
        analysis = _make_analysis(text_density=0.75, is_mostly_text=True)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.use_ocr_route is True
        assert plan.resize is False
        assert any("OCR" in r for r in plan.reasons)

    def test_photo_does_not_get_ocr_route(self):
        analysis = _make_analysis(text_density=0.1, is_mostly_text=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.use_ocr_route is False

    def test_borderline_text_density_below_threshold(self):
        analysis = _make_analysis(text_density=0.55, is_mostly_text=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.use_ocr_route is False

    def test_ocr_route_estimates_low_tokens(self):
        analysis = _make_analysis(width=1920, height=1080, text_density=0.8, is_mostly_text=True)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.use_ocr_route is True
        assert plan.estimated_tokens_after == 200  # rough text estimate
        assert plan.estimated_tokens_before > plan.estimated_tokens_after


class TestResize:
    def test_oversized_image_gets_resized_anthropic(self):
        analysis = _make_analysis(width=4000, height=3000)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is True
        assert plan.target_width is not None
        assert plan.target_height is not None
        assert max(plan.target_width, plan.target_height) <= 1568

    def test_oversized_image_gets_resized_openai(self):
        analysis = _make_analysis(width=4000, height=3000)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.resize is True
        assert max(plan.target_width, plan.target_height) <= 2048

    def test_small_image_not_resized(self):
        analysis = _make_analysis(width=400, height=300)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is False

    def test_exact_max_not_resized(self):
        analysis = _make_analysis(width=1568, height=1000)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is False

    def test_resize_maintains_aspect_ratio(self):
        analysis = _make_analysis(width=4000, height=2000)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is True
        ratio_original = 4000 / 2000
        ratio_resized = plan.target_width / plan.target_height
        assert abs(ratio_original - ratio_resized) < 0.01

    def test_resize_reduces_token_estimate_anthropic(self):
        # Anthropic auto-downscales to 1568px + 1.15MP cap internally too,
        # so our pre-resize saves bandwidth/latency but token estimate matches.
        # The real token savings come from OCR routing and detail mode.
        analysis = _make_analysis(width=4000, height=3000)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is True
        assert plan.estimated_tokens_after <= plan.estimated_tokens_before

    def test_resize_openai_may_not_reduce_tokens(self):
        # OpenAI internally downscales too, so pre-resize may not change token count
        # but it still reduces payload size and latency
        analysis = _make_analysis(width=4000, height=3000)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.resize is True
        assert plan.estimated_tokens_after <= plan.estimated_tokens_before


class TestDetailMode:
    def test_small_openai_image_gets_low_detail(self):
        analysis = _make_analysis(width=400, height=400)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.force_detail_low is True
        assert plan.estimated_tokens_after == 85

    def test_large_openai_image_stays_high_detail(self):
        analysis = _make_analysis(width=1024, height=1024)
        plan = plan_optimization(analysis, "gpt-4o")

        # Large image without detail_override stays high
        assert plan.force_detail_low is False

    def test_detail_override_low(self):
        analysis = _make_analysis(width=1024, height=1024)
        plan = plan_optimization(analysis, "gpt-4o", detail_override="low")

        assert plan.force_detail_low is True

    def test_detail_override_high_prevents_low(self):
        analysis = _make_analysis(width=400, height=400)
        plan = plan_optimization(analysis, "gpt-4o", detail_override="high")

        assert plan.force_detail_low is False

    def test_anthropic_ignores_detail_mode(self):
        # Anthropic doesn't have detail modes
        analysis = _make_analysis(width=400, height=400)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.force_detail_low is False


class TestJPEGRecompression:
    def test_png_without_alpha_gets_recompressed(self):
        analysis = _make_analysis(fmt="png", has_transparency=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.recompress_jpeg is True
        assert any("jpeg" in r.lower() for r in plan.reasons)

    def test_png_with_alpha_not_recompressed(self):
        analysis = _make_analysis(fmt="png", has_transparency=True)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.recompress_jpeg is False

    def test_jpeg_not_recompressed(self):
        analysis = _make_analysis(fmt="jpeg", has_transparency=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.recompress_jpeg is False

    def test_bmp_gets_recompressed(self):
        analysis = _make_analysis(fmt="bmp", has_transparency=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.recompress_jpeg is True

    def test_tiff_gets_recompressed(self):
        analysis = _make_analysis(fmt="tiff", has_transparency=False)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.recompress_jpeg is True


class TestCombinedOptimizations:
    def test_large_png_gets_resize_and_recompress(self):
        analysis = _make_analysis(width=4000, height=3000, fmt="png", has_transparency=False)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.resize is True
        assert plan.recompress_jpeg is True
        assert len(plan.reasons) >= 2

    def test_text_image_skips_other_optimizations(self):
        # OCR route should short-circuit — no resize or recompress needed
        analysis = _make_analysis(
            width=4000, height=3000, text_density=0.8, is_mostly_text=True, fmt="png"
        )
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.use_ocr_route is True
        assert plan.resize is False
        assert plan.recompress_jpeg is False

    def test_small_jpeg_photo_no_optimization(self):
        analysis = _make_analysis(width=600, height=400, text_density=0.05, fmt="jpeg")
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.use_ocr_route is False
        assert plan.resize is False
        assert plan.recompress_jpeg is False
        assert plan.force_detail_low is False


class TestTokenEstimates:
    def test_savings_positive_for_ocr_route(self):
        # OCR route always saves tokens — text extraction is ~200 tokens vs 1000+ for vision
        analysis = _make_analysis(width=1920, height=1080, text_density=0.8, is_mostly_text=True)
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.estimated_tokens_before > 0
        assert plan.estimated_tokens_after > 0
        assert plan.estimated_tokens_before > plan.estimated_tokens_after

    def test_savings_positive_for_low_detail(self):
        # Low detail mode (85 tokens) vs high detail for OpenAI
        analysis = _make_analysis(width=400, height=400)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.force_detail_low is True
        assert plan.estimated_tokens_after == 85
        assert plan.estimated_tokens_before >= plan.estimated_tokens_after

    def test_savings_positive_for_ocr(self):
        analysis = _make_analysis(width=1920, height=1080, text_density=0.8, is_mostly_text=True)
        plan = plan_optimization(analysis, "gpt-4o")

        assert plan.estimated_tokens_before > plan.estimated_tokens_after

    def test_no_savings_for_already_small_jpeg(self):
        analysis = _make_analysis(width=600, height=400, fmt="jpeg")
        plan = plan_optimization(analysis, "claude-sonnet-4-6")

        assert plan.estimated_tokens_before == plan.estimated_tokens_after
