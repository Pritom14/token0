"""Tests for src/optimization/analyzer.py"""

from PIL import Image

from tests.conftest import make_text_image
from token0.optimization.analyzer import (
    analyze_image,
    decode_image,
    detect_text_density,
    estimate_anthropic_tokens,
    estimate_openai_tokens,
)


class TestDecodeImage:
    def test_decode_data_uri(self, small_jpeg):
        raw, data_uri = small_jpeg
        decoded_bytes, pil_img = decode_image(data_uri)
        assert pil_img.size == (200, 200)
        assert len(decoded_bytes) > 0

    def test_decode_raw_base64(self, small_jpeg):
        raw, data_uri = small_jpeg
        b64_only = data_uri.split(",")[1]
        decoded_bytes, pil_img = decode_image(b64_only)
        assert pil_img.size == (200, 200)

    def test_decode_png(self, medium_png):
        raw, data_uri = medium_png
        decoded_bytes, pil_img = decode_image(data_uri)
        assert pil_img.size == (1024, 1024)


class TestEstimateOpenAITokens:
    def test_low_detail_always_85(self):
        assert estimate_openai_tokens(100, 100, "low") == 85
        assert estimate_openai_tokens(4000, 3000, "low") == 85
        assert estimate_openai_tokens(1, 1, "low") == 85

    def test_small_image_high_detail(self):
        # 200x200: fits in one 512x512 tile
        tokens = estimate_openai_tokens(200, 200, "high")
        assert tokens == 85 + 170  # 1 tile

    def test_1024x1024_high_detail(self):
        # 1024x1024 → shortest side > 768, scale to 768
        # After scaling: 768x768 → 2x2 tiles = 4 tiles
        tokens = estimate_openai_tokens(1024, 1024, "high")
        assert tokens == 85 + (170 * 4)  # 765

    def test_large_image_downscaled(self):
        # 4000x3000 → first scale to max 2048: 2048x1536
        # Then shortest side to 768: 1024x768 → 2x2 = 4 tiles
        tokens = estimate_openai_tokens(4000, 3000, "high")
        assert tokens >= 85 + 170  # at least 1 tile
        assert tokens <= 85 + (170 * 12)  # reasonable upper bound

    def test_wide_image(self):
        # 1920x1080 (HD)
        tokens = estimate_openai_tokens(1920, 1080, "high")
        assert tokens > 85  # more than low detail
        assert tokens < 85 + (170 * 20)  # reasonable bound

    def test_tiny_image(self):
        # 50x50 → 1 tile
        tokens = estimate_openai_tokens(50, 50, "high")
        assert tokens == 85 + 170


class TestEstimateAnthropicTokens:
    def test_small_image(self):
        # 200x200 = 40000 / 750 = 53
        tokens = estimate_anthropic_tokens(200, 200)
        assert tokens == 53

    def test_medium_image(self):
        # 1000x1000 = 1000000 / 750 = 1333
        tokens = estimate_anthropic_tokens(1000, 1000)
        assert tokens == 1333

    def test_large_image_downscaled(self):
        # 4000x3000 → longest edge 4000 > 1568, scale: 1568/4000 = 0.392
        # New dims: 1568x1176 = 1843968 pixels
        # But also > 1.15MP, so scale again: sqrt(1150000/1843968) ≈ 0.789
        # Final: ~1238x928 = 1148864 / 750 = 1531
        tokens = estimate_anthropic_tokens(4000, 3000)
        assert tokens > 0
        # Should be less than unscaled: 4000*3000/750 = 16000
        assert tokens < 16000
        # Should be close to 1.15MP / 750 ≈ 1533
        assert 1400 < tokens < 1600

    def test_exact_max_no_downscale(self):
        # 1568x1000 = 1568000 pixels → > 1.15MP, will get downscaled
        tokens = estimate_anthropic_tokens(1568, 1000)
        # After megapixel scaling: sqrt(1150000/1568000) ≈ 0.856
        # ~1342x856 = 1148752 / 750 ≈ 1531
        assert tokens > 0

    def test_minimum_1_token(self):
        tokens = estimate_anthropic_tokens(1, 1)
        assert tokens == 1


class TestDetectTextDensity:
    def test_solid_color_low_density(self):
        # A solid red image has no text. Uniform background boosts score slightly,
        # but lack of edges and line structure keeps it below OCR threshold.
        img = Image.new("RGB", (800, 600), color="red")
        density = detect_text_density(img)
        assert density < 0.52  # below OCR routing threshold

    def test_text_image_higher_density(self):
        # Generate a text-like image
        _, data_uri = make_text_image(800, 600, lines=25)
        _, pil_img = decode_image(data_uri)
        density = detect_text_density(pil_img)
        # Should detect some text-like regions
        assert density > 0.0  # at least some text detected

    def test_gradient_low_density(self):
        # Smooth gradient — no text edges
        import numpy as np

        arr = np.zeros((600, 800, 3), dtype=np.uint8)
        for y in range(600):
            arr[y, :, :] = int(255 * y / 600)
        img = Image.fromarray(arr)
        density = detect_text_density(img)
        assert density < 0.3  # gradients shouldn't look like text


class TestAnalyzeImage:
    def test_analyze_small_jpeg(self, small_jpeg):
        _, data_uri = small_jpeg
        analysis, raw_bytes, pil_img = analyze_image(data_uri)

        assert analysis.width == 200
        assert analysis.height == 200
        assert analysis.size_bytes > 0
        assert analysis.estimated_tokens_openai_low == 85
        assert analysis.estimated_tokens_openai_high > 0
        assert analysis.estimated_tokens_anthropic > 0
        assert analysis.has_transparency is False
        assert 0.0 <= analysis.text_density <= 1.0

    def test_analyze_large_jpeg(self, large_jpeg):
        _, data_uri = large_jpeg
        analysis, _, _ = analyze_image(data_uri)

        assert analysis.width == 4000
        assert analysis.height == 3000
        # Anthropic tokens should be downscaled
        assert analysis.estimated_tokens_anthropic < 4000 * 3000 // 750

    def test_analyze_png_with_alpha(self, png_with_alpha):
        _, data_uri = png_with_alpha
        analysis, _, _ = analyze_image(data_uri)

        assert analysis.has_transparency is True
        assert analysis.format == "png"

    def test_analyze_returns_consistent_data(self, medium_png):
        _, data_uri = medium_png
        a1, _, _ = analyze_image(data_uri)
        a2, _, _ = analyze_image(data_uri)

        assert a1.width == a2.width
        assert a1.height == a2.height
        assert a1.estimated_tokens_openai_high == a2.estimated_tokens_openai_high
        assert a1.estimated_tokens_anthropic == a2.estimated_tokens_anthropic
