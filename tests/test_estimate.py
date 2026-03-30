"""Tests for the /v1/estimate endpoint."""

import asyncio
import base64
import io

from PIL import Image


def _make_image_data_uri(width: int = 800, height: int = 600, fmt: str = "JPEG") -> str:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


class TestEstimateEndpoint:
    def test_estimate_single_image(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        req = EstimateRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(type="text", text="What's in this image?"),
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=_make_image_data_uri(800, 600)),
                        ),
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))

        assert result.model == "gpt-4o"
        assert result.provider == "openai"
        assert len(result.images) == 1
        assert result.images[0].original_tokens > 0
        assert result.total_original_tokens > 0

    def test_estimate_returns_savings(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        # Large image — should trigger resize, yielding savings
        req = EstimateRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=_make_image_data_uri(3000, 2000)),
                        )
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))
        assert result.total_original_tokens >= result.total_optimized_tokens

    def test_estimate_text_only_no_images(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import Message

        req = EstimateRequest(
            model="gpt-4o",
            messages=[Message(role="user", content="Just a text message")],
        )
        result = asyncio.run(estimate(req))
        assert result.images == []
        assert result.total_original_tokens == 0

    def test_estimate_remote_url_skipped_with_note(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        req = EstimateRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url="https://example.com/image.jpg"),
                        )
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))
        assert result.images == []
        assert result.note is not None
        assert "remote" in result.note.lower()

    def test_estimate_multiple_images(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        req = EstimateRequest(
            model="claude-sonnet-4-6",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=_make_image_data_uri(800, 600)),
                        ),
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=_make_image_data_uri(400, 300)),
                        ),
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))
        assert len(result.images) == 2
        assert result.provider == "anthropic"

    def test_estimate_cost_saved_is_non_negative(self):
        from token0.api.v1.estimate import EstimateRequest, estimate
        from token0.models.request import ContentPart, ImageUrl, Message

        req = EstimateRequest(
            model="gpt-4o",
            messages=[
                Message(
                    role="user",
                    content=[
                        ContentPart(
                            type="image_url",
                            image_url=ImageUrl(url=_make_image_data_uri(100, 100)),
                        )
                    ],
                )
            ],
        )
        result = asyncio.run(estimate(req))
        assert result.total_cost_saved_usd >= 0
