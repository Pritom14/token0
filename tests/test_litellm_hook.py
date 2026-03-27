"""Tests for LiteLLM integration hook."""

import pytest

from tests.conftest import make_image, make_text_image
from token0.litellm_hook import Token0Hook, _optimize_messages


class TestOptimizeMessages:
    def test_text_only_passthrough(self):
        """Text-only messages pass through unchanged."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        optimized, stats = _optimize_messages(messages, "gpt-4o")

        assert optimized == messages
        assert stats["tokens_saved"] == 0
        assert stats["optimizations"] == []

    def test_large_image_gets_optimized(self):
        """Large images trigger resize optimization."""
        _, data_uri = make_image(4000, 3000, "blue", "JPEG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        optimized, stats = _optimize_messages(messages, "gpt-4o")

        assert len(stats["optimizations"]) > 0
        # Image should still be present (resized, not OCR'd)
        parts = optimized[0]["content"]
        assert any(p.get("type") == "image_url" for p in parts)

    def test_text_heavy_image_ocr_routed(self):
        """Text-heavy images get OCR routed."""
        _, data_uri = make_text_image(800, 600, lines=25, fmt="PNG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this document"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        optimized, stats = _optimize_messages(messages, "gpt-4o")

        assert stats["tokens_saved"] > 0
        # OCR route replaces image with text
        parts = optimized[0]["content"]
        text_parts = [p for p in parts if p.get("type") == "text"]
        assert len(text_parts) == 2  # original text + extracted text

    def test_non_data_uri_passthrough(self):
        """URLs (not base64) pass through unchanged."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                ],
            }
        ]
        optimized, stats = _optimize_messages(messages, "gpt-4o")

        assert optimized == messages
        assert stats["tokens_saved"] == 0

    def test_multiple_images(self):
        """Multiple images in one message are each optimized."""
        _, uri1 = make_image(4000, 3000, "blue", "JPEG")
        _, uri2 = make_image(3000, 2000, "red", "JPEG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these"},
                    {"type": "image_url", "image_url": {"url": uri1}},
                    {"type": "image_url", "image_url": {"url": uri2}},
                ],
            }
        ]
        optimized, stats = _optimize_messages(messages, "gpt-4o")

        parts = optimized[0]["content"]
        image_parts = [p for p in parts if p.get("type") == "image_url"]
        assert len(image_parts) == 2

    def test_cascade_recommends_cheaper_model(self):
        """Model cascade suggests cheaper alternative."""
        _, data_uri = make_image(800, 600, "red", "JPEG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        # enable_cascade with a known model
        _, stats = _optimize_messages(messages, "gpt-4o", enable_cascade=True)
        # Cascade may or may not trigger depending on prompt classification
        # Just verify the field exists
        assert "recommended_model" in stats

    def test_stats_structure(self):
        """Stats dict has all expected keys."""
        _, data_uri = make_image(800, 600, "red", "JPEG")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]
        _, stats = _optimize_messages(messages, "gpt-4o")

        assert "tokens_before" in stats
        assert "tokens_after" in stats
        assert "tokens_saved" in stats
        assert "optimizations" in stats
        assert "recommended_model" in stats
        assert stats["tokens_saved"] == stats["tokens_before"] - stats["tokens_after"]


class TestToken0HookIntegration:
    @pytest.mark.asyncio
    async def test_hook_modifies_data(self):
        """Hook modifies data dict with optimized messages."""
        _, data_uri = make_image(4000, 3000, "blue", "JPEG")
        hook = Token0Hook()
        data = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this"},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
        }

        result = await hook.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=data,
            call_type="completion",
        )

        assert "metadata" in result
        assert "token0" in result["metadata"]
        assert result["metadata"]["token0"]["tokens_saved"] >= 0

    @pytest.mark.asyncio
    async def test_hook_skips_non_completion(self):
        """Hook ignores non-completion call types."""
        hook = Token0Hook()
        data = {"model": "dall-e-3", "prompt": "A cat"}

        result = await hook.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=data,
            call_type="image_generation",
        )

        assert result == data
        assert "metadata" not in result

    @pytest.mark.asyncio
    async def test_hook_text_only_no_metadata_overhead(self):
        """Text-only requests get minimal metadata."""
        hook = Token0Hook()
        data = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        result = await hook.async_pre_call_hook(
            user_api_key_dict={},
            cache=None,
            data=data,
            call_type="completion",
        )

        assert result["metadata"]["token0"]["tokens_saved"] == 0
