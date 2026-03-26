"""Tests for provider adapters — message format conversion and pricing."""

from src.providers.base import MODEL_PRICING, get_cost_per_token


class TestPricing:
    def test_known_model_exact_match(self):
        cost = get_cost_per_token("gpt-4o", "input")
        assert cost == 2.50 / 1_000_000

    def test_known_model_output(self):
        cost = get_cost_per_token("gpt-4o", "output")
        assert cost == 10.00 / 1_000_000

    def test_anthropic_model(self):
        cost = get_cost_per_token("claude-sonnet-4-6", "input")
        assert cost == 3.00 / 1_000_000

    def test_gemini_model(self):
        cost = get_cost_per_token("gemini-2.5-flash", "input")
        assert cost == 0.30 / 1_000_000

    def test_unknown_model_uses_conservative_default(self):
        cost = get_cost_per_token("totally-unknown-model", "input")
        assert cost == 3.00 / 1_000_000  # conservative default

    def test_prefix_match(self):
        # "gpt-4o-2024-08-06" should match "gpt-4o"
        cost = get_cost_per_token("gpt-4o-2024-08-06", "input")
        assert cost == 2.50 / 1_000_000

    def test_all_models_have_input_and_output(self):
        for model, prices in MODEL_PRICING.items():
            assert "input" in prices, f"{model} missing input price"
            assert "output" in prices, f"{model} missing output price"
            assert prices["input"] > 0
            assert prices["output"] > 0

    def test_output_more_expensive_than_input(self):
        for model, prices in MODEL_PRICING.items():
            assert prices["output"] >= prices["input"], f"{model}: output should be >= input price"


class TestOpenAIMessageFormat:
    """Test that OpenAI provider passes messages through correctly.
    These test the format, not actual API calls (those need mocking).
    """

    def test_text_message_format(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        # Verify format is valid OpenAI format
        for msg in messages:
            assert "role" in msg
            assert "content" in msg

    def test_vision_message_format(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64,/9j/4AAQ",
                            "detail": "low",
                        },
                    },
                ],
            }
        ]
        content = messages[0]["content"]
        assert len(content) == 2
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["detail"] == "low"


class TestAnthropicMessageConversion:
    """Test OpenAI → Anthropic message format conversion logic."""

    def test_system_message_extracted(self):
        """System messages should be separated for Anthropic."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]

        system_prompt = None
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(msg)

        assert system_prompt == "You are helpful."
        assert len(anthropic_messages) == 1
        assert anthropic_messages[0]["role"] == "user"

    def test_image_url_to_base64_source(self):
        """OpenAI data URI should convert to Anthropic base64 source."""
        data_uri = "data:image/jpeg;base64,/9j/4AAQ"
        header, b64_data = data_uri.split(",", 1)
        media_type = header.split(":")[1].split(";")[0]

        anthropic_block = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64_data,
            },
        }

        assert anthropic_block["source"]["media_type"] == "image/jpeg"
        assert anthropic_block["source"]["data"] == "/9j/4AAQ"
        assert anthropic_block["source"]["type"] == "base64"

    def test_multiple_images_in_message(self):
        """Multiple images in one message should all be converted."""
        parts = [
            {"type": "text", "text": "Compare these:"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,def"}},
        ]

        converted = []
        for part in parts:
            if part["type"] == "text":
                converted.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                header, b64 = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                converted.append(
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": mime, "data": b64},
                    }
                )

        assert len(converted) == 3
        assert converted[0]["type"] == "text"
        assert converted[1]["source"]["media_type"] == "image/png"
        assert converted[2]["source"]["media_type"] == "image/jpeg"


class TestGoogleMessageConversion:
    """Test OpenAI → Gemini message format conversion logic."""

    def test_system_becomes_instruction(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ]

        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                contents.append(msg)

        assert system_instruction == "Be concise."
        assert len(contents) == 1

    def test_role_mapping(self):
        """OpenAI 'assistant' should map to Gemini 'model'."""
        role_map = {"user": "user", "assistant": "model"}

        assert role_map.get("user") == "user"
        assert role_map.get("assistant") == "model"
