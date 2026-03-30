"""Tests for the LangChain callback handler."""

import base64
import io

import pytest
from PIL import Image


def _make_image_data_uri(width: int = 800, height: int = 600) -> str:
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class TestToken0Callback:
    def test_import(self):
        from token0.langchain_callback import Token0Callback

        cb = Token0Callback()
        assert cb is not None

    def test_init_defaults(self):
        from token0.langchain_callback import Token0Callback

        cb = Token0Callback()
        assert cb.enable_cascade is False
        assert cb.detail_override is None

    def test_init_custom(self):
        from token0.langchain_callback import Token0Callback

        cb = Token0Callback(enable_cascade=True, detail_override="low")
        assert cb.enable_cascade is True
        assert cb.detail_override == "low"

    def test_text_only_message_unchanged(self):
        """Text-only messages should pass through without modification."""
        from token0.langchain_callback import Token0Callback

        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        cb = Token0Callback()
        msg = HumanMessage(content="Hello, what is 2+2?")
        original_content = msg.content

        cb.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            messages=[[msg]],
        )

        assert msg.content == original_content

    def test_image_message_content_is_list(self):
        """After optimization, image message content remains a list."""
        from token0.langchain_callback import Token0Callback

        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        cb = Token0Callback()
        content = [
            {"type": "text", "text": "Describe this image"},
            {
                "type": "image_url",
                "image_url": {"url": _make_image_data_uri(800, 600)},
            },
        ]
        msg = HumanMessage(content=content)

        cb.on_chat_model_start(
            serialized={"kwargs": {"model_name": "gpt-4o"}},
            messages=[[msg]],
        )

        assert isinstance(msg.content, list)

    def test_empty_serialized_does_not_crash(self):
        """Missing model name in serialized should not crash."""
        from token0.langchain_callback import Token0Callback

        try:
            from langchain_core.messages import HumanMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        cb = Token0Callback()
        msg = HumanMessage(content="hello")

        # Should not raise
        cb.on_chat_model_start(serialized={}, messages=[[msg]])

    def test_extract_model_name(self):
        from token0.langchain_callback import _extract_model_name

        assert _extract_model_name({"kwargs": {"model_name": "gpt-4o"}}) == "gpt-4o"
        model_id = "claude-sonnet-4-6"
        assert _extract_model_name({"kwargs": {"model": model_id}}) == model_id
        assert _extract_model_name({}) == ""

    def test_role_for_messages(self):
        from token0.langchain_callback import _role_for

        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError:
            pytest.skip("langchain-core not installed")

        assert _role_for(HumanMessage(content="hi")) == "user"
        assert _role_for(AIMessage(content="hi")) == "assistant"
        assert _role_for(SystemMessage(content="hi")) == "system"
