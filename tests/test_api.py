"""Integration tests for API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_image
from token0.main import app
from token0.providers.base import ProviderResponse
from token0.storage.redis import MemoryCache


@pytest.fixture(autouse=True)
def _init_cache():
    """Initialize in-memory cache for all API tests."""
    import token0.storage.redis as redis_mod

    redis_mod.pool = MemoryCache()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_provider_response():
    return ProviderResponse(
        content="This is a test response about the image.",
        model="gpt-4o-2024-08-06",
        prompt_tokens=500,
        completion_tokens=50,
        total_tokens=550,
        finish_reason="stop",
    )


def _mock_db_session():
    """Create a properly mocked async database session."""
    mock_session_instance = MagicMock()
    mock_session_instance.add = MagicMock()
    mock_session_instance.commit = AsyncMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "token0"


class TestChatCompletions:
    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_text_only_passthrough(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """Text-only messages should pass through without optimization."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello, how are you?"}],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gpt-4o-2024-08-06"
        assert len(data["choices"]) == 1
        assert (
            data["choices"][0]["message"]["content"] == "This is a test response about the image."
        )
        assert "token0" in data
        assert data["token0"]["tokens_saved"] == 0  # no images to optimize

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_image_message_gets_optimized(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """Image messages should be analyzed and optimized."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        # Create a large test image
        _, data_uri = make_image(4000, 3000, "red", "PNG")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should have applied optimizations (resize + jpeg conversion for large PNG)
        assert data["token0"]["tokens_saved"] >= 0
        assert len(data["token0"]["optimizations_applied"]) > 0

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_optimization_disabled(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """token0_optimize=false should passthrough without optimization."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        _, data_uri = make_image(4000, 3000, "red", "PNG")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                "token0_optimize": False,
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["token0"]["tokens_saved"] == 0
        assert len(data["token0"]["optimizations_applied"]) == 0

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_small_image_low_detail(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """Small images sent to OpenAI should get low-detail mode."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        _, data_uri = make_image(300, 300, "blue", "JPEG")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Classify this"},
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        # Verify provider was called with low detail
        call_args = mock_provider.chat_completion.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        content = messages[0]["content"]
        image_part = [p for p in content if p["type"] == "image_url"][0]
        assert image_part["image_url"]["detail"] == "low"

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_anthropic_model_routing(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """Claude model names should route to Anthropic provider."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        # Verify _get_provider was called with anthropic
        mock_get_provider.assert_called_with("anthropic", api_key="test-key")

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_response_format(self, mock_session, mock_get_provider, client, mock_provider_response):
        """Response should follow OpenAI-compatible format with token0 extras."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Hello"}],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        data = response.json()

        # OpenAI-compatible fields
        assert "id" in data
        assert data["id"].startswith("token0-")
        assert data["object"] == "chat.completion"
        assert "model" in data
        assert "choices" in data
        assert "usage" in data
        assert "prompt_tokens" in data["usage"]
        assert "completion_tokens" in data["usage"]
        assert "total_tokens" in data["usage"]

        # Token0 extras
        assert "token0" in data
        assert "original_prompt_tokens_estimate" in data["token0"]
        assert "optimized_prompt_tokens" in data["token0"]
        assert "tokens_saved" in data["token0"]
        assert "cost_saved_usd" in data["token0"]
        assert "optimizations_applied" in data["token0"]

    def test_missing_provider_key_returns_400(self, client):
        """Should error if no provider key is available."""
        # Clear any default keys
        with patch("token0.api.v1.chat.settings") as mock_settings:
            mock_settings.openai_api_key = ""
            mock_settings.anthropic_api_key = ""
            mock_settings.google_api_key = ""

            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert response.status_code == 400


class TestMultipleImages:
    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_multiple_images_all_optimized(
        self, mock_session, mock_get_provider, client, mock_provider_response
    ):
        """Multiple images in a single message should each be independently optimized."""
        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(return_value=mock_provider_response)
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        _, img1_uri = make_image(4000, 3000, "red", "PNG")
        _, img2_uri = make_image(300, 300, "blue", "JPEG")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Compare these images"},
                            {"type": "image_url", "image_url": {"url": img1_uri}},
                            {"type": "image_url", "image_url": {"url": img2_uri}},
                        ],
                    }
                ],
            },
            headers={"X-Provider-Key": "test-key"},
        )

        assert response.status_code == 200
        data = response.json()
        # Should have optimizations from both images
        assert len(data["token0"]["optimizations_applied"]) > 0
