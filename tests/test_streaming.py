"""Tests for streaming chat completions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import make_image
from token0.main import app
from token0.providers.base import StreamChunk
from token0.storage.redis import MemoryCache


@pytest.fixture(autouse=True)
def _init_cache():
    import token0.storage.redis as redis_mod

    redis_mod.pool = MemoryCache()


@pytest.fixture
def client():
    return TestClient(app)


def _mock_db_session():
    mock_session_instance = MagicMock()
    mock_session_instance.add = MagicMock()
    mock_session_instance.commit = AsyncMock()
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    return mock_ctx


async def _fake_stream(*args, **kwargs):
    """Simulate a provider streaming 3 chunks + final."""
    yield StreamChunk(delta_content="Hello", model="gpt-4o")
    yield StreamChunk(delta_content=" world", model="gpt-4o")
    yield StreamChunk(delta_content="!", model="gpt-4o")
    yield StreamChunk(
        finish_reason="stop",
        model="gpt-4o",
        prompt_tokens=100,
        completion_tokens=3,
    )


class TestStreamingBasics:
    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_returns_sse_format(self, mock_session, mock_get_provider, client):
        """stream=true returns text/event-stream with proper SSE format."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = response.text.strip().split("\n\n")
        # Should have data chunks + [DONE]
        assert lines[-1] == "data: [DONE]"

        # Each non-DONE line should be valid JSON
        for line in lines[:-1]:
            assert line.startswith("data: ")
            data = json.loads(line[6:])
            assert data["object"] == "chat.completion.chunk"
            assert "choices" in data
            assert data["choices"][0]["index"] == 0

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_content_accumulates(self, mock_session, mock_get_provider, client):
        """Streaming chunks contain the expected content deltas."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        full_content = ""
        lines = response.text.strip().split("\n\n")
        for line in lines:
            if line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            delta = data["choices"][0]["delta"]
            if "content" in delta:
                full_content += delta["content"]

        assert full_content == "Hello world!"

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_final_chunk_has_token0_stats(self, mock_session, mock_get_provider, client):
        """Final streaming chunk includes token0 optimization stats."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        lines = response.text.strip().split("\n\n")
        # Find the chunk with finish_reason
        for line in lines:
            if line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data["choices"][0]["finish_reason"] == "stop":
                assert "token0" in data
                assert "tokens_saved" in data["token0"]
                assert "optimizations_applied" in data["token0"]
                return

        pytest.fail("No final chunk with finish_reason found")

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_choice_uses_delta_not_message(self, mock_session, mock_get_provider, client):
        """Streaming chunks use 'delta' key, not 'message'."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        lines = response.text.strip().split("\n\n")
        for line in lines:
            if line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            choice = data["choices"][0]
            assert "delta" in choice
            assert "message" not in choice


class TestStreamingWithOptimization:
    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_with_image_still_optimizes(self, mock_session, mock_get_provider, client):
        """Images are optimized before streaming starts."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        _, data_uri = make_image(4000, 3000, "blue", "JPEG")

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                        ],
                    }
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        # Find final chunk with token0 stats
        lines = response.text.strip().split("\n\n")
        for line in lines:
            if line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "token0" in data:
                assert data["token0"]["tokens_saved"] >= 0
                assert len(data["token0"]["optimizations_applied"]) > 0
                return

        pytest.fail("No token0 stats in stream")

    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_text_only_zero_overhead(self, mock_session, mock_get_provider, client):
        """Text-only streaming adds zero extra tokens."""
        mock_provider = AsyncMock()
        mock_provider.stream_chat_completion = _fake_stream
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": True,
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        assert response.status_code == 200

        lines = response.text.strip().split("\n\n")
        for line in lines:
            if line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if "token0" in data:
                assert data["token0"]["tokens_saved"] == 0
                return


class TestStreamFalseRegression:
    @patch("token0.api.v1.chat._get_provider")
    @patch("token0.api.v1.chat.async_session")
    def test_stream_false_returns_json(self, mock_session, mock_get_provider, client):
        """stream=false still returns a normal JSON response."""
        from token0.providers.base import ProviderResponse

        mock_provider = AsyncMock()
        mock_provider.chat_completion = AsyncMock(
            return_value=ProviderResponse(
                content="Test response",
                model="gpt-4o",
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                finish_reason="stop",
            )
        )
        mock_get_provider.return_value = mock_provider
        mock_session.return_value = _mock_db_session()

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "stream": False,
                "messages": [
                    {"role": "user", "content": "Hello"},
                ],
            },
            headers={"X-Provider-Key": "sk-test"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"]["content"] == "Test response"
        assert "token0" in data
