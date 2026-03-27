"""Ollama provider adapter — local models via OpenAI-compatible API."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from token0.providers.base import BaseProvider, ProviderResponse, StreamChunk


class OllamaProvider(BaseProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        model: str = "llava:7b",
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="ollama")
        self.default_model = model

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        kwargs = {"model": model or self.default_model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        return ProviderResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason=choice.finish_reason,
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        stream = await self.client.chat.completions.create(**kwargs)
        async for chunk in stream:
            sc = StreamChunk(model=chunk.model)
            if chunk.choices:
                delta = chunk.choices[0].delta
                sc.delta_content = delta.content if delta else None
                sc.finish_reason = chunk.choices[0].finish_reason
            if chunk.usage:
                sc.prompt_tokens = chunk.usage.prompt_tokens
                sc.completion_tokens = chunk.usage.completion_tokens
            yield sc
