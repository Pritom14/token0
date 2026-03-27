"""OpenAI provider adapter."""

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from token0.providers.base import BaseProvider, ProviderResponse, StreamChunk


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        kwargs = {"model": model, "messages": messages}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature

        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0]

        return ProviderResponse(
            content=choice.message.content or "",
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            finish_reason=choice.finish_reason,
            raw_response=response.model_dump(),
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
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
