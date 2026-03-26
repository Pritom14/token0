"""Ollama provider adapter — local models via OpenAI-compatible API."""

from openai import AsyncOpenAI

from src.providers.base import BaseProvider, ProviderResponse


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llava:7b"):
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
