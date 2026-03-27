"""Base provider interface — all LLM adapters implement this."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str | None = None
    raw_response: dict | None = None


@dataclass
class StreamChunk:
    delta_content: str | None = None
    finish_reason: str | None = None
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


# Per-million-token pricing (USD). Input/output.
MODEL_PRICING = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # Anthropic
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    # Google
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 2.50, "output": 15.00},
}


def get_cost_per_token(model: str, direction: str = "input") -> float:
    """Get cost per single token in USD."""
    # Try exact match first, then prefix match
    pricing = MODEL_PRICING.get(model)
    if not pricing:
        for key, val in MODEL_PRICING.items():
            if model.startswith(key) or key.startswith(model):
                pricing = val
                break
    if not pricing:
        pricing = {"input": 3.00, "output": 15.00}  # conservative default

    return pricing[direction] / 1_000_000


class BaseProvider(ABC):
    @abstractmethod
    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        """Send a chat completion request to the provider."""
        ...

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion. Override per provider."""
        # Default: fall back to non-streaming and yield as single chunk
        response = await self.chat_completion(model, messages, max_tokens, temperature)
        yield StreamChunk(
            delta_content=response.content,
            finish_reason=response.finish_reason,
            model=response.model,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )
        # Make this a proper async generator
        return  # noqa: B901
