"""Anthropic (Claude) provider adapter."""

from collections.abc import AsyncIterator

import anthropic

from token0.providers.base import BaseProvider, ProviderResponse, StreamChunk


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[dict]]:
        """Convert OpenAI message format to Anthropic format."""
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"] if isinstance(msg["content"], str) else ""
                continue

            if isinstance(msg["content"], str):
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            elif isinstance(msg["content"], list):
                content_blocks = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_blocks.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64_data = url.split(",", 1)
                            media_type = header.split(":")[1].split(";")[0]
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": b64_data,
                                    },
                                }
                            )
                        elif "base64" in part.get("image_url", {}):
                            content_blocks.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": part["image_url"].get(
                                            "media_type", "image/jpeg"
                                        ),
                                        "data": part["image_url"]["base64"],
                                    },
                                }
                            )
                anthropic_messages.append({"role": msg["role"], "content": content_blocks})

        return system_prompt, anthropic_messages

    def _build_kwargs(
        self,
        model: str,
        anthropic_messages: list[dict],
        system_prompt: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict:
        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature
        return kwargs

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        kwargs = self._build_kwargs(
            model, anthropic_messages, system_prompt, max_tokens, temperature
        )

        response = await self.client.messages.create(**kwargs)

        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text += block.text

        return ProviderResponse(
            content=content_text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            finish_reason=response.stop_reason,
            raw_response=response.model_dump(),
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_prompt, anthropic_messages = self._convert_messages(messages)
        kwargs = self._build_kwargs(
            model, anthropic_messages, system_prompt, max_tokens, temperature
        )

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield StreamChunk(delta_content=text, model=model)

            # Final message has usage info
            final = await stream.get_final_message()
            yield StreamChunk(
                finish_reason=final.stop_reason,
                model=final.model,
                prompt_tokens=final.usage.input_tokens,
                completion_tokens=final.usage.output_tokens,
            )
