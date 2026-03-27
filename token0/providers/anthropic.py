"""Anthropic (Claude) provider adapter."""

import anthropic

from token0.providers.base import BaseProvider, ProviderResponse


class AnthropicProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        # Convert from OpenAI message format to Anthropic format
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"] if isinstance(msg["content"], str) else ""
                continue

            # Convert content format
            if isinstance(msg["content"], str):
                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
            elif isinstance(msg["content"], list):
                content_blocks = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        content_blocks.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        # Convert OpenAI image_url format to Anthropic source format
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            # Parse data URI: data:image/jpeg;base64,/9j/...
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
                        # Token0 internal format (already processed)
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

        kwargs = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens or 4096,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        if temperature is not None:
            kwargs["temperature"] = temperature

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
