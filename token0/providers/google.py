"""Google Gemini provider adapter."""

import base64
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from token0.providers.base import BaseProvider, ProviderResponse, StreamChunk


class GoogleProvider(BaseProvider):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def _convert_messages(self, messages: list[dict]) -> tuple[str | None, list[types.Content]]:
        """Convert OpenAI format to Gemini format."""
        gemini_contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"] if isinstance(msg["content"], str) else ""
                continue

            role = "user" if msg["role"] == "user" else "model"

            if isinstance(msg["content"], str):
                gemini_contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )
            elif isinstance(msg["content"], list):
                parts = []
                for part in msg["content"]:
                    if part["type"] == "text":
                        parts.append(types.Part.from_text(text=part["text"]))
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        if url.startswith("data:"):
                            header, b64_data = url.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                            parts.append(
                                types.Part.from_bytes(
                                    data=base64.b64decode(b64_data),
                                    mime_type=mime_type,
                                )
                            )
                gemini_contents.append(types.Content(role=role, parts=parts))

        return system_instruction, gemini_contents

    def _build_config(
        self,
        system_instruction: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig()
        if max_tokens:
            config.max_output_tokens = max_tokens
        if temperature is not None:
            config.temperature = temperature
        if system_instruction:
            config.system_instruction = system_instruction
        return config

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ProviderResponse:
        system_instruction, gemini_contents = self._convert_messages(messages)
        config = self._build_config(system_instruction, max_tokens, temperature)

        response = await self.client.aio.models.generate_content(
            model=model,
            contents=gemini_contents,
            config=config,
        )

        content_text = response.text or ""
        prompt_tokens = response.usage_metadata.prompt_token_count or 0
        completion_tokens = response.usage_metadata.candidates_token_count or 0

        return ProviderResponse(
            content=content_text,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="stop",
        )

    async def stream_chat_completion(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[StreamChunk]:
        system_instruction, gemini_contents = self._convert_messages(messages)
        config = self._build_config(system_instruction, max_tokens, temperature)

        async for chunk in self.client.aio.models.generate_content_stream(
            model=model,
            contents=gemini_contents,
            config=config,
        ):
            text = chunk.text or ""
            sc = StreamChunk(delta_content=text, model=model)
            if chunk.usage_metadata:
                sc.prompt_tokens = chunk.usage_metadata.prompt_token_count
                sc.completion_tokens = chunk.usage_metadata.candidates_token_count
            yield sc

        # Final chunk with stop
        yield StreamChunk(finish_reason="stop", model=model)
