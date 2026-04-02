"""Instructor integration — Token0 as a pre-call hook.

Hooks into instructor's COMPLETION_KWARGS event to optimize vision tokens
before every LLM call. Works with any instructor-supported provider.

Usage:
    import instructor
    import openai
    from token0.instructor_hook import Token0Hook

    client = instructor.from_openai(openai.OpenAI())
    hook = Token0Hook()
    client.on("completion:kwargs", hook)

    # All calls now get image optimization automatically
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the total on this invoice?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
            ]
        }],
        response_model=MyModel,
    )

Works with any instructor provider — OpenAI, Anthropic, Google, Ollama, etc.
No proxy required — runs as an in-process pre-call hook.
"""

import logging
from typing import Any

from token0.optimization.message_optimizer import optimize_messages

logger = logging.getLogger("token0.instructor")


class Token0Hook:
    """Instructor pre-call hook that optimizes vision tokens before LLM calls.

    Attach to any instructor client via client.on("completion:kwargs", Token0Hook()).

    Args:
        enable_cascade: Auto-route simple tasks to cheaper models (default: False).
        detail_override: Force "low" or "high" detail mode for OpenAI (default: auto).
    """

    def __init__(
        self,
        enable_cascade: bool = False,
        detail_override: str | None = None,
    ):
        self.enable_cascade = enable_cascade
        self.detail_override = detail_override

    def __call__(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Optimize images in kwargs["messages"] before the LLM call."""
        messages = kwargs.get("messages")
        if not messages:
            return kwargs

        model = kwargs.get("model", "")
        optimized_messages, stats = optimize_messages(
            messages,
            model,
            detail_override=self.detail_override,
            enable_cascade=self.enable_cascade,
        )

        kwargs["messages"] = optimized_messages

        if stats["tokens_saved"] > 0:
            logger.info(
                "token0: %d tokens saved (%s)",
                stats["tokens_saved"],
                ", ".join(stats["optimizations"]),
            )

        # Cascade: switch to cheaper model if recommended
        if stats.get("recommended_model"):
            logger.info("token0: cascade %s -> %s", model, stats["recommended_model"])
            kwargs["model"] = stats["recommended_model"]

        return kwargs
