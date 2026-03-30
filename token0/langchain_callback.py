"""LangChain callback handler that optimizes vision tokens before LLM calls.

Usage:
    from token0.langchain_callback import Token0Callback
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", callbacks=[Token0Callback()])

    # All calls through this llm instance now get image optimization
    response = llm.invoke([HumanMessage(content=[
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    ])])

Works with any LangChain chat model (ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI, etc.)
No proxy required — runs as a pre-call callback.

Note: This implementation mutates message content in-place inside on_chat_model_start.
This works because LangChain's callbacks fire synchronously before the model converts
messages to API format. Compatible with langchain-core >= 0.2.
"""

import logging
from typing import Any

try:
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages import BaseMessage

    _langchain_available = True
except ImportError:
    _langchain_available = False
    BaseCallbackHandler = object  # type: ignore[assignment,misc]
    BaseMessage = object  # type: ignore[assignment]

from token0.litellm_hook import _optimize_messages

logger = logging.getLogger("token0.langchain")


def _extract_model_name(serialized: dict) -> str:
    """Extract model name from LangChain's serialized callback data."""
    kwargs = serialized.get("kwargs", {})
    return kwargs.get("model_name") or kwargs.get("model") or ""


def _role_for(message: BaseMessage) -> str:
    """Map LangChain message type to role string."""
    class_name = type(message).__name__
    if "Human" in class_name:
        return "user"
    if "AI" in class_name or "Assistant" in class_name:
        return "assistant"
    if "System" in class_name:
        return "system"
    return "user"


class Token0Callback(BaseCallbackHandler):
    """LangChain callback that optimizes vision tokens before LLM calls.

    Drop-in for any LangChain chat model:
        llm = ChatOpenAI(model="gpt-4o", callbacks=[Token0Callback()])

    Args:
        enable_cascade: Auto-route simple tasks to cheaper models (default: False,
            since the LangChain model is already set by the caller).
        detail_override: Force "low" or "high" detail mode for OpenAI (default: auto).
    """

    def __init__(
        self,
        enable_cascade: bool = False,
        detail_override: str | None = None,
    ):
        if not _langchain_available:
            raise ImportError(
                "langchain-core is required for the Token0Callback integration. "
                "Install it with: pip install langchain-core"
            )
        self.enable_cascade = enable_cascade
        self.detail_override = detail_override

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        """Optimize images in messages before the LLM call."""
        model = _extract_model_name(serialized)

        for message_list in messages:
            for message in message_list:
                if not isinstance(message.content, list):
                    continue

                # Wrap in the dict format _optimize_messages expects
                msg_dicts = [{"role": _role_for(message), "content": message.content}]

                optimized_dicts, stats = _optimize_messages(
                    msg_dicts,
                    model,
                    detail_override=self.detail_override,
                    enable_cascade=self.enable_cascade,
                )

                # Mutate in-place — LangChain reads this before sending to provider
                if optimized_dicts:
                    message.content = optimized_dicts[0]["content"]

                if stats["tokens_saved"] > 0:
                    logger.info(
                        "token0: %d tokens saved (%s)",
                        stats["tokens_saved"],
                        ", ".join(stats["optimizations"]),
                    )
