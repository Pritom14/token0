"""LiteLLM integration.

LiteLLM Proxy (config.yaml) — runs as a pre-call hook:

    litellm_settings:
      callbacks: ["token0.litellm_hook.proxy_handler_instance"]

LiteLLM SDK — pre-call hooks only run inside the proxy, so use the drop-in wrappers:

    from token0.litellm_hook import completion, acompletion
    response = completion(model="gpt-4.1", messages=[...])   # same args as litellm
"""

import asyncio
import logging

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:
    raise ImportError(
        "litellm is required for the Token0Hook integration. Install it with: pip install litellm"
    )

from token0.optimization.message_optimizer import optimize_messages

logger = logging.getLogger("token0.litellm")


class Token0Hook(CustomLogger):
    """LiteLLM hook that optimizes vision tokens before LLM calls."""

    def __init__(
        self,
        enable_cascade: bool = False,
        detail_override: str | None = None,
    ):
        self.enable_cascade = enable_cascade
        self.detail_override = detail_override

    async def async_pre_call_hook(
        self,
        user_api_key_dict,
        cache,
        data: dict,
        call_type: str,
    ) -> dict:
        """Optimize images in messages before the LLM call."""
        if call_type not in ("completion", "acompletion"):
            return data

        messages = data.get("messages")
        if not messages:
            return data

        model = data.get("model", "")
        # OCR / image work is CPU-bound; run it off the proxy's event loop
        optimized_messages, stats = await asyncio.to_thread(
            optimize_messages,
            messages,
            model,
            detail_override=self.detail_override,
            enable_cascade=self.enable_cascade,
        )

        data["messages"] = optimized_messages

        if stats["tokens_saved"] > 0:
            logger.info(
                "token0: %d tokens saved (%s)",
                stats["tokens_saved"],
                ", ".join(stats["optimizations"]),
            )

        # Attach stats for downstream logging/callbacks
        data.setdefault("metadata", {})
        data["metadata"]["token0"] = stats

        # Apply model cascade if recommended
        if stats.get("recommended_model"):
            data["model"] = stats["recommended_model"]
            logger.info("token0: cascade %s → %s", model, stats["recommended_model"])

        return data


def _prepare(kwargs: dict, enable_cascade: bool, detail_override: str | None) -> dict:
    messages = kwargs.get("messages")
    if not messages:
        return kwargs
    model = kwargs.get("model", "")
    optimized_messages, stats = optimize_messages(
        messages, model, detail_override=detail_override, enable_cascade=enable_cascade
    )
    kwargs = {**kwargs, "messages": optimized_messages}
    if stats.get("recommended_model"):
        kwargs["model"] = stats["recommended_model"]
    metadata = dict(kwargs.get("metadata") or {})
    metadata["token0"] = stats
    kwargs["metadata"] = metadata
    if stats["tokens_saved"] > 0:
        logger.info(
            "token0: %d tokens saved (%s)", stats["tokens_saved"], ", ".join(stats["optimizations"])
        )
    return kwargs


def completion(
    *args, token0_enable_cascade: bool = False, token0_detail: str | None = None, **kwargs
):
    """Drop-in for ``litellm.completion`` that optimizes images first.

    Stats: ``response._hidden_params["token0"]``.
    """
    import litellm

    if args:
        kwargs.setdefault("model", args[0])
        args = args[1:]
    kwargs = _prepare(kwargs, token0_enable_cascade, token0_detail)
    response = litellm.completion(*args, **kwargs)
    _attach_stats(response, kwargs)
    return response


async def acompletion(
    *args, token0_enable_cascade: bool = False, token0_detail: str | None = None, **kwargs
):
    """Drop-in for ``litellm.acompletion`` that optimizes images first."""
    import litellm

    if args:
        kwargs.setdefault("model", args[0])
        args = args[1:]
    # OCR / image work is CPU-bound; keep it off the event loop
    kwargs = await asyncio.to_thread(_prepare, kwargs, token0_enable_cascade, token0_detail)
    response = await litellm.acompletion(*args, **kwargs)
    _attach_stats(response, kwargs)
    return response


def _attach_stats(response, kwargs: dict) -> None:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        hidden["token0"] = kwargs.get("metadata", {}).get("token0")


# Instance for LiteLLM Proxy configs: callbacks: ["token0.litellm_hook.proxy_handler_instance"]
proxy_handler_instance = Token0Hook()

# Backwards-compatible alias
_optimize_messages = optimize_messages
