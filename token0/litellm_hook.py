"""LiteLLM integration — Token0 as a pre-call hook.

Usage in litellm proxy_config.yaml:

    litellm_settings:
      callbacks: ["token0.litellm_hook.Token0Hook"]

Or programmatically:

    import litellm
    from token0.litellm_hook import Token0Hook
    litellm.callbacks = [Token0Hook()]
"""

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
        if call_type != "completion":
            return data

        messages = data.get("messages")
        if not messages:
            return data

        model = data.get("model", "")
        optimized_messages, stats = optimize_messages(
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


# Backwards-compatible alias
_optimize_messages = optimize_messages
