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

from token0.optimization.analyzer import analyze_image
from token0.optimization.router import plan_optimization
from token0.optimization.transformer import transform_image

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
        optimized_messages, stats = _optimize_messages(
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


def _optimize_messages(
    messages: list[dict],
    model: str,
    detail_override: str | None = None,
    enable_cascade: bool = False,
) -> tuple[list[dict], dict]:
    """Optimize images in a list of message dicts.

    Returns (optimized_messages, stats_dict).
    """
    optimized = []
    total_before = 0
    total_after = 0
    optimizations = []
    recommended_model = None

    for msg in messages:
        content = msg.get("content")

        # Text-only message — pass through
        if isinstance(content, str) or content is None:
            optimized.append(msg)
            continue

        # Multi-part content — check for images
        if not isinstance(content, list):
            optimized.append(msg)
            continue

        opt_parts = []
        for part in content:
            if part.get("type") != "image_url":
                opt_parts.append(part)
                continue

            image_url = part.get("image_url", {})
            url = image_url.get("url", "")

            # Only optimize base64 data URIs
            if not url.startswith("data:"):
                opt_parts.append(part)
                continue

            # PDF pre-processing: extract text layer if available
            from token0.optimization.pdf import (
                decode_pdf,
                estimate_pdf_tokens,
                extract_pdf_text,
                is_pdf_data_uri,
            )

            if is_pdf_data_uri(url):
                try:
                    pdf_bytes = decode_pdf(url)
                    pdf_text = extract_pdf_text(pdf_bytes)
                    if pdf_text:
                        token_count = estimate_pdf_tokens(pdf_text)
                        total_before += 765
                        total_after += token_count
                        optimizations.append("pdf → text layer extracted")
                        opt_parts.append(
                            {"type": "text", "text": f"[Extracted text from PDF]:\n{pdf_text}"}
                        )
                    else:
                        opt_parts.append(part)  # no text layer — passthrough
                except Exception:
                    logger.warning("token0: PDF extraction failed, passing through", exc_info=True)
                    opt_parts.append(part)
                continue

            try:
                analysis, raw_bytes, pil_image = analyze_image(url)
                plan = plan_optimization(
                    analysis,
                    model,
                    detail_override=detail_override,
                    enable_cascade=enable_cascade,
                )

                total_before += plan.estimated_tokens_before
                total_after += plan.estimated_tokens_after
                optimizations.extend(plan.reasons)

                if plan.recommended_model and recommended_model is None:
                    recommended_model = plan.recommended_model

                if plan.use_ocr_route:
                    result = transform_image(plan, analysis, raw_bytes, pil_image)
                    opt_parts.append(
                        {
                            "type": "text",
                            "text": f"[Extracted text from image]:\n{result['content']}",
                        }
                    )
                elif any([plan.resize, plan.recompress_jpeg, plan.force_detail_low]):
                    result = transform_image(plan, analysis, raw_bytes, pil_image)
                    detail = "low" if plan.force_detail_low else image_url.get("detail", "auto")
                    opt_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{result['media_type']};base64,{result['base64']}",
                                "detail": detail,
                            },
                        }
                    )
                else:
                    opt_parts.append(part)

            except Exception:
                logger.warning("token0: failed to optimize image, passing through", exc_info=True)
                opt_parts.append(part)

        optimized.append({"role": msg["role"], "content": opt_parts})

    stats = {
        "tokens_before": total_before,
        "tokens_after": total_after,
        "tokens_saved": total_before - total_after,
        "optimizations": optimizations,
        "recommended_model": recommended_model,
    }

    return optimized, stats
