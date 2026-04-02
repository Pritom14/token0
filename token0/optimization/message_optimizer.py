"""Shared message optimization logic — no litellm or langchain dependency.

Used by both litellm_hook.py and langchain_callback.py.
"""

import logging

from token0.optimization.analyzer import analyze_image
from token0.optimization.prompt_classifier import extract_prompt_text
from token0.optimization.router import plan_optimization
from token0.optimization.saliency import apply_saliency_crop, detect_roi
from token0.optimization.transformer import transform_image

logger = logging.getLogger("token0.optimizer")


def optimize_messages(
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
    prompt_text = extract_prompt_text(messages)

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

                # Saliency crop — trim to region the prompt asks about
                saliency = detect_roi(prompt_text, pil_image)
                if saliency.cropped:
                    pil_image = apply_saliency_crop(pil_image, saliency)
                    # Re-encode cropped image to bytes for downstream steps
                    import io as _io
                    fmt = "JPEG" if analysis.format == "jpg" else analysis.format.upper()
                    buf = _io.BytesIO()
                    pil_image.save(buf, format=fmt)
                    raw_bytes = buf.getvalue()
                    kw, pct = saliency.matched_keyword, saliency.savings_pct
                    optimizations.append(f"saliency crop ({kw!r}: {pct:.0%} pixels removed)")
                    logger.debug("token0: saliency crop on %r, savings=%.0f%%", kw, pct * 100)

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
