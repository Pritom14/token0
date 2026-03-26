"""Main /v1/chat/completions endpoint — the core proxy."""

import time
import uuid

from fastapi import APIRouter, Header, HTTPException

from src.config import settings
from src.models.db import Request
from src.models.request import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Token0Usage,
    UsageInfo,
)
from src.optimization.analyzer import analyze_image
from src.optimization.cache import get_cached_response, make_cache_key, set_cached_response
from src.optimization.prompt_classifier import classify_prompt_detail, extract_prompt_text
from src.optimization.router import OptimizationPlan, get_provider_from_model, plan_optimization
from src.optimization.transformer import transform_image
from src.providers.anthropic import AnthropicProvider
from src.providers.base import BaseProvider, get_cost_per_token
from src.providers.google import GoogleProvider
from src.providers.openai import OpenAIProvider
from src.storage.postgres import async_session

router = APIRouter()


def _get_provider(provider_name: str, api_key: str | None = None) -> BaseProvider:
    """Instantiate the right provider with API key."""
    if provider_name == "openai":
        key = api_key or settings.openai_api_key
        if not key:
            raise HTTPException(400, "OpenAI API key required. Pass via X-Provider-Key header.")
        return OpenAIProvider(api_key=key)
    elif provider_name == "anthropic":
        key = api_key or settings.anthropic_api_key
        if not key:
            raise HTTPException(400, "Anthropic API key required. Pass via X-Provider-Key header.")
        return AnthropicProvider(api_key=key)
    elif provider_name == "google":
        key = api_key or settings.google_api_key
        if not key:
            raise HTTPException(400, "Google API key required. Pass via X-Provider-Key header.")
        return GoogleProvider(api_key=key)
    else:
        raise HTTPException(400, f"Unsupported provider: {provider_name}")


@router.post("/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    authorization: str | None = Header(None),
    x_provider_key: str | None = Header(None),
    x_token0_key: str | None = Header(None),
):
    start_time = time.time()

    # --- Step 1: Classify the prompt ---
    prompt_text = extract_prompt_text(
        [{"role": m.role, "content": m.content} for m in request.messages]
    )
    prompt_detail = classify_prompt_detail(prompt_text) if request.token0_optimize else "auto"

    # --- Step 2: Determine provider and model ---
    actual_model = request.model
    model_cascaded_to = None
    provider_name = get_provider_from_model(request.model)

    # --- Step 3: Process messages — find and optimize images ---
    optimized_messages = []
    total_tokens_before = 0
    total_tokens_after = 0
    optimizations_applied = []
    plans: list[OptimizationPlan] = []
    cache_key = None
    first_pil_image = None  # for cache key generation

    for msg in request.messages:
        if isinstance(msg.content, str):
            optimized_messages.append({"role": msg.role, "content": msg.content})
            continue

        optimized_parts = []
        for part in msg.content:
            if part.type == "text":
                optimized_parts.append({"type": "text", "text": part.text})
            elif part.type == "image_url" and part.image_url and request.token0_optimize:
                image_data = part.image_url.url
                analysis, raw_bytes, pil_image = analyze_image(image_data)

                # Save first image for cache key
                if first_pil_image is None:
                    first_pil_image = pil_image

                plan = plan_optimization(
                    analysis,
                    request.model,
                    detail_override=request.token0_detail_override,
                    prompt_detail=prompt_detail,
                    enable_cascade=request.token0_enable_cascade,
                )
                plans.append(plan)

                # Check for model cascade recommendation
                if plan.recommended_model and model_cascaded_to is None:
                    model_cascaded_to = plan.recommended_model
                    actual_model = plan.recommended_model

                total_tokens_before += plan.estimated_tokens_before
                total_tokens_after += plan.estimated_tokens_after
                optimizations_applied.extend(plan.reasons)

                if plan.use_ocr_route:
                    result = transform_image(plan, analysis, raw_bytes, pil_image)
                    optimized_parts.append(
                        {
                            "type": "text",
                            "text": f"[Extracted text from image]:\n{result['content']}",
                        }
                    )
                elif any([plan.resize, plan.recompress_jpeg, plan.force_detail_low]):
                    result = transform_image(plan, analysis, raw_bytes, pil_image)
                    detail = "low" if plan.force_detail_low else (part.image_url.detail or "auto")
                    optimized_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{result['media_type']};base64,{result['base64']}",
                                "detail": detail,
                            },
                        }
                    )
                else:
                    optimized_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data, "detail": part.image_url.detail},
                        }
                    )
            elif part.type == "image_url" and part.image_url:
                optimized_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url, "detail": part.image_url.detail},
                    }
                )

        optimized_messages.append({"role": msg.role, "content": optimized_parts})

    # --- Step 4: Check semantic cache ---
    cache_hit = False
    if request.token0_enable_cache and first_pil_image is not None and prompt_text:
        cache_key = make_cache_key(first_pil_image, prompt_text, actual_model)
        cached = await get_cached_response(cache_key)
        if cached:
            cache_hit = True
            latency_ms = int((time.time() - start_time) * 1000)

            # Log cache hit
            tokens_saved = total_tokens_before  # saved everything
            cost_per_input_token = get_cost_per_token(request.model, "input")
            optimizations_applied.append("cache hit — 0 tokens")

            async with async_session() as session:
                db_request = Request(
                    provider=provider_name,
                    model=actual_model,
                    customer_id="00000000-0000-0000-0000-000000000000",
                    image_count=len(plans),
                    optimization_type="cache_hit",
                    tokens_original_estimate=total_tokens_before,
                    tokens_actual=0,
                    tokens_saved=total_tokens_before,
                    cost_original_estimate=total_tokens_before * cost_per_input_token,
                    cost_actual=0,
                    cost_saved=total_tokens_before * cost_per_input_token,
                    response_tokens=0,
                    latency_ms=latency_ms,
                    optimization_details={"cache_hit": True, "cache_key": cache_key},
                )
                session.add(db_request)
                await session.commit()

            return ChatResponse(
                id=f"token0-{uuid.uuid4().hex[:12]}",
                model=cached["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=cached["content"]),
                        finish_reason=cached.get("finish_reason", "stop"),
                    )
                ],
                usage=UsageInfo(
                    prompt_tokens=0,
                    completion_tokens=cached.get("completion_tokens", 0),
                    total_tokens=cached.get("completion_tokens", 0),
                ),
                token0=Token0Usage(
                    original_prompt_tokens_estimate=total_tokens_before,
                    optimized_prompt_tokens=0,
                    tokens_saved=total_tokens_before,
                    cost_saved_usd=round(total_tokens_before * cost_per_input_token, 6),
                    optimizations_applied=optimizations_applied,
                    cache_hit=True,
                    model_cascaded_to=model_cascaded_to,
                ),
            )

    # --- Step 5: Resolve provider for actual model (may have been cascaded) ---
    actual_provider_name = get_provider_from_model(actual_model)
    provider = _get_provider(actual_provider_name, api_key=x_provider_key)

    # --- Step 6: Forward to provider ---
    provider_response = await provider.chat_completion(
        model=actual_model,
        messages=optimized_messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    latency_ms = int((time.time() - start_time) * 1000)

    # --- Step 7: Cache the response ---
    if cache_key and not cache_hit:
        await set_cached_response(
            cache_key,
            {
                "model": provider_response.model,
                "content": provider_response.content,
                "finish_reason": provider_response.finish_reason,
                "completion_tokens": provider_response.completion_tokens,
            },
        )

    # --- Step 8: Calculate savings ---
    tokens_saved = max(0, total_tokens_before - total_tokens_after)
    # If model was cascaded, factor in the price difference
    original_cost_per_token = get_cost_per_token(request.model, "input")
    actual_cost_per_token = get_cost_per_token(actual_model, "input")

    cost_before = total_tokens_before * original_cost_per_token
    cost_after = total_tokens_after * actual_cost_per_token
    cost_saved = max(0, cost_before - cost_after)

    # --- Step 9: Log to database ---
    async with async_session() as session:
        db_request = Request(
            provider=actual_provider_name,
            model=actual_model,
            customer_id="00000000-0000-0000-0000-000000000000",
            image_count=len(plans),
            optimization_type=", ".join(set(optimizations_applied)) or "none",
            tokens_original_estimate=total_tokens_before,
            tokens_actual=provider_response.prompt_tokens,
            tokens_saved=tokens_saved,
            cost_original_estimate=cost_before,
            cost_actual=cost_after,
            cost_saved=cost_saved,
            response_tokens=provider_response.completion_tokens,
            latency_ms=latency_ms,
            optimization_details={
                "plans": [
                    {
                        "reasons": p.reasons,
                        "before": p.estimated_tokens_before,
                        "after": p.estimated_tokens_after,
                    }
                    for p in plans
                ],
                "model_cascaded": model_cascaded_to,
                "prompt_detail": prompt_detail,
            },
        )
        session.add(db_request)
        await session.commit()

    return ChatResponse(
        id=f"token0-{uuid.uuid4().hex[:12]}",
        model=provider_response.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=provider_response.content),
                finish_reason=provider_response.finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=provider_response.prompt_tokens,
            completion_tokens=provider_response.completion_tokens,
            total_tokens=provider_response.total_tokens,
        ),
        token0=Token0Usage(
            original_prompt_tokens_estimate=total_tokens_before,
            optimized_prompt_tokens=provider_response.prompt_tokens,
            tokens_saved=tokens_saved,
            cost_saved_usd=round(cost_saved, 6),
            optimizations_applied=optimizations_applied,
            cache_hit=False,
            model_cascaded_to=model_cascaded_to,
        ),
    )
