"""Main /v1/chat/completions endpoint — the core proxy."""

import json
import time
import uuid

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from token0.config import settings
from token0.models.db import Request
from token0.models.request import (
    ChatRequest,
    ChatResponse,
    Choice,
    Message,
    Token0Usage,
    UsageInfo,
)
from token0.optimization.analyzer import analyze_image
from token0.optimization.cache import (
    get_cached_response,
    make_cache_key,
    set_cached_response,
)
from token0.optimization.prompt_classifier import (
    classify_prompt_detail,
    extract_prompt_text,
)
from token0.optimization.router import (
    OptimizationPlan,
    get_provider_from_model,
    plan_optimization,
)
from token0.optimization.transformer import transform_image
from token0.optimization.video import process_video
from token0.providers.anthropic import AnthropicProvider
from token0.providers.base import BaseProvider, get_cost_per_token
from token0.providers.google import GoogleProvider
from token0.providers.ollama import OllamaProvider
from token0.providers.openai import OpenAIProvider
from token0.storage.postgres import async_session

router = APIRouter()


def _get_provider(provider_name: str, api_key: str | None = None) -> BaseProvider:
    """Instantiate the right provider with API key."""
    if provider_name == "openai":
        key = api_key or settings.openai_api_key
        if not key:
            raise HTTPException(
                400,
                "OpenAI API key required. Pass via X-Provider-Key header.",
            )
        return OpenAIProvider(api_key=key)
    elif provider_name == "anthropic":
        key = api_key or settings.anthropic_api_key
        if not key:
            raise HTTPException(
                400,
                "Anthropic API key required. Pass via X-Provider-Key header.",
            )
        return AnthropicProvider(api_key=key)
    elif provider_name == "google":
        key = api_key or settings.google_api_key
        if not key:
            raise HTTPException(
                400,
                "Google API key required. Pass via X-Provider-Key header.",
            )
        return GoogleProvider(api_key=key)
    elif provider_name == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url)
    else:
        raise HTTPException(400, f"Unsupported provider: {provider_name}")


def _optimize_messages(request: ChatRequest, prompt_detail: str):
    """Run the optimization pipeline on messages.

    Returns (optimized_messages, total_tokens_before, total_tokens_after,
    optimizations_applied, plans, model_cascaded_to, actual_model,
    first_pil_image).
    """
    actual_model = request.model
    model_cascaded_to = None
    optimized_messages = []
    total_tokens_before = 0
    total_tokens_after = 0
    optimizations_applied = []
    plans: list[OptimizationPlan] = []
    first_pil_image = None

    for msg in request.messages:
        if isinstance(msg.content, str):
            optimized_messages.append({"role": msg.role, "content": msg.content})
            continue

        optimized_parts = []
        for part in msg.content:
            if part.type == "text":
                optimized_parts.append({"type": "text", "text": part.text})
            elif part.type == "video_url" and part.video_url and request.token0_optimize:
                # Video optimization: extract keyframes, dedup, optimize each
                prompt_text = extract_prompt_text(request.messages)
                video_frames, video_stats = process_video(
                    part.video_url.url,
                    prompt=prompt_text,
                )
                optimizations_applied.append(
                    f"video: {video_stats['total_video_frames']} frames → "
                    f"{video_stats['frames_selected']} keyframes "
                    f"({video_stats['frame_reduction_pct']}% reduction)"
                )
                # Each keyframe goes through the image optimization pipeline
                for frame_img in video_frames:
                    import base64 as b64mod
                    import io as iomod

                    buf = iomod.BytesIO()
                    frame_img.save(buf, format="JPEG", quality=85)
                    frame_bytes = buf.getvalue()
                    frame_b64 = b64mod.b64encode(frame_bytes).decode()
                    frame_data = f"data:image/jpeg;base64,{frame_b64}"

                    frame_analysis, frame_raw, frame_pil = analyze_image(frame_data)
                    if first_pil_image is None:
                        first_pil_image = frame_pil

                    frame_plan = plan_optimization(
                        frame_analysis,
                        request.model,
                        detail_override=request.token0_detail_override,
                        prompt_detail=prompt_detail,
                        enable_cascade=request.token0_enable_cascade,
                    )
                    plans.append(frame_plan)
                    total_tokens_before += frame_plan.estimated_tokens_before
                    total_tokens_after += frame_plan.estimated_tokens_after

                    if frame_plan.use_ocr_route:
                        result = transform_image(frame_plan, frame_analysis, frame_raw, frame_pil)
                        optimized_parts.append(
                            {
                                "type": "text",
                                "text": f"[Extracted text from video frame]:\n{result['content']}",
                            }
                        )
                    elif any(
                        [frame_plan.resize, frame_plan.recompress_jpeg, frame_plan.force_detail_low]
                    ):
                        result = transform_image(frame_plan, frame_analysis, frame_raw, frame_pil)
                        detail = "low" if frame_plan.force_detail_low else "auto"
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
                                "image_url": {
                                    "url": frame_data,
                                    "detail": "auto",
                                },
                            }
                        )
                # Also estimate tokens for all dropped frames (what it would have cost)
                tokens_per_frame_avg = 765  # GPT-4o high detail estimate
                dropped_frames = video_stats["total_video_frames"] - video_stats["frames_selected"]
                total_tokens_before += dropped_frames * tokens_per_frame_avg

            elif part.type == "image_url" and part.image_url and request.token0_optimize:
                image_data = part.image_url.url
                analysis, raw_bytes, pil_image = analyze_image(image_data)

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
                            "text": (f"[Extracted text from image]:\n{result['content']}"),
                        }
                    )
                elif any([plan.resize, plan.recompress_jpeg, plan.force_detail_low]):
                    result = transform_image(plan, analysis, raw_bytes, pil_image)
                    detail = "low" if plan.force_detail_low else (part.image_url.detail or "auto")
                    optimized_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{result['media_type']};base64,{result['base64']}"),
                                "detail": detail,
                            },
                        }
                    )
                else:
                    optimized_parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                                "detail": part.image_url.detail,
                            },
                        }
                    )
            elif part.type == "image_url" and part.image_url:
                optimized_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": part.image_url.url,
                            "detail": part.image_url.detail,
                        },
                    }
                )

        optimized_messages.append({"role": msg.role, "content": optimized_parts})

    return (
        optimized_messages,
        total_tokens_before,
        total_tokens_after,
        optimizations_applied,
        plans,
        model_cascaded_to,
        actual_model,
        first_pil_image,
    )


async def _log_request(
    provider_name: str,
    actual_model: str,
    plans: list,
    optimization_type: str,
    tokens_original: int,
    tokens_actual: int,
    tokens_saved: int,
    cost_original: float,
    cost_actual: float,
    cost_saved: float,
    response_tokens: int,
    latency_ms: int,
    details: dict,
):
    """Log request to database."""
    async with async_session() as session:
        db_request = Request(
            provider=provider_name,
            model=actual_model,
            customer_id="00000000-0000-0000-0000-000000000000",
            image_count=len(plans),
            optimization_type=optimization_type,
            tokens_original_estimate=tokens_original,
            tokens_actual=tokens_actual,
            tokens_saved=tokens_saved,
            cost_original_estimate=cost_original,
            cost_actual=cost_actual,
            cost_saved=cost_saved,
            response_tokens=response_tokens,
            latency_ms=latency_ms,
            optimization_details=details,
        )
        session.add(db_request)
        await session.commit()


async def _handle_stream(
    request: ChatRequest,
    provider: BaseProvider,
    actual_model: str,
    optimized_messages: list[dict],
    total_tokens_before: int,
    total_tokens_after: int,
    optimizations_applied: list[str],
    plans: list[OptimizationPlan],
    model_cascaded_to: str | None,
    provider_name: str,
    prompt_detail: str,
    cache_key: str | None,
    start_time: float,
):
    """Generate SSE stream for streaming responses."""
    request_id = f"token0-{uuid.uuid4().hex[:12]}"
    full_content = ""
    prompt_tokens = 0
    completion_tokens = 0

    async for chunk in provider.stream_chat_completion(
        model=actual_model,
        messages=optimized_messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    ):
        if chunk.delta_content:
            full_content += chunk.delta_content

        if chunk.prompt_tokens is not None:
            prompt_tokens = chunk.prompt_tokens
        if chunk.completion_tokens is not None:
            completion_tokens = chunk.completion_tokens

        # Build OpenAI-compatible SSE chunk
        sse_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": chunk.model or actual_model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": chunk.finish_reason,
                }
            ],
        }

        if chunk.delta_content is not None:
            sse_data["choices"][0]["delta"]["content"] = chunk.delta_content

        # On final chunk, attach token0 stats
        if chunk.finish_reason:
            tokens_saved = max(0, total_tokens_before - total_tokens_after)
            original_cpt = get_cost_per_token(request.model, "input")
            actual_cpt = get_cost_per_token(actual_model, "input")
            cost_saved = max(
                0,
                total_tokens_before * original_cpt - total_tokens_after * actual_cpt,
            )

            sse_data["token0"] = {
                "original_prompt_tokens_estimate": total_tokens_before,
                "optimized_prompt_tokens": prompt_tokens or total_tokens_after,
                "tokens_saved": tokens_saved,
                "cost_saved_usd": round(cost_saved, 6),
                "optimizations_applied": optimizations_applied,
                "cache_hit": False,
                "model_cascaded_to": model_cascaded_to,
            }

        yield f"data: {json.dumps(sse_data)}\n\n"

    yield "data: [DONE]\n\n"

    # Post-stream: cache and log
    latency_ms = int((time.time() - start_time) * 1000)

    if cache_key:
        await set_cached_response(
            cache_key,
            {
                "model": actual_model,
                "content": full_content,
                "finish_reason": "stop",
                "completion_tokens": completion_tokens,
            },
        )

    tokens_saved = max(0, total_tokens_before - total_tokens_after)
    original_cpt = get_cost_per_token(request.model, "input")
    actual_cpt = get_cost_per_token(actual_model, "input")

    await _log_request(
        provider_name=provider_name,
        actual_model=actual_model,
        plans=plans,
        optimization_type=(", ".join(set(optimizations_applied)) or "none"),
        tokens_original=total_tokens_before,
        tokens_actual=prompt_tokens or total_tokens_after,
        tokens_saved=tokens_saved,
        cost_original=total_tokens_before * original_cpt,
        cost_actual=total_tokens_after * actual_cpt,
        cost_saved=max(
            0,
            total_tokens_before * original_cpt - total_tokens_after * actual_cpt,
        ),
        response_tokens=completion_tokens,
        latency_ms=latency_ms,
        details={
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
            "streamed": True,
        },
    )


async def _stream_cached_response(
    cached: dict,
    request_id: str,
    total_tokens_before: int,
    optimizations_applied: list[str],
    model_cascaded_to: str | None,
    original_model: str,
):
    """Simulate streaming from a cached response."""
    content = cached["content"]
    model = cached["model"]

    # Yield content in small chunks to simulate streaming
    chunk_size = 20
    for i in range(0, len(content), chunk_size):
        text_chunk = content[i : i + chunk_size]
        is_last = (i + chunk_size) >= len(content)

        sse_data = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text_chunk},
                    "finish_reason": "stop" if is_last else None,
                }
            ],
        }

        if is_last:
            cost_per_token = get_cost_per_token(original_model, "input")
            optimizations_applied.append("cache hit — 0 tokens")
            sse_data["token0"] = {
                "original_prompt_tokens_estimate": total_tokens_before,
                "optimized_prompt_tokens": 0,
                "tokens_saved": total_tokens_before,
                "cost_saved_usd": round(total_tokens_before * cost_per_token, 6),
                "optimizations_applied": optimizations_applied,
                "cache_hit": True,
                "model_cascaded_to": model_cascaded_to,
            }

        yield f"data: {json.dumps(sse_data)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/chat/completions")
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

    # --- Step 2: Determine provider ---
    provider_name = get_provider_from_model(request.model)

    # --- Step 3: Optimize messages ---
    (
        optimized_messages,
        total_tokens_before,
        total_tokens_after,
        optimizations_applied,
        plans,
        model_cascaded_to,
        actual_model,
        first_pil_image,
    ) = _optimize_messages(request, prompt_detail)

    # --- Step 4: Check semantic cache ---
    cache_key = None
    if request.token0_enable_cache and first_pil_image is not None and prompt_text:
        cache_key = make_cache_key(first_pil_image, prompt_text, actual_model)
        cached = await get_cached_response(cache_key)
        if cached:
            latency_ms = int((time.time() - start_time) * 1000)
            cost_per_input_token = get_cost_per_token(request.model, "input")
            optimizations_list = list(optimizations_applied)

            # Log cache hit
            await _log_request(
                provider_name=provider_name,
                actual_model=actual_model,
                plans=plans,
                optimization_type="cache_hit",
                tokens_original=total_tokens_before,
                tokens_actual=0,
                tokens_saved=total_tokens_before,
                cost_original=total_tokens_before * cost_per_input_token,
                cost_actual=0,
                cost_saved=total_tokens_before * cost_per_input_token,
                response_tokens=0,
                latency_ms=latency_ms,
                details={"cache_hit": True, "cache_key": cache_key},
            )

            # Stream cached response
            if request.stream:
                request_id = f"token0-{uuid.uuid4().hex[:12]}"
                return StreamingResponse(
                    _stream_cached_response(
                        cached,
                        request_id,
                        total_tokens_before,
                        optimizations_list,
                        model_cascaded_to,
                        request.model,
                    ),
                    media_type="text/event-stream",
                )

            # Non-streaming cached response
            optimizations_list.append("cache hit — 0 tokens")
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
                    optimizations_applied=optimizations_list,
                    cache_hit=True,
                    model_cascaded_to=model_cascaded_to,
                ),
            )

    # --- Step 5: Resolve provider ---
    actual_provider_name = get_provider_from_model(actual_model)
    provider = _get_provider(actual_provider_name, api_key=x_provider_key)

    # --- Step 6: Streaming or non-streaming ---
    if request.stream:
        return StreamingResponse(
            _handle_stream(
                request=request,
                provider=provider,
                actual_model=actual_model,
                optimized_messages=optimized_messages,
                total_tokens_before=total_tokens_before,
                total_tokens_after=total_tokens_after,
                optimizations_applied=optimizations_applied,
                plans=plans,
                model_cascaded_to=model_cascaded_to,
                provider_name=actual_provider_name,
                prompt_detail=prompt_detail,
                cache_key=cache_key,
                start_time=start_time,
            ),
            media_type="text/event-stream",
        )

    # --- Non-streaming path ---
    provider_response = await provider.chat_completion(
        model=actual_model,
        messages=optimized_messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    latency_ms = int((time.time() - start_time) * 1000)

    # Cache the response
    if cache_key:
        await set_cached_response(
            cache_key,
            {
                "model": provider_response.model,
                "content": provider_response.content,
                "finish_reason": provider_response.finish_reason,
                "completion_tokens": provider_response.completion_tokens,
            },
        )

    # Calculate savings
    tokens_saved = max(0, total_tokens_before - total_tokens_after)
    original_cost_per_token = get_cost_per_token(request.model, "input")
    actual_cost_per_token = get_cost_per_token(actual_model, "input")
    cost_before = total_tokens_before * original_cost_per_token
    cost_after = total_tokens_after * actual_cost_per_token
    cost_saved = max(0, cost_before - cost_after)

    # Log to database
    await _log_request(
        provider_name=actual_provider_name,
        actual_model=actual_model,
        plans=plans,
        optimization_type=(", ".join(set(optimizations_applied)) or "none"),
        tokens_original=total_tokens_before,
        tokens_actual=provider_response.prompt_tokens,
        tokens_saved=tokens_saved,
        cost_original=cost_before,
        cost_actual=cost_after,
        cost_saved=cost_saved,
        response_tokens=provider_response.completion_tokens,
        latency_ms=latency_ms,
        details={
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
