"""POST /v1/estimate — pre-call token cost estimator.

Takes an image + prompt + model, returns predicted token count and dollar cost
BEFORE making any LLM call. No provider API key required.

Useful for:
- Understanding image costs before committing to a provider
- Comparing GPT-4o vs Claude vs Gemini costs for a specific image
- Building cost dashboards without proxying real calls
"""

from fastapi import APIRouter
from pydantic import BaseModel

from token0.models.request import Message
from token0.optimization.analyzer import analyze_image
from token0.optimization.prompt_classifier import classify_prompt_detail, extract_prompt_text
from token0.optimization.router import get_provider_from_model, plan_optimization
from token0.providers.base import get_cost_per_token

router = APIRouter()


class ImageEstimate(BaseModel):
    original_tokens: int
    optimized_tokens: int
    tokens_saved: int
    cost_saved_usd: float
    optimizations: list[str]


class EstimateRequest(BaseModel):
    model: str
    messages: list[Message]


class EstimateResponse(BaseModel):
    model: str
    provider: str
    images: list[ImageEstimate]
    total_original_tokens: int
    total_optimized_tokens: int
    total_tokens_saved: int
    total_cost_saved_usd: float
    note: str | None = None


@router.post("/estimate", response_model=EstimateResponse)
async def estimate(request: EstimateRequest) -> EstimateResponse:
    """Estimate token cost savings for a request without making any LLM call."""
    provider_name = get_provider_from_model(request.model)
    # Convert Pydantic messages to dicts for prompt_classifier compatibility
    messages_as_dicts = [
        {"role": m.role, "content": m.content if isinstance(m.content, str)
         else [{"type": p.type, "text": p.text} for p in m.content]}
        for m in request.messages
    ]
    prompt_text = extract_prompt_text(messages_as_dicts)
    prompt_detail = classify_prompt_detail(prompt_text)

    image_estimates: list[ImageEstimate] = []
    skipped_remote = False

    for msg in request.messages:
        if isinstance(msg.content, str):
            continue
        for part in msg.content:
            if part.type != "image_url" or not part.image_url:
                continue

            url = part.image_url.url

            if not url.startswith("data:"):
                skipped_remote = True
                continue  # can't analyze remote URLs without fetching

            try:
                analysis, _, _ = analyze_image(url)
                plan = plan_optimization(
                    analysis,
                    request.model,
                    detail_override=part.image_url.detail,
                    prompt_detail=prompt_detail,
                    enable_cascade=False,  # cascade changes model pricing; excluded for clarity
                )
                cost_per_token = get_cost_per_token(request.model, "input")
                cost_before = plan.estimated_tokens_before * cost_per_token
                cost_after = plan.estimated_tokens_after * cost_per_token

                image_estimates.append(
                    ImageEstimate(
                        original_tokens=plan.estimated_tokens_before,
                        optimized_tokens=plan.estimated_tokens_after,
                        tokens_saved=max(0, plan.estimated_tokens_before - plan.estimated_tokens_after),  # noqa: E501
                        cost_saved_usd=round(max(0.0, cost_before - cost_after), 6),
                        optimizations=plan.reasons,
                    )
                )
            except Exception:
                continue  # skip unanalyzable images silently

    total_orig = sum(e.original_tokens for e in image_estimates)
    total_opt = sum(e.optimized_tokens for e in image_estimates)
    total_saved_tokens = sum(e.tokens_saved for e in image_estimates)
    total_saved_usd = round(sum(e.cost_saved_usd for e in image_estimates), 6)

    note = None
    if skipped_remote:
        note = "Remote image URLs were skipped — provide base64 data URIs for full estimates."

    return EstimateResponse(
        model=request.model,
        provider=provider_name,
        images=image_estimates,
        total_original_tokens=total_orig,
        total_optimized_tokens=total_opt,
        total_tokens_saved=total_saved_tokens,
        total_cost_saved_usd=total_saved_usd,
        note=note,
    )
