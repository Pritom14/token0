"""Optimization router — decides which optimizations to apply to each image."""

import math
from dataclasses import dataclass

from token0.optimization.analyzer import ImageAnalysis


@dataclass
class OptimizationPlan:
    # What to do
    resize: bool = False
    target_width: int | None = None
    target_height: int | None = None

    use_ocr_route: bool = False  # Extract text via OCR instead of sending image
    force_detail_low: bool = False  # Switch to low-detail mode (85 tokens flat)
    recompress_jpeg: bool = False
    jpeg_quality: int = 85

    # Model cascade
    recommended_model: str | None = None  # If set, suggests a cheaper model

    # Why
    reasons: list[str] | None = None
    estimated_tokens_before: int = 0
    estimated_tokens_after: int = 0

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


def get_provider_from_model(model: str) -> str:
    """Infer provider from model name."""
    model_lower = model.lower()
    if any(k in model_lower for k in ("gpt", "o1", "o3", "o4")):
        return "openai"
    if any(k in model_lower for k in ("claude", "sonnet", "opus", "haiku")):
        return "anthropic"
    if any(k in model_lower for k in ("gemini", "palm")):
        return "google"
    # Known Ollama/local models
    ollama_models = (
        "moondream",
        "llava",
        "minicpm",
        "bakllava",
        "cogvlm",
        "yi-vl",
        "llava-phi",
        "nanollava",
        "llama3.2-vision",
        "llama3.2",
        "gemma3",
        "granite3.2",
        "granite3",
        "qwen2.5vl",
        "qwen3-vl",
    )
    if any(k in model_lower for k in ollama_models):
        return "ollama"
    return "openai"  # default


def _tile_optimized_resize(width: int, height: int) -> tuple[int, int]:
    """Resize to minimize OpenAI tile count.

    OpenAI tiles at 512px. A 770x770 image = 4 tiles (765 tokens).
    Resizing to 768x768 still = 4 tiles. But 512x512 = 1 tile (255 tokens).

    Strategy: scale down to fit in fewer tiles while keeping the shortest side ≥ 512.
    After OpenAI's internal scaling (max 2048, then shortest side to 768),
    we want the result to land on exact tile boundaries.
    """
    # Simulate OpenAI's internal scaling
    max_dim = max(width, height)
    if max_dim > 2048:
        scale = 2048 / max_dim
        width = int(width * scale)
        height = int(height * scale)

    min_dim = min(width, height)
    if min_dim > 768:
        scale = 768 / min_dim
        width = int(width * scale)
        height = int(height * scale)

    # Current tile count
    current_tiles = math.ceil(width / 512) * math.ceil(height / 512)

    # Try reducing to fit in fewer tiles
    # Find the largest dimensions that use one fewer row or column of tiles
    best_w, best_h = width, height
    best_tiles = current_tiles

    for target_cols in range(1, math.ceil(width / 512) + 1):
        for target_rows in range(1, math.ceil(height / 512) + 1):
            tiles = target_cols * target_rows
            if tiles >= current_tiles:
                continue
            # Max dimensions for this tile count
            tw = target_cols * 512
            th = target_rows * 512
            # Scale original to fit, maintaining aspect ratio
            scale_w = tw / width
            scale_h = th / height
            scale = min(scale_w, scale_h)
            nw = int(width * scale)
            nh = int(height * scale)
            # Only accept if we're not shrinking too much (keep at least 60% of pixels)
            if nw * nh >= width * height * 0.4 and tiles < best_tiles:
                best_w, best_h = nw, nh
                best_tiles = tiles

    return best_w, best_h


# Model cascade: cheaper alternatives for simple tasks
MODEL_CASCADE = {
    # expensive model → cheaper alternative for simple tasks
    "gpt-4o": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1-mini",
    "gpt-4.1-mini": "gpt-4.1-nano",
    "claude-opus-4-6": "claude-sonnet-4-6",
    "claude-sonnet-4-6": "claude-haiku-4-5-20251001",
    "gemini-2.5-pro": "gemini-2.5-flash",
}


def plan_optimization(
    analysis: ImageAnalysis,
    model: str,
    detail_override: str | None = None,
    prompt_detail: str = "auto",
    enable_cascade: bool = True,
    allow_ocr: bool = True,
) -> OptimizationPlan:
    """Decide the optimal compression strategy for a given image + target model.

    Args:
        analysis: Image analysis results
        model: Target model name
        detail_override: Force 'low' or 'high' detail (user override)
        prompt_detail: Result from prompt classifier ('low', 'high', 'auto')
        enable_cascade: Whether to suggest cheaper models for simple tasks
        allow_ocr: Set False to plan without the OCR route (fallback after OCR
            turned out more expensive than the image)
    """
    from token0.config import settings

    provider = get_provider_from_model(model)
    plan = OptimizationPlan()

    # --- Optimization 1: OCR Route ---
    if analysis.is_mostly_text and prompt_detail == "layout":
        plan.reasons.append(
            f"text_density={analysis.text_density:.2f} but prompt needs layout — skip OCR"
        )
    elif analysis.is_mostly_text and allow_ocr:
        estimated_image_tokens = _estimate_tokens(analysis, provider, "high", model)
        estimated_ocr_tokens = min(
            estimate_ocr_tokens(analysis, provider, model), estimated_image_tokens
        )

        # Some Ollama models encode images with extremely few tokens — far fewer
        # than our estimation formula predicts. For these, OCR text output
        # (200-700 tokens) costs MORE than just sending the image.
        # Only skip OCR for models confirmed to use <50 tokens per image.
        _ultra_efficient_models = ("llama3.2-vision", "llama3.2")
        is_ultra_efficient = any(k in model.lower() for k in _ultra_efficient_models)
        if provider == "ollama" and is_ultra_efficient:
            plan.reasons.append(
                f"text_density={analysis.text_density:.2f} but Ollama model — "
                f"skip OCR (image tokens ~{estimated_image_tokens} likely cheaper than OCR text)"
            )
        elif estimated_image_tokens <= OCR_MIN_IMAGE_TOKENS:
            plan.reasons.append(
                f"text_density={analysis.text_density:.2f} but image is only "
                f"{estimated_image_tokens} tokens — skip OCR"
            )
        else:
            plan.use_ocr_route = True
            plan.reasons.append(
                f"text_density={analysis.text_density:.2f} "
                f"> {settings.text_density_threshold} — OCR route"
            )
            plan.estimated_tokens_before = estimated_image_tokens
            plan.estimated_tokens_after = estimated_ocr_tokens
            # Still suggest cascade for OCR route (text processing is simple)
            if enable_cascade and model in MODEL_CASCADE:
                plan.recommended_model = MODEL_CASCADE[model]
                plan.reasons.append(f"cascade → {plan.recommended_model} (text task)")
            return plan

    # Layout prompts (UI / computer-use): the caller maps coordinates the model returns
    # back onto the original image, so dimensions must not change and detail must stay.
    keep_dimensions = prompt_detail == "layout"
    if keep_dimensions:
        plan.reasons.append("prompt needs layout — keep original dimensions")

    # --- Optimization 2: Prompt-Aware Detail Mode (NEW) ---
    if provider == "openai" and detail_override != "high" and not keep_dimensions:
        # Priority: user override > prompt classification > size heuristic
        if detail_override == "low":
            plan.force_detail_low = True
            plan.reasons.append("detail_override=low")
        elif prompt_detail == "low":
            plan.force_detail_low = True
            plan.reasons.append("prompt-aware → low detail (simple task)")
        elif prompt_detail == "auto" and analysis.width <= 512 and analysis.height <= 512:
            plan.force_detail_low = True
            plan.reasons.append("image ≤512px — low detail equivalent")

    # --- Optimization 2b: Low detail only sees 512px — don't upload more ---
    if plan.force_detail_low and max(analysis.width, analysis.height) > 512:
        scale = 512 / max(analysis.width, analysis.height)
        plan.resize = True
        plan.target_width = max(1, int(analysis.width * scale))
        plan.target_height = max(1, int(analysis.height * scale))
        plan.reasons.append(
            f"low detail → resize {analysis.width}x{analysis.height} "
            f"→ {plan.target_width}x{plan.target_height}"
        )

    # --- Optimization 3: Tile-Optimized Resize (NEW for OpenAI) ---
    # Skipped for text/detail prompts: shrinking below the provider's working size
    # (768px short side) makes small print illegible.
    if (
        provider == "openai"
        and not plan.force_detail_low
        and not keep_dimensions
        and prompt_detail != "high"
    ):
        opt_w, opt_h = _tile_optimized_resize(analysis.width, analysis.height)
        # Check if tile-optimized resize saves tiles
        current_tokens = _estimate_tokens(analysis, provider, "high", model)
        from token0.optimization.analyzer import estimate_openai_tokens

        optimized_tokens = estimate_openai_tokens(opt_w, opt_h, "high")

        if optimized_tokens < current_tokens:
            plan.resize = True
            plan.target_width = opt_w
            plan.target_height = opt_h
            plan.reasons.append(
                f"tile-optimized resize {analysis.width}x{analysis.height} → {opt_w}x{opt_h} "
                f"({current_tokens} → {optimized_tokens} tokens)"
            )

    # --- Optimization 4: Standard Smart Resize ---
    if not plan.resize and not keep_dimensions and provider == "anthropic":
        from token0.optimization.analyzer import (
            CLAUDE_STANDARD_LIMITS,
            claude_image_limits,
            claude_resized_size,
        )

        limits = claude_image_limits(model)
        reason = ""
        if limits != CLAUDE_STANDARD_LIMITS and prompt_detail == "low":
            # High-res tier costs up to ~3x the tokens; a simple prompt doesn't need it.
            limits, reason = CLAUDE_STANDARD_LIMITS, " (simple prompt, standard-tier size)"
        target_w, target_h = claude_resized_size(analysis.width, analysis.height, *limits)
        if (target_w, target_h) != (analysis.width, analysis.height):
            plan.resize = True
            plan.target_width, plan.target_height = target_w, target_h
            plan.reasons.append(
                f"resize {analysis.width}x{analysis.height} → {target_w}x{target_h}{reason}"
            )
    elif not plan.resize and not keep_dimensions:
        max_dim = settings.max_image_dimension
        if provider == "openai":
            max_dim = 2048

        if max(analysis.width, analysis.height) > max_dim:
            scale = max_dim / max(analysis.width, analysis.height)
            plan.resize = True
            plan.target_width = int(analysis.width * scale)
            plan.target_height = int(analysis.height * scale)
            plan.reasons.append(
                f"resize {analysis.width}x{analysis.height} "
                f"→ {plan.target_width}x{plan.target_height}"
            )

    # --- Optimization 5: JPEG Recompression ---
    if analysis.format in ("png", "bmp", "tiff") and not analysis.has_transparency:
        plan.recompress_jpeg = True
        plan.jpeg_quality = settings.jpeg_quality
        plan.reasons.append(f"convert {analysis.format} → jpeg q={settings.jpeg_quality}")

    # --- Optimization 6: Model Cascade (NEW) ---
    if enable_cascade and prompt_detail == "low" and model in MODEL_CASCADE:
        plan.recommended_model = MODEL_CASCADE[model]
        plan.reasons.append(f"cascade → {plan.recommended_model} (simple task)")

    # Calculate token estimates
    plan.estimated_tokens_before = _estimate_tokens(analysis, provider, "high", model)

    if plan.force_detail_low:
        plan.estimated_tokens_after = 85
    elif plan.resize:
        plan.estimated_tokens_after = _estimate_tokens_for_dims(
            plan.target_width, plan.target_height, provider, model
        )
    else:
        plan.estimated_tokens_after = plan.estimated_tokens_before

    return plan


# Below this many image tokens, OCR text can't realistically be cheaper — don't try.
OCR_MIN_IMAGE_TOKENS = 200

# Text pixels (at the resolution the provider actually processes) per OCR token.
# Measured on a 200dpi document page (~960) and a 1080p web screenshot (~430);
# rough by nature — the runtime check in transformer.resolve_ocr() uses the real text.
OCR_TEXT_PIXELS_PER_TOKEN = 700


def _processed_dims(
    width: int, height: int, provider: str, model: str | None = None
) -> tuple[int, int]:
    """Approximate dimensions the provider downsamples an image to before tokenizing."""
    if provider == "openai":
        scale = min(1.0, 2048 / max(width, height))
        w, h = width * scale, height * scale
        scale = min(1.0, 768 / min(w, h))
        return int(w * scale), int(h * scale)
    from token0.optimization.analyzer import claude_image_limits, claude_resized_size

    return claude_resized_size(width, height, *claude_image_limits(model))


def estimate_ocr_tokens(
    analysis: ImageAnalysis, provider: str = "openai", model: str | None = None
) -> int:
    """Rough token count of the text OCR would extract, from the text-covered area."""
    w, h = _processed_dims(analysis.width, analysis.height, provider, model)
    return max(30, int(w * h * analysis.text_density / OCR_TEXT_PIXELS_PER_TOKEN))


def _estimate_tokens(
    analysis: ImageAnalysis, provider: str, detail: str = "high", model: str | None = None
) -> int:
    from token0.optimization.analyzer import estimate_anthropic_tokens

    if provider == "openai":
        if detail == "low":
            return analysis.estimated_tokens_openai_low
        return analysis.estimated_tokens_openai_high
    elif provider == "anthropic" and model:
        return estimate_anthropic_tokens(analysis.width, analysis.height, model)
    else:
        return analysis.estimated_tokens_anthropic


def _estimate_tokens_for_dims(
    width: int, height: int, provider: str, model: str | None = None
) -> int:
    from token0.optimization.analyzer import estimate_anthropic_tokens, estimate_openai_tokens

    if provider == "openai":
        return estimate_openai_tokens(width, height, "high")
    elif provider == "anthropic":
        return estimate_anthropic_tokens(width, height, model)
    else:
        return estimate_anthropic_tokens(width, height)
