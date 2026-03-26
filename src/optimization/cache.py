"""Semantic response cache — cache LLM responses for similar image+prompt pairs.

Uses perceptual image hash + prompt text to create a cache key.
If a similar request was seen before, return the cached response (0 tokens).

Works in both lite mode (in-memory dict) and full mode (Redis).
"""

import hashlib
import json

from PIL import Image

from src.storage.redis import get_redis


def _image_hash(pil_image: Image.Image, hash_size: int = 16) -> str:
    """Compute a perceptual hash (average hash) for an image.

    Perceptual hashes are similar for visually similar images,
    unlike cryptographic hashes which change completely for any pixel change.
    """
    # Resize to small square, convert to grayscale
    img = pil_image.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    # Create binary hash: 1 if pixel > average, 0 otherwise
    bits = "".join("1" if p > avg else "0" for p in pixels)
    # Convert to hex for compact storage
    return hex(int(bits, 2))[2:].zfill(hash_size * hash_size // 4)


def _prompt_hash(prompt: str) -> str:
    """Hash the prompt text. Normalize whitespace and case."""
    normalized = " ".join(prompt.lower().strip().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def make_cache_key(pil_image: Image.Image, prompt: str, model: str) -> str:
    """Create a cache key from image + prompt + model."""
    img_h = _image_hash(pil_image)
    prompt_h = _prompt_hash(prompt)
    return f"token0:cache:{model}:{img_h}:{prompt_h}"


async def get_cached_response(cache_key: str) -> dict | None:
    """Look up a cached response. Returns parsed dict or None."""
    cache = get_redis()
    try:
        cached = await cache.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass
    return None


async def set_cached_response(
    cache_key: str,
    response: dict,
    ttl_seconds: int = 3600,
) -> None:
    """Cache a response. Default TTL: 1 hour."""
    cache = get_redis()
    try:
        await cache.set(cache_key, json.dumps(response), ex=ttl_seconds)
    except Exception:
        pass  # cache failures shouldn't break the request
