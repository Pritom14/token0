"""Semantic response cache — cache LLM responses for similar image+prompt pairs.

Uses perceptual image hash + prompt text to create a cache key.
If a similar request was seen before, return the cached response (0 tokens).

v0.3: QJL-compressed fuzzy matching — similar (not just identical) images
can hit the cache using Hamming distance on compressed binary signatures.
Inspired by Google's TurboQuant (arXiv 2504.19874) QJL technique.

Works in both lite mode (in-memory dict) and full mode (Redis).
"""

import hashlib
import json
import logging

import numpy as np
from PIL import Image

from token0.storage.redis import get_redis

logger = logging.getLogger("token0.cache")

# ---------------------------------------------------------------------------
# JL projection matrix — fixed per process, seeded for determinism
# Projects 256-dim perceptual hash into 128-dim compressed signature
# ---------------------------------------------------------------------------
_JL_DIM = 128  # compressed signature size (bits)
_HASH_SIZE = 16  # perceptual hash grid (16x16 = 256 bits)
_HASH_DIM = _HASH_SIZE * _HASH_SIZE  # 256
_HAMMING_THRESHOLD = 18  # max Hamming distance for fuzzy match (~14% of 128)

_rng = np.random.RandomState(seed=42)
_JL_MATRIX = _rng.randn(_JL_DIM, _HASH_DIM).astype(np.float32)

# In-memory fuzzy index: model -> list of (signature_bytes, cache_key)
_fuzzy_index: dict[str, list[tuple[bytes, str]]] = {}


def _image_hash(pil_image: Image.Image, hash_size: int = _HASH_SIZE) -> str:
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


def _hash_to_vector(hash_hex: str) -> np.ndarray:
    """Convert a hex perceptual hash to a numpy float vector (+1/-1)."""
    hash_int = int(hash_hex, 16)
    bits = format(hash_int, f"0{_HASH_DIM}b")
    return np.array([1.0 if b == "1" else -1.0 for b in bits], dtype=np.float32)


def _jl_compress(hash_hex: str) -> bytes:
    """Compress a perceptual hash to a QJL binary signature.

    1. Convert hash to +1/-1 vector (256-dim)
    2. Project through random JL matrix (256 -> 128)
    3. Take sign bits -> 128-bit signature (16 bytes)
    """
    vec = _hash_to_vector(hash_hex)
    projected = _JL_MATRIX @ vec
    # Sign quantization: 1 if positive, 0 if negative
    sign_bits = (projected > 0).astype(np.uint8)
    # Pack into bytes (128 bits -> 16 bytes)
    return np.packbits(sign_bits).tobytes()


def _hamming_distance(sig_a: bytes, sig_b: bytes) -> int:
    """Compute Hamming distance between two binary signatures."""
    a = np.frombuffer(sig_a, dtype=np.uint8)
    b = np.frombuffer(sig_b, dtype=np.uint8)
    # XOR and popcount
    xor = np.bitwise_xor(a, b)
    return sum(bin(byte).count("1") for byte in xor)


def _prompt_hash(prompt: str) -> str:
    """Hash the prompt text. Normalize whitespace and case."""
    normalized = " ".join(prompt.lower().strip().split())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def make_cache_key(pil_image: Image.Image, prompt: str, model: str) -> str:
    """Create a cache key from image + prompt + model."""
    img_h = _image_hash(pil_image)
    prompt_h = _prompt_hash(prompt)
    return f"token0:cache:{model}:{img_h}:{prompt_h}"


def _parse_cache_key(cache_key: str) -> tuple[str, str, str]:
    """Extract (model, img_hash, prompt_hash) from a cache key."""
    parts = cache_key.split(":")
    # token0:cache:{model}:{img_hash}:{prompt_hash}
    return parts[2], parts[3], parts[4]


async def get_cached_response(
    cache_key: str,
    fuzzy: bool = True,
) -> dict | None:
    """Look up a cached response. Tries exact match first, then fuzzy.

    Returns parsed dict or None.
    """
    cache = get_redis()

    # Step 1: Exact match (O(1), fast)
    try:
        cached = await cache.get(cache_key)
        if cached:
            return json.loads(cached)
    except Exception:
        pass

    # Step 2: Fuzzy match via QJL signatures
    if not fuzzy:
        return None

    try:
        model, img_hash, prompt_hash = _parse_cache_key(cache_key)
        query_sig = _jl_compress(img_hash)

        # Search fuzzy index for this model
        candidates = _fuzzy_index.get(model, [])
        best_key = None
        best_distance = _HAMMING_THRESHOLD + 1

        for sig, stored_key in candidates:
            # Must match same prompt hash (fuzzy is image-only)
            _, _, stored_prompt_hash = _parse_cache_key(stored_key)
            if stored_prompt_hash != prompt_hash:
                continue

            dist = _hamming_distance(query_sig, sig)
            if dist < best_distance:
                best_distance = dist
                best_key = stored_key

        if best_key is not None:
            cached = await cache.get(best_key)
            if cached:
                logger.info(
                    "fuzzy cache hit: hamming_distance=%d (threshold=%d)",
                    best_distance,
                    _HAMMING_THRESHOLD,
                )
                return json.loads(cached)
    except Exception:
        logger.debug("fuzzy cache lookup failed", exc_info=True)

    return None


async def set_cached_response(
    cache_key: str,
    response: dict,
    ttl_seconds: int = 3600,
) -> None:
    """Cache a response and index its QJL signature for fuzzy matching."""
    cache = get_redis()
    try:
        await cache.set(cache_key, json.dumps(response), ex=ttl_seconds)

        # Add to fuzzy index
        model, img_hash, _ = _parse_cache_key(cache_key)
        sig = _jl_compress(img_hash)

        if model not in _fuzzy_index:
            _fuzzy_index[model] = []

        # Avoid duplicate entries for same key
        _fuzzy_index[model] = [(s, k) for s, k in _fuzzy_index[model] if k != cache_key]
        _fuzzy_index[model].append((sig, cache_key))
    except Exception:
        pass  # cache failures shouldn't break the request


def clear_fuzzy_index() -> None:
    """Clear the in-memory fuzzy index. Useful for tests."""
    _fuzzy_index.clear()


def get_fuzzy_index_size() -> int:
    """Return total entries across all models in the fuzzy index."""
    return sum(len(entries) for entries in _fuzzy_index.values())
