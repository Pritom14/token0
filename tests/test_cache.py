"""Tests for semantic cache with QJL fuzzy matching."""

import numpy as np
import pytest
from PIL import Image

from token0.optimization.cache import (
    _hamming_distance,
    _image_hash,
    _jl_compress,
    _prompt_hash,
    clear_fuzzy_index,
    get_cached_response,
    get_fuzzy_index_size,
    make_cache_key,
    set_cached_response,
)
from token0.storage.redis import MemoryCache


@pytest.fixture(autouse=True)
def _init_cache():
    """Initialize in-memory cache and clear fuzzy index for each test."""
    import token0.storage.redis as redis_mod

    redis_mod._memory_cache.clear()
    redis_mod.pool = MemoryCache()
    clear_fuzzy_index()
    yield
    clear_fuzzy_index()
    redis_mod._memory_cache.clear()


def _make_pil_image(width=800, height=600, color="red"):
    """Create a PIL Image directly."""
    return Image.new("RGB", (width, height), color=color)


def _make_gradient_image(width=800, height=600, seed=0):
    """Create a gradient image with unique content (not solid color)."""
    rng = np.random.RandomState(seed=seed)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _make_slightly_different_image(base_seed=0, noise_level=10):
    """Create an image that's visually similar but not identical to a gradient."""
    base = _make_gradient_image(seed=base_seed)
    pixels = np.array(base)
    rng = np.random.RandomState(seed=123)
    noise = rng.randint(-noise_level, noise_level + 1, pixels.shape, dtype=np.int16)
    noisy = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


class TestPerceptualHash:
    def test_identical_images_same_hash(self):
        """Identical images produce the same hash."""
        img1 = _make_pil_image(800, 600, "red")
        img2 = _make_pil_image(800, 600, "red")
        assert _image_hash(img1) == _image_hash(img2)

    def test_different_images_different_hash(self):
        """Completely different images produce different hashes."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_gradient_image(seed=99)
        assert _image_hash(img1) != _image_hash(img2)

    def test_hash_is_hex_string(self):
        """Hash is a valid hex string of expected length."""
        img = _make_pil_image()
        h = _image_hash(img)
        assert isinstance(h, str)
        int(h, 16)  # should not raise
        assert len(h) == 64  # 256 bits / 4 = 64 hex chars


class TestJLCompression:
    def test_signature_size(self):
        """Compressed signature is exactly 16 bytes (128 bits)."""
        img = _make_pil_image()
        h = _image_hash(img)
        sig = _jl_compress(h)
        assert isinstance(sig, bytes)
        assert len(sig) == 16

    def test_identical_hashes_same_signature(self):
        """Same hash always produces same signature (deterministic)."""
        img = _make_pil_image()
        h = _image_hash(img)
        sig1 = _jl_compress(h)
        sig2 = _jl_compress(h)
        assert sig1 == sig2

    def test_similar_images_close_signatures(self):
        """Visually similar images have small Hamming distance."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_slightly_different_image(base_seed=0, noise_level=5)

        h1 = _image_hash(img1)
        h2 = _image_hash(img2)
        sig1 = _jl_compress(h1)
        sig2 = _jl_compress(h2)

        dist = _hamming_distance(sig1, sig2)
        # Similar images should have small distance
        assert dist < 30  # well under half of 128

    def test_different_images_far_signatures(self):
        """Very different images have large Hamming distance."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_gradient_image(seed=99)

        sig1 = _jl_compress(_image_hash(img1))
        sig2 = _jl_compress(_image_hash(img2))

        dist = _hamming_distance(sig1, sig2)
        # Very different images should be far apart
        assert dist > 20


class TestHammingDistance:
    def test_identical_signatures_zero_distance(self):
        """Identical signatures have distance 0."""
        sig = b"\xff" * 16
        assert _hamming_distance(sig, sig) == 0

    def test_completely_different_max_distance(self):
        """Opposite signatures have distance 128."""
        sig_a = b"\x00" * 16
        sig_b = b"\xff" * 16
        assert _hamming_distance(sig_a, sig_b) == 128

    def test_one_bit_flip(self):
        """Single bit difference = distance 1."""
        sig_a = b"\x00" * 16
        sig_b = b"\x01" + b"\x00" * 15
        assert _hamming_distance(sig_a, sig_b) == 1


class TestExactCacheMatch:
    @pytest.mark.asyncio
    async def test_exact_match_hit(self):
        """Exact key match returns cached response."""
        img = _make_pil_image()
        key = make_cache_key(img, "describe this", "gpt-4o")
        response = {"model": "gpt-4o", "content": "A red rectangle"}

        await set_cached_response(key, response)
        result = await get_cached_response(key)

        assert result is not None
        assert result["content"] == "A red rectangle"

    @pytest.mark.asyncio
    async def test_exact_match_miss(self):
        """Non-existent key returns None."""
        img = _make_pil_image()
        key = make_cache_key(img, "describe this", "gpt-4o")
        result = await get_cached_response(key)
        assert result is None

    @pytest.mark.asyncio
    async def test_different_prompts_no_match(self):
        """Different prompts don't match even for same image."""
        img = _make_pil_image()
        key1 = make_cache_key(img, "describe this", "gpt-4o")
        key2 = make_cache_key(img, "classify this", "gpt-4o")

        await set_cached_response(key1, {"content": "description"})
        result = await get_cached_response(key2)

        assert result is None

    @pytest.mark.asyncio
    async def test_different_models_no_match(self):
        """Different models don't share cache."""
        img = _make_pil_image()
        key1 = make_cache_key(img, "describe this", "gpt-4o")
        key2 = make_cache_key(img, "describe this", "claude-sonnet-4-6")

        await set_cached_response(key1, {"content": "gpt response"})
        result = await get_cached_response(key2)

        assert result is None


class TestFuzzyCacheMatch:
    @pytest.mark.asyncio
    async def test_fuzzy_match_similar_image(self):
        """Similar images hit cache via fuzzy matching."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_slightly_different_image(base_seed=0, noise_level=5)
        prompt = "describe this image"

        # Cache response for img1
        key1 = make_cache_key(img1, prompt, "gpt-4o")
        await set_cached_response(key1, {"content": "A gradient image"})

        # Query with img2 (similar but not identical)
        key2 = make_cache_key(img2, prompt, "gpt-4o")

        # If hashes are identical, it's an exact match (still valid)
        # If hashes differ, fuzzy should find it
        result = await get_cached_response(key2)
        assert result is not None
        assert result["content"] == "A gradient image"

    @pytest.mark.asyncio
    async def test_fuzzy_no_match_very_different(self):
        """Very different images don't fuzzy match."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_gradient_image(seed=99)
        prompt = "describe this"

        key1 = make_cache_key(img1, prompt, "gpt-4o")
        await set_cached_response(key1, {"content": "image 1"})

        key2 = make_cache_key(img2, prompt, "gpt-4o")
        result = await get_cached_response(key2)

        # Very different images should not fuzzy match
        assert result is None

    @pytest.mark.asyncio
    async def test_fuzzy_requires_same_prompt(self):
        """Fuzzy match only works for same prompt, different image."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_slightly_different_image(base_seed=0, noise_level=5)

        key1 = make_cache_key(img1, "describe this", "gpt-4o")
        await set_cached_response(key1, {"content": "A gradient image"})

        # Different prompt — should NOT match even with similar image
        key2 = make_cache_key(img2, "classify this", "gpt-4o")
        result = await get_cached_response(key2)
        assert result is None

    @pytest.mark.asyncio
    async def test_fuzzy_disabled(self):
        """fuzzy=False skips fuzzy matching."""
        img1 = _make_gradient_image(seed=0)
        img2 = _make_slightly_different_image(base_seed=0, noise_level=5)
        prompt = "describe this"

        key1 = make_cache_key(img1, prompt, "gpt-4o")
        await set_cached_response(key1, {"content": "cached"})

        key2 = make_cache_key(img2, prompt, "gpt-4o")

        # If keys are different, fuzzy=False should miss
        if key1 != key2:
            result = await get_cached_response(key2, fuzzy=False)
            assert result is None

    @pytest.mark.asyncio
    async def test_fuzzy_index_tracks_entries(self):
        """Fuzzy index grows as entries are cached."""
        assert get_fuzzy_index_size() == 0

        img = _make_pil_image()
        key = make_cache_key(img, "test", "gpt-4o")
        await set_cached_response(key, {"content": "test"})

        assert get_fuzzy_index_size() == 1

    @pytest.mark.asyncio
    async def test_fuzzy_index_no_duplicates(self):
        """Re-caching same key doesn't duplicate in fuzzy index."""
        img = _make_pil_image()
        key = make_cache_key(img, "test", "gpt-4o")

        await set_cached_response(key, {"content": "v1"})
        await set_cached_response(key, {"content": "v2"})

        assert get_fuzzy_index_size() == 1


class TestPromptHash:
    def test_normalized_whitespace(self):
        """Extra whitespace is normalized."""
        assert _prompt_hash("  hello   world  ") == _prompt_hash("hello world")

    def test_case_insensitive(self):
        """Case is normalized."""
        assert _prompt_hash("Hello World") == _prompt_hash("hello world")

    def test_different_prompts_different_hash(self):
        """Different prompts produce different hashes."""
        assert _prompt_hash("describe this") != _prompt_hash("classify this")
