"""Benchmark: QJL fuzzy cache vs exact-match-only cache.

Demonstrates token savings from fuzzy cache hits on similar images.
Simulates a real workload: same product/document photographed multiple
times with slight variations (lighting, angle, compression artifacts).

Usage:
    python -m benchmarks.bench_fuzzy_cache
"""

import asyncio
import time

import numpy as np
from PIL import Image

from token0.optimization.cache import (
    _hamming_distance,
    _image_hash,
    _jl_compress,
    clear_fuzzy_index,
    get_cached_response,
    get_fuzzy_index_size,
    make_cache_key,
    set_cached_response,
)
from token0.storage.redis import MemoryCache

# Estimated tokens per image (GPT-4o high detail, ~800x600)
TOKENS_PER_IMAGE = 765
COST_PER_TOKEN = 2.50 / 1_000_000  # GPT-4o input price


def _make_base_image(seed: int, width=800, height=600) -> Image.Image:
    """Create a unique base image (simulates a product photo or document)."""
    rng = np.random.RandomState(seed=seed)
    pixels = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _add_variation(base: Image.Image, variation_seed: int, noise_level: int = 15) -> Image.Image:
    """Add slight variation to an image (simulates re-photo, compression, etc.)."""
    pixels = np.array(base)
    rng = np.random.RandomState(seed=variation_seed)
    noise = rng.randint(-noise_level, noise_level + 1, pixels.shape, dtype=np.int16)
    noisy = np.clip(pixels.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


async def run_benchmark():
    import token0.storage.redis as redis_mod

    redis_mod._memory_cache.clear()
    redis_mod.pool = MemoryCache()
    clear_fuzzy_index()

    print("=" * 80)
    print("  QJL Fuzzy Cache Benchmark")
    print("=" * 80)

    # --- Setup: create base images and variations ---
    num_unique_images = 20
    variations_per_image = 5  # each base image has 5 slight variations
    prompt = "describe this product image"

    base_images = [_make_base_image(seed=i) for i in range(num_unique_images)]
    variation_images = []
    for i, base in enumerate(base_images):
        for v in range(variations_per_image):
            variation_images.append((i, _add_variation(base, variation_seed=i * 100 + v)))

    total_requests = num_unique_images + len(variation_images)
    print(f"\n  Setup: {num_unique_images} unique images, {variations_per_image} variations each")
    print(f"  Total requests: {total_requests}")
    print(f"  Tokens per image (GPT-4o): {TOKENS_PER_IMAGE}")

    # --- Benchmark 1: Exact-match only ---
    print("\n  --- Exact Match Only ---\n")
    redis_mod._memory_cache.clear()
    clear_fuzzy_index()

    exact_hits = 0
    exact_misses = 0
    start = time.time()

    # First pass: cache base images
    for i, base in enumerate(base_images):
        key = make_cache_key(base, prompt, "gpt-4o")
        await set_cached_response(key, {"content": f"response_{i}"})

    # Second pass: query with variations (exact match only)
    for base_idx, var_img in variation_images:
        key = make_cache_key(var_img, prompt, "gpt-4o")
        result = await get_cached_response(key, fuzzy=False)
        if result:
            exact_hits += 1
        else:
            exact_misses += 1

    exact_time = time.time() - start
    exact_tokens_used = exact_misses * TOKENS_PER_IMAGE
    exact_cost = exact_tokens_used * COST_PER_TOKEN

    print(f"  Hits:   {exact_hits}/{len(variation_images)}")
    print(f"  Misses: {exact_misses}/{len(variation_images)}")
    print(f"  Tokens used: {exact_tokens_used:,}")
    print(f"  Cost: ${exact_cost:.4f}")
    print(f"  Time: {exact_time * 1000:.1f}ms")

    # --- Benchmark 2: Fuzzy match (QJL) ---
    print("\n  --- QJL Fuzzy Match ---\n")
    redis_mod._memory_cache.clear()
    clear_fuzzy_index()

    fuzzy_hits = 0
    fuzzy_misses = 0
    start = time.time()

    # First pass: cache base images
    for i, base in enumerate(base_images):
        key = make_cache_key(base, prompt, "gpt-4o")
        await set_cached_response(key, {"content": f"response_{i}"})

    # Second pass: query with variations (fuzzy match enabled)
    for base_idx, var_img in variation_images:
        key = make_cache_key(var_img, prompt, "gpt-4o")
        result = await get_cached_response(key, fuzzy=True)
        if result:
            fuzzy_hits += 1
        else:
            fuzzy_misses += 1

    fuzzy_time = time.time() - start
    fuzzy_tokens_used = fuzzy_misses * TOKENS_PER_IMAGE
    fuzzy_cost = fuzzy_tokens_used * COST_PER_TOKEN

    print(f"  Hits:   {fuzzy_hits}/{len(variation_images)}")
    print(f"  Misses: {fuzzy_misses}/{len(variation_images)}")
    print(f"  Tokens used: {fuzzy_tokens_used:,}")
    print(f"  Cost: ${fuzzy_cost:.4f}")
    print(f"  Time: {fuzzy_time * 1000:.1f}ms")
    print(f"  Fuzzy index size: {get_fuzzy_index_size()} entries")

    # --- Hamming distance analysis ---
    print("\n  --- Hamming Distance Analysis ---\n")
    distances_similar = []
    distances_different = []

    for i, base in enumerate(base_images[:5]):
        base_hash = _image_hash(base)
        base_sig = _jl_compress(base_hash)

        # Similar: variations of same base
        for v in range(variations_per_image):
            var = _add_variation(base, variation_seed=i * 100 + v)
            var_hash = _image_hash(var)
            var_sig = _jl_compress(var_hash)
            distances_similar.append(_hamming_distance(base_sig, var_sig))

        # Different: other base images
        for j in range(5):
            if i == j:
                continue
            other_hash = _image_hash(base_images[j])
            other_sig = _jl_compress(other_hash)
            distances_different.append(_hamming_distance(base_sig, other_sig))

    print(
        f"  Similar images:   avg={np.mean(distances_similar):.1f}, "
        f"min={min(distances_similar)}, max={max(distances_similar)}"
    )
    print(
        f"  Different images: avg={np.mean(distances_different):.1f}, "
        f"min={min(distances_different)}, max={max(distances_different)}"
    )

    # --- Summary ---
    print(f"\n  {'=' * 70}")
    print("  SUMMARY")
    print(f"  {'=' * 70}")
    print(f"  {'':30s} {'Exact':>12s} {'Fuzzy (QJL)':>12s} {'Improvement':>12s}")
    print(f"  {'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(
        f"  {'Cache hits':30s} {exact_hits:>12d} {fuzzy_hits:>12d} "
        f"{'+' + str(fuzzy_hits - exact_hits):>12s}"
    )
    print(
        f"  {'Cache misses':30s} {exact_misses:>12d} {fuzzy_misses:>12d} "
        f"{exact_misses - fuzzy_misses:>12d}"
    )
    print(
        f"  {'Tokens used':30s} {exact_tokens_used:>12,} {fuzzy_tokens_used:>12,} "
        f"{exact_tokens_used - fuzzy_tokens_used:>12,}"
    )

    if exact_tokens_used > 0:
        savings_pct = (exact_tokens_used - fuzzy_tokens_used) / exact_tokens_used * 100
        print(f"  {'Token savings':30s} {'':>12s} {'':>12s} {savings_pct:>11.1f}%")

    print(
        f"  {'Cost (GPT-4o)':30s} ${exact_cost:>11.4f} ${fuzzy_cost:>11.4f} "
        f"${exact_cost - fuzzy_cost:>11.4f}"
    )

    # Scale projections
    print("\n  At scale (100K images/day, 20% are variations):")
    daily_variations = 20_000
    exact_miss_rate = exact_misses / len(variation_images)
    fuzzy_miss_rate = fuzzy_misses / len(variation_images)
    daily_exact_tokens = daily_variations * TOKENS_PER_IMAGE * exact_miss_rate
    daily_fuzzy_tokens = daily_variations * TOKENS_PER_IMAGE * fuzzy_miss_rate
    monthly_exact = daily_exact_tokens * 30 * COST_PER_TOKEN
    monthly_fuzzy = daily_fuzzy_tokens * 30 * COST_PER_TOKEN
    print(f"  Exact-only monthly cost:  ${monthly_exact:,.2f}")
    print(f"  Fuzzy cache monthly cost: ${monthly_fuzzy:,.2f}")
    print(f"  Monthly savings:          ${monthly_exact - monthly_fuzzy:,.2f}")
    print(f"  {'=' * 70}")

    # Memory overhead
    sig_bytes = get_fuzzy_index_size() * 16  # 16 bytes per signature
    key_bytes = get_fuzzy_index_size() * 80  # ~80 bytes per cache key string
    print(
        f"\n  Memory overhead: {(sig_bytes + key_bytes) / 1024:.1f} KB "
        f"for {get_fuzzy_index_size()} entries "
        f"({sig_bytes} bytes signatures + {key_bytes} bytes keys)"
    )
    print(f"  At 1M entries: ~{(1_000_000 * 96) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
