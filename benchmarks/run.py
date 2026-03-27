"""Token0 Benchmark — Measures actual savings vs direct LLM calls.

Tests ALL input types:
  - Single images (various sizes/formats)
  - Text-only (regression — should add zero overhead)
  - Multi-image (multiple images in one request)
  - Multi-turn (conversation with image history)
  - Mixed content (text-heavy + image in same request)
  - Task types (classification, extraction, description, comparison)

Usage:
    python -m benchmarks.run                              # defaults: moondream, all tests
    python -m benchmarks.run --model llava:7b              # use llava
    python -m benchmarks.run --suite images                # only image tests
    python -m benchmarks.run --suite text                  # only text passthrough
    python -m benchmarks.run --suite multi                 # multi-image + multi-turn
    python -m benchmarks.run --suite all                   # everything
"""

import argparse
import asyncio
import base64
import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from token0.optimization.analyzer import analyze_image
from token0.optimization.router import plan_optimization
from token0.optimization.transformer import transform_image
from token0.providers.ollama import OllamaProvider

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BENCHMARK_DIR, "images")
RESULTS_DIR = os.path.join(BENCHMARK_DIR, "results")


class TestCategory(str, Enum):
    IMAGE = "image"
    TEXT = "text"
    MULTI_IMAGE = "multi_image"
    MULTI_TURN = "multi_turn"
    TASK_TYPE = "task_type"


@dataclass
class BenchmarkResult:
    test_name: str
    category: str
    description: str

    # Direct path
    direct_prompt_tokens: int = 0
    direct_completion_tokens: int = 0
    direct_total_tokens: int = 0
    direct_response: str = ""
    direct_latency_ms: int = 0

    # Optimized path
    optimized_prompt_tokens: int = 0
    optimized_completion_tokens: int = 0
    optimized_total_tokens: int = 0
    optimized_response: str = ""
    optimized_latency_ms: int = 0
    optimizations_applied: list[str] = field(default_factory=list)

    @property
    def token_savings_pct(self) -> float:
        if self.direct_prompt_tokens == 0:
            return 0.0
        saved = self.direct_prompt_tokens - self.optimized_prompt_tokens
        return round(100 * saved / self.direct_prompt_tokens, 1)

    @property
    def prompt_token_diff(self) -> int:
        return self.direct_prompt_tokens - self.optimized_prompt_tokens

    @property
    def latency_delta_ms(self) -> int:
        return self.optimized_latency_ms - self.direct_latency_ms


def load_image_as_data_uri(image_path: str) -> str:
    with open(image_path, "rb") as f:
        raw = f.read()
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
    mime = mime_map.get(ext, "image/jpeg")
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def optimize_message(messages: list[dict], model: str) -> tuple[list[dict], list[str], int, int]:
    """Run Token0 optimization on messages.

    Returns (optimized_messages, optimizations, tokens_before, tokens_after).
    """
    optimized = []
    all_optimizations = []
    total_before = 0
    total_after = 0

    for msg in messages:
        if isinstance(msg.get("content"), str):
            optimized.append(msg)
            continue

        parts = msg["content"]
        opt_parts = []
        for part in parts:
            if part["type"] == "text":
                opt_parts.append(part)
            elif part["type"] == "image_url":
                data_uri = part["image_url"]["url"]
                analysis, raw_bytes, pil_image = analyze_image(data_uri)
                plan = plan_optimization(analysis, model)

                total_before += plan.estimated_tokens_before
                total_after += plan.estimated_tokens_after
                all_optimizations.extend(plan.reasons)

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
                    detail = "low" if plan.force_detail_low else "auto"
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
                    if not plan.reasons:
                        all_optimizations.append("passthrough")

        optimized.append({"role": msg["role"], "content": opt_parts})

    return optimized, all_optimizations, total_before, total_after


async def run_test(
    provider: OllamaProvider,
    model: str,
    test_name: str,
    category: str,
    description: str,
    messages_direct: list[dict],
    messages_for_optimization: list[dict] | None = None,
) -> BenchmarkResult:
    """Run a single benchmark test: direct vs optimized."""
    if messages_for_optimization is None:
        messages_for_optimization = messages_direct

    print(f"  [{test_name}] — {description}")

    # Direct
    print("    Direct...", end="", flush=True)
    start = time.time()
    try:
        direct_resp = await provider.chat_completion(
            model=model, messages=messages_direct, max_tokens=200
        )
        direct_latency = int((time.time() - start) * 1000)
        print(f" {direct_latency}ms, {direct_resp.prompt_tokens} prompt tokens")
    except Exception as e:
        print(f" ERROR: {e}")
        return BenchmarkResult(test_name=test_name, category=category, description=description)

    # Optimized
    print("    Token0...", end="", flush=True)
    opt_messages, optimizations, est_before, est_after = optimize_message(
        messages_for_optimization, model
    )
    start = time.time()
    try:
        opt_resp = await provider.chat_completion(
            model=model, messages=opt_messages, max_tokens=200
        )
        opt_latency = int((time.time() - start) * 1000)
        print(f" {opt_latency}ms, {opt_resp.prompt_tokens} prompt tokens")
    except Exception as e:
        print(f" ERROR: {e}")
        return BenchmarkResult(test_name=test_name, category=category, description=description)

    result = BenchmarkResult(
        test_name=test_name,
        category=category,
        description=description,
        direct_prompt_tokens=direct_resp.prompt_tokens,
        direct_completion_tokens=direct_resp.completion_tokens,
        direct_total_tokens=direct_resp.total_tokens,
        direct_response=direct_resp.content,
        direct_latency_ms=direct_latency,
        optimized_prompt_tokens=opt_resp.prompt_tokens,
        optimized_completion_tokens=opt_resp.completion_tokens,
        optimized_total_tokens=opt_resp.total_tokens,
        optimized_response=opt_resp.content,
        optimized_latency_ms=opt_latency,
        optimizations_applied=optimizations,
    )

    pct = result.token_savings_pct
    sign = "+" if result.latency_delta_ms > 0 else ""
    opts_str = ", ".join(optimizations[:2]) if optimizations else "none"
    print(
        f"    → Tokens: {direct_resp.prompt_tokens} → {opt_resp.prompt_tokens} "
        f"({pct}% saved) | Latency: {sign}{result.latency_delta_ms}ms | {opts_str}"
    )
    print()

    return result


# ============================================================================
# TEST SUITES
# ============================================================================


async def suite_images(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Single image tests — the core optimization benchmarks."""
    results = []

    test_cases = [
        (
            "large_photo",
            "large_photo.jpg",
            "Describe what you see in one sentence.",
            "resize 4000x3000 → model max",
        ),
        (
            "document_png",
            "document_screenshot.png",
            "What type of document is this?",
            "OCR routing for text-heavy PNG",
        ),
        (
            "small_photo",
            "small_photo.jpg",
            "What is the dominant color and shape?",
            "detail mode for small image",
        ),
        (
            "receipt",
            "receipt.png",
            "List the items and prices on this receipt.",
            "OCR routing for receipt",
        ),
        (
            "optimized_jpg",
            "already_optimized.jpg",
            "Describe the contents briefly.",
            "passthrough — already optimal",
        ),
        (
            "large_screenshot",
            "large_screenshot.png",
            "What interface is shown here?",
            "resize + PNG→JPEG + possible OCR",
        ),
    ]

    for name, filename, prompt, desc in test_cases:
        path = os.path.join(IMAGES_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {filename} not found")
            continue

        data_uri = load_image_as_data_uri(path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        result = await run_test(provider, model, f"img/{name}", TestCategory.IMAGE, desc, messages)
        results.append(result)

    return results


async def suite_text(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Text-only tests — regression checks. Token0 should add ZERO overhead."""
    results = []

    text_cases = [
        ("short_text", "What is 2+2?", "short text — zero overhead check"),
        (
            "medium_text",
            "Explain the difference between TCP and UDP in networking. Be concise.",
            "medium text — zero overhead",
        ),
        (
            "long_text",
            (
                "You are an expert software architect. Review the following design decision: "
                "We chose to use PostgreSQL over MongoDB for our e-commerce "
                "platform because we need strong ACID transactions for payment "
                "processing, complex JOIN queries for inventory management "
                "across warehouses, and we already have team expertise in SQL. "
                "The tradeoff is that our product catalog with variable "
                "attributes might be slightly less natural to model in a "
                "relational schema compared to a document store. "
                "What are the pros and cons of this decision? Be concise."
            ),
            "long text — ensure no token inflation",
        ),
        ("system_prompt", None, "system + user message — no image, no overhead"),
    ]

    for name, prompt, desc in text_cases:
        if name == "system_prompt":
            messages = [
                {"role": "system", "content": "You are a helpful coding assistant. Be concise."},
                {
                    "role": "user",
                    "content": "Write a Python function to check if a number is prime.",
                },
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        result = await run_test(provider, model, f"text/{name}", TestCategory.TEXT, desc, messages)
        results.append(result)

    return results


async def suite_multi_image(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Multi-image tests — multiple images in one request."""
    results = []

    # Test 1: Two images comparison
    large_path = os.path.join(IMAGES_DIR, "large_photo.jpg")
    small_path = os.path.join(IMAGES_DIR, "small_photo.jpg")
    if os.path.exists(large_path) and os.path.exists(small_path):
        large_uri = load_image_as_data_uri(large_path)
        small_uri = load_image_as_data_uri(small_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these two images. What are the differences?"},
                    {"type": "image_url", "image_url": {"url": large_uri}},
                    {"type": "image_url", "image_url": {"url": small_uri}},
                ],
            }
        ]

        result = await run_test(
            provider,
            model,
            "multi/two_images",
            TestCategory.MULTI_IMAGE,
            "two images — both should be independently optimized",
            messages,
        )
        results.append(result)

    # Test 2: Three images (photo + document + receipt)
    receipt_path = os.path.join(IMAGES_DIR, "receipt.png")
    doc_path = os.path.join(IMAGES_DIR, "document_screenshot.png")
    if os.path.exists(small_path) and os.path.exists(receipt_path) and os.path.exists(doc_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe each of these three images briefly."},
                    {"type": "image_url", "image_url": {"url": load_image_as_data_uri(small_path)}},
                    {
                        "type": "image_url",
                        "image_url": {"url": load_image_as_data_uri(receipt_path)},
                    },
                    {"type": "image_url", "image_url": {"url": load_image_as_data_uri(doc_path)}},
                ],
            }
        ]

        result = await run_test(
            provider,
            model,
            "multi/three_mixed",
            TestCategory.MULTI_IMAGE,
            "3 mixed images — photo + receipt + document",
            messages,
        )
        results.append(result)

    return results


async def suite_multi_turn(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Multi-turn conversation tests — image in history, follow-up questions."""
    results = []

    large_path = os.path.join(IMAGES_DIR, "large_photo.jpg")
    if not os.path.exists(large_path):
        return results

    large_uri = load_image_as_data_uri(large_path)

    # Turn 1: send image + question
    # Turn 2: follow-up text question (image still in history)
    messages_turn2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image_url", "image_url": {"url": large_uri}},
            ],
        },
        {
            "role": "assistant",
            "content": "I see a colorful abstract image with gradient patterns.",
        },
        {
            "role": "user",
            "content": "What colors are most dominant?",
        },
    ]

    result = await run_test(
        provider,
        model,
        "turn/follow_up",
        TestCategory.MULTI_TURN,
        "multi-turn — image in turn 1, text follow-up in turn 2 (image re-sent in history)",
        messages_turn2,
    )
    results.append(result)

    # Turn 3: another follow-up
    messages_turn3 = messages_turn2 + [
        {
            "role": "assistant",
            "content": (
                "The dominant colors appear to be various shades of blue, green, and warm tones."
            ),
        },
        {
            "role": "user",
            "content": "Is there any text visible in the image?",
        },
    ]

    result = await run_test(
        provider,
        model,
        "turn/third_turn",
        TestCategory.MULTI_TURN,
        "multi-turn — 3 turns deep, image still in history (context accumulation test)",
        messages_turn3,
    )
    results.append(result)

    return results


async def suite_task_types(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Different task types with same image.

    Test if optimization preserves accuracy across tasks.
    """
    results = []

    screenshot_path = os.path.join(IMAGES_DIR, "large_screenshot.png")
    if not os.path.exists(screenshot_path):
        return results

    screenshot_uri = load_image_as_data_uri(screenshot_path)
    tasks = [
        (
            "task/classify",
            "Classify this image in one word: photo, screenshot, document, or chart?",
            "classification task",
        ),
        (
            "task/describe",
            "Describe everything you see in this image in detail.",
            "detailed description task",
        ),
        ("task/extract", "List all text or labels visible in this image.", "text extraction task"),
        (
            "task/question",
            "Is this image showing a mobile app or a desktop application?",
            "yes/no question task",
        ),
    ]

    for name, prompt, desc in tasks:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": screenshot_uri}},
                ],
            }
        ]

        result = await run_test(provider, model, name, TestCategory.TASK_TYPE, desc, messages)
        results.append(result)

    return results


REAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "real")


async def suite_real_world(provider: OllamaProvider, model: str) -> list[BenchmarkResult]:
    """Real-world images — actual photos, documents, receipts, screenshots."""
    results = []

    if not os.path.exists(REAL_IMAGES_DIR):
        print("  SKIP: No real-world images found. Run generate_test_images.py first.")
        return results

    test_cases = [
        # Real photos (should be resized, NOT OCR routed)
        (
            "photo_nature.jpg",
            "Describe this landscape photo in detail. What natural features do you see?",
            "REAL photo 4000x2047 — should resize only, NOT OCR",
        ),
        (
            "photo_street.jpg",
            "What is happening in this street scene? Describe the people and surroundings.",
            "REAL photo 3000x1988 — should resize only, NOT OCR",
        ),
        # Real receipt (should be OCR routed — dense text)
        (
            "receipt_real.jpg",
            "Read this receipt. What store is it from and what is the total amount?",
            "REAL receipt 2448x3264 — should OCR route (text-heavy)",
        ),
        # Real document with actual readable text (should be OCR routed)
        (
            "document_invoice.png",
            "Read this invoice. What is the total amount due and who is it billed to?",
            "REAL invoice 850x1100 — should OCR route (typed text)",
        ),
        # Real screenshot from user's machine
        (
            "screenshot_real.png",
            "What application or website is shown in this screenshot? Describe the UI elements.",
            "REAL screenshot 2066x766 — should resize + detect text content",
        ),
    ]

    for filename, prompt, desc in test_cases:
        path = os.path.join(REAL_IMAGES_DIR, filename)
        if not os.path.exists(path):
            print(f"  SKIP: {filename} not found")
            continue

        from PIL import Image as PILImage

        with PILImage.open(path) as img:
            w, h = img.size
        file_kb = os.path.getsize(path) / 1024
        print(f"  [{filename}] ({w}x{h}, {file_kb:.0f}KB)")

        data_uri = load_image_as_data_uri(path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        result = await run_test(
            provider,
            model,
            f"real/{filename.split('.')[0]}",
            "real_world",
            desc,
            messages,
        )
        results.append(result)

    return results


# ============================================================================
# RUNNER
# ============================================================================

SUITES = {
    "images": suite_images,
    "text": suite_text,
    "multi": suite_multi_image,
    "turns": suite_multi_turn,
    "tasks": suite_task_types,
    "real": suite_real_world,
}


def print_summary(results: list[BenchmarkResult], model: str):
    if not results:
        print("No results.")
        return

    # Group by category
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    print(f"\n{'=' * 120}")
    print(f"  RESULTS SUMMARY — {model}")
    print(f"{'=' * 120}")

    for cat, cat_results in categories.items():
        print(f"\n  [{cat.upper()}]")
        print(
            f"  {'Test':<30} {'Direct':>10} {'Token0':>10} "
            f"{'Saved':>8} {'Savings':>8} {'Latency Δ':>10}  Optimizations"
        )
        print(f"  {'-' * 30} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 10}  {'-' * 30}")

        for r in cat_results:
            sign = "+" if r.latency_delta_ms > 0 else ""
            opts = (
                ", ".join(r.optimizations_applied[:2]) if r.optimizations_applied else "passthrough"
            )
            if len(opts) > 30:
                opts = opts[:27] + "..."
            print(
                f"  {r.test_name:<30} "
                f"{r.direct_prompt_tokens:>10} {r.optimized_prompt_tokens:>10} "
                f"{r.prompt_token_diff:>8} {r.token_savings_pct:>7.1f}% "
                f"{sign}{r.latency_delta_ms:>9}ms  {opts}"
            )

    # Grand totals
    total_direct = sum(r.direct_prompt_tokens for r in results)
    total_optimized = sum(r.optimized_prompt_tokens for r in results)
    total_saved = total_direct - total_optimized
    total_pct = round(100 * total_saved / total_direct, 1) if total_direct > 0 else 0

    print(f"\n  {'=' * 120}")
    print(
        f"  GRAND TOTAL: {total_direct} → {total_optimized} "
        f"prompt tokens ({total_saved} saved, {total_pct}%)"
    )

    # Text overhead check
    text_results = [r for r in results if r.category == TestCategory.TEXT]
    if text_results:
        overhead = sum(r.optimized_prompt_tokens - r.direct_prompt_tokens for r in text_results)
        print(
            f"  TEXT OVERHEAD CHECK: {overhead} extra tokens "
            f"across {len(text_results)} text-only requests",
            end="",
        )
        if overhead == 0:
            print(" — PASS (zero overhead)")
        elif overhead <= len(text_results):
            print(" — OK (negligible)")
        else:
            print(" — WARNING: unexpected overhead!")

    print(f"  {'=' * 120}")

    # Response comparison
    print("\n  RESPONSE COMPARISON (first 100 chars)")
    print(f"  {'-' * 100}")
    for r in results:
        d = r.direct_response[:100].replace("\n", " ").strip()
        o = r.optimized_response[:100].replace("\n", " ").strip()
        match = "SAME" if d == o else "DIFF"
        print(f"  [{r.test_name}] ({match})")
        print(f"    D: {d}")
        print(f"    O: {o}")
        print()


def save_results(results: list[BenchmarkResult], model: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_{model.replace(':', '_')}_{timestamp}.json"
    path = os.path.join(RESULTS_DIR, filename)

    data = {
        "model": model,
        "timestamp": timestamp,
        "results": [
            {
                "test_name": r.test_name,
                "category": r.category,
                "description": r.description,
                "direct": {
                    "prompt_tokens": r.direct_prompt_tokens,
                    "completion_tokens": r.direct_completion_tokens,
                    "total_tokens": r.direct_total_tokens,
                    "response": r.direct_response,
                    "latency_ms": r.direct_latency_ms,
                },
                "optimized": {
                    "prompt_tokens": r.optimized_prompt_tokens,
                    "completion_tokens": r.optimized_completion_tokens,
                    "total_tokens": r.optimized_total_tokens,
                    "response": r.optimized_response,
                    "latency_ms": r.optimized_latency_ms,
                    "optimizations": r.optimizations_applied,
                },
                "savings": {
                    "tokens_saved": r.prompt_token_diff,
                    "savings_pct": r.token_savings_pct,
                    "latency_delta_ms": r.latency_delta_ms,
                },
            }
            for r in results
        ],
        "totals": {
            "total_direct": sum(r.direct_prompt_tokens for r in results),
            "total_optimized": sum(r.optimized_prompt_tokens for r in results),
            "total_saved": sum(r.prompt_token_diff for r in results),
            "overall_savings_pct": round(
                100
                * sum(r.prompt_token_diff for r in results)
                / max(sum(r.direct_prompt_tokens for r in results), 1),
                1,
            ),
        },
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved: {path}")


async def run_benchmark(model: str, suite: str = "all"):
    # Ensure test images exist
    if not os.path.exists(IMAGES_DIR) or len(os.listdir(IMAGES_DIR)) < 6:
        print("Generating test images...")
        from benchmarks.generate_test_images import generate_all

        generate_all()
        print()

    provider = OllamaProvider(model=model)
    all_results: list[BenchmarkResult] = []

    print(f"{'=' * 120}")
    print(f"  Token0 Benchmark — Model: {model} — Suite: {suite}")
    print(f"{'=' * 120}\n")

    if suite == "all":
        suites_to_run = list(SUITES.keys())
    else:
        suites_to_run = [suite]

    for suite_name in suites_to_run:
        if suite_name not in SUITES:
            print(f"  Unknown suite: {suite_name}")
            continue
        print(f"\n  --- Suite: {suite_name} ---\n")
        results = await SUITES[suite_name](provider, model)
        all_results.extend(results)

    print_summary(all_results, model)
    save_results(all_results, model)


def main():
    parser = argparse.ArgumentParser(description="Token0 Benchmark")
    parser.add_argument("--model", default="moondream", help="Ollama vision model")
    parser.add_argument(
        "--suite", default="all", choices=["all", *SUITES.keys()], help="Test suite to run"
    )
    args = parser.parse_args()
    asyncio.run(run_benchmark(model=args.model, suite=args.suite))


if __name__ == "__main__":
    main()
