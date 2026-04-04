"""Benchmark: AX tree routing vs screenshot images on real Ollama vision models.

Compares two input modalities for the same UI:
  - Screenshot: PIL image (base64 JPEG data URI)
  - AX Tree: serialized accessibility tree as plain text

Measures real prompt_tokens from Ollama for both, calculates savings.

Usage:
    python -m benchmarks.bench_ax_tree_models
    python -m benchmarks.bench_ax_tree_models --model moondream
    python -m benchmarks.bench_ax_tree_models --model llava:7b --model minicpm-v
"""

import argparse
import asyncio
import base64
import io
import time
from typing import Optional

from PIL import Image, ImageDraw

from token0.optimization.ax_tree import serialize_ax_tree
from token0.providers.ollama import OllamaProvider

VISION_MODELS = [
    "moondream",
    "llava:7b",
    "llava-llama3",
    "minicpm-v",
    "gemma3:4b",
    "granite3.2-vision",
    "llama3.2-vision",
]


def _pil_to_data_uri(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64 JPEG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _create_login_form_screenshot() -> Image.Image:
    """Create a login form screenshot: header, email/password fields, login button, forgot link."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Gray header bar
    draw.rectangle([0, 0, 800, 50], fill="lightgray")

    # "Sign In" heading (top center)
    draw.text((300, 80), "Sign In", fill="black")

    # Email label
    draw.text((200, 180), "Email", fill="black")
    # Email input box
    draw.rectangle([200, 200, 600, 230], outline="black")

    # Password label
    draw.text((200, 260), "Password", fill="black")
    # Password input box
    draw.rectangle([200, 280, 600, 310], outline="black")

    # Blue "Log In" button
    draw.rectangle([300, 330, 500, 370], fill="blue")
    draw.text((340, 345), "Log In", fill="white")

    # "Forgot password?" link
    draw.text((310, 400), "Forgot password?", fill="blue")

    return img


def _create_todo_list_screenshot() -> Image.Image:
    """Create a todo list screenshot with 3 tasks (one checked) and add button."""
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # "My Tasks" heading
    draw.text((300, 40), "My Tasks", fill="black")

    # Task row 1: Buy groceries (checked)
    draw.rectangle([200, 120, 220, 140], fill="green")  # checked box
    draw.text((230, 120), "Buy groceries", fill="black")

    # Task row 2: Write report (unchecked)
    draw.rectangle([200, 180, 220, 200], outline="black")  # empty box
    draw.text((230, 180), "Write report", fill="black")

    # Task row 3: Call dentist (unchecked)
    draw.rectangle([200, 240, 220, 260], outline="black")  # empty box
    draw.text((230, 240), "Call dentist", fill="black")

    # Green "Add Task" button
    draw.rectangle([300, 340, 500, 380], fill="green")
    draw.text((340, 355), "Add Task", fill="white")

    return img


def _create_login_ax_tree() -> dict:
    """Return login form accessibility tree."""
    return {
        "role": "WebArea",
        "name": "Sign In",
        "children": [
            {"role": "heading", "name": "Sign In", "children": []},
            {"role": "textbox", "name": "Email", "value": "", "children": []},
            {"role": "textbox", "name": "Password", "value": "", "children": []},
            {"role": "button", "name": "Log In", "children": []},
            {"role": "link", "name": "Forgot password?", "children": []},
        ],
    }


def _create_todo_ax_tree() -> str:
    """Return todo list tree as serialized text (to include checked state)."""
    # Manually build the tree to preserve "checked" state info
    tree_text = """WebArea "My Tasks"
  heading "My Tasks"
  list "Tasks"
    checkbox "Buy groceries" [checked]
    checkbox "Write report"
    checkbox "Call dentist"
  button "Add Task"
"""
    return tree_text.strip()


async def run_ax_tree_scenario(
    model: str,
    provider: OllamaProvider,
    scenario_name: str,
    question: str,
    screenshot: Image.Image,
    ax_tree: str,
    required_substrings: list[str],
) -> Optional[dict]:
    """Run a single AX tree scenario: screenshot vs tree. Returns result dict or None on error."""
    print(f"\n  Scenario: {scenario_name}")
    print(f'  Question: "{question}"')

    # --- Screenshot path ---
    print("    Screenshot: ", end="", flush=True)
    data_uri = _pil_to_data_uri(screenshot)
    screenshot_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri, "detail": "auto"},
                },
            ],
        }
    ]

    screenshot_start = time.time()
    try:
        screenshot_resp = await provider.chat_completion(
            model=model, messages=screenshot_messages, max_tokens=200
        )
        screenshot_latency = int((time.time() - screenshot_start) * 1000)
        screenshot_tokens = screenshot_resp.prompt_tokens
        screenshot_text = screenshot_resp.content
        print(f"{screenshot_tokens:,} tokens | {screenshot_latency}ms")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # --- Tree path ---
    print("    AX Tree:       ", end="", flush=True)
    tree_question = f"{question}\n\nUI Accessibility Tree:\n{ax_tree}"
    tree_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": tree_question}],
        }
    ]

    tree_start = time.time()
    try:
        tree_resp = await provider.chat_completion(
            model=model, messages=tree_messages, max_tokens=200
        )
        tree_latency = int((time.time() - tree_start) * 1000)
        tree_tokens = tree_resp.prompt_tokens
        tree_text = tree_resp.content
        print(f"{tree_tokens:,} tokens | {tree_latency}ms", end="")

        # Calculate savings
        saved = screenshot_tokens - tree_tokens
        pct = (saved / screenshot_tokens * 100) if screenshot_tokens > 0 else 0
        print(f" ({-pct:.1f}%)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # --- Verify screenshot answer contains key items (tree may interpret differently) ---
    screenshot_lower = screenshot_text.lower()
    screenshot_has_items = all(
        substring.lower() in screenshot_lower for substring in required_substrings
    )

    print(f"    Screenshot captured key items: {'YES' if screenshot_has_items else 'NO'}")
    print(f'    Screenshot: "{screenshot_text[:60]}..."')
    print(f'    Tree:       "{tree_text[:60]}..."')

    return {
        "scenario": scenario_name,
        "question": question,
        "screenshot_tokens": screenshot_tokens,
        "tree_tokens": tree_tokens,
        "tokens_saved": saved,
        "savings_pct": round(pct, 1),
        "screenshot_latency_ms": screenshot_latency,
        "tree_latency_ms": tree_latency,
        "screenshot_answer": screenshot_text,
        "tree_answer": tree_text,
        "screenshot_captured_items": screenshot_has_items,
    }


async def run_all_benchmarks(models: list[str]):
    """Run AX tree benchmarks for all models."""
    provider = OllamaProvider(base_url="http://localhost:11434/v1")

    print("=" * 80)
    print("  AX Tree Routing Benchmark — Real Ollama Models")
    print("=" * 80)

    # Create test scenarios
    scenarios = [
        {
            "name": "Login Form",
            "question": "List every interactive element on this page (buttons, links, inputs).",
            "screenshot": _create_login_form_screenshot(),
            "ax_tree": serialize_ax_tree(_create_login_ax_tree()),
            "required_substrings": ["email", "password", "log in"],
        },
        {
            "name": "Todo List",
            "question": "How many tasks are shown and which ones are completed?",
            "screenshot": _create_todo_list_screenshot(),
            "ax_tree": _create_todo_ax_tree(),
            "required_substrings": ["buy groceries"],
        },
    ]

    all_results = {}

    for model in models:
        print(f"\n{'=' * 80}")
        print(f"  Model: {model}")
        print(f"{'=' * 80}")

        model_results = []

        # Check if model is available
        try:
            await provider.chat_completion(
                model=model,
                messages=[{"role": "user", "content": [{"type": "text", "text": "test"}]}],
                max_tokens=5,
            )
        except Exception as e:
            print(f"  SKIPPED: Model not available ({e})")
            continue

        for scenario in scenarios:
            result = await run_ax_tree_scenario(
                model=model,
                provider=provider,
                scenario_name=scenario["name"],
                question=scenario["question"],
                screenshot=scenario["screenshot"],
                ax_tree=scenario["ax_tree"],
                required_substrings=scenario["required_substrings"],
            )
            if result:
                model_results.append(result)

        all_results[model] = model_results

        # Print model summary
        if model_results:
            total_screenshot = sum(r["screenshot_tokens"] for r in model_results)
            total_tree = sum(r["tree_tokens"] for r in model_results)
            total_saved = total_screenshot - total_tree
            total_pct = (total_saved / total_screenshot * 100) if total_screenshot > 0 else 0

            print(f"\n  --- {model} Summary ---")
            print(f"  {'Scenario':<20s} {'Screenshot':>12s} {'Tree':>8s} {'Savings':>8s}")
            print(f"  {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 8}")
            for r in model_results:
                print(
                    f"  {r['scenario']:<20s} {r['screenshot_tokens']:>12,} "
                    f"{r['tree_tokens']:>8,} {r['savings_pct']:>7.1f}%"
                )
            print(f"  {'TOTAL':<20s} {total_screenshot:>12,} {total_tree:>8,} {total_pct:>7.1f}%")

    # --- Grand summary across all models ---
    print(f"\n{'=' * 80}")
    print("  Grand Summary — All Models")
    print(f"{'=' * 80}")
    print(f"\n  {'Model':<20s} {'Screenshot':>12s} {'Tree':>12s} {'Savings':>8s}")
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 8}")

    for model, results in all_results.items():
        if results:
            total_screenshot = sum(r["screenshot_tokens"] for r in results)
            total_tree = sum(r["tree_tokens"] for r in results)
            total_saved = total_screenshot - total_tree
            pct = (total_saved / total_screenshot * 100) if total_screenshot > 0 else 0
            print(f"  {model:<20s} {total_screenshot:>12,} {total_tree:>12,} {pct:>7.1f}%")

    print(f"\n{'=' * 80}\n")

    # --- Cloud API extrapolation ---
    # Tree tokens are text — roughly constant across all models and providers.
    # Screenshot tokens for OpenAI/Anthropic are calculated from their published formulas.
    # We use the average tree tokens measured across all Ollama models as our estimate.
    successful = {m: r for m, r in all_results.items() if r}
    if not successful:
        return

    all_tree_tokens = [t for r in successful.values() for s in r for t in [s["tree_tokens"]]]
    avg_tree_tokens_per_scenario = sum(all_tree_tokens) / len(all_tree_tokens)
    num_scenarios = len(scenarios)
    total_avg_tree = avg_tree_tokens_per_scenario * num_scenarios

    # OpenAI GPT-4o: 800x600 JPEG → tile formula (512px tiles)
    # tiles = ceil(800/512) * ceil(600/512) = 2 * 2 = 4 tiles
    # tokens = 85 + 170 * 4 = 765 per image
    openai_screenshot_per_scenario = 765
    openai_total_screenshot = openai_screenshot_per_scenario * num_scenarios

    # Anthropic Claude: pixels / 750
    # 800 * 600 / 750 = 640 per image
    anthropic_screenshot_per_scenario = 640
    anthropic_total_screenshot = anthropic_screenshot_per_scenario * num_scenarios

    def _savings(before, after):
        saved = before - after
        pct = saved / before * 100 if before else 0
        return saved, pct

    openai_saved, openai_pct = _savings(openai_total_screenshot, total_avg_tree)
    anthropic_saved, anthropic_pct = _savings(anthropic_total_screenshot, total_avg_tree)

    # Pricing (input tokens)
    openai_price_per_m = 2.50  # GPT-4o
    anthropic_price_per_m = 3.00  # Claude Sonnet

    openai_cost_before = openai_total_screenshot * openai_price_per_m / 1_000_000
    openai_cost_after = total_avg_tree * openai_price_per_m / 1_000_000
    anthropic_cost_before = anthropic_total_screenshot * anthropic_price_per_m / 1_000_000
    anthropic_cost_after = total_avg_tree * anthropic_price_per_m / 1_000_000

    print("=" * 80)
    print("  Cloud API Extrapolation (based on avg Ollama tree token measurements)")
    print("=" * 80)
    avg_str = f"{avg_tree_tokens_per_scenario:.0f}"
    print(f"\n  Avg tree tokens/scenario across Ollama models: {avg_str}")
    print(f"  Total tree tokens ({num_scenarios} scenarios): {total_avg_tree:.0f}")
    print()
    hdr = f"  {'Provider':<22} {'Screenshot':>12} {'Tree':>8} {'Savings':>9} {'$/1M saved':>12}"
    print(hdr)
    print(f"  {'-' * 22} {'-' * 12} {'-' * 8} {'-' * 9} {'-' * 12}")

    for label, shot_tok, pct, cb, ca in [
        ("OpenAI GPT-4o", openai_total_screenshot, openai_pct,
         openai_cost_before, openai_cost_after),
        ("Anthropic Claude", anthropic_total_screenshot, anthropic_pct,
         anthropic_cost_before, anthropic_cost_after),
    ]:
        saved_per_m = (cb - ca) * 1_000_000
        print(
            f"  {label:<22} {shot_tok:>12,} {total_avg_tree:>8.0f}"
            f" {pct:>8.1f}%  ${saved_per_m:>10,.0f}"
        )

    print()
    print("  At-scale (100K UI agent calls/day, 30 days):")
    print(f"  {'Provider':<22} {'Direct/mo':>12} {'Token0/mo':>12} {'Saved/mo':>12}")
    print(f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 12}")
    calls = 100_000 * 30
    for label, cost_before, cost_after in [
        ("OpenAI GPT-4o", openai_cost_before, openai_cost_after),
        ("Anthropic Claude", anthropic_cost_before, anthropic_cost_after),
    ]:
        mo_before = cost_before * calls
        mo_after = cost_after * calls
        saved_mo = mo_before - mo_after
        print(f"  {label:<22} ${mo_before:>10,.0f}  ${mo_after:>10,.0f}  ${saved_mo:>10,.0f}")

    print()
    print("  Notes:")
    print("  - Screenshot tokens: OpenAI tile formula (85 + 170×tiles), Anthropic w×h/750")
    print("  - Tree tokens: measured from real Ollama calls — text tokenization is")
    print("    provider-agnostic (~4 chars/token, consistent across OpenAI/Anthropic/Ollama)")
    print("  - Image size: 800×600 synthetic screenshots (matches our benchmark)")
    print(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="AX tree routing benchmark against Ollama models")
    parser.add_argument(
        "--model", action="append", help="Ollama model(s) to test (can specify multiple)"
    )
    args = parser.parse_args()

    models = args.model or VISION_MODELS
    asyncio.run(run_all_benchmarks(models))


if __name__ == "__main__":
    main()
