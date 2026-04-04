"""Benchmark: AX tree routing on REAL browser pages via Playwright.

Requires Ollama running locally with moondream and/or llava:7b pulled.
Playwright + Chromium are installed automatically on first run.

Usage:
    python -m benchmarks.bench_ax_tree_real
"""

import asyncio
import base64
import subprocess
import sys
import time
from typing import Optional

from token0.optimization.ax_tree import (
    has_opaque_nodes,
    serialize_ax_tree,
)
from token0.providers.ollama import OllamaProvider

FAST_MODELS = ["moondream", "llava:7b"]

URLS = [
    {
        "url": "https://github.com",
        "name": "GitHub Home",
        "question": (
            "List every interactive element visible "
            "(buttons, links, search inputs)."
        ),
        "required_substrings": ["sign"],
    },
    {
        "url": "https://news.ycombinator.com",
        "name": "Hacker News",
        "question": (
            "How many story links are visible? "
            "Name the first 3 stories."
        ),
        "required_substrings": [],
    },
    {
        "url": "https://en.wikipedia.org/wiki/Main_Page",
        "name": "Wikipedia",
        "question": (
            "What search and navigation elements are available "
            "on this page?"
        ),
        "required_substrings": ["search"],
    },
]

_INTERACTIVE_ROLES = frozenset(
    {
        "button",
        "link",
        "textbox",
        "searchbox",
        "combobox",
        "checkbox",
        "radio",
        "slider",
        "spinbutton",
        "switch",
        "tab",
        "menuitem",
        "menuitemcheckbox",
        "menuitemradio",
        "option",
        "treeitem",
    }
)
_STRUCTURAL_ROLES = frozenset(
    {
        "heading",
        "list",
        "listitem",
        "table",
        "row",
        "cell",
        "navigation",
        "main",
        "banner",
        "contentinfo",
        "complementary",
        "form",
        "search",
        "dialog",
        "alertdialog",
        "tablist",
        "toolbar",
        "menu",
        "menubar",
        "tree",
        "grid",
        "treegrid",
        "WebArea",
        "RootWebArea",
    }
)
_WRAPPER_ROLES = frozenset(
    {
        "generic",
        "none",
        "presentation",
        "group",
        "Section",
    }
)


def _ensure_playwright():
    """Install Playwright if missing, then install Chromium."""
    try:
        import playwright  # noqa: F401
    except ImportError:
        print("Installing playwright...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "playwright"]
        )
    print("Installing Chromium...")
    subprocess.check_call(
        [sys.executable, "-m", "playwright", "install", "chromium"]
    )


def prune_ax_tree(node: Optional[dict], depth: int = 0, max_depth: int = 6):
    """Prune AX tree to interactive/structural nodes only."""
    if node is None:
        return None

    role = node.get("role", "")
    name = node.get("name", "")
    value = node.get("value")
    children = node.get("children", [])

    # Hard depth limit
    if depth > max_depth:
        if role in _INTERACTIVE_ROLES and name:
            return {"role": role, "name": name[:80]}
        return None

    # Prune children first
    pruned_children = []
    for child in children:
        pruned = prune_ax_tree(child, depth + 1, max_depth)
        if pruned:
            pruned_children.append(pruned)

    # Collapse wrappers with 1 child
    if (
        role in _WRAPPER_ROLES
        and not name
        and len(pruned_children) == 1
    ):
        return pruned_children[0]

    is_interactive = role in _INTERACTIVE_ROLES
    is_structural = role in _STRUCTURAL_ROLES
    has_name = bool(name)
    has_children = len(pruned_children) > 0

    keep = (
        is_interactive
        or (is_structural and (has_name or has_children))
        or (has_name and has_children)
    )

    if depth == 0:
        keep = True

    if not keep and not has_children:
        return None

    if not keep and has_children and len(pruned_children) == 1:
        return pruned_children[0]

    if not keep and has_children and len(pruned_children) > 1:
        return {"role": role, "children": pruned_children}

    # Build result
    result: dict = {"role": role}
    if has_name:
        result["name"] = name[:80]
    if is_interactive and value:
        result["value"] = str(value)[:80]
    if has_children:
        result["children"] = pruned_children

    # Hard cap
    serialized = str(result)
    if len(serialized) > 8000:
        result["children"] = pruned_children[:10]

    return result


async def capture_page(browser, url: str, timeout_ms: int = 30000):
    """Capture screenshot and AX snapshot from real page."""
    page = None
    try:
        page = await browser.new_page(
            viewport={"width": 1280, "height": 720}
        )
        await page.goto(url, wait_until="networkidle", timeout=timeout_ms)
        await page.wait_for_timeout(2000)
        screenshot_bytes = await page.screenshot(
            type="jpeg", quality=85, full_page=False
        )

        # Build simple AX tree from DOM structure
        ax_snapshot = await _extract_ax_tree(page)
        return screenshot_bytes, ax_snapshot
    finally:
        if page:
            await page.close()


async def _extract_ax_tree(page):
    """Extract a simple AX tree via JavaScript evaluation."""
    tree = await page.evaluate(
        """
        () => {
            function buildTree(node) {
                if (!node) return null;
                const role = node.getAttribute('role') ||
                            node.tagName.toLowerCase();
                const ariaLabel = node.getAttribute('aria-label');
                const ariaPressed = node.getAttribute('aria-pressed');
                const name = ariaLabel || node.getAttribute('title') ||
                            (node.textContent ?
                            node.textContent.trim().slice(0, 100) : '');

                const children = [];
                for (let child of node.children) {
                    const subtree = buildTree(child);
                    if (subtree) children.push(subtree);
                }

                const result = {role, name};
                if (ariaPressed) result.value = ariaPressed;
                if (children.length > 0) result.children = children;
                return result;
            }
            return buildTree(document.documentElement);
        }
        """
    )
    return tree


def _bytes_to_data_uri(jpeg_bytes: bytes) -> str:
    """Convert JPEG bytes to base64 data URI."""
    b64 = base64.b64encode(jpeg_bytes).decode()
    return f"data:image/jpeg;base64,{b64}"


async def _run_real_scenario(
    model: str,
    provider: OllamaProvider,
    name: str,
    question: str,
    screenshot_uri: str,
    ax_tree_text: str,
    required_substrings: list,
    has_opaque: bool,
) -> Optional[dict]:
    """Run single scenario: screenshot vs AX tree."""
    print(f"\n  Scenario: {name}")
    print(f'  Question: "{question}"')
    if has_opaque:
        print("    NOTE: opaque nodes detected — benchmarking both paths")

    # Screenshot path
    print("    Screenshot: ", end="", flush=True)
    screenshot_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {"url": screenshot_uri, "detail": "auto"},
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

    # Tree path
    print("    AX Tree:       ", end="", flush=True)
    tree_question = f"{question}\n\nUI Accessibility Tree:\n{ax_tree_text}"
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

        saved = screenshot_tokens - tree_tokens
        pct = (saved / screenshot_tokens * 100) if screenshot_tokens > 0 else 0
        print(f" ({-pct:.1f}%)")
    except Exception as e:
        print(f"ERROR: {e}")
        return None

    # Verify key substrings
    screenshot_lower = screenshot_text.lower()
    screenshot_has_items = all(
        substring.lower() in screenshot_lower
        for substring in required_substrings
    )

    print(
        f"    Screenshot captured key items: "
        f"{'YES' if screenshot_has_items else 'NO'}"
    )
    print(f'    Screenshot: "{screenshot_text[:60]}..."')
    print(f'    Tree:       "{tree_text[:60]}..."')

    return {
        "scenario": name,
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
        "has_opaque": has_opaque,
    }


async def run_real_benchmarks():
    """Run benchmarks on real pages via Playwright."""
    _ensure_playwright()

    from playwright.async_api import async_playwright

    provider = OllamaProvider(base_url="http://localhost:11434/v1")

    print("=" * 80)
    print("  AX Tree Routing Benchmark — Real Browser Pages")
    print("=" * 80)

    # Phase 1: Capture all pages
    print("\n" + "=" * 80)
    print("  Phase 1: Capturing Real Pages")
    print("=" * 80)

    captures = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        for url_info in URLS:
            url = url_info["url"]
            name = url_info["name"]
            print(f"\n  {name}: ", end="", flush=True)
            try:
                screenshot_bytes, ax_snapshot = await capture_page(
                    browser, url
                )
                if ax_snapshot is None:
                    print("FAILED: No AX snapshot")
                    continue

                pruned = prune_ax_tree(ax_snapshot)
                tree_text = serialize_ax_tree(pruned)
                opaque = has_opaque_nodes(pruned)

                captures[name] = {
                    "url": url,
                    "screenshot_bytes": screenshot_bytes,
                    "screenshot_uri": _bytes_to_data_uri(screenshot_bytes),
                    "tree_text": tree_text,
                    "has_opaque": opaque,
                }

                print(
                    f"OK ({len(tree_text)} chars, "
                    f"opaque={opaque})"
                )
            except Exception as e:
                print(f"FAILED: {e}")

        await browser.close()

    if not captures:
        print("\nNo captures succeeded. Exiting.")
        return

    # Phase 2: Benchmark each model
    print("\n" + "=" * 80)
    print("  Phase 2: Benchmarking Models")
    print("=" * 80)

    all_results = {}

    for model in FAST_MODELS:
        print(f"\n{'=' * 80}")
        print(f"  Model: {model}")
        print(f"{'=' * 80}")

        model_results = []

        # Check model availability
        try:
            await provider.chat_completion(
                model=model,
                messages=[
                    {"role": "user",
                     "content": [{"type": "text", "text": "test"}]}
                ],
                max_tokens=5,
            )
        except Exception as e:
            print(f"  SKIPPED: Model not available ({e})")
            continue

        for url_info in URLS:
            name = url_info["name"]
            if name not in captures:
                continue

            cap = captures[name]
            result = await _run_real_scenario(
                model=model,
                provider=provider,
                name=name,
                question=url_info["question"],
                screenshot_uri=cap["screenshot_uri"],
                ax_tree_text=cap["tree_text"],
                required_substrings=url_info.get(
                    "required_substrings", []
                ),
                has_opaque=cap["has_opaque"],
            )
            if result:
                model_results.append(result)

        all_results[model] = model_results

        # Summary table
        if model_results:
            total_screenshot = sum(
                r["screenshot_tokens"] for r in model_results
            )
            total_tree = sum(r["tree_tokens"] for r in model_results)
            total_saved = total_screenshot - total_tree
            total_pct = (
                (total_saved / total_screenshot * 100)
                if total_screenshot > 0
                else 0
            )

            print(f"\n  --- {model} Summary ---")
            print(
                f"  {'Scenario':<20s} {'Screenshot':>12s} "
                f"{'Tree':>8s} {'Savings':>8s}"
            )
            print(
                f"  {'-' * 20} {'-' * 12} {'-' * 8} {'-' * 8}"
            )
            for r in model_results:
                print(
                    f"  {r['scenario']:<20s} "
                    f"{r['screenshot_tokens']:>12,} "
                    f"{r['tree_tokens']:>8,} "
                    f"{r['savings_pct']:>7.1f}%"
                )
            print(
                f"  {'TOTAL':<20s} {total_screenshot:>12,} "
                f"{total_tree:>8,} {total_pct:>7.1f}%"
            )

    # Grand summary
    print(f"\n{'=' * 80}")
    print("  Grand Summary — All Models")
    print(f"{'=' * 80}")
    print(
        f"\n  {'Model':<20s} {'Screenshot':>12s} "
        f"{'Tree':>12s} {'Savings':>8s}"
    )
    print(f"  {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 8}")

    for model, results in all_results.items():
        if results:
            total_screenshot = sum(
                r["screenshot_tokens"] for r in results
            )
            total_tree = sum(r["tree_tokens"] for r in results)
            total_saved = total_screenshot - total_tree
            pct = (
                (total_saved / total_screenshot * 100)
                if total_screenshot > 0
                else 0
            )
            print(
                f"  {model:<20s} {total_screenshot:>12,} "
                f"{total_tree:>12,} {pct:>7.1f}%"
            )

    print(f"\n{'=' * 80}\n")

    # Cloud extrapolation
    successful = {m: r for m, r in all_results.items() if r}
    if not successful:
        return

    all_tree_tokens = [
        s["tree_tokens"]
        for r in successful.values()
        for s in r
    ]
    avg_tree_tokens_per_scenario = (
        sum(all_tree_tokens) / len(all_tree_tokens)
    )
    num_scenarios = len(captures)
    total_avg_tree = avg_tree_tokens_per_scenario * num_scenarios

    # Real 1280x720 viewport
    # OpenAI: ceil(1280/512) * ceil(720/512) = 3 * 2 = 6 tiles
    # tokens = 85 + 170 * 6 = 1105
    openai_screenshot_per_scenario = 1105
    openai_total_screenshot = (
        openai_screenshot_per_scenario * num_scenarios
    )

    # Anthropic: 1280 * 720 / 750 = 1229
    anthropic_screenshot_per_scenario = 1229
    anthropic_total_screenshot = (
        anthropic_screenshot_per_scenario * num_scenarios
    )

    def _savings(before, after):
        saved = before - after
        pct = saved / before * 100 if before else 0
        return saved, pct

    openai_saved, openai_pct = _savings(
        openai_total_screenshot, total_avg_tree
    )
    anthropic_saved, anthropic_pct = _savings(
        anthropic_total_screenshot, total_avg_tree
    )

    openai_price_per_m = 2.50
    anthropic_price_per_m = 3.00

    openai_cost_before = (
        openai_total_screenshot * openai_price_per_m / 1_000_000
    )
    openai_cost_after = (
        total_avg_tree * openai_price_per_m / 1_000_000
    )
    anthropic_cost_before = (
        anthropic_total_screenshot * anthropic_price_per_m / 1_000_000
    )
    anthropic_cost_after = (
        total_avg_tree * anthropic_price_per_m / 1_000_000
    )

    print("=" * 80)
    print(
        "  Cloud API Extrapolation "
        "(based on avg Ollama tree token measurements)"
    )
    print("=" * 80)
    avg_str = f"{avg_tree_tokens_per_scenario:.0f}"
    print(f"\n  Avg tree tokens/scenario: {avg_str}")
    print(
        f"  Total tree tokens "
        f"({num_scenarios} scenarios): {total_avg_tree:.0f}"
    )
    print()
    hdr = (
        f"  {'Provider':<22} {'Screenshot':>12} {'Tree':>8} "
        f"{'Savings':>9} {'$/1M saved':>12}"
    )
    print(hdr)
    print(
        f"  {'-' * 22} {'-' * 12} {'-' * 8} "
        f"{'-' * 9} {'-' * 12}"
    )

    for label, shot_tok, pct, cb, ca in [
        (
            "OpenAI GPT-4o",
            openai_total_screenshot,
            openai_pct,
            openai_cost_before,
            openai_cost_after,
        ),
        (
            "Anthropic Claude",
            anthropic_total_screenshot,
            anthropic_pct,
            anthropic_cost_before,
            anthropic_cost_after,
        ),
    ]:
        saved_per_m = (cb - ca) * 1_000_000
        print(
            f"  {label:<22} {shot_tok:>12,} "
            f"{total_avg_tree:>8.0f} {pct:>8.1f}%  "
            f"${saved_per_m:>10,.0f}"
        )

    print()
    print("  At-scale (100K calls/day, 30 days):")
    print(
        f"  {'Provider':<22} {'Direct/mo':>12} "
        f"{'Token0/mo':>12} {'Saved/mo':>12}"
    )
    print(
        f"  {'-' * 22} {'-' * 12} {'-' * 12} {'-' * 12}"
    )
    calls = 100_000 * 30
    for label, cost_before, cost_after in [
        ("OpenAI GPT-4o", openai_cost_before, openai_cost_after),
        (
            "Anthropic Claude",
            anthropic_cost_before,
            anthropic_cost_after,
        ),
    ]:
        mo_before = cost_before * calls
        mo_after = cost_after * calls
        saved_mo = mo_before - mo_after
        print(
            f"  {label:<22} ${mo_before:>10,.0f}  "
            f"${mo_after:>10,.0f}  ${saved_mo:>10,.0f}"
        )

    print()
    print("  Notes:")
    print(
        "  - Real 1280x720 screenshots cost ~1105 tokens (OpenAI) "
        "vs ~765 for synthetic 800x600."
    )
    print(
        "  - AX tree text tokens scale with page complexity, "
        "not resolution — savings are LARGER on real pages."
    )
    print(
        "  - Pricing: OpenAI $2.50/1M, Anthropic $3.00/1M "
        "(input tokens)"
    )

    print(f"\n{'=' * 80}\n")


def main():
    asyncio.run(run_real_benchmarks())


if __name__ == "__main__":
    main()
