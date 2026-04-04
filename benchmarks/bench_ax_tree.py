"""Benchmark: AX tree routing vs raw screenshot token cost.

Measures token savings when token0 routes an accessibility tree to text
instead of passing a screenshot to the LLM.

Three scenarios:
  1. Screenshot only          — baseline (what everyone does today)
  2. AX tree only             — best case (no screenshot at all)
  3. Combo (screenshot + tree, tree is complete) — token0 drops screenshot
  4. Combo (screenshot + tree, tree has canvas)  — token0 keeps screenshot

Usage:
    python -m benchmarks.bench_ax_tree
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Representative AX trees (no real browser needed)
# ---------------------------------------------------------------------------

# Typical GitHub PR page — all interactive elements, no canvas
GITHUB_PR_TREE = {
    "role": "WebArea",
    "name": "Pull request #42 · Pritom14/token0",
    "children": [
        {
            "role": "navigation",
            "name": "Main",
            "children": [
                {"role": "link", "name": "Code", "children": []},
                {"role": "link", "name": "Issues", "children": []},
                {"role": "link", "name": "Pull requests", "children": []},
            ],
        },
        {
            "role": "main",
            "name": "",
            "children": [
                {"role": "heading", "name": "feat: AX tree routing", "children": []},
                {
                    "role": "group",
                    "name": "PR actions",
                    "children": [
                        {"role": "button", "name": "Merge pull request", "children": []},
                        {"role": "button", "name": "Close pull request", "children": []},
                    ],
                },
                {
                    "role": "list",
                    "name": "Commits",
                    "children": [
                        {
                            "role": "listitem",
                            "name": "feat: AX tree routing — accept accessibility_tree content parts",
                            "children": [],
                        },
                        {
                            "role": "listitem",
                            "name": "fix: remove unused pytest import",
                            "children": [],
                        },
                    ],
                },
                {
                    "role": "group",
                    "name": "Review",
                    "children": [
                        {"role": "radio", "name": "Comment", "children": []},
                        {"role": "radio", "name": "Approve", "children": []},
                        {"role": "radio", "name": "Request changes", "children": []},
                        {"role": "button", "name": "Submit review", "children": []},
                    ],
                },
            ],
        },
    ],
}

# Figma editor — has canvas element (opaque, needs screenshot)
FIGMA_TREE = {
    "role": "application",
    "name": "Figma",
    "children": [
        {
            "role": "toolbar",
            "name": "Tools",
            "children": [
                {"role": "button", "name": "Move", "children": []},
                {"role": "button", "name": "Frame", "children": []},
                {"role": "button", "name": "Text", "children": []},
            ],
        },
        {
            "role": "main",
            "name": "Canvas",
            "children": [
                # The actual design is rendered in a canvas — not accessible
                {"role": "canvas", "name": "", "children": []},
            ],
        },
        {
            "role": "complementary",
            "name": "Layers",
            "children": [
                {"role": "treeitem", "name": "Frame 1", "children": []},
                {"role": "treeitem", "name": "Button component", "children": []},
            ],
        },
    ],
}

# macOS Finder — AXUIElement format
FINDER_AXUI_TREE = {
    "AXRole": "AXWindow",
    "AXTitle": "Finder",
    "AXChildren": [
        {
            "AXRole": "AXToolbar",
            "AXTitle": "",
            "AXChildren": [
                {"AXRole": "AXButton", "AXTitle": "Back", "AXEnabled": True, "AXChildren": []},
                {"AXRole": "AXButton", "AXTitle": "Forward", "AXEnabled": False, "AXChildren": []},
                {
                    "AXRole": "AXTextField",
                    "AXTitle": "Search",
                    "AXValue": "",
                    "AXEnabled": True,
                    "AXChildren": [],
                },
            ],
        },
        {
            "AXRole": "AXOutline",
            "AXTitle": "Files",
            "AXChildren": [
                {
                    "AXRole": "AXRow",
                    "AXTitle": "Documents",
                    "AXChildren": [
                        {
                            "AXRole": "AXRow",
                            "AXTitle": "runbookai",
                            "AXChildren": [],
                        },
                        {
                            "AXRole": "AXRow",
                            "AXTitle": "token0",
                            "AXChildren": [],
                        },
                    ],
                },
                {"AXRole": "AXRow", "AXTitle": "Downloads", "AXChildren": []},
                {"AXRole": "AXRow", "AXTitle": "Desktop", "AXChildren": []},
            ],
        },
    ],
}


# ---------------------------------------------------------------------------
# Token estimation helpers (no LLM calls needed)
# ---------------------------------------------------------------------------

# GPT-4o: 1080p screenshot (1920×1080) → high detail
#   = 85 + 170 × ceil(1920/512) × ceil(1080/512) = 85 + 170 × 4 × 3 = 2,125 tiles tokens
#   real-world measurements land around 1,500–5,000 depending on content; use 2,125 as baseline
SCREENSHOT_1080P_TOKENS = 2_125

# Same screenshot but resized by token0 to provider max (2048px longest edge)
# 2048×1152 → tiles: ceil(2048/512)×ceil(1152/512) = 4×3 = 12 tiles = 2125 tokens (same for 1080p)
# For a 4K screenshot (3840×2160) token0 would resize to 2048×1152:
SCREENSHOT_4K_TOKENS_RAW = 8_925   # 4K without any optimization
SCREENSHOT_4K_TOKENS_RESIZED = 2_125  # after token0 resize to 2048px

COST_PER_TOKEN_USD = 2.50 / 1_000_000  # GPT-4o input


def _ax_tokens(tree) -> int:
    from token0.optimization.ax_tree import estimate_ax_tree_tokens, serialize_ax_tree

    return estimate_ax_tree_tokens(serialize_ax_tree(tree))


def _ax_serialized(tree) -> str:
    from token0.optimization.ax_tree import serialize_ax_tree

    return serialize_ax_tree(tree)


def _is_opaque(tree) -> bool:
    from token0.optimization.ax_tree import has_opaque_nodes

    return has_opaque_nodes(tree)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

WIDTH = 72

def _header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def _row(label: str, tokens: int, cost_usd: float, note: str = "") -> None:
    savings_col = f"  {note}" if note else ""
    print(f"  {label:<38} {tokens:>6,} tokens  ${cost_usd:.4f}{savings_col}")


def _divider() -> None:
    print("  " + "-" * (WIDTH - 2))


def run_scenario(name: str, tree, screenshot_tokens: int) -> dict:
    from token0.optimization.ax_tree import (
        estimate_ax_tree_tokens,
        has_opaque_nodes,
        serialize_ax_tree,
    )

    serialized = serialize_ax_tree(tree)
    tree_tokens = estimate_ax_tree_tokens(serialized)
    opaque = has_opaque_nodes(tree)

    if opaque:
        # token0 keeps screenshot, drops tree
        optimized_tokens = screenshot_tokens
        strategy = "screenshot kept (opaque nodes)"
    else:
        # token0 drops screenshot, uses tree text
        optimized_tokens = tree_tokens
        strategy = "tree text used (screenshot dropped)"

    savings = screenshot_tokens - optimized_tokens
    savings_pct = savings / screenshot_tokens * 100 if screenshot_tokens else 0
    cost_before = screenshot_tokens * COST_PER_TOKEN_USD
    cost_after = optimized_tokens * COST_PER_TOKEN_USD

    return {
        "name": name,
        "screenshot_tokens": screenshot_tokens,
        "tree_tokens": tree_tokens,
        "optimized_tokens": optimized_tokens,
        "savings": savings,
        "savings_pct": savings_pct,
        "cost_before": cost_before,
        "cost_after": cost_after,
        "strategy": strategy,
        "opaque": opaque,
        "serialized_chars": len(serialized),
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parent.parent))

    scenarios = [
        ("GitHub PR page (Playwright tree)", GITHUB_PR_TREE, SCREENSHOT_1080P_TOKENS),
        ("Figma editor (canvas — opaque)", FIGMA_TREE, SCREENSHOT_1080P_TOKENS),
        ("macOS Finder (AXUIElement)", FINDER_AXUI_TREE, SCREENSHOT_1080P_TOKENS),
        ("4K screenshot, no tree (baseline)", None, SCREENSHOT_4K_TOKENS_RAW),
        ("4K screenshot + Finder tree", FINDER_AXUI_TREE, SCREENSHOT_4K_TOKENS_RAW),
    ]

    results = []
    for name, tree, shot_tokens in scenarios:
        if tree is None:
            # Baseline: no tree, no optimization
            r = {
                "name": name,
                "screenshot_tokens": shot_tokens,
                "tree_tokens": 0,
                "optimized_tokens": shot_tokens,
                "savings": 0,
                "savings_pct": 0.0,
                "cost_before": shot_tokens * COST_PER_TOKEN_USD,
                "cost_after": shot_tokens * COST_PER_TOKEN_USD,
                "strategy": "no tree provided — passthrough",
                "opaque": False,
                "serialized_chars": 0,
            }
        else:
            r = run_scenario(name, tree, shot_tokens)
        results.append(r)

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------
    _header("AX Tree Routing — Token Savings Benchmark (GPT-4o pricing)")

    for r in results:
        print()
        print(f"  Scenario: {r['name']}")
        print(f"  Strategy: {r['strategy']}")
        if r["serialized_chars"]:
            print(f"  Tree size: {r['serialized_chars']:,} chars → {r['tree_tokens']:,} tokens")
        _divider()
        _row("Screenshot (no optimization)", r["screenshot_tokens"], r["cost_before"])
        _row(
            "token0 optimized",
            r["optimized_tokens"],
            r["cost_after"],
            f"  (-{r['savings_pct']:.1f}%)" if r["savings_pct"] else "",
        )
        if r["savings"] > 0:
            print(f"  >> Saved: {r['savings']:,} tokens  ${r['cost_before'] - r['cost_after']:.4f}/call")

    # ---------------------------------------------------------------------------
    # At-scale projection
    # ---------------------------------------------------------------------------
    _header("At-Scale Projection — GitHub PR agent (100K calls/day)")

    github_r = results[0]  # GitHub PR tree
    calls_per_day = 100_000
    days = 30

    before_daily = github_r["cost_before"] * calls_per_day
    after_daily = github_r["cost_after"] * calls_per_day
    before_monthly = before_daily * days
    after_monthly = after_daily * days

    print(f"\n  Per call:   ${github_r['cost_before']:.4f} → ${github_r['cost_after']:.4f}")
    print(f"  Daily:      ${before_daily:,.2f} → ${after_daily:,.2f}")
    print(f"  Monthly:    ${before_monthly:,.2f} → ${after_monthly:,.2f}")
    print(f"  Saved/mo:   ${before_monthly - after_monthly:,.2f}  ({github_r['savings_pct']:.1f}%)")

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    _header("Summary")
    print(f"\n  {'Scenario':<42} {'Before':>8} {'After':>8} {'Savings':>10}")
    print("  " + "-" * 70)
    for r in results:
        pct = f"-{r['savings_pct']:.1f}%" if r["savings_pct"] else "n/a"
        print(
            f"  {r['name']:<42} {r['screenshot_tokens']:>6,}t  "
            f"{r['optimized_tokens']:>6,}t  {pct:>10}"
        )

    print()
    print("  Notes:")
    print("  - Token counts use GPT-4o tile formula (85 + 170×tiles)")
    print("  - 1080p screenshot = 1920×1080 = 12 tiles = 2,125 tokens")
    print("  - AX tree tokens estimated at 4 chars/token")
    print("  - Figma (canvas) forces screenshot path — no savings expected")
    print()


if __name__ == "__main__":
    main()
