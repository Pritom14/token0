"""AX (Accessibility) Tree routing — convert UI accessibility trees to compact text.

When a UI automation agent provides both a screenshot and an accessibility tree,
token0 picks the cheaper representation:
- Tree is complete (no canvas/iframe/opaque nodes): use text (~4K tokens vs 50K+)
- Tree has opaque elements: fall back to screenshot for visual accuracy

Supported formats:
- Web (Chrome DevTools / Playwright): {"role": "...", "name": "...", "children": [...]}
- macOS AXUIElement: {"AXRole": "...", "AXTitle": "...", "AXChildren": [...]}
- Pre-serialized string: passed through as-is
"""

from __future__ import annotations

import logging

logger = logging.getLogger("token0.ax_tree")

# Roles that cannot be represented textually — require visual rendering.
_OPAQUE_ROLES: frozenset[str] = frozenset(
    {
        "canvas",
        "AXCanvas",
        "embed",
        "object",
        "plugin",
        "img",
        "image",
        "figure",
        "math",
        "meter",
        "progressbar",
        "AXImage",
    }
)

# HTML tag names that are inherently opaque.
_OPAQUE_TAGS: frozenset[str] = frozenset(
    {"canvas", "iframe", "embed", "object", "video", "audio", "svg"}
)


def _normalize_node(node: dict) -> dict:
    """Return a uniform dict from either AXUIElement or Playwright/CDP format."""
    if "AXRole" in node:
        # macOS AXUIElement
        return {
            "role": node.get("AXRole", ""),
            "name": (node.get("AXTitle") or node.get("AXDescription") or node.get("AXValue") or ""),
            "value": node.get("AXValue", ""),
            "enabled": node.get("AXEnabled", True),
            "children": node.get("AXChildren", []),
        }
    # Web / Playwright / Chrome DevTools Protocol
    return {
        "role": node.get("role", ""),
        "name": node.get("name", ""),
        "value": node.get("value", ""),
        "enabled": not node.get("disabled", False),
        "children": node.get("children", []),
    }


def _serialize_node(node: dict, depth: int, lines: list[str]) -> None:
    """Recursively append compact indented lines for one node."""
    n = _normalize_node(node)
    role = n["role"]
    name = n["name"]
    value = str(n["value"]) if n["value"] else ""
    enabled = n["enabled"]

    indent = "  " * depth
    tokens: list[str] = [role]
    if name:
        tokens.append(f'"{name}"')
    if value and value != name:
        tokens.append(f"={value!r}")
    if not enabled:
        tokens.append("[disabled]")

    lines.append(indent + " ".join(tokens))

    for child in n["children"]:
        _serialize_node(child, depth + 1, lines)


def serialize_ax_tree(tree: dict | list | str) -> str:
    """Convert an AX tree to compact indented text for the LLM.

    Args:
        tree: Nested dict (Playwright/AXUIElement), list of root nodes, or
              pre-serialized string (returned as-is).

    Returns:
        Multi-line string representation of the tree.
    """
    if isinstance(tree, str):
        return tree.strip()

    lines: list[str] = []
    if isinstance(tree, list):
        for node in tree:
            _serialize_node(node, 0, lines)
    elif isinstance(tree, dict):
        _serialize_node(tree, 0, lines)
    else:
        return str(tree)

    return "\n".join(lines)


def estimate_ax_tree_tokens(serialized: str) -> int:
    """Estimate LLM token count for a serialized AX tree (~4 chars per token)."""
    return max(10, len(serialized) // 4)


def _node_is_opaque(node: dict) -> bool:
    """Return True if this node or any descendant needs visual rendering."""
    n = _normalize_node(node)
    role = n["role"]

    if role in _OPAQUE_ROLES:
        return True
    if role.lower() in _OPAQUE_TAGS:
        return True

    return any(_node_is_opaque(child) for child in n["children"])


def has_opaque_nodes(tree: dict | list | str) -> bool:
    """Return True when the tree contains elements that require a screenshot fallback.

    Canvas elements, iframes, embedded media, and images without text equivalents
    cannot be described by the tree alone — the screenshot must be kept.

    Args:
        tree: Same formats as serialize_ax_tree.

    Returns:
        True  → keep screenshot, discard tree (tree alone is insufficient).
        False → use tree text only, drop screenshot (90%+ token savings).
    """
    if isinstance(tree, str):
        lower = tree.lower()
        return any(
            kw in lower
            for kw in ("canvas", "iframe", "embed", "<img", "aximage", "axcanvas", "svg")
        )

    if isinstance(tree, list):
        return any(_node_is_opaque(node) for node in tree)

    if isinstance(tree, dict):
        return _node_is_opaque(tree)

    return False
