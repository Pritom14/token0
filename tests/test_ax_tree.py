"""Tests for AX tree serialization, opaque detection, and combo routing."""

from token0.optimization.ax_tree import (
    estimate_ax_tree_tokens,
    has_opaque_nodes,
    serialize_ax_tree,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLAYWRIGHT_TREE = {
    "role": "WebArea",
    "name": "GitHub",
    "children": [
        {
            "role": "navigation",
            "name": "Main",
            "children": [
                {"role": "link", "name": "Home", "children": []},
                {"role": "link", "name": "About", "children": []},
            ],
        },
        {
            "role": "main",
            "name": "",
            "children": [
                {"role": "heading", "name": "Welcome", "children": []},
                {"role": "button", "name": "Get Started", "children": []},
                {"role": "textbox", "name": "Search", "value": "foo", "children": []},
            ],
        },
    ],
}

AXUI_TREE = {
    "AXRole": "AXWindow",
    "AXTitle": "Finder",
    "AXChildren": [
        {
            "AXRole": "AXButton",
            "AXTitle": "Close",
            "AXEnabled": True,
            "AXChildren": [],
        },
        {
            "AXRole": "AXTextField",
            "AXTitle": "Search",
            "AXValue": "query",
            "AXEnabled": True,
            "AXChildren": [],
        },
    ],
}

CANVAS_TREE = {
    "role": "WebArea",
    "name": "App",
    "children": [
        {"role": "button", "name": "OK", "children": []},
        {"role": "canvas", "name": "", "children": []},
    ],
}

IFRAME_TREE = {
    "role": "document",
    "name": "",
    "children": [
        {"role": "iframe", "name": "embedded", "children": []},
    ],
}


# ---------------------------------------------------------------------------
# serialize_ax_tree
# ---------------------------------------------------------------------------


def test_serialize_playwright_tree_contains_roles():
    result = serialize_ax_tree(PLAYWRIGHT_TREE)
    assert "WebArea" in result
    assert "button" in result
    assert "Get Started" in result


def test_serialize_playwright_tree_is_indented():
    result = serialize_ax_tree(PLAYWRIGHT_TREE)
    lines = result.splitlines()
    # Root has no indent; children have at least 2 spaces
    assert lines[0].startswith("WebArea")
    assert any(line.startswith("  ") for line in lines)


def test_serialize_axui_tree_normalizes_roles():
    result = serialize_ax_tree(AXUI_TREE)
    assert "AXWindow" in result
    assert "AXButton" in result
    assert "Close" in result
    assert "Search" in result


def test_serialize_axui_includes_value():
    result = serialize_ax_tree(AXUI_TREE)
    assert "query" in result


def test_serialize_list_of_roots():
    roots = [
        {"role": "button", "name": "OK", "children": []},
        {"role": "button", "name": "Cancel", "children": []},
    ]
    result = serialize_ax_tree(roots)
    assert "OK" in result
    assert "Cancel" in result


def test_serialize_string_passthrough():
    pre = "button: Submit\n  text: Click me"
    assert serialize_ax_tree(pre) == pre


def test_serialize_disabled_node():
    tree = {"role": "button", "name": "Submit", "disabled": True, "children": []}
    result = serialize_ax_tree(tree)
    assert "[disabled]" in result


def test_serialize_value_shown_when_different_from_name():
    tree = {"role": "textbox", "name": "Email", "value": "user@example.com", "children": []}
    result = serialize_ax_tree(tree)
    assert "user@example.com" in result


# ---------------------------------------------------------------------------
# estimate_ax_tree_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_proportional_to_length():
    short = "button OK"
    long_text = "button OK\n" * 100
    assert estimate_ax_tree_tokens(long_text) > estimate_ax_tree_tokens(short)


def test_estimate_tokens_minimum_ten():
    assert estimate_ax_tree_tokens("hi") == 10


def test_estimate_tokens_approx_four_chars():
    text = "a" * 400
    assert estimate_ax_tree_tokens(text) == 100


# ---------------------------------------------------------------------------
# has_opaque_nodes
# ---------------------------------------------------------------------------


def test_no_opaque_in_clean_playwright_tree():
    assert has_opaque_nodes(PLAYWRIGHT_TREE) is False


def test_no_opaque_in_axui_tree():
    assert has_opaque_nodes(AXUI_TREE) is False


def test_canvas_role_is_opaque():
    assert has_opaque_nodes(CANVAS_TREE) is True


def test_iframe_role_is_opaque():
    assert has_opaque_nodes(IFRAME_TREE) is True


def test_opaque_detected_in_nested_child():
    nested = {
        "role": "main",
        "name": "",
        "children": [
            {
                "role": "section",
                "name": "",
                "children": [
                    {"role": "canvas", "name": "", "children": []},
                ],
            }
        ],
    }
    assert has_opaque_nodes(nested) is True


def test_opaque_string_contains_canvas():
    assert has_opaque_nodes("button OK\ncanvas [OPAQUE]") is True


def test_opaque_string_contains_iframe():
    assert has_opaque_nodes("main\n  iframe embedded") is True


def test_clean_string_is_not_opaque():
    assert has_opaque_nodes("button OK\nlink Home\ntextbox Search") is False


def test_opaque_list_of_roots():
    roots = [
        {"role": "button", "name": "OK", "children": []},
        {"role": "canvas", "name": "", "children": []},
    ]
    assert has_opaque_nodes(roots) is True


def test_clean_list_of_roots():
    roots = [
        {"role": "button", "name": "OK", "children": []},
        {"role": "link", "name": "Home", "children": []},
    ]
    assert has_opaque_nodes(roots) is False


def test_axui_aximage_is_opaque():
    tree = {
        "AXRole": "AXGroup",
        "AXTitle": "",
        "AXChildren": [
            {"AXRole": "AXImage", "AXTitle": "", "AXChildren": []},
        ],
    }
    assert has_opaque_nodes(tree) is True
