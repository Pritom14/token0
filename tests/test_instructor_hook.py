"""Tests for the instructor integration hook.

All tests are mock-only — no real LLM calls, no instructor dependency required.
"""

from unittest.mock import patch

from token0.instructor_hook import Token0Hook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MOCK_STATS_NO_SAVINGS = {
    "tokens_before": 100,
    "tokens_after": 100,
    "tokens_saved": 0,
    "optimizations": [],
    "recommended_model": None,
}

_MOCK_STATS_WITH_SAVINGS = {
    "tokens_before": 765,
    "tokens_after": 85,
    "tokens_saved": 680,
    "optimizations": ["prompt-aware -> low detail (simple task)"],
    "recommended_model": None,
}

_MOCK_STATS_CASCADE = {
    "tokens_before": 765,
    "tokens_after": 85,
    "tokens_saved": 680,
    "optimizations": ["cascade -> gpt-4o-mini"],
    "recommended_model": "gpt-4o-mini",
}

_IMAGE_MESSAGE = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
        ],
    }
]

_TEXT_MESSAGE = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_hook_is_callable():
    hook = Token0Hook()
    assert callable(hook)


def test_empty_messages_passthrough():
    hook = Token0Hook()
    kwargs = {"model": "gpt-4o", "messages": []}
    result = hook(kwargs)
    assert result == {"model": "gpt-4o", "messages": []}


def test_missing_messages_passthrough():
    hook = Token0Hook()
    kwargs = {"model": "gpt-4o"}
    result = hook(kwargs)
    assert result == {"model": "gpt-4o"}


def test_text_only_passthrough():
    hook = Token0Hook()
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_TEXT_MESSAGE, _MOCK_STATS_NO_SAVINGS),
    ):
        result = hook({"model": "gpt-4o", "messages": _TEXT_MESSAGE})
    assert result["messages"] == _TEXT_MESSAGE


# ---------------------------------------------------------------------------
# Image optimization
# ---------------------------------------------------------------------------


def test_image_messages_are_optimized():
    hook = Token0Hook()
    optimized = [{"role": "user", "content": [{"type": "text", "text": "[Extracted text]"}]}]
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(optimized, _MOCK_STATS_WITH_SAVINGS),
    ) as mock_opt:
        result = hook({"model": "gpt-4o", "messages": _IMAGE_MESSAGE})

    mock_opt.assert_called_once_with(
        _IMAGE_MESSAGE, "gpt-4o", detail_override=None, enable_cascade=False
    )
    assert result["messages"] == optimized


def test_detail_override_passed_through():
    hook = Token0Hook(detail_override="low")
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_NO_SAVINGS),
    ) as mock_opt:
        hook({"model": "gpt-4o", "messages": _IMAGE_MESSAGE})

    _, call_kwargs = mock_opt.call_args
    assert call_kwargs.get("detail_override") == "low" or mock_opt.call_args[0][2] == "low"


def test_enable_cascade_passed_through():
    hook = Token0Hook(enable_cascade=True)
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_NO_SAVINGS),
    ) as mock_opt:
        hook({"model": "gpt-4o", "messages": _IMAGE_MESSAGE})

    args, kwargs = mock_opt.call_args
    enable_cascade = kwargs.get("enable_cascade", args[3] if len(args) > 3 else False)
    assert enable_cascade is True


# ---------------------------------------------------------------------------
# Model cascade
# ---------------------------------------------------------------------------


def test_cascade_updates_model():
    hook = Token0Hook(enable_cascade=True)
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_CASCADE),
    ):
        result = hook({"model": "gpt-4o", "messages": _IMAGE_MESSAGE})

    assert result["model"] == "gpt-4o-mini"


def test_no_cascade_leaves_model_unchanged():
    hook = Token0Hook()
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_WITH_SAVINGS),
    ):
        result = hook({"model": "gpt-4o", "messages": _IMAGE_MESSAGE})

    assert result["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# kwargs passthrough
# ---------------------------------------------------------------------------


def test_extra_kwargs_preserved():
    hook = Token0Hook()
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_NO_SAVINGS),
    ):
        result = hook(
            {
                "model": "gpt-4o",
                "messages": _IMAGE_MESSAGE,
                "temperature": 0.7,
                "max_tokens": 512,
            }
        )

    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 512


def test_no_model_key_still_works():
    hook = Token0Hook()
    with patch(
        "token0.instructor_hook.optimize_messages",
        return_value=(_IMAGE_MESSAGE, _MOCK_STATS_NO_SAVINGS),
    ):
        result = hook({"messages": _IMAGE_MESSAGE})

    assert "messages" in result
