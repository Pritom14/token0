"""OCR is only used when it is actually cheaper, and never for layout/agent prompts."""

from tests.conftest import make_text_image
from token0.optimization import transformer
from token0.optimization.message_optimizer import optimize_messages
from token0.optimization.prompt_classifier import classify_messages, classify_prompt_detail


def _msgs(prompt, data_uri, system=None):
    msgs = [{"role": "system", "content": system}] if system else []
    msgs.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    )
    return msgs


def _fake_ocr(text, detected=True):
    return lambda raw: {"type": "text", "content": text, "detected": detected}


def _has_image(parts):
    return any(p.get("type") == "image_url" for p in parts)


def test_ocr_more_expensive_than_image_falls_back(monkeypatch):
    monkeypatch.setattr(transformer, "_ocr_extract", _fake_ocr("word " * 2000))
    _, uri = make_text_image(800, 600, lines=25)
    optimized, stats = optimize_messages(_msgs("Read this document", uri), "gpt-4o")
    assert _has_image(optimized[0]["content"])
    assert any("sent image instead" in o for o in stats["optimizations"])
    assert stats["tokens_after"] <= stats["tokens_before"]


def test_no_text_detected_keeps_image(monkeypatch):
    monkeypatch.setattr(
        transformer, "_ocr_extract", _fake_ocr("[No text detected in image]", False)
    )
    _, uri = make_text_image(800, 600, lines=25)
    optimized, stats = optimize_messages(_msgs("Read this document", uri), "gpt-4o")
    assert _has_image(optimized[0]["content"])


def test_layout_prompt_never_ocr(monkeypatch):
    called = []
    monkeypatch.setattr(transformer, "_ocr_extract", lambda raw: called.append(1))
    _, uri = make_text_image(800, 600, lines=25)
    optimized, stats = optimize_messages(_msgs("Click the add to cart button", uri), "gpt-4o")
    assert not called
    assert _has_image(optimized[0]["content"])
    assert any("prompt needs layout" in o for o in stats["optimizations"])


def test_layout_cue_in_system_prompt(monkeypatch):
    called = []
    monkeypatch.setattr(transformer, "_ocr_extract", lambda raw: called.append(1))
    _, uri = make_text_image(800, 600, lines=25)
    msgs = _msgs("next step", uri, system="You are a browser agent. Use the screenshot to act.")
    optimize_messages(msgs, "gpt-4o")
    assert not called


def test_ocr_tokens_reported_from_real_text(monkeypatch):
    monkeypatch.setattr(transformer, "_ocr_extract", _fake_ocr("Invoice 42 total 1,299"))
    _, uri = make_text_image(800, 600, lines=25)
    _, stats = optimize_messages(_msgs("Read this document", uri), "gpt-4o")
    assert stats["tokens_after"] == transformer.estimate_text_tokens("Invoice 42 total 1,299")


def test_classifier_layout():
    assert classify_prompt_detail("Click the login button") == "layout"
    assert classify_prompt_detail("Where is the search box on this web page?") == "layout"
    assert classify_prompt_detail("Extract all text from this receipt") == "high"
    assert classify_messages([{"role": "user", "content": "classify this image"}]) == "low"
