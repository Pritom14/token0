"""Prompt classifier — determines if a task needs high or low detail vision.

Low-detail tasks: classification, yes/no questions, color identification, counting,
simple description, sentiment analysis. These work fine at 85 tokens (OpenAI low-detail).

High-detail tasks: text extraction, OCR, reading labels, detailed description,
spatial reasoning, comparing fine details, medical/technical analysis.
"""

import re

# Patterns that indicate LOW detail is sufficient (simple tasks)
LOW_DETAIL_PATTERNS = [
    # Classification
    r"\b(classify|categorize|categorise|label|identify|what type|what kind)\b",
    r"\b(is this|is it|does this|does it|are these|are there)\b",
    r"\b(yes or no|true or false)\b",
    # Simple questions
    r"\b(what color|what colour|dominant color|main color)\b",
    r"\b(how many|count the|number of)\b",
    r"\b(what is this|what's this|what are these)\b",
    r"\b(which one|pick one|choose|select)\b",
    # Brief/simple descriptions
    r"\b(describe briefly|brief description|one sentence|one word|in short)\b",
    r"\b(summarize|summary|overview|gist)\b",
    r"\b(what emotion|what mood|sentiment|feeling)\b",
    # Object detection (just presence, not details)
    r"\b(is there a|can you see|do you see|are there any)\b",
    r"\b(contains|includes|has a|shows a)\b",
]

# Patterns that indicate HIGH detail is required (complex tasks)
HIGH_DETAIL_PATTERNS = [
    # Text extraction / reading
    r"\b(read|extract|transcribe|ocr|what does it say|what text)\b",
    r"\b(list all text|list the text|all visible text|readable text)\b",
    r"\b(labels|captions|annotations|watermark)\b",
    # Detailed analysis
    r"\b(in detail|detailed|thoroughly|comprehensive|every|all elements)\b",
    r"\b(describe everything|explain everything|analyze in)\b",
    r"\b(specific|precisely|exactly|pixel|resolution)\b",
    # Spatial / positional
    r"\b(where exactly|position of|location of|coordinates)\b",
    r"\b(top left|top right|bottom left|bottom right|center of)\b",
    # Comparison of fine details
    r"\b(difference between|compare|spot the|find the error|what changed)\b",
    # Technical / specialized
    r"\b(diagnos|medical|x-ray|scan|pathology|microscop)\b",
    r"\b(code|syntax|error message|stack trace|log)\b",
    r"\b(small print|fine print|footnote)\b",
]

# Patterns that indicate the model must see the image's layout (UI / computer-use
# agents). OCR loses positions and low detail loses small targets, so neither applies.
LAYOUT_PATTERNS = [
    r"\b(click|double[- ]click|right[- ]click|tap|hover|scroll|drag|swipe)\b",
    r"\b(type into|fill in|fill out|press the|press enter|select the)\b",
    r"\b(button|checkbox|dropdown|drop-down|text ?box|input field|form field|menu item|tab bar)\b",
    r"\b(cursor|mouse|coordinates?|bounding box|x,\s*y|pixel position)\b",
    r"\b(where is|locate|point to|navigate to|go to the)\b",
    r"\b(screenshot|web ?page|browser|screen|ui element|user interface)\b",
]

_low_compiled = [re.compile(p, re.IGNORECASE) for p in LOW_DETAIL_PATTERNS]
_layout_compiled = [re.compile(p, re.IGNORECASE) for p in LAYOUT_PATTERNS]
_high_compiled = [re.compile(p, re.IGNORECASE) for p in HIGH_DETAIL_PATTERNS]


def classify_prompt_detail(prompt: str) -> str:
    """Classify a prompt as needing 'low' or 'high' detail vision.

    Returns:
        'low' — simple task, 85 tokens sufficient
        'high' — complex task, needs full resolution
        'layout' — UI/agent task: positions matter, never OCR or low detail
        'auto' — uncertain, use default behavior
    """
    if not prompt:
        return "auto"

    prompt_lower = prompt.lower().strip()

    if needs_layout(prompt_lower):
        return "layout"

    # Score both directions
    low_score = sum(1 for p in _low_compiled if p.search(prompt_lower))
    high_score = sum(1 for p in _high_compiled if p.search(prompt_lower))

    # Short prompts (< 10 words) are usually simple tasks
    word_count = len(prompt_lower.split())
    if word_count <= 6 and high_score == 0:
        low_score += 1

    if high_score > low_score:
        return "high"
    elif low_score > 0 and high_score == 0:
        return "low"
    elif low_score > high_score:
        return "low"

    return "auto"


def needs_layout(text: str) -> bool:
    """True if the task depends on where things are in the image (UI / agent actions)."""
    return any(p.search(text) for p in _layout_compiled)


def extract_all_text(messages: list) -> str:
    """All text across system/user messages — agents often put the task in the system prompt."""
    chunks = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        chunks.append(part.get("text") or "")
                elif getattr(part, "type", None) == "text":
                    chunks.append(getattr(part, "text", "") or "")
    return "\n".join(chunks)


def classify_messages(messages: list) -> str:
    """Classify detail need for a request; layout cues anywhere in the conversation win."""
    if needs_layout(extract_all_text(messages)):
        return "layout"
    return classify_prompt_detail(extract_prompt_text(messages))


def extract_prompt_text(messages: list) -> str:
    """Extract the text prompt from messages (the last user message's text content)."""
    for msg in reversed(messages):
        role = msg.get("role") or getattr(msg, "role", None)
        content = msg.get("content") or getattr(msg, "content", None)

        if role != "user":
            continue

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
                elif hasattr(part, "type") and part.type == "text":
                    texts.append(part.text or "")
            if texts:
                return " ".join(texts)

    return ""
