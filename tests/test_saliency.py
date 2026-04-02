"""Tests for saliency-based ROI cropping."""

import pytest
from PIL import Image

from token0.optimization.saliency import SaliencyResult, apply_saliency_crop, detect_roi


def _make_image(w: int = 800, h: int = 1000) -> Image.Image:
    return Image.new("RGB", (w, h), color=(200, 200, 200))


# ---------------------------------------------------------------------------
# detect_roi — keyword matching
# ---------------------------------------------------------------------------


def test_footer_keyword_crops_bottom():
    img = _make_image()
    result = detect_roi("What is the total amount on this invoice?", img)
    assert result.cropped is True
    assert result.matched_keyword is not None
    # Bottom crop — top edge should be > 50% down
    _, top, _, bottom = result.crop_box
    assert top > img.height * 0.5
    assert bottom == img.height


def test_header_keyword_crops_top():
    img = _make_image()
    result = detect_roi("Read the header text", img)
    assert result.cropped is True
    left, top, right, bottom = result.crop_box
    assert top == 0
    assert bottom < img.height * 0.5


def test_top_right_keyword():
    img = _make_image()
    result = detect_roi("What is the date on this document?", img)
    assert result.cropped is True
    left, top, right, bottom = result.crop_box
    assert left > 0  # right half
    assert top == 0


def test_bottom_right_keyword():
    img = _make_image()
    result = detect_roi("What does the signature say at the bottom right?", img)
    assert result.cropped is True
    # "signature" matches footer rule (full-width bottom strip) — still a valid crop
    _, top, _, bottom = result.crop_box
    assert top > img.height * 0.5
    assert bottom == img.height


def test_no_match_returns_not_cropped():
    img = _make_image()
    result = detect_roi("Describe this image", img)
    assert result.cropped is False
    assert result.crop_box is None
    assert result.savings_pct == 0.0


def test_empty_prompt_returns_not_cropped():
    img = _make_image()
    result = detect_roi("", img)
    assert result.cropped is False


def test_tiny_image_skipped():
    img = _make_image(100, 100)
    result = detect_roi("What is the total?", img)
    assert result.cropped is False


def test_savings_pct_is_meaningful():
    img = _make_image()
    result = detect_roi("Read the header", img)
    assert result.cropped is True
    assert result.savings_pct >= 0.20


# ---------------------------------------------------------------------------
# apply_saliency_crop
# ---------------------------------------------------------------------------


def test_crop_produces_correct_dimensions():
    img = _make_image(800, 1000)
    result = detect_roi("What is the total?", img)
    assert result.cropped
    cropped = apply_saliency_crop(img, result)
    left, top, right, bottom = result.crop_box
    assert cropped.size == (right - left, bottom - top)


def test_no_crop_returns_original():
    img = _make_image()
    result = SaliencyResult(cropped=False, crop_box=None, matched_keyword=None, savings_pct=0.0)
    out = apply_saliency_crop(img, result)
    assert out is img


# ---------------------------------------------------------------------------
# Integration: detect_roi → apply_saliency_crop produces smaller image
# ---------------------------------------------------------------------------


def test_cropped_image_is_smaller():
    img = _make_image(800, 1000)
    result = detect_roi("What is the invoice total?", img)
    assert result.cropped
    cropped = apply_saliency_crop(img, result)
    orig_area = img.width * img.height
    crop_area = cropped.width * cropped.height
    assert crop_area < orig_area


def test_center_keyword():
    img = _make_image()
    result = detect_roi("What is in the center of this image?", img)
    assert result.cropped is True
    left, top, right, bottom = result.crop_box
    assert left > 0 and top > 0
    assert right < img.width and bottom < img.height
