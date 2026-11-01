"""Tests for saliency-based ROI cropping."""

from PIL import Image, ImageDraw

from token0.optimization.saliency import SaliencyResult, apply_saliency_crop, detect_roi


def _make_image(w: int = 800, h: int = 1000) -> Image.Image:
    """A canvas with content everywhere (a 20 px checkerboard), so the content box is the
    whole image and region rules map onto the canvas exactly."""
    img = Image.new("RGB", (w, h), color=(220, 220, 220))
    d = ImageDraw.Draw(img)
    for y in range(0, h, 20):
        for x in range((y // 20) % 2 * 20, w, 40):
            d.rectangle([x, y, x + 19, y + 19], fill=(40, 40, 40))
    return img


def _short_invoice() -> Image.Image:
    """A 1700x2200 white page whose seven text rows end a third of the way down."""
    img = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(img)
    for i in range(7):
        y = 200 + i * 90
        d.rectangle([140, y, 600, y + 30], fill="black")  # label
        d.rectangle([1100, y, 1400, y + 30], fill="black")  # amount
    return img


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


# ---------------------------------------------------------------------------
# Content-aware regions
# ---------------------------------------------------------------------------


def test_bottom_crop_follows_the_content_not_the_blank_page():
    # "total" means the bottom of what's written; on a short invoice the bottom 40% of the
    # page is blank paper, which used to be what got sent.
    img = _short_invoice()
    result = detect_roi("What is the total due on this invoice?", img)
    assert result.cropped
    _, top, _, bottom = result.crop_box
    last_row = 200 + 6 * 90  # the total row, y 740-770
    assert top < last_row and bottom >= last_row + 30
    assert bottom < 1100  # stops near the end of the writing
    cropped = apply_saliency_crop(img, result)
    assert cropped.convert("L").getextrema()[0] < 50  # the crop holds dark text


def test_region_holding_almost_no_content_is_not_cropped():
    # Text in a left column plus a tiny mark at the top right: the top-right region of the
    # content box holds ~0% of the content, so the whole image is kept.
    img = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(img)
    for i in range(18):
        d.rectangle([100, 300 + i * 90, 600, 330 + i * 90], fill="black")
    d.rectangle([1600, 100, 1620, 120], fill="black")
    result = detect_roi("What is the date on this document?", img)
    assert result.cropped is False


def test_blank_image_is_not_cropped():
    result = detect_roi("What is the total?", Image.new("RGB", (800, 1000), "white"))
    assert result.cropped is False
