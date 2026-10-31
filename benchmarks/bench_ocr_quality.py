"""OCR extraction quality on dense tables and low-contrast UI.

The OCR route replaces an image with the text EasyOCR extracts, so it is only safe
when that text is complete and correct. This benchmark renders images with known
ground truth, runs them through token0's real routing + OCR path, and reports:

- word recall: share of ground-truth words found verbatim in the OCR text
- numeric accuracy: share of numeric cells (amounts, dates, ids) found verbatim
- rows on own line (tables): share of rows output as one line, cells in order
- route: what token0 actually sends for claude-sonnet-4-6 (OCR text or the image)

Usage:
    python benchmarks/bench_ocr_quality.py --render   # regenerate images (needs playwright)
    python benchmarks/bench_ocr_quality.py            # run on the committed images
"""

import argparse
import base64
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images", "ocr_quality")
MODEL = "claude-sonnet-4-6"

FONT = "-apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"


# ---------------------------------------------------------------- ground truth


def table_rows(seed: int, n: int) -> list[list[str]]:
    rng = random.Random(seed)
    names = ["Acme Corp", "Globex", "Initech", "Umbrella", "Hooli", "Stark Ind", "Wayne Ent"]
    status = ["Paid", "Pending", "Overdue", "Refunded"]
    rows = [["Invoice", "Date", "Customer", "Qty", "Unit", "Amount", "Tax", "Status"]]
    for _ in range(n):
        qty = rng.randint(1, 240)
        unit = rng.randint(100, 99999) / 100
        amount = qty * unit
        rows.append(
            [
                f"INV-{rng.randint(10000, 99999)}",
                f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                rng.choice(names),
                str(qty),
                f"{unit:,.2f}",
                f"{amount:,.2f}",
                f"{amount * 0.18:,.2f}",
                rng.choice(status),
            ]
        )
    return rows


UI_TEXT = [
    "Workspace settings",
    "Workspace name",
    "acme-prod-eu",
    "Shown on invoices and in email footers",
    "Billing email",
    "billing@acme.example",
    "Default currency",
    "EUR",
    "Webhook endpoint",
    "https://api.acme.example/hooks/payments",
    "Last delivery 14 Mar 2026 09:41",
    "API keys",
    "sk_live_7f3a",
    "Created 12 Mar 2026",
    "Last used 2 hours ago",
    "Active",
    "sk_test_19c0",
    "Created 03 Jan 2026",
    "Never used",
    "Revoked",
    "Cancel",
    "Save changes",
    "Delete workspace",
    "This cannot be undone",
]


def contrast_ratio(fg: str, bg: str) -> float:
    def lum(hex_color: str) -> float:
        c = [int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# --------------------------------------------------------------------- render


def table_html(rows: list[list[str]], font_px: int) -> str:
    head = "".join(f"<th>{c}</th>" for c in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
    return f"""<html><body style="margin:16px;background:#fff;font:{font_px}px {FONT};color:#222">
<table style="border-collapse:collapse">
<style>td,th{{border:1px solid #ddd;padding:3px 8px;text-align:left;white-space:nowrap}}
th{{background:#f4f4f4}}</style>
<tr>{head}</tr>{body}</table></body></html>"""


def ui_html(fg: str, bg: str, panel: str) -> str:
    t = UI_TEXT

    def field(label: str, value: str, hint: str = "") -> str:
        hint_div = f'<div style="font-size:12px;margin-top:4px">{hint}</div>' if hint else ""
        return (
            f'<div style="margin:14px 0"><div style="font-size:13px">{label}</div>'
            f'<div style="border:1px solid {fg}55;background:{panel};padding:7px 10px;'
            f'margin-top:4px;font-size:14px;width:380px">{value}</div>{hint_div}</div>'
        )

    def key(k: str, created: str, used: str, status: str) -> str:
        badge = "border:1px solid " + fg + ";padding:0 6px;border-radius:9px;font-size:12px"
        return (
            f'<div style="display:flex;gap:28px;font-size:13px;padding:8px 0;'
            f'border-bottom:1px solid {fg}33"><span>{k}</span><span>{created}</span>'
            f'<span>{used}</span><span style="{badge}">{status}</span></div>'
        )

    return f"""<html><body style="margin:0;background:{bg};color:{fg};font-family:{FONT}">
<div style="padding:28px 36px;width:640px">
<div style="font-size:20px;font-weight:600;margin-bottom:10px">{t[0]}</div>
{field(t[1], t[2], t[3])}{field(t[4], t[5])}{field(t[6], t[7])}{field(t[8], t[9], t[10])}
<div style="font-size:16px;font-weight:600;margin:24px 0 6px">{t[11]}</div>
{key(t[12], t[13], t[14], t[15])}{key(t[16], t[17], t[18], t[19])}
<div style="margin-top:24px;display:flex;gap:12px;font-size:14px">
<span style="border:1px solid {fg}88;padding:7px 14px">{t[20]}</span>
<span style="border:1px solid {fg}88;padding:7px 14px;font-weight:600">{t[21]}</span></div>
<div style="margin-top:28px;font-size:14px;opacity:.6">{t[22]}</div>
<div style="font-size:12px;opacity:.6">{t[23]}</div>
</div></body></html>"""


CASES_TABLE = [  # (name, font px, rows, device scale)
    ("table_11px_1x", 11, 40, 1),
    ("table_13px_1x", 13, 30, 1),
    ("table_15px_1x", 15, 25, 1),
    ("table_11px_2x", 11, 40, 2),
    ("table_13px_2x", 13, 30, 2),
]

CASES_UI = [  # (name, fg, bg, panel, device scale)
    ("ui_light_7.0", "#595959", "#ffffff", "#ffffff", 1),
    ("ui_light_4.5", "#767676", "#ffffff", "#ffffff", 1),
    ("ui_light_3.0", "#949494", "#ffffff", "#ffffff", 1),
    ("ui_light_2.0", "#b3b3b3", "#ffffff", "#ffffff", 1),
    ("ui_light_1.5", "#cbcbcb", "#ffffff", "#ffffff", 1),
    ("ui_dark_7.0", "#b4b4b4", "#1e1e1e", "#262626", 1),
    ("ui_dark_4.5", "#8f8f8f", "#1e1e1e", "#262626", 1),
    ("ui_dark_3.0", "#717171", "#1e1e1e", "#262626", 1),
    ("ui_dark_2.0", "#585858", "#1e1e1e", "#262626", 1),
    ("ui_dark_1.5", "#454545", "#1e1e1e", "#262626", 1),
    ("ui_light_3.0_2x", "#949494", "#ffffff", "#ffffff", 2),
    ("ui_dark_3.0_2x", "#717171", "#1e1e1e", "#262626", 2),
]


def render() -> None:
    from playwright.sync_api import sync_playwright

    os.makedirs(IMAGES_DIR, exist_ok=True)
    truth = {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        jobs = []
        for i, (name, px, n, scale) in enumerate(CASES_TABLE):
            rows = table_rows(seed=i % 3, n=n)
            jobs.append((name, table_html(rows, px), scale, {"kind": "table", "rows": rows}))
        for name, fg, bg, panel, scale in CASES_UI:
            meta = {"kind": "ui", "lines": UI_TEXT, "contrast": round(contrast_ratio(fg, bg), 2)}
            jobs.append((name, ui_html(fg, bg, panel), scale, meta))
        for name, html, scale, meta in jobs:
            page = browser.new_page(
                device_scale_factor=scale, viewport={"width": 760, "height": 400}
            )
            page.set_content(html)
            page.screenshot(path=os.path.join(IMAGES_DIR, f"{name}.png"), full_page=True)
            page.close()
            truth[name] = meta
            print(f"  rendered {name}")
        browser.close()
    with open(os.path.join(IMAGES_DIR, "truth.json"), "w") as f:
        json.dump(truth, f, indent=1)


# -------------------------------------------------------------------- measure

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._:/,-]*[A-Za-z0-9]|[A-Za-z0-9]")


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def recall(expected: list[str], got: list[str]) -> float:
    have = Counter(got)
    hit = 0
    for w in expected:
        if have[w] > 0:
            have[w] -= 1
            hit += 1
    return hit / len(expected) if expected else 1.0


def row_integrity(rows: list[list[str]], lines: list[str]) -> float:
    """Share of table rows that come out on their own line, with cells in order."""
    ids = [r[0] for r in rows[1:]]
    ok = 0
    for row in rows[1:]:
        line = next((ln for ln in lines if row[0] in ln), None)
        if line is None or sum(i in line for i in ids) != 1:
            continue
        pos = 0
        for cell in row:
            idx = line.find(cell, pos)
            if idx < 0:
                break
            pos = idx + len(cell)
        else:
            ok += 1
    return ok / len(ids)


def measure() -> list[dict]:
    from token0.optimization.analyzer import analyze_image
    from token0.optimization.router import plan_optimization
    from token0.optimization.transformer import _ocr_extract, resolve_ocr

    with open(os.path.join(IMAGES_DIR, "truth.json")) as f:
        truth = json.load(f)

    out = []
    for name, meta in truth.items():
        raw = open(os.path.join(IMAGES_DIR, f"{name}.png"), "rb").read()
        analysis, raw_bytes, _ = analyze_image(base64.b64encode(raw).decode())
        plan, result = resolve_ocr(
            plan_optimization(analysis, MODEL),
            raw_bytes,
            lambda a=analysis: plan_optimization(a, MODEL, allow_ocr=False),
        )
        # Score the OCR text whether or not token0 chose it, so quality is visible
        # even for images routed as pixels.
        text = (result or _ocr_extract(raw_bytes))["content"]
        got = words(text)

        if meta["kind"] == "table":
            cells = [c for r in meta["rows"] for c in r]
            numeric = [c for c in cells if any(ch.isdigit() for ch in c)]
            expected = [w for c in cells for w in words(c)]
            row = {
                "word_recall": recall(expected, got),
                "numeric_exact": recall(numeric, got),
                "row_integrity": row_integrity(meta["rows"], text.split("\n")),
            }
        else:
            expected = [w for line in meta["lines"] for w in words(line)]
            row = {
                "contrast": meta["contrast"],
                "word_recall": recall(expected, got),
                "lines_exact": sum(line in text for line in meta["lines"]) / len(meta["lines"]),
            }
        row.update(
            name=name,
            size=f"{analysis.width}x{analysis.height}",
            text_density=round(analysis.text_density, 2),
            route="ocr" if plan.use_ocr_route else "image",
            low_confidence=round((result or {}).get("low_confidence_share", -1), 2),
            tokens_image=plan_optimization(analysis, MODEL, allow_ocr=False).estimated_tokens_after,
            tokens_sent=plan.estimated_tokens_after,
        )
        out.append(row)
        print(f"  {name}: {row}", flush=True)
    return out


def report(rows: list[dict]) -> str:
    pct = lambda x: f"{x * 100:.0f}%"  # noqa: E731
    lines = [
        "### Dense tables",
        "",
        "| Image | Size | Word recall | Numeric cells exact | Rows on own line | token0 route |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        if "numeric_exact" in r:
            lines.append(
                f"| {r['name']} | {r['size']} | {pct(r['word_recall'])} | "
                f"{pct(r['numeric_exact'])} | {pct(r['row_integrity'])} | {r['route']} |"
            )
    lines += [
        "",
        "### Low-contrast UI",
        "",
        "| Image | Contrast | Word recall | Lines exact | token0 route |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if "contrast" in r:
            lines.append(
                f"| {r['name']} | {r['contrast']}:1 | {pct(r['word_recall'])} | "
                f"{pct(r['lines_exact'])} | {r['route']} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()
    if args.render:
        render()
    results = measure()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "ocr_quality.json"), "w") as f:
        json.dump(results, f, indent=1)
    print()
    print(report(results))
