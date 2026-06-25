#!/usr/bin/env python3
"""Generate deterministic synthetic Thai OCR test fixtures.

Renders 14 known Thai strings to individual PNG images using a license-clean
Thai font (Noto Sans Thai / Sarabun, both SIL OFL), plus a ``ground_truth.json``
mapping each PNG to its exact source string. The ground truth IS the Python
source string rendered, so the corpus is fully deterministic, offline, and
reproducible.

Thai needs complex shaping (tone marks stack above base consonants, some
vowels reorder), so we render with Pillow's RAQM layout engine when available
and fall back to BASIC layout otherwise (with a printed caveat).

Run from the repo root::

    python3 tests/fixtures/thai/make_thai_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, features

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE / "fonts"
GROUND_TRUTH = HERE / "ground_truth.json"

# Prefer Noto Sans Thai (high-quality reference face), fall back to Sarabun.
# Both are SIL Open Font License (OFL.txt ships next to each in fonts/).
FONT_CANDIDATES = [
    FONT_DIR / "NotoSansThai-VF.ttf",
    FONT_DIR / "Sarabun-Regular.ttf",
]

FONT_SIZE = 44
PADDING = 28
BG_COLOR = (250, 250, 248)
TEXT_COLOR = (15, 15, 20)

# 14 Thai strings spanning three categories. The string is the ground truth.
#   - common words/phrases
#   - tone marks / stacked vowels + diacritics
#   - digit mixing (Thai numerals, Arabic numerals, Thai + English)
STRINGS: list[str] = [
    # common words / phrases
    "สวัสดีครับ",  # hello (polite, male)
    "ขอบคุณมากครับ",  # thank you very much
    "ประเทศไทย",  # Thailand
    "ภาษาไทย",  # Thai language
    "กรุงเทพมหานคร",  # Bangkok (full name)
    "ยินดีต้อนรับ",  # welcome
    # tone marks / stacked vowels + diacritics
    "น้ำ",  # water (stacked sara am + mai tho)
    "ที่นี่",  # here (mai ek on both syllables)
    "ผู้ใหญ่",  # adult / elder (sara uu + tone, mai ek)
    "เพื่อน",  # friend (sara uea + mai ek)
    # digit mixing
    "ปี ๒๕๖๙",  # year 2569 (Thai numerals)
    "ราคา 1,250 บาท",  # price 1,250 baht (Arabic numerals)
    "เบอร์โทร 081-234-5678",  # phone number
    "Thailand ประเทศไทย 2026",  # Thai + English + Arabic numerals
]


def _select_font() -> Path:
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Thai font found in {FONT_DIR}. Expected one of: "
        f"{', '.join(f.name for f in FONT_CANDIDATES)}"
    )


def _layout_engine() -> tuple[int | None, bool]:
    """Return (layout_engine, raqm_available)."""
    if features.check("raqm"):
        return ImageFont.Layout.RAQM, True
    return ImageFont.Layout.BASIC, False


def _render(text: str, font: ImageFont.FreeTypeFont, out: Path) -> None:
    # Measure on a scratch image; anchor at top-left ("la" = left/ascender).
    scratch = Image.new("RGB", (1, 1), BG_COLOR)
    draw = ImageDraw.Draw(scratch)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font, anchor="la")
    text_w = right - left
    text_h = bottom - top

    width = text_w + PADDING * 2
    height = text_h + PADDING * 2
    img = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(img)
    # Offset by (-left, -top) so the text's actual ink box is padded evenly.
    draw.text((PADDING - left, PADDING - top), text, font=font, fill=TEXT_COLOR, anchor="la")
    img.save(out)


def main() -> None:
    assert len(STRINGS) == 14, f"expected 14 strings, got {len(STRINGS)}"

    font_path = _select_font()
    layout_engine, raqm = _layout_engine()
    font = ImageFont.truetype(str(font_path), FONT_SIZE, layout_engine=layout_engine)

    # Remove any stale fixtures for a clean, reproducible run.
    for old in HERE.glob("thai_*.png"):
        old.unlink()

    records: list[dict[str, str]] = []
    for idx, text in enumerate(STRINGS):
        name = f"thai_{idx:02d}.png"
        _render(text, font, HERE / name)
        records.append({"file": name, "text": text})

    GROUND_TRUTH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    engine_name = "RAQM" if raqm else "BASIC"
    print(f"font: {font_path.name}")
    print(f"raqm available: {raqm} (layout engine: {engine_name})")
    if not raqm:
        print(
            "CAVEAT: RAQM not available -- Thai complex shaping (stacked tone "
            "marks / vowel reordering) may be imperfect with BASIC layout."
        )
    print(f"wrote {len(records)} PNGs to {HERE} and {GROUND_TRUTH.name}")


if __name__ == "__main__":
    main()
