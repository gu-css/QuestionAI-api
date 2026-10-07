#!/usr/bin/env python3
"""Generates valid PNG icon assets for the Chrome Extension.
Authoritative source: ORIGINAL_REQUEST.md AC R3 & PROJECT.md
"""
from pathlib import Path
from PIL import Image, ImageDraw

ICONS_DIR = Path(__file__).resolve().parent.parent / "extension" / "icons"
ICONS_DIR.mkdir(parents=True, exist_ok=True)


def create_icon(size: int, filename: str) -> None:
    """Create a shield/alert icon of specified dimensions in PNG format."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Circular/rounded alert badge background
    margin = max(1, size // 16)
    draw.ellipse([margin, margin, size - margin, size - margin], fill="#E53E3E")

    # Center exclamation mark or alert symbol
    center_x = size // 2
    if size >= 48:
        # Exclamation stem
        line_w = max(2, size // 12)
        top_y = size // 4
        bottom_y = int(size * 0.6)
        draw.line([(center_x, top_y), (center_x, bottom_y)], fill="#FFFFFF", width=line_w)
        # Exclamation dot
        dot_r = max(1, size // 16)
        dot_y = int(size * 0.75)
        draw.ellipse(
            [center_x - dot_r, dot_y - dot_r, center_x + dot_r, dot_y + dot_r],
            fill="#FFFFFF",
        )
    else:
        # High-contrast center mark for 16x16
        inner_m = max(2, size // 4)
        draw.ellipse(
            [inner_m, inner_m, size - inner_m, size - inner_m],
            fill="#FFFFFF",
        )

    out_path = ICONS_DIR / filename
    img.save(out_path, format="PNG")
    print(f"Generated {out_path} ({size}x{size}, {out_path.stat().st_size} bytes)")


def main() -> None:
    create_icon(16, "icon16.png")
    create_icon(48, "icon48.png")
    create_icon(128, "icon128.png")
    print("✅ All extension icons generated successfully!")


if __name__ == "__main__":
    main()
