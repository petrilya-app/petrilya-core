"""Generate hero images for the landing page.

Produces:
  docs/hero-dish.png    a styled before/after of a petri dish
  docs/og-image.png     a 1200x630 social-share card
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def synthetic_dish(size: int = 900, n_colonies: int = 180, seed: int = 11) -> np.ndarray:
    img = Image.new("L", (size, size), color=232)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = int(size * 0.46)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=210)
    inner = int(radius * 0.93)
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=190)

    rng = np.random.default_rng(seed)
    placed = []
    attempts = 0
    while len(placed) < n_colonies and attempts < n_colonies * 30:
        attempts += 1
        x = rng.integers(cx - inner, cx + inner)
        y = rng.integers(cy - inner, cy + inner)
        if (x - cx) ** 2 + (y - cy) ** 2 > (inner - 22) ** 2:
            continue
        r = int(rng.integers(7, 16))
        if any((x - px) ** 2 + (y - py) ** 2 < (r + pr + 2) ** 2 for px, py, pr in placed):
            continue
        placed.append((int(x), int(y), r))
        shade = int(rng.integers(105, 160))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=shade)

    img = img.filter(ImageFilter.GaussianBlur(0.7))
    return np.array(img), placed


def composite_overlay(base: np.ndarray, colonies: list[tuple[int, int, int]]) -> Image.Image:
    """Apply colored translucent disks over the base photo."""
    rgb = np.stack([base] * 3, axis=-1).astype(np.uint8)
    out = Image.fromarray(rgb).convert("RGBA")
    overlay = Image.new("RGBA", out.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = np.random.default_rng(42)
    palette = [
        (91, 142, 217),   # blue
        (111, 191, 161),  # green
        (218, 145, 88),   # orange
        (200, 100, 130),  # pink
        (170, 130, 220),  # purple
        (110, 180, 220),  # teal
    ]
    for (x, y, r) in colonies:
        color = palette[int(rng.integers(0, len(palette)))]
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(*color, 130))
    out = Image.alpha_composite(out, overlay)
    return out


def crop_circle(img: Image.Image) -> Image.Image:
    """Crop a square image to a centered circle on a dark background."""
    size = img.size[0]
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    bg = Image.new("RGBA", img.size, (15, 17, 21, 255))
    bg.paste(img, (0, 0), mask)
    return bg


def make_hero_dish(out_path: Path) -> None:
    base, colonies = synthetic_dish(size=900, n_colonies=200)
    overlaid = composite_overlay(base, colonies)
    circled = crop_circle(overlaid)
    circled.save(out_path, "PNG", optimize=True)
    print(f"Saved {out_path} ({circled.size[0]}x{circled.size[1]})")


def make_og_image(out_path: Path) -> None:
    """1200x630 social-share image."""
    W, H = 1200, 630
    canvas = Image.new("RGB", (W, H), (15, 17, 21))

    # gradient blob
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    blob_draw = ImageDraw.Draw(blob)
    for i in range(80, 0, -1):
        alpha = int(80 * (1 - i / 80))
        blob_draw.ellipse(
            (W - 400 - i, -200 - i, W + 200 + i, 400 + i),
            fill=(91, 142, 217, alpha // 4),
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), blob).convert("RGB")

    # mini dish
    base, colonies = synthetic_dish(size=440, n_colonies=110, seed=3)
    overlaid = composite_overlay(base, colonies)
    circled = crop_circle(overlaid)
    canvas.paste(circled.convert("RGB"), (W - 480, (H - 440) // 2))

    # text
    draw = ImageDraw.Draw(canvas)
    try:
        font_big = ImageFont.truetype("seguibl.ttf", 64)
        font_sub = ImageFont.truetype("segoeui.ttf", 26)
        font_small = ImageFont.truetype("segoeui.ttf", 22)
    except OSError:
        font_big = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((80, 200), "Petrilya", font=font_big, fill=(230, 233, 239))
    draw.text(
        (80, 290),
        "AI colony counter\nfor biology labs.",
        font=font_sub,
        fill=(180, 188, 200),
        spacing=6,
    )
    draw.text((80, 530), "petrilya.com", font=font_small, fill=(91, 142, 217))

    canvas.save(out_path, "PNG", optimize=True)
    print(f"Saved {out_path} ({W}x{H})")


def main() -> None:
    repo = Path(__file__).resolve().parent.parent
    docs = repo / "docs"
    docs.mkdir(exist_ok=True)
    make_hero_dish(docs / "hero-dish.png")
    make_og_image(docs / "og-image.png")


if __name__ == "__main__":
    main()
