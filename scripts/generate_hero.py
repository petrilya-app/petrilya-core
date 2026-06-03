"""Generate hero images for the landing page.

Produces:
  docs/hero-dish.png    a photoreal petri dish with AI detection overlay
  docs/og-image.png     a 1200x630 social-share card
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ----------------------------------------------------------------------
# Photoreal Petri dish
# ----------------------------------------------------------------------

PALETTE_DETECT = [
    (91, 142, 217),   # accent blue
    (111, 191, 161),  # accent green
    (218, 145, 88),   # warm orange
    (200, 100, 130),  # muted pink
    (170, 130, 220),  # soft purple
    (110, 180, 220),  # teal
]


def _agar_background(size: int, cx: int, cy: int, dish_r: int, inner_r: int, rng) -> Image.Image:
    """Render the agar interior with a soft radial gradient + film grain."""
    yy, xx = np.ogrid[:size, :size]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    arr = np.full((size, size, 3), 11, dtype=np.uint8)  # outer dark background

    # Agar interior — warm cream with subtle darkening toward the rim
    agar_mask = dist < inner_r
    t = np.clip(dist / inner_r, 0, 1)
    center_col = np.array([222, 218, 202], dtype=np.float32)
    edge_col = np.array([184, 178, 162], dtype=np.float32)
    agar_rgb = center_col * (1 - t[..., None]) + edge_col * t[..., None]

    # Subtle gradient brightening from top-left (lab light from above)
    light = np.clip(1 - ((xx - cx) - (yy - cy)) / (inner_r * 2.5), 0.85, 1.08)
    agar_rgb = np.clip(agar_rgb * light[..., None], 0, 255)

    # Film grain (gaussian noise) — keeps the agar from looking plasticky
    noise = rng.normal(0, 4.5, (size, size, 3))
    agar_rgb = np.clip(agar_rgb + noise, 0, 255).astype(np.uint8)
    arr[agar_mask] = agar_rgb[agar_mask]

    # Dish plastic rim — light, with a subtle inner shadow
    rim_mask = (dist >= inner_r) & (dist < dish_r)
    rim_t = (dist - inner_r) / max(1, dish_r - inner_r)
    rim_inner = np.array([196, 195, 190], dtype=np.float32)
    rim_outer = np.array([238, 236, 232], dtype=np.float32)
    rim_rgb = rim_inner * (1 - rim_t[..., None]) + rim_outer * rim_t[..., None]
    rim_rgb = np.clip(rim_rgb, 0, 255).astype(np.uint8)
    arr[rim_mask] = rim_rgb[rim_mask]

    return Image.fromarray(arr)


def _draw_colonies(
    dish_img: Image.Image,
    cx: int,
    cy: int,
    inner_r: int,
    rng,
    n_target: int = 220,
) -> tuple[Image.Image, list[tuple[int, int, int]]]:
    """Layer realistic colony blobs onto the agar; return list of (x,y,r)."""
    size = dish_img.size[0]
    overlay = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colonies: list[tuple[int, int, int]] = []
    margin = 26
    attempts = 0
    while len(colonies) < n_target and attempts < n_target * 30:
        attempts += 1
        # uniform-area sampling within the agar disc
        angle = rng.uniform(0, 2 * np.pi)
        rad = (inner_r - margin) * np.sqrt(rng.uniform(0, 1))
        x = cx + rad * np.cos(angle)
        y = cy + rad * np.sin(angle)
        r = int(rng.integers(5, 14))
        if any((x - px) ** 2 + (y - py) ** 2 < (r + pr + 1) ** 2 for px, py, pr in colonies):
            continue
        colonies.append((int(x), int(y), r))

        # Natural colony tone: mostly cream/pale-yellow, occasionally pink/tan
        col_pick = rng.uniform()
        if col_pick < 0.55:
            base = np.array([242, 236, 215])  # cream
        elif col_pick < 0.78:
            base = np.array([235, 223, 188])  # pale yellow
        elif col_pick < 0.92:
            base = np.array([220, 205, 188])  # light tan
        else:
            base = np.array([245, 226, 218])  # very faint pink
        base = np.clip(base + rng.integers(-10, 10, size=3), 0, 255)
        c = tuple(int(v) for v in base)

        # slight elliptical asymmetry
        rx = r + rng.integers(-1, 2)
        ry = r + rng.integers(-1, 2)

        # drop shadow
        draw.ellipse(
            (x - rx - 1.4, y - ry + 0.5, x + rx + 1.4, y + ry + 2.5),
            fill=(20, 18, 14, 55),
        )
        # body
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(*c, 252))
        # specular highlight (top-left)
        hx = x - rx * 0.35
        hy = y - ry * 0.35
        hr = max(2.0, rx * 0.55)
        highlight = tuple(int(v) for v in np.clip(base + 25, 0, 255))
        draw.ellipse((hx - hr * 0.55, hy - hr * 0.55, hx + hr * 0.55, hy + hr * 0.55),
                     fill=(*highlight, 110))

    # tiny gaussian blur sells the photographic look
    overlay = overlay.filter(ImageFilter.GaussianBlur(0.55))
    return Image.alpha_composite(dish_img.convert("RGBA"), overlay), colonies


def _ai_detection_layer(size: int, colonies: list[tuple[int, int, int]]) -> Image.Image:
    """Thin colored detection rings (not flat fills) — looks like real CV output."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, (x, y, r) in enumerate(colonies):
        color = PALETTE_DETECT[i % len(PALETTE_DETECT)]
        # super-faint inner tint
        draw.ellipse(
            (x - r - 1, y - r - 1, x + r + 1, y + r + 1),
            fill=(*color, 38),
        )
        # outline
        draw.ellipse(
            (x - r - 1, y - r - 1, x + r + 1, y + r + 1),
            outline=(*color, 235),
            width=2,
        )
    return layer


def _id_labels(
    size: int,
    colonies: list[tuple[int, int, int]],
    rng,
    n_labels: int = 5,
) -> Image.Image:
    """Floating ID chips connected to a handful of colonies."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("segoeui.ttf", 13)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 13)
        except OSError:
            font = ImageFont.load_default()

    # pick well-spaced colonies
    if not colonies:
        return layer
    picks: list[int] = []
    candidates = list(range(len(colonies)))
    rng.shuffle(candidates)
    for idx in candidates:
        x, y, _ = colonies[idx]
        if all(
            (x - colonies[p][0]) ** 2 + (y - colonies[p][1]) ** 2 > (size * 0.18) ** 2
            for p in picks
        ):
            picks.append(idx)
        if len(picks) >= n_labels:
            break

    for idx in picks:
        x, y, r = colonies[idx]
        label = f"#{idx + 1:03d}"
        text_w = draw.textlength(label, font=font)
        text_h = 14
        # offset the chip up-and-right or up-and-left based on position
        side = 1 if x < size * 0.55 else -1
        lx = x + side * (r + 18)
        ly = y - r - 22
        if side == -1:
            lx -= text_w + 12

        chip = (lx - 6, ly - 3, lx + text_w + 6, ly + text_h + 3)
        # connector
        draw.line(
            ((chip[0] + chip[2]) / 2, chip[3], x + (r * 0.6 * -side), y - r * 0.4),
            fill=(91, 142, 217, 180),
            width=1,
        )
        # pill
        draw.rounded_rectangle(
            chip,
            radius=5,
            fill=(15, 19, 26, 215),
            outline=(91, 142, 217, 200),
            width=1,
        )
        draw.text((lx, ly), label, fill=(225, 232, 245, 245), font=font)
    return layer


def make_hero_dish(out_path: Path, size: int = 1100) -> None:
    rng = np.random.default_rng(7)
    cx = cy = size // 2
    dish_r = int(size * 0.46)
    inner_r = int(dish_r * 0.92)

    img = _agar_background(size, cx, cy, dish_r, inner_r, rng)
    img, colonies = _draw_colonies(img, cx, cy, inner_r, rng, n_target=210)
    img = Image.alpha_composite(img, _ai_detection_layer(size, colonies))
    img = Image.alpha_composite(img, _id_labels(size, colonies, rng, n_labels=4))

    # outer vignette ring around the dish for grounding
    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    for i in range(14, 0, -1):
        gd.ellipse(
            (cx - dish_r - i * 3, cy - dish_r - i * 3, cx + dish_r + i * 3, cy + dish_r + i * 3),
            outline=(91, 142, 217, max(0, 6 - i // 2)),
            width=2,
        )
    glow = glow.filter(ImageFilter.GaussianBlur(8))
    img = Image.alpha_composite(glow, img)

    # crop to a square circle on dark background
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    bg = Image.new("RGBA", (size, size), (10, 13, 18, 255))
    bg.paste(img, (0, 0), mask)

    bg.save(out_path, "PNG", optimize=True)
    print(f"Saved {out_path} ({bg.size[0]}x{bg.size[1]})")


# ----------------------------------------------------------------------
# OG image
# ----------------------------------------------------------------------

def _mini_dish(size: int = 480) -> Image.Image:
    rng = np.random.default_rng(3)
    cx = cy = size // 2
    dish_r = int(size * 0.46)
    inner_r = int(dish_r * 0.92)
    img = _agar_background(size, cx, cy, dish_r, inner_r, rng)
    img, colonies = _draw_colonies(img, cx, cy, inner_r, rng, n_target=130)
    img = Image.alpha_composite(img, _ai_detection_layer(size, colonies))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    bg = Image.new("RGBA", (size, size), (10, 13, 18, 255))
    bg.paste(img, (0, 0), mask)
    return bg


def make_og_image(out_path: Path) -> None:
    W, H = 1200, 630
    canvas = Image.new("RGB", (W, H), (10, 13, 18))
    blob = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bd = ImageDraw.Draw(blob)
    for i in range(80, 0, -1):
        alpha = max(0, 80 - i) // 4
        bd.ellipse(
            (W - 400 - i, -200 - i, W + 200 + i, 400 + i),
            fill=(91, 142, 217, alpha),
        )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), blob).convert("RGB")

    mini = _mini_dish(440)
    canvas.paste(mini.convert("RGB"), (W - 480, (H - 440) // 2))

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
