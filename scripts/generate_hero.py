"""Generate hero images for the landing page.

Photorealistic Petri dish: translucent agar over a subtly blurred lab
background bleed, varied colony morphology and color (yellow / amber /
cream / white / red), AI detection overlay, ID labels. Cropped to a
circle with transparency outside.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ----------------------------------------------------------------------
# Palettes
# ----------------------------------------------------------------------

# Realistic colony colors weighted by frequency in nature.
# (mean_rgb, variance, weight)
COLONY_PALETTES = [
    ((232, 205, 100), 22, 3.0),  # vibrant yellow
    ((245, 230, 198), 14, 2.5),  # cream / pale
    ((222, 158, 78),  20, 1.5),  # amber/orange
    ((250, 246, 235),  8, 1.2),  # off-white
    ((198, 142, 70),  16, 0.9),  # dark amber
    ((212, 105, 92),  20, 0.5),  # red-orange
    ((164, 122, 84),  14, 0.4),  # brown
]
_COLONY_WEIGHTS = np.array([p[2] for p in COLONY_PALETTES])
_COLONY_WEIGHTS = _COLONY_WEIGHTS / _COLONY_WEIGHTS.sum()

# Subtle AI overlay rings — kept mostly cool to contrast warm colonies.
PALETTE_DETECT = [
    (91, 142, 217),    # accent blue
    (111, 191, 161),   # accent green
    (170, 130, 220),   # soft purple
    (110, 180, 220),   # teal
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _lab_background(size: int, rng) -> Image.Image:
    """Soft, blurred out-of-focus lab scene that shows through the agar."""
    yy, xx = np.indices((size, size), dtype=np.float32)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    # vertical gradient: lighter near top
    arr[..., 0] = 18 + (yy / size) * 9
    arr[..., 1] = 24 + (yy / size) * 11
    arr[..., 2] = 32 + (yy / size) * 14

    # Diffuse highlights to suggest equipment / light
    glow = np.zeros((size, size), dtype=np.float32)
    for _ in range(14):
        nx = rng.uniform(0, size)
        ny = rng.uniform(0, size)
        nr = rng.uniform(90, 200)
        ni = rng.uniform(10, 26)
        glow += ni * np.exp(-((xx - nx) ** 2 + (yy - ny) ** 2) / (2 * nr ** 2))
    arr[..., 0] += glow * 0.55
    arr[..., 1] += glow * 0.7
    arr[..., 2] += glow * 0.95

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    bg = Image.fromarray(arr).filter(ImageFilter.GaussianBlur(18))
    return bg.convert("RGBA")


def _translucent_agar(
    size: int, cx: int, cy: int, inner_r: int, rng
) -> Image.Image:
    """Semi-transparent warm cream agar layer."""
    yy, xx = np.indices((size, size), dtype=np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(dist / inner_r, 0, 1)

    R = 244 - t * 32
    G = 230 - t * 38
    B = 175 - t * 38
    inside = dist < inner_r
    A = np.where(inside, 200 - t * 32, 0)

    grain = rng.normal(0, 3.5, (size, size))
    R = np.clip(R + grain, 0, 255)
    G = np.clip(G + grain, 0, 255)
    B = np.clip(B + grain, 0, 255)

    rgba = np.stack([R, G, B, A], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _plastic_rim(size: int, cx: int, cy: int, dish_r: int, inner_r: int) -> Image.Image:
    """Light, slightly translucent dish rim with glass highlight."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # outer band — soft gray ring
    draw.ellipse(
        (cx - dish_r, cy - dish_r, cx + dish_r, cy + dish_r),
        fill=(202, 207, 208, 145),
    )
    # inner cutout — agar will fill this
    draw.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        fill=(0, 0, 0, 0),
    )

    # bright highlight arc on top-left of the rim (glass reflection)
    hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hd.arc(
        (cx - dish_r + 3, cy - dish_r + 3, cx + dish_r - 3, cy + dish_r - 3),
        start=205, end=335,
        fill=(255, 255, 255, 150),
        width=3,
    )
    layer = Image.alpha_composite(layer, hl)
    return layer


def _place_colonies(
    cx: int, cy: int, inner_r: int, rng, n_target: int = 175
) -> list[tuple[int, int, int]]:
    margin = 32
    colonies: list[tuple[int, int, int]] = []
    attempts = 0
    while len(colonies) < n_target and attempts < n_target * 30:
        attempts += 1
        angle = rng.uniform(0, 2 * np.pi)
        rad = (inner_r - margin) * np.sqrt(rng.uniform(0, 1))
        x = cx + rad * np.cos(angle)
        y = cy + rad * np.sin(angle)

        s = rng.uniform()
        if s < 0.5:
            r = rng.uniform(4, 9)
        elif s < 0.85:
            r = rng.uniform(9, 16)
        else:
            r = rng.uniform(16, 24)
        r = int(r)

        # allow slight overlap (touching clusters)
        if any((x - px) ** 2 + (y - py) ** 2 < (r + pr - 1) ** 2 for px, py, pr in colonies):
            continue
        colonies.append((int(x), int(y), r))
    return colonies


def _draw_colonies(
    size: int, colonies: list[tuple[int, int, int]], rng
) -> Image.Image:
    """Realistic colored colonies with halo, body, inner ring, highlight."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for (x, y, r) in colonies:
        pidx = int(rng.choice(len(COLONY_PALETTES), p=_COLONY_WEIGHTS))
        mean_col, var, _ = COLONY_PALETTES[pidx]
        color = tuple(int(v) for v in np.clip(
            np.array(mean_col) + rng.integers(-var, var, 3), 0, 255
        ))

        # faint outer halo (zone of growth around some colonies)
        if rng.uniform() < 0.42:
            halo_r = int(r * 1.35)
            halo_col = tuple(int(v) for v in np.clip(np.array(color) + 25, 0, 255))
            draw.ellipse(
                (x - halo_r, y - halo_r, x + halo_r, y + halo_r),
                fill=(*halo_col, 55),
            )

        # drop shadow under the colony body
        draw.ellipse(
            (x - r - 1.4, y - r + 0.4, x + r + 1.4, y + r + 2.6),
            fill=(15, 12, 8, 90),
        )

        # main body
        rx = r + rng.integers(-1, 2)
        ry = r + rng.integers(-1, 2)
        draw.ellipse((x - rx, y - ry, x + rx, y + ry), fill=(*color, 252))

        # darker center ~30% of colonies
        if rng.uniform() < 0.32:
            inner_col = tuple(max(0, int(c * 0.72)) for c in color)
            ir = int(min(rx, ry) * 0.55)
            draw.ellipse((x - ir, y - ir, x + ir, y + ir), fill=(*inner_col, 210))

        # specular highlight top-left
        hr = max(2, int(min(rx, ry) * 0.5))
        hx = x - rx * 0.32
        hy = y - ry * 0.32
        bright = tuple(int(v) for v in np.clip(np.array(color) + 38, 0, 255))
        draw.ellipse(
            (hx - hr * 0.55, hy - hr * 0.55, hx + hr * 0.55, hy + hr * 0.55),
            fill=(*bright, 120),
        )

    return layer.filter(ImageFilter.GaussianBlur(0.55))


def _ai_detection_layer(
    size: int, colonies: list[tuple[int, int, int]], rng, detection_rate: float = 0.85
) -> Image.Image:
    """Thin colored rings on a fraction of colonies — like real CV output."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, (x, y, r) in enumerate(colonies):
        if rng.uniform() > detection_rate:
            continue
        c = PALETTE_DETECT[i % len(PALETTE_DETECT)]
        draw.ellipse(
            (x - r - 1.5, y - r - 1.5, x + r + 1.5, y + r + 1.5),
            outline=(*c, 225),
            width=2,
        )
    return layer


def _id_labels(
    size: int,
    colonies: list[tuple[int, int, int]],
    rng,
    n_labels: int = 4,
) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    try:
        font = ImageFont.truetype("segoeuib.ttf", 22)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except OSError:
            font = ImageFont.load_default()

    if not colonies:
        return layer

    candidates = list(range(len(colonies)))
    rng.shuffle(candidates)
    picks: list[int] = []
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
        text_h = 26
        side = 1 if x < size * 0.55 else -1
        lx = x + side * (r + 30)
        ly = y - r - 40
        if side == -1:
            lx -= text_w + 22
        chip = (lx - 12, ly - 6, lx + text_w + 12, ly + text_h + 6)
        cx_chip = (chip[0] + chip[2]) / 2
        draw.line(
            (cx_chip, chip[3], x + (r * 0.6 * -side), y - r * 0.4),
            fill=(91, 142, 217, 200),
            width=2,
        )
        draw.rounded_rectangle(
            chip,
            radius=8,
            fill=(13, 17, 24, 235),
            outline=(91, 142, 217, 225),
            width=2,
        )
        draw.text((lx, ly), label, fill=(232, 240, 252, 250), font=font)
    return layer


# ----------------------------------------------------------------------
# Main composers
# ----------------------------------------------------------------------

def _compose_dish(size: int, rng, n_colonies: int = 175) -> tuple[Image.Image, list]:
    cx = cy = size // 2
    dish_r = int(size * 0.46)
    inner_r = int(dish_r * 0.93)

    # 1. lab background (will show through translucent agar)
    canvas = _lab_background(size, rng)
    # 2. soft drop shadow under the dish
    sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.ellipse(
        (cx - dish_r - 8, cy - dish_r + 22, cx + dish_r + 8, cy + dish_r + 34),
        fill=(0, 0, 0, 100),
    )
    sh = sh.filter(ImageFilter.GaussianBlur(22))
    canvas = Image.alpha_composite(canvas, sh)
    # 3. plastic rim
    canvas = Image.alpha_composite(canvas, _plastic_rim(size, cx, cy, dish_r, inner_r))
    # 4. translucent agar
    canvas = Image.alpha_composite(canvas, _translucent_agar(size, cx, cy, inner_r, rng))
    # 5. colonies
    colonies = _place_colonies(cx, cy, inner_r, rng, n_target=n_colonies)
    canvas = Image.alpha_composite(canvas, _draw_colonies(size, colonies, rng))
    # 6. AI detection overlay
    canvas = Image.alpha_composite(canvas, _ai_detection_layer(size, colonies, rng))
    # 7. final glass rim highlight on top of everything
    rim_hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rh = ImageDraw.Draw(rim_hl)
    rh.arc(
        (cx - dish_r + 4, cy - dish_r + 4, cx + dish_r - 4, cy + dish_r - 4),
        start=210, end=325,
        fill=(255, 255, 255, 95),
        width=2,
    )
    canvas = Image.alpha_composite(canvas, rim_hl)
    return canvas, colonies


def make_hero_dish(out_path: Path, size: int = 1200) -> None:
    rng = np.random.default_rng(9)
    cx = cy = size // 2
    dish_r = int(size * 0.46)

    full, colonies = _compose_dish(size, rng, n_colonies=180)

    # Clip everything outside the dish (plus a small ring for the shadow)
    pad = 14
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(
        (cx - dish_r - pad, cy - dish_r - pad,
         cx + dish_r + pad, cy + dish_r + pad),
        fill=255,
    )
    # Feather the mask edge a bit so the dish doesn't have a hard outline
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    final.paste(full, (0, 0), mask)

    # Labels are drawn on top of the masked dish so they can extend out
    final = Image.alpha_composite(final, _id_labels(size, colonies, rng, n_labels=4))

    final.save(out_path, "PNG", optimize=True)
    print(f"Saved {out_path} ({final.size[0]}x{final.size[1]})")


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

    # Small dish in the right side, mirroring main hero look
    mini_size = 440
    rng = np.random.default_rng(2)
    mini, _ = _compose_dish(mini_size, rng, n_colonies=110)
    mini_mask = Image.new("L", (mini_size, mini_size), 0)
    ImageDraw.Draw(mini_mask).ellipse((0, 0, mini_size, mini_size), fill=255)
    mini_mask = mini_mask.filter(ImageFilter.GaussianBlur(2))
    mini_final = Image.new("RGBA", (mini_size, mini_size), (0, 0, 0, 0))
    mini_final.paste(mini, (0, 0), mini_mask)
    canvas_rgba = canvas.convert("RGBA")
    canvas_rgba.alpha_composite(mini_final, (W - 480, (H - mini_size) // 2))
    canvas = canvas_rgba.convert("RGB")

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
