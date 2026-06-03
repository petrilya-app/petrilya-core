"""Generate hero images for the landing page.

Aim: a clean iStock-style Petri dish photo.
  * light blue-gray out-of-focus lab background
  * thin glass-like plastic rim with a top highlight
  * pale cream agar — translucent so the background shows through
  * ~70 colonies in a yellow/orange/cream/white palette, varied sizes
    (4-44 px), all lit from the top-left for a single light source
  * thin colored AI detection rings on most colonies
  * four floating ID chips
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


# ----------------------------------------------------------------------
# Palettes (warm yellows / oranges / creams — like the iStock reference)
# ----------------------------------------------------------------------

COLONY_PALETTES = [
    # (mean_rgb, variance, weight)
    ((248, 200,  90), 22, 3.0),   # vivid yellow (dominant)
    ((250, 240, 220), 10, 2.4),   # cream / off-white
    ((230, 150,  60), 22, 2.0),   # orange amber
    ((252, 250, 245),  6, 1.5),   # bright white
    ((200, 100,  55), 22, 0.7),   # red-orange
    ((175, 120,  55), 16, 0.5),   # dark amber / brown
]
_W = np.array([p[2] for p in COLONY_PALETTES])
_W /= _W.sum()

DETECT_PALETTE = [
    (91, 142, 217),    # blue
    (111, 191, 161),   # green
    (170, 130, 220),   # purple
    (110, 180, 220),   # teal
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _lab_background(size: int) -> Image.Image:
    """Clean light blue-gray gradient — soft and unobtrusive."""
    yy = np.linspace(0, 1, size, dtype=np.float32).reshape(-1, 1)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    # cool, almost photo-studio gradient
    arr[..., 0] = 130 + (1 - yy) * 50      # R
    arr[..., 1] = 145 + (1 - yy) * 50      # G
    arr[..., 2] = 165 + (1 - yy) * 55      # B
    bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return bg.filter(ImageFilter.GaussianBlur(25)).convert("RGBA")


def _agar(size: int, cx: int, cy: int, inner_r: int, rng) -> Image.Image:
    """Light cream agar, partially translucent so background shows through."""
    yy, xx = np.indices((size, size), dtype=np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    t = np.clip(dist / inner_r, 0, 1)

    # cream agar with subtle warming toward the centre (lamp under it)
    R = 248 - t * 14
    G = 240 - t * 18
    B = 218 - t * 24
    inside = dist < inner_r
    # alpha gradient — slightly more transparent toward the rim
    A = np.where(inside, 232 - t * 30, 0)

    # very faint grain
    grain = rng.normal(0, 2.5, (size, size))
    R = np.clip(R + grain, 0, 255)
    G = np.clip(G + grain, 0, 255)
    B = np.clip(B + grain, 0, 255)

    rgba = np.stack([R, G, B, A], axis=-1).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _plastic_rim(size: int, cx: int, cy: int, dish_r: int, inner_r: int) -> Image.Image:
    """Thin glass-like plastic ring with a top highlight."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # outer plastic — light, slightly cool
    draw.ellipse(
        (cx - dish_r, cy - dish_r, cx + dish_r, cy + dish_r),
        fill=(210, 216, 220, 175),
    )
    # cut interior
    draw.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        fill=(0, 0, 0, 0),
    )

    # bright highlight arc on the top half (single light source)
    hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hl)
    hd.arc(
        (cx - dish_r + 3, cy - dish_r + 3, cx + dish_r - 3, cy + dish_r - 3),
        start=205, end=335,
        fill=(255, 255, 255, 180),
        width=4,
    )
    # slight darker bottom arc (shadow side)
    hd.arc(
        (cx - dish_r + 3, cy - dish_r + 3, cx + dish_r - 3, cy + dish_r - 3),
        start=20, end=160,
        fill=(40, 50, 60, 75),
        width=3,
    )
    layer = Image.alpha_composite(layer, hl)

    # tiny inner ring shadow at the agar/rim boundary
    inner_shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    isd = ImageDraw.Draw(inner_shadow)
    isd.ellipse(
        (cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r),
        outline=(40, 45, 55, 95),
        width=4,
    )
    inner_shadow = inner_shadow.filter(ImageFilter.GaussianBlur(2))
    layer = Image.alpha_composite(layer, inner_shadow)
    return layer


def _place_colonies(
    cx: int, cy: int, inner_r: int, rng, n_target: int = 80
) -> list[tuple[int, int, int]]:
    """Place colonies with strong size variation and breathing room."""
    margin = 48
    colonies: list[tuple[int, int, int]] = []
    attempts = 0
    while len(colonies) < n_target and attempts < n_target * 60:
        attempts += 1
        angle = rng.uniform(0, 2 * np.pi)
        rad = (inner_r - margin) * np.sqrt(rng.uniform(0, 1))
        x = cx + rad * np.cos(angle)
        y = cy + rad * np.sin(angle)

        s = rng.uniform()
        if s < 0.35:
            r = rng.uniform(4, 8)
        elif s < 0.65:
            r = rng.uniform(8, 16)
        elif s < 0.88:
            r = rng.uniform(16, 26)
        else:
            r = rng.uniform(26, 44)
        r = int(r)

        # require breathing room (no overlap, plus small gap)
        if any(
            (x - px) ** 2 + (y - py) ** 2 < (r + pr + 4) ** 2
            for px, py, pr in colonies
        ):
            continue
        colonies.append((int(x), int(y), r))
    return colonies


def _draw_colonies(
    size: int, colonies: list[tuple[int, int, int]], rng
) -> Image.Image:
    """3D-looking colonies, all lit from top-left."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for (x, y, r) in colonies:
        pidx = int(rng.choice(len(COLONY_PALETTES), p=_W))
        mean_col, var, _ = COLONY_PALETTES[pidx]
        color = np.clip(np.array(mean_col) + rng.integers(-var, var, 3), 0, 255)

        # very subtle outer ring of slower growth (more on bigger colonies)
        if r > 10 and rng.uniform() < 0.55:
            halo_r = int(r * 1.18)
            halo = np.clip(color + 18, 0, 255).astype(int)
            draw.ellipse(
                (x - halo_r, y - halo_r, x + halo_r, y + halo_r),
                fill=(int(halo[0]), int(halo[1]), int(halo[2]), 70),
            )

        # drop shadow under the colony (offset to bottom-right of light)
        draw.ellipse(
            (x - r - 1, y - r + 1.5, x + r + 2, y + r + 4),
            fill=(20, 22, 28, 95),
        )

        # base body — full opacity for crisp 3D look
        draw.ellipse((x - r, y - r, x + r, y + r), fill=tuple(int(v) for v in color) + (255,))

        # darker ring near the edge (real colonies thicken at the boundary)
        if rng.uniform() < 0.45:
            inset = max(1, int(r * 0.18))
            edge = np.clip(color * 0.82, 0, 255).astype(int)
            draw.ellipse(
                (x - r + inset, y - r + inset, x + r - inset, y + r - inset),
                outline=(int(edge[0]), int(edge[1]), int(edge[2]), 130),
                width=max(1, int(r * 0.08)),
            )

        # specular highlight — single light source from top-left
        hr = max(2, int(r * 0.55))
        hx = x - r * 0.36
        hy = y - r * 0.36
        bright = np.clip(color + 45, 0, 255).astype(int)
        draw.ellipse(
            (hx - hr * 0.55, hy - hr * 0.55, hx + hr * 0.55, hy + hr * 0.55),
            fill=(int(bright[0]), int(bright[1]), int(bright[2]), 170),
        )
        # tiny brighter speck inside the highlight
        if r > 7:
            sx = x - r * 0.42
            sy = y - r * 0.42
            sr = max(1, int(r * 0.18))
            draw.ellipse((sx - sr, sy - sr, sx + sr, sy + sr),
                         fill=(255, 255, 255, 140))

    # very slight blur softens edges into the agar
    return layer.filter(ImageFilter.GaussianBlur(0.45))


def _ai_detection_layer(
    size: int, colonies: list[tuple[int, int, int]], rng, detection_rate: float = 0.85
) -> Image.Image:
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for i, (x, y, r) in enumerate(colonies):
        if rng.uniform() > detection_rate:
            continue
        c = DETECT_PALETTE[i % len(DETECT_PALETTE)]
        # ring width scales gently with colony size
        w = 2 if r < 14 else 3
        draw.ellipse(
            (x - r - 2, y - r - 2, x + r + 2, y + r + 2),
            outline=(*c, 230),
            width=w,
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

    # prefer medium-large colonies for labels
    candidates = sorted(range(len(colonies)), key=lambda i: -colonies[i][2])[:int(len(colonies) * 0.5)]
    rng.shuffle(candidates)
    picks: list[int] = []
    for idx in candidates:
        x, y, _ = colonies[idx]
        if all(
            (x - colonies[p][0]) ** 2 + (y - colonies[p][1]) ** 2 > (size * 0.22) ** 2
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
        lx = x + side * (r + 36)
        ly = y - r - 46
        if side == -1:
            lx -= text_w + 22
        chip = (lx - 12, ly - 6, lx + text_w + 12, ly + text_h + 6)
        cx_chip = (chip[0] + chip[2]) / 2
        draw.line(
            (cx_chip, chip[3], x + (r * 0.6 * -side), y - r * 0.4),
            fill=(91, 142, 217, 220),
            width=2,
        )
        draw.rounded_rectangle(
            chip,
            radius=8,
            fill=(13, 17, 24, 235),
            outline=(91, 142, 217, 230),
            width=2,
        )
        draw.text((lx, ly), label, fill=(232, 240, 252, 250), font=font)
    return layer


# ----------------------------------------------------------------------
# Composer
# ----------------------------------------------------------------------

def _compose_dish(size: int, rng, n_colonies: int = 80) -> tuple[Image.Image, list]:
    cx = cy = size // 2
    dish_r = int(size * 0.46)
    inner_r = int(dish_r * 0.94)

    canvas = _lab_background(size)

    # outer drop shadow
    sh = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sh)
    sd.ellipse(
        (cx - dish_r - 6, cy - dish_r + 26, cx + dish_r + 6, cy + dish_r + 38),
        fill=(0, 0, 0, 110),
    )
    sh = sh.filter(ImageFilter.GaussianBlur(24))
    canvas = Image.alpha_composite(canvas, sh)

    canvas = Image.alpha_composite(canvas, _plastic_rim(size, cx, cy, dish_r, inner_r))
    canvas = Image.alpha_composite(canvas, _agar(size, cx, cy, inner_r, rng))

    colonies = _place_colonies(cx, cy, inner_r, rng, n_target=n_colonies)
    canvas = Image.alpha_composite(canvas, _draw_colonies(size, colonies, rng))
    canvas = Image.alpha_composite(canvas, _ai_detection_layer(size, colonies, rng))

    # final glass rim hairline on top
    rim_hl = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    rh = ImageDraw.Draw(rim_hl)
    rh.arc(
        (cx - dish_r + 4, cy - dish_r + 4, cx + dish_r - 4, cy + dish_r - 4),
        start=215, end=325,
        fill=(255, 255, 255, 115),
        width=2,
    )
    canvas = Image.alpha_composite(canvas, rim_hl)

    return canvas, colonies


def make_hero_dish(out_path: Path, size: int = 1200) -> None:
    rng = np.random.default_rng(13)
    cx = cy = size // 2
    dish_r = int(size * 0.46)

    full, colonies = _compose_dish(size, rng, n_colonies=80)

    pad = 16
    final = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse(
        (cx - dish_r - pad, cy - dish_r - pad,
         cx + dish_r + pad, cy + dish_r + pad),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(2))
    final.paste(full, (0, 0), mask)

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

    mini_size = 440
    rng = np.random.default_rng(2)
    mini, _ = _compose_dish(mini_size, rng, n_colonies=55)
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
