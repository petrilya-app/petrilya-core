"""Generate marketing-grade screenshots of the Petrilya UI.

Boots the main window with mock-engine data and produces three deliverables:

  1. screenshot-raw.png       Pixel-perfect window grab at the chosen DPR.
                              Use for documentation, GitHub README, issues.

  2. screenshot-framed.png    The raw shot wrapped in a macOS-style title
                              bar with rounded corners and a soft drop
                              shadow, on a transparent background.

  3. screenshot-hero.png      Framed shot composed onto an editorial dark
                              background with a subtle amber + teal glow.
                              Drop directly into the landing hero.
                              Also emitted as .webp for the website.

Typical use
-----------
    # all three artefacts, defaults (1600×1000 @ 2x, sample image)
    petrilya-screenshots

    # use a real dish photo of your own
    petrilya-screenshots --image C:/path/to/dish.jpg

    # only the hero composition
    petrilya-screenshots --scenes hero

    # bigger output (UHD)
    petrilya-screenshots --size 1920x1200 --scale 2

Outputs land in docs/ by default (override with --out).

If you haven't `pip install -e .`'d yet, the module is also runnable as:
    python -m petrilya._tools.screenshots [args]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# HiDPI MUST be set before QApplication is constructed. We pre-seed the
# env var here so an early import of PySide6 elsewhere can't break it.
# QT_ENABLE_HIGHDPI_SCALING=0 disables system display scaling (Windows 125%)
# so QT_SCALE_FACTOR multiplies cleanly — otherwise you get 2 × 1.25 = 2.5x.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "0")
os.environ.setdefault("QT_SCALE_FACTOR", "2")

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[2]   # …/petrilya/_tools/.. → repo
DOCS = REPO / "docs"

# Folders the script will hunt for a sample photo when --image isn't passed.
# Restricted to the repo so we never accidentally pick up a CS2 screenshot
# from the user's desktop or similar unrelated images.
SAMPLE_DIRS: tuple[Path, ...] = (
    REPO / "tests" / "samples",
    REPO,                           # numbered dish photos at repo root
    REPO / "docs",
)

# Filenames in the search dirs that are NOT dish photos (website assets, etc).
_NOT_A_DISH_HINTS = (
    "screenshot", "hero-dish", "og-image", "favicon",
    "colony-", "agar.png", "dish.png",  # decorative website images
)


# ====================================================================== sample
def find_sample_image() -> Path | None:
    """Pick the largest dish-like candidate from the repo's sample dirs."""
    seen: set[Path] = set()
    candidates: list[Path] = []
    for d in SAMPLE_DIRS:
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p in seen or p.is_dir():
                continue
            seen.add(p)
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
                continue
            low = p.name.lower()
            if any(h in low for h in _NOT_A_DISH_HINTS):
                continue
            candidates.append(p)
    # bias toward bigger files — those are usually full-resolution dish photos
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0] if candidates else None


def synthetic_dish(size: int = 900) -> np.ndarray:
    """Render a synthetic petri dish if no real image is available."""
    img = Image.new("L", (size, size), color=235)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = int(size * 0.46)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=210)
    inner = int(radius * 0.93)
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=190)
    rng = np.random.default_rng(7)
    for _ in range(140):
        for _retry in range(30):
            x = int(rng.integers(cx - inner, cx + inner))
            y = int(rng.integers(cy - inner, cy + inner))
            if (x - cx) ** 2 + (y - cy) ** 2 < (inner - 20) ** 2:
                break
        r = int(rng.integers(7, 16))
        shade = int(rng.integers(110, 160))
        draw.ellipse((x - r, y - r, x + r, y + r), fill=shade)
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return np.array(img)


def load_image(path: Path | None) -> np.ndarray:
    if path and path.exists():
        print(f"  image: {path}")
        return np.array(Image.open(path).convert("RGB"))
    found = find_sample_image()
    if found:
        print(f"  image: {found}  (auto-detected)")
        return np.array(Image.open(found).convert("RGB"))
    print("  image: synthetic (no sample found)")
    return synthetic_dish(900)


# ======================================================================= Qt
def boot_and_grab(image_arr: np.ndarray, size: tuple[int, int]) -> Image.Image:
    """Open MainWindow, populate it with mock results, return a PIL grab."""
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from petrilya.metrics.colony import compute_colony_metrics
    from petrilya.ui.main_window import MainWindow
    from petrilya.ui.mock_engine import mock_segment

    app = QApplication.instance() or QApplication(sys.argv)

    win = MainWindow()
    win.resize(*size)
    win.show()
    app.processEvents()

    win.current_image = image_arr
    win.current_image_path = Path("demo.jpg")
    win.canvas.set_image(image_arr)

    # mock_engine wants grayscale
    if image_arr.ndim == 3:
        gray = np.dot(image_arr[..., :3], [0.2126, 0.7152, 0.0722]).astype(np.uint8)
    else:
        gray = image_arr
    masks, _ = mock_segment(gray, n_colonies=200, delay_seconds=0)
    metrics = compute_colony_metrics(masks, scale_um_per_px=None)

    win.canvas.set_masks(masks)
    win.current_metrics = metrics
    win.last_elapsed = 1.18
    win.last_engine_name = "cellpose-cyto3"
    if hasattr(win, "summary_label"):
        win.summary_label.setText(
            f"<b>{len(metrics)}</b> colonies in <b>1.18 s</b><br>"
            f"<span style='color:#8a93a4'>engine: cellpose-cyto3 · unit: px</span>"
        )
    for attr, args in (
        ("_populate_table", (metrics,)),
        ("_enable_exports", (True,)),
    ):
        fn = getattr(win, attr, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception:
                pass
    if hasattr(win, "status_label"):
        win.status_label.setText(f"Done — {len(metrics)} colonies in 1.18 s")

    app.processEvents()
    if hasattr(win.canvas, "fit_to_window"):
        win.canvas.fit_to_window()
    app.processEvents()

    # Enable before/after split at 60% — shows off the slider in the shot.
    if hasattr(win.canvas, "set_split_visible") and hasattr(win.canvas, "set_split_x"):
        try:
            iw = image_arr.shape[1]
            win.canvas.set_split_x(iw * 0.55)
            win.canvas.set_split_visible(True)
        except Exception:
            pass
    app.processEvents()

    pix = win.grab()
    qimg = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    return Image.frombuffer(
        "RGBA", (w, h), bytes(qimg.bits()), "raw", "RGBA", 0, 1
    ).copy()


# ============================================================== composition
def _try_font(size: int) -> ImageFont.ImageFont:
    for cand in ("Inter-Medium.ttf", "Arial.ttf", "arial.ttf",
                 "DejaVuSans.ttf", "Helvetica.ttc", "SegoeUI.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def round_corners(img: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, img.size[0], img.size[1]), radius=radius, fill=255
    )
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(img, (0, 0), mask=mask)
    return out


def add_chrome(shot: Image.Image,
               title: str = "Petrilya — colony counter") -> Image.Image:
    """Glue a macOS-style title bar onto the top of `shot`."""
    bar_h = 56
    w, h = shot.size

    bar = Image.new("RGBA", (w, bar_h), (38, 41, 48, 255))
    bd = ImageDraw.Draw(bar)
    # subtle top sheen + bottom hairline
    bd.line([(0, 0), (w, 0)], fill=(255, 255, 255, 22))
    bd.line([(0, bar_h - 1), (w, bar_h - 1)], fill=(0, 0, 0, 80))
    # traffic lights
    for i, color in enumerate(("#ff5f57", "#febc2e", "#28c840")):
        cx = 34 + i * 30
        cy = bar_h // 2
        r = 10
        bd.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
        # subtle inner highlight
        bd.ellipse((cx - r + 2, cy - r + 2, cx - 3, cy - 3),
                   fill=(255, 255, 255, 60))
    # centred title
    font = _try_font(22)
    tw = bd.textlength(title, font=font)
    bd.text(((w - tw) // 2, bar_h // 2 - 13), title,
            font=font, fill=(180, 184, 190, 255))

    canvas = Image.new("RGBA", (w, h + bar_h), (0, 0, 0, 0))
    canvas.paste(bar, (0, 0))
    canvas.paste(shot, (0, bar_h))
    return canvas


def frame_with_shadow(shot: Image.Image, *, radius: int = 22,
                       shadow_blur: int = 60, shadow_y: int = 40,
                       shadow_alpha: int = 130) -> Image.Image:
    """Round the corners and place on a transparent canvas with a drop shadow."""
    pad = shadow_blur * 2 + shadow_y
    w, h = shot.size
    cw, ch = w + pad * 2, h + pad * 2

    shadow = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        (pad, pad + shadow_y, pad + w, pad + shadow_y + h),
        radius=radius, fill=(0, 0, 0, shadow_alpha),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

    rounded = round_corners(shot, radius)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas = Image.alpha_composite(canvas, shadow)
    canvas.paste(rounded, (pad, pad), rounded)
    return canvas


def make_hero(framed: Image.Image,
              bg: tuple[int, int, int] = (10, 13, 18)) -> Image.Image:
    """Editorial composition: framed shot on a soft amber/teal glow."""
    fw, fh = framed.size
    w = int(fw * 1.18)
    h = int(fh * 1.20)

    canvas = Image.new("RGBA", (w, h), (*bg, 255))

    # amber glow, upper-left
    glow_a = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow_a).ellipse(
        (-w // 3, -h // 3, w // 2, h // 2), fill=(240, 200, 74, 36),
    )
    glow_a = glow_a.filter(ImageFilter.GaussianBlur(140))

    # teal glow, lower-right
    glow_t = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(glow_t).ellipse(
        (w // 2, h // 2, w + w // 3, h + h // 3), fill=(122, 183, 208, 26),
    )
    glow_t = glow_t.filter(ImageFilter.GaussianBlur(160))

    canvas = Image.alpha_composite(canvas, glow_a)
    canvas = Image.alpha_composite(canvas, glow_t)

    x = (w - fw) // 2
    y = (h - fh) // 2
    canvas.paste(framed, (x, y), framed)
    return canvas.convert("RGB")


# =========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate marketing screenshots of the Petrilya UI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--image", type=Path,
                    help="real petri dish photo (jpg/png/tiff)")
    ap.add_argument("--size", default="1600x1000",
                    help="logical window size, e.g. 1600x1000 (default)")
    ap.add_argument("--scale", type=int, default=2, choices=(1, 2, 3),
                    help="device pixel ratio: 2 = retina (default)")
    ap.add_argument("--scenes", default="raw,framed,hero",
                    help="comma-separated subset of raw,framed,hero")
    ap.add_argument("--out", type=Path, default=DOCS,
                    help="output directory (default: docs/)")
    ap.add_argument("--prefix", default="screenshot",
                    help="output filename prefix (default: screenshot)")
    ap.add_argument("--title", default="Petrilya — colony counter",
                    help="title for the macOS chrome bar")
    args = ap.parse_args(argv)

    os.environ["QT_SCALE_FACTOR"] = str(args.scale)

    try:
        w_str, h_str = args.size.lower().split("x")
        size = (int(w_str), int(h_str))
    except (ValueError, AttributeError):
        ap.error(f"--size must look like 1600x1000, got: {args.size!r}")

    scenes = {s.strip() for s in args.scenes.split(",") if s.strip()}
    unknown = scenes - {"raw", "framed", "hero"}
    if unknown:
        ap.error(f"unknown scenes: {', '.join(sorted(unknown))}")

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Booting Petrilya UI at {size[0]}×{size[1]} @ {args.scale}x …")
    arr = load_image(args.image)
    shot = boot_and_grab(arr, size)
    print(f"  captured {shot.width}×{shot.height} px")

    written: list[Path] = []

    if "raw" in scenes:
        p = args.out / f"{args.prefix}-raw.png"
        shot.save(p, optimize=True)
        written.append(p)

    chromed = None
    framed = None
    if {"framed", "hero"} & scenes:
        chromed = add_chrome(shot, title=args.title)
        framed = frame_with_shadow(chromed)
        if "framed" in scenes:
            p = args.out / f"{args.prefix}-framed.png"
            framed.save(p, optimize=True)
            written.append(p)

    if "hero" in scenes:
        assert framed is not None
        hero = make_hero(framed)
        p_png = args.out / f"{args.prefix}-hero.png"
        hero.save(p_png, optimize=True)
        written.append(p_png)

        p_webp = args.out / f"{args.prefix}-hero.webp"
        hero.save(p_webp, quality=88, method=6)
        written.append(p_webp)

    print("\nDone:")
    for p in written:
        try:
            rel = p.relative_to(REPO)
        except ValueError:
            rel = p
        size_kb = p.stat().st_size / 1024
        print(f"  ✓ {rel}   ({size_kb:,.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
