"""Generate a marketing screenshot of the running UI with mock data.

Loads a real test image (or generates a synthetic petri-dish-looking one
if no image is supplied), runs the mock segmentation, and saves a PNG
of the full main window to docs/screenshot.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication

from petrilya.metrics.colony import compute_colony_metrics
from petrilya.ui.main_window import MainWindow
from petrilya.ui.mock_engine import mock_segment


def synthetic_dish(size: int = 900) -> np.ndarray:
    """Render a synthetic petri dish photo if no real one is available."""
    img = Image.new("L", (size, size), color=235)
    draw = ImageDraw.Draw(img)
    cx = cy = size // 2
    radius = int(size * 0.46)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=210)
    inner = int(radius * 0.93)
    draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=190)

    rng = np.random.default_rng(7)
    for _ in range(140):
        while True:
            x = rng.integers(cx - inner, cx + inner)
            y = rng.integers(cy - inner, cy + inner)
            if (x - cx) ** 2 + (y - cy) ** 2 < (inner - 20) ** 2:
                break
        r = rng.integers(7, 16)
        shade = rng.integers(110, 160)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=int(shade))

    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return np.array(img)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    image_arg = sys.argv[1] if len(sys.argv) > 1 else None

    if image_arg and Path(image_arg).exists():
        arr = np.array(Image.open(image_arg).convert("L"))
    else:
        arr = synthetic_dish(900)

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.resize(1400, 880)
    window.show()
    app.processEvents()

    window.current_image = arr
    window.current_image_path = Path("demo.jpg")
    window.canvas.set_image(arr)

    masks, _ = mock_segment(arr, n_colonies=140, delay_seconds=0)
    metrics = compute_colony_metrics(masks, scale_um_per_px=None)

    window.canvas.set_masks(masks)
    window.current_metrics = metrics
    window.last_elapsed = 1.18
    window.last_engine_name = "cellpose-cyto3"
    window.summary_label.setText(
        f"<b>{len(metrics)}</b> colonies found in <b>1.18s</b><br>"
        f"<span style='color:#aaa'>engine: cellpose-cyto3 | unit: px</span>"
    )
    window._populate_table(metrics)
    window._enable_exports(True)
    window.status_label.setText(f"Done — {len(metrics)} colonies in 1.18s")

    app.processEvents()
    window.canvas.fit_to_window()
    app.processEvents()

    out = repo_root / "docs" / "screenshot.png"
    pix: QPixmap = window.grab()
    pix.save(str(out), "PNG")
    print(f"Saved {out} ({pix.width()}x{pix.height()})")


if __name__ == "__main__":
    main()
